"""
Parameterized vacancy formation driver (ported from ex_scripts Step3-1_V).

Ef^V(A) = E_vac - E_bulk + mu_A, with mu/masses from Family config.
"""

from __future__ import annotations

import csv
import os
import random
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .lammps_cell import (
    Cell,
    die,
    final_energy,
    find_potential,
    nearest_neighbors,
    parse_lammps_data,
    write_lammps_data,
    write_minimize_input,
)

DEFAULT_SAMPLES_PER_ELEMENT = 125
DEFAULT_SITE_SEED = 12345
MAX_RANKS = 13
SPARE_CORES = 3
CSV_NAME = "v_form_e.csv"
CASES_DIRNAME = "v_cases"


@dataclass(frozen=True)
class VacancyRunConfig:
    aged: Path
    pot_dir: Path
    species: tuple[str, ...]
    masses: Mapping[str, float]
    mu_eV: Mapping[str, float]
    lmp: str = "lmp"
    samples_per_element: int = DEFAULT_SAMPLES_PER_ELEMENT
    site_seed: int = DEFAULT_SITE_SEED
    nworkers: Optional[int] = None
    expected_natoms: Optional[int] = 5488
    family_name: str = ""
    skip_bulk: bool = False


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    if not raw.isdigit():
        die(f"{name} must be a non-negative integer, got {raw!r}")
    return int(raw)


def mass_to_element_map(masses: Mapping[str, float]) -> dict[float, str]:
    return {float(v): str(k) for k, v in masses.items()}


def csv_fields(species: Sequence[str]) -> list[str]:
    nn_cols = [f"nn_{el}" for el in species]
    return [
        "site_id",
        "removed_element",
        "element_nearest_neighbors",
        *nn_cols,
        "E_bulk_eV",
        "E_vacancy_eV",
        "mu_eV",
        "formation_energy_vacancy_eV",
    ]


def nn_counts(cell: Cell, neighbor_indexes: list[int], species: Sequence[str]) -> dict[str, int]:
    counts = {name: 0 for name in species}
    for j in neighbor_indexes:
        element = cell.masses[cell.atoms[j][1]]
        if element in counts:
            counts[element] += 1
    return counts


def nn_string(counts: dict[str, int], species: Sequence[str]) -> str:
    return " ".join(f"{name}:{counts[name]}" for name in species)


def host_cores() -> int:
    cores = os.cpu_count()
    if cores is None or cores < 1:
        return 1
    return cores


def resolve_workers(requested: Optional[int] = None) -> int:
    cores = host_cores()
    limit = cores - SPARE_CORES
    if limit < 1:
        limit = 1
    if limit > MAX_RANKS:
        limit = MAX_RANKS
    if requested is None:
        nworkers = env_int("NWORKERS", limit)
    else:
        nworkers = int(requested)
    if nworkers < 1:
        die("NWORKERS must be >= 1")
    if nworkers > limit:
        print(
            f"WARNING: NWORKERS={nworkers} would fill the host; using {limit} ranks.",
            flush=True,
        )
        nworkers = limit
    return nworkers


def rank_line(nworkers: int) -> str:
    cores = host_cores()
    idle = max(0, cores - nworkers)
    return f"{nworkers} serial lmp tasks, one core each, {idle} cores left idle"


_CORE_LOCK = threading.Lock()
_FREE_CORES: list[int] | None = None


def reset_core_pool(nworkers: int) -> None:
    global _FREE_CORES
    with _CORE_LOCK:
        _FREE_CORES = list(range(nworkers))


def take_core() -> int:
    global _FREE_CORES
    with _CORE_LOCK:
        if _FREE_CORES is None:
            die("core pool not initialized")
        if not _FREE_CORES:
            die("no free core for a LAMMPS task")
        return _FREE_CORES.pop()


def give_core(core: int) -> None:
    global _FREE_CORES
    with _CORE_LOCK:
        if _FREE_CORES is None:
            _FREE_CORES = []
        _FREE_CORES.append(core)


def lammps_child_env() -> dict:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["OMP_PROC_BIND"] = "false"
    for name in ("GOMP_CPU_AFFINITY", "KMP_AFFINITY", "OMP_PLACES"):
        env.pop(name, None)
    return env


def lammps_argv(lmp: str, input_name: str, core: int) -> list[str]:
    """Serial lmp; pin with taskset when available (Linux)."""
    base = [lmp, "-in", input_name, "-log", "log.lammps"]
    if shutil.which("taskset"):
        return ["taskset", "-c", str(core), *base]
    return base


def run_lammps(work: Path, lmp: str, input_name: str = "in.lmp") -> float:
    log_path = work / "log.lammps"
    screen_path = work / "screen.log"
    core = take_core()
    argv = lammps_argv(lmp, input_name, core)
    try:
        with screen_path.open("w", encoding="utf-8") as screen:
            try:
                proc = subprocess.run(
                    argv,
                    cwd=str(work),
                    env=lammps_child_env(),
                    stdout=screen,
                    stderr=subprocess.STDOUT,
                )
            except FileNotFoundError:
                raise RuntimeError(f"cannot start {argv[0]!r} for LAMMPS in {work}")
    finally:
        give_core(core)
    if not log_path.is_file():
        raise RuntimeError(
            f"LAMMPS wrote no log in {work} (exit {proc.returncode}). See {screen_path}"
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"LAMMPS failed in {work} (exit {proc.returncode}). See {log_path}")
    return final_energy(text, work=work)


def load_done(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    done = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add(int(row["site_id"]))
    return done


def sample_indexes_per_element(
    bulk: Cell,
    *,
    species: Sequence[str],
    n_per_element: int,
    done: set[int],
    rng: random.Random,
) -> list[int]:
    pools: dict[str, list[int]] = {el: [] for el in species}
    for index, atom in enumerate(bulk.atoms):
        site_id = atom[0]
        if site_id in done:
            continue
        element = bulk.masses[atom[1]]
        if element in pools:
            pools[element].append(index)
    pending: list[int] = []
    for element in species:
        pool = pools[element]
        rng.shuffle(pool)
        if len(pool) < n_per_element:
            die(
                f"need {n_per_element} unused {element} sites, only {len(pool)} left "
                f"(done={len(done)})"
            )
        chosen = pool[:n_per_element]
        pending.extend(chosen)
        print(
            f"sample {element}: {n_per_element} sites (pool_available={len(pool)})",
            flush=True,
        )
    rng.shuffle(pending)
    return pending


def relax_bulk(
    work_dir: Path,
    aged: Path,
    tabgap: Path,
    eam: Path,
    lmp: str,
    *,
    masses: Mapping[str, float],
    expected_natoms: Optional[int],
    family_name: str,
    skip_bulk: bool,
) -> tuple[Cell, float]:
    bulk_dir = work_dir / "bulk"
    bulk_dir.mkdir(exist_ok=True)
    m2e = mass_to_element_map(masses)
    start = parse_lammps_data(aged, m2e)
    if expected_natoms is not None and len(start.atoms) != int(expected_natoms):
        die(
            f"aged cell has {len(start.atoms)} atoms, expected {expected_natoms} "
            f"(got {aged})"
        )
    title = f"{family_name or 'alloy'} formation-energy cell"
    if skip_bulk and (bulk_dir / "relaxed.data").is_file() and (bulk_dir / "log.lammps").is_file():
        bulk = parse_lammps_data(bulk_dir / "relaxed.data", m2e)
        e_bulk = final_energy(
            (bulk_dir / "log.lammps").read_text(encoding="utf-8", errors="replace"),
            work=bulk_dir,
        )
        print(f"SKIP_BULK: reuse E_bulk = {e_bulk:.6f} eV for {len(bulk.atoms)} atoms", flush=True)
        return bulk, e_bulk

    write_lammps_data(bulk_dir / "start.data", start, start.atoms, element_masses=masses, title=title)
    write_minimize_input(
        bulk_dir / "in.lmp", "start.data", start, tabgap, eam, relax_box=True
    )
    print("Minimizing the aged cell (this loads the potential once).", flush=True)
    try:
        e_bulk = run_lammps(bulk_dir, lmp)
    except RuntimeError as exc:
        die(str(exc))
    bulk = parse_lammps_data(bulk_dir / "relaxed.data", m2e)
    print(f"E_bulk = {e_bulk:.6f} eV for {len(bulk.atoms)} atoms", flush=True)
    return bulk, e_bulk


def prepare_vacancy_jobs(
    work_dir: Path,
    bulk: Cell,
    pending: list[int],
    tabgap: Path,
    eam: Path,
    *,
    species: Sequence[str],
    masses: Mapping[str, float],
    mu_eV: Mapping[str, float],
    family_name: str,
):
    site_meta: dict[int, dict] = {}
    jobs: list[tuple[str, int, Path]] = []
    title = f"{family_name or 'alloy'} vacancy cell"
    for index in pending:
        atom = bulk.atoms[index]
        site_id = atom[0]
        element = bulk.masses[atom[1]]
        counts = nn_counts(bulk, nearest_neighbors(bulk, index), species)
        if element not in mu_eV:
            die(f"no mu configured for removed element {element!r}")
        mu = float(mu_eV[element])
        site_meta[site_id] = {"element": element, "counts": counts, "mu": mu}

        vac_dir = work_dir / CASES_DIRNAME / f"vac_{site_id}"
        vac_dir.mkdir(parents=True, exist_ok=True)
        vacancy_atoms = [other for other in bulk.atoms if other[0] != site_id]
        write_lammps_data(
            vac_dir / "start.data", bulk, vacancy_atoms, element_masses=masses, title=title
        )
        write_minimize_input(
            vac_dir / "in.lmp", "start.data", bulk, tabgap, eam, relax_box=False
        )
        jobs.append(("vac", site_id, vac_dir))
    return site_meta, jobs


def run_vacancy_jobs(
    work_dir: Path,
    bulk_energy: float,
    pending: list[int],
    site_meta: dict,
    jobs: list,
    lmp: str,
    nworkers: int,
    *,
    species: Sequence[str],
) -> dict[str, object]:
    csv_path = work_dir / CSV_NAME
    new_file = not csv_path.is_file()
    fields = csv_fields(species)
    csv_lock = threading.Lock()
    n_done = 0
    n_jobs_done = 0
    failures: list[str] = []
    e_bulk = bulk_energy

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
            handle.flush()

        with ThreadPoolExecutor(max_workers=nworkers) as pool:
            future_map = {
                pool.submit(run_lammps, work_dir_job, lmp): (kind, site_id, work_dir_job)
                for kind, site_id, work_dir_job in jobs
            }
            for fut in as_completed(future_map):
                kind, site_id, work_dir_job = future_map[fut]
                try:
                    energy = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{kind} site {site_id} ({work_dir_job}): {exc}")
                    print(f"FAIL {kind} site {site_id}: {exc}", flush=True)
                    continue
                n_jobs_done += 1
                meta = site_meta[site_id]
                element = meta["element"]
                counts = meta["counts"]
                mu = meta["mu"]
                e_vac = energy
                e_f_v = e_vac - e_bulk + mu
                row = {
                    "site_id": site_id,
                    "removed_element": element,
                    "element_nearest_neighbors": nn_string(counts, species),
                    "E_bulk_eV": f"{e_bulk:.8f}",
                    "E_vacancy_eV": f"{e_vac:.8f}",
                    "mu_eV": f"{mu:.6f}",
                    "formation_energy_vacancy_eV": f"{e_f_v:.8f}",
                }
                for el in species:
                    row[f"nn_{el}"] = counts[el]
                with csv_lock:
                    writer.writerow(row)
                    handle.flush()
                    n_done += 1
                print(
                    f"site {site_id} {element}  NN {nn_string(counts, species)}  "
                    f"Ef_V={e_f_v:.3f}  ({n_done}/{len(pending)})",
                    flush=True,
                )
                if n_jobs_done % max(1, nworkers) == 0 or n_jobs_done == len(jobs):
                    print(f"jobs finished {n_jobs_done}/{len(jobs)}", flush=True)

    if failures:
        die(
            f"{len(failures)} LAMMPS job(s) failed; {n_done}/{len(pending)} sites written. "
            f"First error: {failures[0]}"
        )
    print(f"Wrote {csv_path} ({n_done} sites)", flush=True)
    return {"csv": str(csv_path), "n_done": n_done, "n_jobs": len(jobs)}


def run_vacancy(work_dir: Path, cfg: VacancyRunConfig) -> dict[str, object]:
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    aged = Path(cfg.aged).expanduser().resolve()
    pot_dir = Path(cfg.pot_dir).expanduser().resolve()
    lmp = cfg.lmp
    species = tuple(cfg.species)
    masses = {str(k): float(v) for k, v in dict(cfg.masses).items()}
    mu_eV = {str(k): float(v) for k, v in dict(cfg.mu_eV).items()}

    if os.environ.get("SAMPLES_PER_ELEMENT", "").strip():
        n_per_element = env_int("SAMPLES_PER_ELEMENT", cfg.samples_per_element)
    elif os.environ.get("MAX_SITES", "").strip():
        total = env_int("MAX_SITES", cfg.samples_per_element * len(species))
        n_per_element = max(1, total // len(species))
        print(
            f"NOTE: MAX_SITES={total} → SAMPLES_PER_ELEMENT={n_per_element}",
            flush=True,
        )
    else:
        n_per_element = int(cfg.samples_per_element)
    site_seed = env_int("SITE_SEED", cfg.site_seed)
    nworkers = resolve_workers(cfg.nworkers)

    if not aged.is_file():
        die(f"missing aged cell: {aged}")
    local_aged = root / "aged.data"
    if aged.resolve() != local_aged.resolve():
        shutil.copy2(aged, local_aged)

    tabgap, eam = find_potential(pot_dir)
    reset_core_pool(nworkers)

    bulk, e_bulk = relax_bulk(
        root,
        local_aged if local_aged.is_file() else aged,
        tabgap,
        eam,
        lmp,
        masses=masses,
        expected_natoms=cfg.expected_natoms,
        family_name=cfg.family_name,
        skip_bulk=cfg.skip_bulk,
    )

    csv_path = root / CSV_NAME
    done = load_done(csv_path)
    rng = random.Random(site_seed)
    pending = sample_indexes_per_element(
        bulk,
        species=species,
        n_per_element=n_per_element,
        done=done,
        rng=rng,
    )
    print(
        f"Sampling {len(pending)} vacancies "
        f"({n_per_element} per element × {len(species)}; "
        f"SITE_SEED={site_seed}, already_done={len(done)})",
        flush=True,
    )
    print(f"Parallel LAMMPS: {rank_line(nworkers)} ({len(pending)} minimizations).", flush=True)

    site_meta, jobs = prepare_vacancy_jobs(
        root,
        bulk,
        pending,
        tabgap,
        eam,
        species=species,
        masses=masses,
        mu_eV=mu_eV,
        family_name=cfg.family_name,
    )
    job_stats = run_vacancy_jobs(
        root,
        e_bulk,
        pending,
        site_meta,
        jobs,
        lmp,
        nworkers,
        species=species,
    )
    return {
        "status": "ok",
        "e_bulk": e_bulk,
        "n_atoms": len(bulk.atoms),
        "n_samples": len(pending),
        "samples_per_element": n_per_element,
        "nworkers": nworkers,
        **job_stats,
    }
