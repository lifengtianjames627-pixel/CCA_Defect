"""
Parameterized SIA anneal driver (ported from ex_scripts Step3-2_SIA).

Parts 1–2 + driver: Zhou-style random insertion, anneal, Ef^SIA.
Element-pair chemistry labeling via ``sia_pair`` (no geometry / orientation).
"""

from __future__ import annotations

import csv
import math
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from . import vacancy_driver as vac
from .lammps_cell import Cell, find_potential, parse_lammps_data, write_lammps_data
from .sia_pair import label_element_pair, summarize_pairs

DEFAULT_ANNEAL_K = 1000.0
DEFAULT_INSERTS_PER_ELEMENT = 200
DEFAULT_MD_STEPS = 5000
DEFAULT_COOL_STEPS = 20000
MIN_INSERT_A = 0.8
MAX_INSERT_A = 1.4
TOO_CLOSE_A = 0.7
PLACEMENT_TRIES = 200

# Geometry / orientation intentionally omitted.
SIA_FIELDS = [
    "job",
    "anneal_K",
    "site_id",
    "host_element",
    "inserted_element",
    "insert_x",
    "insert_y",
    "insert_z",
    "E_bulk_eV",
    "E_SIA_eV",
    "mu_eV",
    "formation_energy_SIA_eV",
    "pair",
    "element_a",
    "element_b",
    "class_key",
    "status",
    "note",
]

PAIR_SUMMARY_FIELDS = [
    "pair",
    "class_key",
    "count",
    "E_min_eV",
    "E_median_eV",
    "E_max_eV",
]

# Hard exclusions from the template Part-3 geometry classifier.
EXCLUDED_GEOMETRY_FIELDS = (
    "geometry",
    "orientation",
    "load_classifier",
    "classify_sia_geometry",
    "sia_classes_",
)


@dataclass(frozen=True)
class SiaRunConfig:
    aged: Path
    pot_dir: Path
    species: tuple[str, ...]
    masses: Mapping[str, float]
    mu_eV: Mapping[str, float]
    lmp: str = "lmp"
    anneal_k: float = DEFAULT_ANNEAL_K
    inserts_per_element: int = DEFAULT_INSERTS_PER_ELEMENT
    md_steps: int = DEFAULT_MD_STEPS
    cool_steps: int = DEFAULT_COOL_STEPS
    site_seed: int = vac.DEFAULT_SITE_SEED
    nworkers: Optional[int] = None
    expected_natoms: Optional[int] = 5488
    family_name: str = ""
    skip_bulk: bool = False


def die(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        die(f"{name} must be a float, got {raw!r}")
    if not math.isfinite(value):
        die(f"{name} must be finite, got {value!r}")
    return value


def energy_csv_name(temperature: float) -> str:
    return f"sia_formation_energies_{int(temperature)}K.csv"


def pair_csv_name(temperature: float) -> str:
    return f"sia_pairs_{int(temperature)}K.csv"


def cases_dirname(temperature: float) -> str:
    return f"cases_{int(temperature)}K"


def random_direction(rng: random.Random):
    z = 2.0 * rng.random() - 1.0
    phi = 2.0 * math.pi * rng.random()
    xy = math.sqrt(max(0.0, 1.0 - z * z))
    return (xy * math.cos(phi), xy * math.sin(phi), z)


def wrap_point(x: float, y: float, z: float, cell: Cell):
    lx, ly, lz = cell.lengths
    x = cell.lo[0] + (x - cell.lo[0]) % lx
    y = cell.lo[1] + (y - cell.lo[1]) % ly
    z = cell.lo[2] + (z - cell.lo[2]) % lz
    return x, y, z


def closest_distance(cell: Cell, x: float, y: float, z: float) -> float:
    from .lammps_cell import minimum_image

    best = None
    for atom in cell.atoms:
        dx, dy, dz = minimum_image(x - atom[2], y - atom[3], z - atom[4], cell)
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if best is None or dist < best:
            best = dist
    return 0.0 if best is None else best


def type_of_element(cell: Cell, element: str) -> int:
    for type_id in cell.masses:
        if cell.masses[type_id] == element:
            return type_id
    die(f"aged cell has no atom type for {element}")
    return 0


def place_random_interstitial(cell: Cell, index: int, element: str, rng: random.Random):
    host = cell.atoms[index]
    type_id = type_of_element(cell, element)
    new_id = max(atom[0] for atom in cell.atoms) + 1
    span = MAX_INSERT_A - MIN_INSERT_A
    for _ in range(PLACEMENT_TRIES):
        direction = random_direction(rng)
        length = MIN_INSERT_A + span * rng.random()
        x = host[2] + direction[0] * length
        y = host[3] + direction[1] * length
        z = host[4] + direction[2] * length
        x, y, z = wrap_point(x, y, z, cell)
        if closest_distance(cell, x, y, z) < TOO_CLOSE_A:
            continue
        atoms = list(cell.atoms)
        atoms.append((new_id, type_id, x, y, z))
        return atoms, (x, y, z)
    return None


def insertion_job(element: str, site_id: int) -> str:
    return f"{element}:{site_id}"


def sample_insertions(
    bulk: Cell,
    *,
    species: Sequence[str],
    n_per_element: int,
    done: set,
    rng: random.Random,
):
    plans = []
    for element in species:
        pool = []
        for index, atom in enumerate(bulk.atoms):
            site_id = atom[0]
            if insertion_job(element, site_id) in done:
                continue
            pool.append(index)
        rng.shuffle(pool)
        if len(pool) < n_per_element:
            die(
                f"need {n_per_element} unused sites for inserted {element}, "
                f"only {len(pool)} left"
            )
        chosen = pool[:n_per_element]
        for index in chosen:
            plans.append((element, index))
        print(
            f"insert {element}: {n_per_element} random sites "
            f"(pool_available={len(pool)})",
            flush=True,
        )
    rng.shuffle(plans)
    return plans


def write_anneal_input(
    path: Path,
    cell: Cell,
    tabgap: Path,
    eam: Path,
    temperature: float,
    md_steps: int,
    cool_steps: int,
    seed: int,
) -> None:
    from .lammps_cell import pair_block as _pair_block

    text = f"""units           metal
atom_style      atomic
boundary        p p p
atom_modify     map array

read_data       start.data

{_pair_block(cell, tabgap, eam)}
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

# Stage 1. Pull apart any remaining overlap. This is not the formation energy.
minimize        1.0e-4 1.0e-4 500 5000

# Stage 2. Short molecular dynamics at the anneal temperature.
velocity        all create {temperature:.4f} {seed} mom yes rot yes dist gaussian
fix             hot all nvt temp {temperature:.4f} {temperature:.4f} 0.1
timestep        0.001
thermo          500
run             {md_steps}
unfix           hot

# Stage 3. Linear cool to 1 K. The final basin is found by minimization, not by this thermostat.
variable        Tcool equal ramp({temperature:.4f},1.0)
fix             cool all nvt temp v_Tcool v_Tcool 0.1
run             {cool_steps}
unfix           cool
velocity        all set 0.0 0.0 0.0

# Stage 4. Thorough minimization at the bulk volume.
minimize        1.0e-8 1.0e-6 5000 50000
write_data      relaxed.data
"""
    path.write_text(text, encoding="utf-8")


def lammps_seed(site_seed: int, site_id: int, element: str, species: Sequence[str]) -> int:
    offset = list(species).index(element) + 1
    value = site_seed + site_id * 10 + offset
    return 1 if value < 1 else value


def load_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        # Accept our schema; refuse geometry-laden legacy headers mixed mid-run.
        if "geometry" in fields or "orientation" in fields:
            die(
                f"{path} still has geometry/orientation columns. "
                "Move it aside (geometry classification is not used)."
            )
        if fields and fields != SIA_FIELDS:
            die(f"{path} header does not match SIA_FIELDS. Move the file aside and rerun.")
        for row in reader:
            done.add(row["job"])
    return done


def blank_row(job, temperature, site_id, host, inserted, status, note) -> dict:
    return {
        "job": job,
        "anneal_K": f"{temperature:.1f}",
        "site_id": site_id,
        "host_element": host,
        "inserted_element": inserted,
        "insert_x": "",
        "insert_y": "",
        "insert_z": "",
        "E_bulk_eV": "",
        "E_SIA_eV": "",
        "mu_eV": "",
        "formation_energy_SIA_eV": "",
        "pair": "",
        "element_a": "",
        "element_b": "",
        "class_key": "",
        "status": status,
        "note": note,
    }


def write_pair_summary(csv_path: Path, summary_path: Path) -> None:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize_pairs(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def anneal_one(
    work: Path,
    cell: Cell,
    bulk: Cell,
    settings: dict,
    element: str,
    site_id: int,
    host: str,
    index: int,
    rng: random.Random,
    *,
    masses: Mapping[str, float],
    mu_eV: Mapping[str, float],
    species: Sequence[str],
    family_name: str,
) -> dict:
    temperature = settings["temperature"]
    job = insertion_job(element, site_id)
    placed = place_random_interstitial(cell, index, element, rng)
    if placed is None:
        return blank_row(
            job, temperature, site_id, host, element, "failed", "no open insertion point"
        )
    atoms, point = placed
    work.mkdir(parents=True, exist_ok=True)
    write_lammps_data(
        work / "start.data",
        cell,
        atoms,
        element_masses=masses,
        title=f"{family_name or 'alloy'} SIA start",
    )
    seed = lammps_seed(settings["site_seed"], site_id, element, species)
    write_anneal_input(
        work / "in.anneal.lmp",
        cell,
        settings["tabgap"],
        settings["eam"],
        temperature,
        settings["md_steps"],
        settings["cool_steps"],
        seed,
    )
    try:
        e_sia = vac.run_lammps(work, settings["lmp"], "in.anneal.lmp")
    except RuntimeError as exc:
        row = blank_row(job, temperature, site_id, host, element, "failed", str(exc))
        row["insert_x"] = f"{point[0]:.6f}"
        row["insert_y"] = f"{point[1]:.6f}"
        row["insert_z"] = f"{point[2]:.6f}"
        return row

    m2e = vac.mass_to_element_map(masses)
    relaxed = parse_lammps_data(work / "relaxed.data", m2e)
    label = label_element_pair(bulk, relaxed)
    if element not in mu_eV:
        die(f"no mu configured for inserted element {element!r}")
    mu = float(mu_eV[element])
    e_bulk = settings["e_bulk"]
    e_f = e_sia - e_bulk - mu
    pair = label.get("pair", "unknown")
    return {
        "job": job,
        "anneal_K": f"{temperature:.1f}",
        "site_id": site_id,
        "host_element": host,
        "inserted_element": element,
        "insert_x": f"{point[0]:.6f}",
        "insert_y": f"{point[1]:.6f}",
        "insert_z": f"{point[2]:.6f}",
        "E_bulk_eV": f"{e_bulk:.8f}",
        "E_SIA_eV": f"{e_sia:.8f}",
        "mu_eV": f"{mu:.6f}",
        "formation_energy_SIA_eV": f"{e_f:.8f}",
        "pair": pair,
        "element_a": label.get("element_a", ""),
        "element_b": label.get("element_b", ""),
        "class_key": pair,  # chemistry only — not geometry|orientation
        "status": "ok",
        "note": label.get("note", ""),
    }


def run_sia_jobs(
    work_dir: Path,
    bulk: Cell,
    plans: list,
    settings: dict,
    *,
    masses: Mapping[str, float],
    mu_eV: Mapping[str, float],
    species: Sequence[str],
    family_name: str,
) -> tuple[int, int]:
    temperature = settings["temperature"]
    csv_path = work_dir / energy_csv_name(temperature)
    new_file = not csv_path.is_file()
    lock = __import__("threading").Lock()
    n_done = 0
    n_fail = 0
    nworkers = settings["nworkers"]

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIA_FIELDS)
        if new_file:
            writer.writeheader()
            handle.flush()

        def work_one(plan):
            element, index = plan
            atom = bulk.atoms[index]
            site_id = atom[0]
            host = bulk.masses[atom[1]]
            rng = random.Random(
                lammps_seed(settings["site_seed"], site_id, element, species)
            )
            folder = (
                work_dir
                / cases_dirname(temperature)
                / f"sia_{element}_{site_id}"
            )
            return anneal_one(
                folder,
                bulk,
                bulk,
                settings,
                element,
                site_id,
                host,
                index,
                rng,
                masses=masses,
                mu_eV=mu_eV,
                species=species,
                family_name=family_name,
            )

        with ThreadPoolExecutor(max_workers=nworkers) as pool:
            futures = [pool.submit(work_one, plan) for plan in plans]
            for fut in as_completed(futures):
                row = fut.result()
                with lock:
                    writer.writerow(row)
                    handle.flush()
                    n_done += 1
                    if row["status"] != "ok":
                        n_fail += 1
                print(
                    f"{row['job']} {row['status']} pair={row['class_key']} "
                    f"Ef={row['formation_energy_SIA_eV']} ({n_done}/{len(plans)})",
                    flush=True,
                )
    return n_done, n_fail


def run_sia(work_dir: Path, cfg: SiaRunConfig) -> dict[str, object]:
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    aged = Path(cfg.aged).expanduser().resolve()
    pot_dir = Path(cfg.pot_dir).expanduser().resolve()
    species = tuple(cfg.species)
    masses = {str(k): float(v) for k, v in dict(cfg.masses).items()}
    mu_eV = {str(k): float(v) for k, v in dict(cfg.mu_eV).items()}

    temperature = env_float("ANNEAL_K", float(cfg.anneal_k))
    if temperature < 1.0:
        die(f"ANNEAL_K must be at least 1 K, got {temperature}")
    n_per_element = vac.env_int(
        "INSERTS_PER_ELEMENT", int(cfg.inserts_per_element)
    )
    if n_per_element < 1:
        die("INSERTS_PER_ELEMENT must be >= 1")
    md_steps = vac.env_int("MD_STEPS", int(cfg.md_steps))
    cool_steps = vac.env_int("COOL_STEPS", int(cfg.cool_steps))
    if md_steps < 1 or cool_steps < 1:
        die("MD_STEPS and COOL_STEPS must be >= 1")
    site_seed = vac.env_int("SITE_SEED", int(cfg.site_seed))
    nworkers = vac.resolve_workers(cfg.nworkers)

    if not aged.is_file():
        die(f"missing aged cell: {aged}")
    local_aged = root / "aged.data"
    if aged.resolve() != local_aged.resolve():
        shutil.copy2(aged, local_aged)

    tabgap, eam = find_potential(pot_dir)
    vac.reset_core_pool(nworkers)

    bulk, e_bulk = vac.relax_bulk(
        root,
        local_aged if local_aged.is_file() else aged,
        tabgap,
        eam,
        cfg.lmp,
        masses=masses,
        expected_natoms=cfg.expected_natoms,
        family_name=cfg.family_name,
        skip_bulk=cfg.skip_bulk,
    )

    settings = {
        "lmp": cfg.lmp,
        "temperature": temperature,
        "md_steps": md_steps,
        "cool_steps": cool_steps,
        "site_seed": site_seed,
        "nworkers": nworkers,
        "tabgap": tabgap,
        "eam": eam,
        "e_bulk": e_bulk,
    }

    csv_path = root / energy_csv_name(temperature)
    done = load_done(csv_path)
    rng = random.Random(site_seed)
    plans = sample_insertions(
        bulk,
        species=species,
        n_per_element=n_per_element,
        done=done,
        rng=rng,
    )
    print(
        f"Annealing at {temperature:.0f} K. "
        f"{len(plans)} new insertions "
        f"({n_per_element} per element, already_done={len(done)}). "
        f"MD {md_steps} steps, cool {cool_steps} steps. "
        f"{vac.rank_line(nworkers)}. "
        "Labeling chemical pairs only (no geometry/orientation).",
        flush=True,
    )
    n_done, n_fail = run_sia_jobs(
        root,
        bulk,
        plans,
        settings,
        masses=masses,
        mu_eV=mu_eV,
        species=species,
        family_name=cfg.family_name,
    )
    pair_path = root / pair_csv_name(temperature)
    write_pair_summary(csv_path, pair_path)
    print(
        f"Wrote {csv_path} ({n_done} new rows, {n_fail} failed); "
        f"pair summary → {pair_path.name}",
        flush=True,
    )
    if n_fail:
        die(f"{n_fail} anneal(s) failed. Pair table still includes finished rows.")
    return {
        "status": "ok",
        "e_bulk": e_bulk,
        "n_atoms": len(bulk.atoms),
        "n_done": n_done,
        "n_fail": n_fail,
        "anneal_k": temperature,
        "energy_csv": str(csv_path),
        "pair_csv": str(pair_path),
        "nworkers": nworkers,
    }
