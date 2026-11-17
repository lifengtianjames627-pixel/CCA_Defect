#!/usr/bin/env python3
"""
0 K vacancy formation energies on the aged W-Ta-Cr-V cell.

The executable code is split into four parts:
Part 1 (read and write a cell), Part 2 (one minimization), Part 3 (which sites),
and Part 4 (the driver).
main only calls those parts in order.

Interstitials are not computed here. They are a separate anneal in SIA_form_e.py.
The old combined script preset a same-species <111> dumbbell. That start is retired.

5488-atom branch. Sampling is still a fixed number of removals per chemical
species. Default is 125 vacancies for each of Cr, Ta, V, and W.

Ef^V(A) = E_vac - E_bulk + mu_A, with the equiatomic tabGAP chemical potentials.

Usage (on the server, from this directory):

    python3 V_form_e.py

Independent vacancy minimizations use at most 13 serial lmp processes,
one core each, so three cores on this 16-core host stay idle. Each
process minimizes a different site. Bulk relaxation stays one process
at the start. Do not wrap these in mpirun -np 1: that binds every
process to core 0.

Environment:
    AGED      LAMMPS data file
              (default: the aged cell under W-Ta-Cr-V/Metropolis_MC)
    POT_DIR   directory with one .tabgap and one .eam.fs
    LMP       lammps binary (default: lmp)
    SAMPLES_PER_ELEMENT  vacancy trials per species (default: 125)
    SITE_SEED RNG seed for site sampling (default: 12345)
    NWORKERS  concurrent serial lmp processes, one core each (default: 13)
"""

from __future__ import annotations

import csv
import math
import os
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_alloy_root() -> Path:
    """
    The W-Ta-Cr-V directory.

    The file may sit in W-Ta-Cr-V, in form_e_5488, or in form_e_5488/1000K.
    Walk upward until the directory that holds Metropolis_MC is found.
    """
    for folder in (HERE, *HERE.parents):
        if (folder / "Metropolis_MC" / "MC_Metropolis_5488" / "aged.data").is_file():
            return folder
    die("could not find the W-Ta-Cr-V Metropolis aged cell above this script")
    return HERE


ALLOY_ROOT = find_alloy_root()
DEFAULT_AGED = ALLOY_ROOT / "Metropolis_MC" / "MC_Metropolis_5488" / "aged.data"
DEFAULT_POT = Path.home() / "HEA_Irr_ML_study/ML_potentials/tabGAP/W-Ta-Cr-V"
DEFAULT_SAMPLES_PER_ELEMENT = 125
DEFAULT_SITE_SEED = 12345
MAX_RANKS = 13
SPARE_CORES = 3
TARGET_NATOMS = 5488
SPECIES = ("Cr", "Ta", "V", "W")

# Substitution-fit chemical potentials for equiatomic random WTaCrV
# with this tabGAP, in eV/atom. Byggmästar et al., eq. (7) and the
# paragraph that follows it. Pure-bcc values differ by about 0.1 eV.
MU_EV = {"Cr": -9.388, "Ta": -11.724, "V": -9.033, "W": -13.029}
MASS_TO_ELEMENT = {
    51.9961: "Cr",
    180.94788: "Ta",
    50.9415: "V",
    183.84: "W",
}


# =============================================================================
# Part 1 — Read and write a cell
# =============================================================================
# A LAMMPS data file in, a Cell out, and the reverse. No energies here.

def die(message: str) -> None:
    """Print the error and stop."""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def env_int(name: str, default: int) -> int:
    """A non-negative integer from the environment, or the default."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    if not raw.isdigit():
        die(f"{name} must be a non-negative integer, got {raw!r}")
    return int(raw)


class Cell:
    """One LAMMPS cell: box, masses, and atoms."""

    def __init__(self, lo, hi, tilt, masses, atoms):
        """Store the box and the atom list. Coordinates stay in Angstroms."""
        self.lo = lo  # (xlo, ylo, zlo)
        self.hi = hi
        self.tilt = tilt  # (xy, xz, yz)
        self.masses = masses  # type -> element
        self.atoms = atoms  # list of (id, type, x, y, z)

    @property
    def lengths(self):
        """Box edge lengths in Angstroms."""
        return tuple(self.hi[i] - self.lo[i] for i in range(3))


def element_of_mass(mass: float) -> str:
    """Element name for a LAMMPS mass. Stops if the mass is not Cr, Ta, V, or W."""
    for known, name in MASS_TO_ELEMENT.items():
        if abs(known - mass) < 1e-3:
            return name
    die(f"unrecognised mass {mass}")
    return ""


def parse_lammps_data(path: Path) -> Cell:
    """Read one LAMMPS data file into a Cell."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    masses: dict[int, str] = {}
    atoms = []
    lo = [0.0, 0.0, 0.0]
    hi = [0.0, 0.0, 0.0]
    tilt = (0.0, 0.0, 0.0)
    section = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        low = stripped.lower()
        if low.startswith("masses"):
            section = "masses"
            continue
        if low.startswith("atoms"):
            section = "atoms"
            continue
        if low.split()[0] in {
            "velocities",
            "bonds",
            "angles",
            "dihedrals",
            "impropers",
        }:
            section = ""
            continue
        parts = stripped.split()
        if section == "" and len(parts) >= 4 and parts[2] in {"xlo", "ylo", "zlo"}:
            if parts[2] == "xlo":
                lo[0], hi[0] = float(parts[0]), float(parts[1])
            elif parts[2] == "ylo":
                lo[1], hi[1] = float(parts[0]), float(parts[1])
            else:
                lo[2], hi[2] = float(parts[0]), float(parts[1])
            continue
        if section == "" and len(parts) >= 6 and parts[3:6] == ["xy", "xz", "yz"]:
            tilt = (float(parts[0]), float(parts[1]), float(parts[2]))
            continue
        if section == "masses" and len(parts) >= 2 and parts[0].isdigit():
            masses[int(parts[0])] = element_of_mass(float(parts[1]))
            continue
        if section == "atoms" and len(parts) >= 5 and parts[0].isdigit():
            atoms.append(
                (int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            )
    if not atoms or not masses:
        die(f"could not read atoms and masses from {path}")
    return Cell(tuple(lo), tuple(hi), tilt, masses, atoms)

# data writer
def write_lammps_data(path: Path, cell: Cell, atoms) -> None:
    """Write a LAMMPS data file from the cell box and the given atom list."""
    xy, xz, yz = cell.tilt
    lines = [
        "W-Ta-Cr-V formation-energy cell",
        "",
        f"{len(atoms)} atoms",
        f"{len(cell.masses)} atom types",
        "",
        f"{cell.lo[0]:.10f} {cell.hi[0]:.10f} xlo xhi",
        f"{cell.lo[1]:.10f} {cell.hi[1]:.10f} ylo yhi",
        f"{cell.lo[2]:.10f} {cell.hi[2]:.10f} zlo zhi",
        f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz",
        "",
        "Masses",
        "",
    ]
    for type_id in sorted(cell.masses):
        element = cell.masses[type_id]
        mass = {v: k for k, v in MASS_TO_ELEMENT.items()}[element]
        lines.append(f"{type_id} {mass}")
    lines += ["", "Atoms # atomic", ""]
    for atom_id, type_id, x, y, z in atoms:
        lines.append(f"{atom_id} {type_id} {x:.10f} {y:.10f} {z:.10f}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Part 2 — One minimization
# =============================================================================
# Write in.lmp, run one LAMMPS rank, and read the final energy from the log.
# The worker pool in Part 4 calls run_lammps once per vacancy. Each call is a
# different directory and a different rank. OpenMP stays at one thread so the
# idle cores stay idle.

def pair_block(cell: Cell, tabgap: Path, eam: Path) -> str:
    """The tabGAP plus EAM pair_coeff lines for this cell."""
    elements = [cell.masses[i] for i in sorted(cell.masses)]
    names = " ".join(elements)
    return f"""pair_style      hybrid/overlay tabgap eam/fs
pair_coeff      * * tabgap {tabgap} {names}
pair_coeff      * * eam/fs {eam} {names}
"""

# the input writer for both the bulk and the defect solids
def write_input(path: Path, data_name: str, cell: Cell, tabgap: Path, eam: Path, relax_box: bool) -> None:
    """One minimize input. The box relaxes only for the defect-free cell."""
    box = ""
    if relax_box: # if true, it is a bulk, allow the change in volume; if false it is a defected vacancy unrelaxed, don't allow the change in volume
        box = "fix             box all box/relax iso 0.0 vmax 0.001\n"
    text = f"""units           metal
atom_style      atomic
boundary        p p p
atom_modify     map array

read_data       {data_name}

{pair_block(cell, tabgap, eam)}
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

{box}minimize        1.0e-8 1.0e-6 5000 50000
write_data      relaxed.data
"""
    path.write_text(text, encoding="utf-8")


def final_energy(log_text: str, *, work: Path | None = None) -> float:
    """The last 'Energy initial, next-to-last, final' value in a LAMMPS log."""
    marker = "Energy initial, next-to-last, final"
    last = None
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        if marker in line:
            numbers = line.split("=")[-1].split()
            if len(numbers) < 3 and i + 1 < len(lines):
                numbers = lines[i + 1].split()
            if len(numbers) >= 3:
                last = float(numbers[2])
    if last is None or not math.isfinite(last):
        where = f" in {work}" if work is not None else ""
        raise RuntimeError(f"LAMMPS log has no final minimization energy{where}")
    return last


def host_cores() -> int:
    """How many cores this process can see. At least one."""
    cores = os.cpu_count()
    if cores is None:
        return 1
    if cores < 1:
        return 1
    return cores


def resolve_workers() -> int:
    """
    How many LAMMPS ranks to start.

    Three cores stay idle, and the count never goes above 13. On this
    16-core host that is 13 ranks. NWORKERS above the limit is cut back.
    """
    cores = host_cores()
    limit = cores - SPARE_CORES
    if limit < 1:
        limit = 1
    if limit > MAX_RANKS:
        limit = MAX_RANKS
    nworkers = env_int("NWORKERS", limit)
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
    """One log line: serial lmp tasks in use, and the cores left idle."""
    cores = host_cores()
    idle = cores - nworkers
    if idle < 0:
        idle = 0
    return f"{nworkers} serial lmp tasks, one core each, {idle} cores left idle"


# mpirun -np 1 binds every singleton rank to core 0. These jobs are independent
# serial LAMMPS runs, so each lmp is pinned to its own core instead.
_CORE_LOCK = threading.Lock()
_FREE_CORES: list[int] | None = None


def take_core() -> int:
    """Reserve one core from the worker set. Released in run_lammps."""
    global _FREE_CORES
    with _CORE_LOCK:
        if _FREE_CORES is None:
            _FREE_CORES = list(range(resolve_workers()))
        if not _FREE_CORES:
            die("no free core for a LAMMPS task")
        return _FREE_CORES.pop()


def give_core(core: int) -> None:
    """Return a core to the pool after its LAMMPS task exits."""
    global _FREE_CORES
    with _CORE_LOCK:
        if _FREE_CORES is None:
            _FREE_CORES = []
        _FREE_CORES.append(core)


def lammps_child_env() -> dict:
    """
    Environment for one serial LAMMPS task.

    OpenMP and BLAS stay at one thread. Binding variables are removed so
    they cannot pull every task back onto core 0; taskset does the pinning.
    """
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
    """One serial lmp, pinned to ``core``. Not an MPI launch."""
    return [
        "taskset",
        "-c",
        str(core),
        lmp,
        "-in",
        input_name,
        "-log",
        "log.lammps",
    ]


def run_lammps(work: Path, lmp: str, input_name: str = "in.lmp") -> float:
    """
    Run one LAMMPS input on its own core and return the last minimize energy.

    The log and the screen capture are different files. Safe to call from
    several worker threads, each with its own directory and core.
    """
    log_path = work / "log.lammps"
    screen_path = work / "screen.log"
    core = take_core()
    argv = lammps_argv(lmp, input_name, core)
    try:
        with screen_path.open("w", encoding="utf-8") as screen:
            try:
                proc = subprocess.run(
                    argv,
                    cwd=work,
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


# =============================================================================
# Part 3 — Which sites
# =============================================================================
# Neighbors are recorded for the CSV. The vacancy is that atom removed.
# find_potential only locates the two potential files.

def minimum_image(dx, dy, dz, cell: Cell):
    """Minimum-image displacement, including the box tilt."""
    lx, ly, lz = cell.lengths
    xy, xz, yz = cell.tilt
    # Convert to fractional-like coordinates for a triclinic box, then wrap.
    dz = dz - lz * round(dz / lz) if lz else dz
    dy = dy - yz * (dz / lz if lz else 0.0)
    dy = dy - ly * round(dy / ly) if ly else dy
    dx = dx - xz * (dz / lz if lz else 0.0) - xy * (dy / ly if ly else 0.0)
    dx = dx - lx * round(dx / lx) if lx else dx
    return dx, dy, dz


def nearest_neighbors(cell: Cell, index: int, count: int = 8) -> list[int]:
    """Indexes of the nearest atoms, default the eight BCC neighbors."""
    _, _, x0, y0, z0 = cell.atoms[index]
    ranked = []
    for j, atom in enumerate(cell.atoms):
        if j == index:
            continue
        dx, dy, dz = minimum_image(atom[2] - x0, atom[3] - y0, atom[4] - z0, cell)
        ranked.append((dx * dx + dy * dy + dz * dz, j))
    ranked.sort()
    return [j for _, j in ranked[:count]]


def nn_counts(cell: Cell, neighbor_indexes: list[int]) -> dict[str, int]:
    """How many of those neighbors are Cr, Ta, V, and W."""
    counts = {name: 0 for name in ("Cr", "Ta", "V", "W")}
    for j in neighbor_indexes:
        element = cell.masses[cell.atoms[j][1]]
        counts[element] = counts.get(element, 0) + 1
    return counts


def nn_string(counts: dict[str, int]) -> str:
    """The neighbor counts as one CSV field."""
    return " ".join(f"{name}:{counts[name]}" for name in ("Cr", "Ta", "V", "W"))


def find_potential(pot_dir: Path) -> tuple[Path, Path]:
    """The single .tabgap and the single .eam.fs in the potential directory."""
    tabgaps = sorted(pot_dir.glob("*.tabgap"))
    eams = sorted(pot_dir.glob("*.eam.fs"))
    if len(tabgaps) != 1 or len(eams) != 1:
        die(f"{pot_dir} needs exactly one .tabgap and one .eam.fs")
    return tabgaps[0].resolve(), eams[0].resolve()


def load_done(path: Path) -> set[int]:
    """Site ids already written to the formation CSV. Empty if the file is new."""
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
    n_per_element: int,
    done: set[int],
    rng: random.Random,
) -> list[int]:
    """Pick n_per_element unused sites of each species (Xinran-style stratification)."""
    pools: dict[str, list[int]] = {el: [] for el in SPECIES}
    for index, atom in enumerate(bulk.atoms):
        site_id = atom[0]
        if site_id in done:
            continue
        element = bulk.masses[atom[1]]
        if element in pools:
            pools[element].append(index)
    pending: list[int] = []
    for element in SPECIES:
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
            f"sample {element}: {n_per_element} sites "
            f"(pool_available={len(pool)})",
            flush=True,
        )
    rng.shuffle(pending)  # mix species order in the run queue
    return pending


# =============================================================================
# Part 4 — Driver
# =============================================================================
# Relax the aged cell once. Sample sites. Write every vacancy input.
# Then minimize them and append one CSV row per finished vacancy.
# main only calls these functions.

def resolve_run():
    """
    Read the environment and the potential files.

    SAMPLES_PER_ELEMENT is the count per species. MAX_SITES, if that variable
    is unset, is the old total budget split across the four species.
    """
    aged = Path(os.environ.get("AGED", str(DEFAULT_AGED))).expanduser()
    pot_dir = Path(os.environ.get("POT_DIR", str(DEFAULT_POT))).expanduser()
    lmp = os.environ.get("LMP", "lmp")
    # Prefer SAMPLES_PER_ELEMENT; keep MAX_SITES as a legacy override meaning
    # total sites (not per element) only if SAMPLES_PER_ELEMENT is unset and
    # MAX_SITES is set — otherwise use the per-element default.
    if os.environ.get("SAMPLES_PER_ELEMENT", "").strip():
        n_per_element = env_int("SAMPLES_PER_ELEMENT", DEFAULT_SAMPLES_PER_ELEMENT)
    elif os.environ.get("MAX_SITES", "").strip():
        # Legacy: treat MAX_SITES as total budget split equally across 4 species.
        total = env_int("MAX_SITES", DEFAULT_SAMPLES_PER_ELEMENT * len(SPECIES))
        n_per_element = max(1, total // len(SPECIES))
        print(
            f"NOTE: MAX_SITES={total} → SAMPLES_PER_ELEMENT={n_per_element} "
            f"(legacy split across {len(SPECIES)} species)",
            flush=True,
        )
    else:
        n_per_element = DEFAULT_SAMPLES_PER_ELEMENT
    site_seed = env_int("SITE_SEED", DEFAULT_SITE_SEED)
    nworkers = resolve_workers()
    if not aged.is_file():
        die(f"missing aged cell: {aged}")
    tabgap, eam = find_potential(pot_dir)
    return aged, lmp, n_per_element, site_seed, nworkers, tabgap, eam

# relax the bulk material
def relax_bulk(aged: Path, tabgap: Path, eam: Path, lmp: str):
    """Minimize the aged cell once, with the box free. Returns the cell and E_bulk."""
    bulk_dir = HERE / "bulk"
    bulk_dir.mkdir(exist_ok=True)
    start = parse_lammps_data(aged)
    if len(start.atoms) != TARGET_NATOMS:
        die(
            f"aged cell has {len(start.atoms)} atoms, expected {TARGET_NATOMS} "
            f"for the 5488-atom formation branch (got {aged})"
        )
    write_lammps_data(bulk_dir / "start.data", start, start.atoms)
    write_input(bulk_dir / "in.lmp", "start.data", start, tabgap, eam, relax_box=True)
    print("Minimizing the aged cell (this loads the potential once).", flush=True)
    try:
        e_bulk = run_lammps(bulk_dir, lmp)
    except RuntimeError as exc:
        die(str(exc))
    bulk = parse_lammps_data(bulk_dir / "relaxed.data")
    print(f"E_bulk = {e_bulk:.6f} eV for {len(bulk.atoms)} atoms", flush=True)
    return bulk, e_bulk

# randomly remove site_id from the atomic lists 
def prepare_vacancy_jobs(bulk: Cell, pending: list[int], tabgap: Path, eam: Path):
    """
    Write every vacancy input before any defect minimization starts.

    Returns the neighbor record for each site and the job list
    (kind, site_id, directory).
    """
    site_meta: dict[int, dict] = {}
    jobs: list[tuple[str, int, Path]] = []
    for index in pending:
        atom = bulk.atoms[index]
        site_id = atom[0]
        element = bulk.masses[atom[1]]
        counts = nn_counts(bulk, nearest_neighbors(bulk, index))
        mu = MU_EV[element]
        site_meta[site_id] = {
            "element": element,
            "counts": counts,
            "mu": mu,
        }

        vac_dir = HERE / "cases" / f"vac_{site_id}"
        vac_dir.mkdir(parents=True, exist_ok=True)
        vacancy_atoms = []
        for other in bulk.atoms:
            if other[0] != site_id:
                vacancy_atoms.append(other)
        write_lammps_data(vac_dir / "start.data", bulk, vacancy_atoms)
        write_input(vac_dir / "in.lmp", "start.data", bulk, tabgap, eam, relax_box=False)
        jobs.append(("vac", site_id, vac_dir))
    return site_meta, jobs


def run_vacancy_jobs(
    bulk_energy: float,
    pending: list[int],
    site_meta: dict,
    jobs: list,
    lmp: str,
    nworkers: int,
) -> None:
    """
    Minimize every prepared vacancy and append one CSV row when it finishes.

    A failed job is recorded and stops the script after the pool is done.
    """
    csv_path = HERE / "v_form_e.csv"
    new_file = not csv_path.is_file()
    fields = [
        "site_id",
        "removed_element",
        "element_nearest_neighbors",
        "nn_Cr",
        "nn_Ta",
        "nn_V",
        "nn_W",
        "E_bulk_eV",
        "E_vacancy_eV",
        "mu_eV",
        "formation_energy_vacancy_eV",
    ]
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
                pool.submit(run_lammps, work_dir, lmp): (kind, site_id, work_dir)
                for kind, site_id, work_dir in jobs
            }
            for fut in as_completed(future_map):
                kind, site_id, work_dir = future_map[fut]
                try:
                    energy = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{kind} site {site_id} ({work_dir}): {exc}")
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
                    "element_nearest_neighbors": nn_string(counts),
                    "nn_Cr": counts["Cr"],
                    "nn_Ta": counts["Ta"],
                    "nn_V": counts["V"],
                    "nn_W": counts["W"],
                    "E_bulk_eV": f"{e_bulk:.8f}",
                    "E_vacancy_eV": f"{e_vac:.8f}",
                    "mu_eV": f"{mu:.6f}",
                    "formation_energy_vacancy_eV": f"{e_f_v:.8f}",
                }
                with csv_lock:
                    writer.writerow(row)
                    handle.flush()
                    n_done += 1
                print(
                    f"site {site_id} {element}  NN {nn_string(counts)}  "
                    f"Ef_V={e_f_v:.3f}  ({n_done}/{len(pending)})",
                    flush=True,
                )
                if n_jobs_done % max(1, nworkers) == 0 or n_jobs_done == len(jobs):
                    print(
                        f"jobs finished {n_jobs_done}/{len(jobs)}",
                        flush=True,
                    )

    if failures:
        die(
            f"{len(failures)} LAMMPS job(s) failed; {n_done}/{len(pending)} sites written. "
            f"First error: {failures[0]}"
        )
    print(f"Wrote {csv_path} ({n_done} sites)", flush=True)


def main() -> None:
    """Relax the bulk, sample sites, write the vacancy inputs, then minimize them."""
    aged, lmp, n_per_element, site_seed, nworkers, tabgap, eam = resolve_run()
    bulk, e_bulk = relax_bulk(aged, tabgap, eam, lmp)

    csv_path = HERE / "v_form_e.csv"
    done = load_done(csv_path)
    rng = random.Random(site_seed)
    pending = sample_indexes_per_element(
        bulk,
        n_per_element=n_per_element,
        done=done,
        rng=rng,
    )
    print(
        f"Sampling {len(pending)} vacancies "
        f"({n_per_element} per element × {len(SPECIES)}; "
        f"SITE_SEED={site_seed}, already_done={len(done)})",
        flush=True,
    )
    print(
        f"Parallel LAMMPS: {rank_line(nworkers)} "
        f"({len(pending)} minimizations).",
        flush=True,
    )

    site_meta, jobs = prepare_vacancy_jobs(bulk, pending, tabgap, eam)
    run_vacancy_jobs(e_bulk, pending, site_meta, jobs, lmp, nworkers)


if __name__ == "__main__":
    main()
