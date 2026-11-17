#!/usr/bin/env python3
"""
Self-interstitial formation energies by simulated annealing on the aged cell.

The executable code is split into four parts:
Part 1 (a random insertion, not a dumbbell), Part 2 (one anneal in LAMMPS),
Part 3 (classify and group the relaxed defects), and Part 4 (the driver).
main only calls those parts in order.

The inserted atom is one of Cr, Ta, V, or W. The lattice site is drawn at
random and is not required to be the same element. Nothing is placed on a
<111>, <110>, or <100> axis. The run then:

1. minimizes once, only long enough to pull apart an overlap
2. runs a short NVT molecular-dynamics segment at ANNEAL_K
3. cools that thermostat linearly to 1 K
4. minimizes to the same tolerance as the vacancy script

Default campaign: 1000 K, 200 insertions of each element, 800 anneals.
Set ANNEAL_K to 1200 or 1400 for a second campaign. Those writes go to a
different CSV, so the 1000 K table is kept. Compare the class counts. A
change in which geometries appear means 1000 K did not finish the search.

Ef^SIA = E_SIA - E_bulk - mu_A. A is the atom that was inserted, because
that is the atom taken from the reservoir. The relaxed dumbbell may be a
different pair. Both chemistries are written down.

Usage, from this directory:

    python3 SIA_form_e.py

Environment:
    ANNEAL_K               thermostat temperature in K (default: 1000)
    INSERTS_PER_ELEMENT    random sites per inserted element (default: 200)
    MD_STEPS               steps at ANNEAL_K, 1 fs each (default: 5000, so 5 ps)
    COOL_STEPS             steps while cooling to 1 K (default: 20000, so 20 ps)
    SITE_SEED              RNG seed (default: 12345)
    NWORKERS               concurrent LAMMPS ranks (default: 13, three cores left idle)
    AGED, POT_DIR, LMP     same meaning as in V_form_e.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import V_form_e as vac

DEFAULT_ANNEAL_K = 1000.0
DEFAULT_INSERTS_PER_ELEMENT = 200
DEFAULT_MD_STEPS = 5000
DEFAULT_COOL_STEPS = 20000
MIN_INSERT_A = 0.8
MAX_INSERT_A = 1.4
TOO_CLOSE_A = 0.7
PLACEMENT_TRIES = 40


# =============================================================================
# Part 1 — Random insertion
# =============================================================================
# One extra atom near a lattice site, in a random direction. No dumbbell axis
# is chosen here. The anneal in Part 2 decides the geometry.

def die(message: str) -> None:
    """Print the error and stop."""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def env_float(name: str, default: float) -> float:
    """A finite float from the environment, or the default."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        die(f"{name} must be a number, got {raw!r}")
        return default
    if not math.isfinite(value):
        die(f"{name} must be finite, got {raw!r}")
    return value


def random_direction(rng: random.Random):
    """One unit vector drawn uniformly on the sphere."""
    z = 2.0 * rng.random() - 1.0
    phi = 2.0 * math.pi * rng.random()
    xy = math.sqrt(max(0.0, 1.0 - z * z))
    return (xy * math.cos(phi), xy * math.sin(phi), z)


def wrap_point(x: float, y: float, z: float, cell: vac.Cell):
    """Fold one cartesian point back into the cell."""
    lx, ly, lz = cell.lengths
    x = cell.lo[0] + (x - cell.lo[0]) % lx
    y = cell.lo[1] + (y - cell.lo[1]) % ly
    z = cell.lo[2] + (z - cell.lo[2]) % lz
    return x, y, z


def closest_distance(cell: vac.Cell, x: float, y: float, z: float) -> float:
    """Shortest minimum-image distance from this point to any atom."""
    best = None
    for atom in cell.atoms:
        dx, dy, dz = vac.minimum_image(x - atom[2], y - atom[3], z - atom[4], cell)
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if best is None or dist < best:
            best = dist
    if best is None:
        return 0.0
    return best


def type_of_element(cell: vac.Cell, element: str) -> int:
    """LAMMPS type id of this element in the aged cell."""
    for type_id in cell.masses:
        if cell.masses[type_id] == element:
            return type_id
    die(f"aged cell has no atom type for {element}")
    return 0


def place_random_interstitial(cell: vac.Cell, index: int, element: str, rng: random.Random):
    """
    The aged cell plus one atom of `element` near this site.

    The offset length is uniform between 0.8 and 1.4 A. The direction is
    isotropic, so the start is not a dumbbell. A point closer than 0.7 A to
    an existing atom is discarded and drawn again. None means no open point
    was found.
    """
    host = cell.atoms[index]
    type_id = type_of_element(cell, element)
    new_id = 0
    for atom in cell.atoms:
        if atom[0] > new_id:
            new_id = atom[0]
    new_id = new_id + 1
    span = MAX_INSERT_A - MIN_INSERT_A
    attempt = 0
    while attempt < PLACEMENT_TRIES:
        attempt = attempt + 1
        direction = random_direction(rng)
        length = MIN_INSERT_A + span * rng.random()
        x = host[2] + direction[0] * length
        y = host[3] + direction[1] * length
        z = host[4] + direction[2] * length
        x, y, z = wrap_point(x, y, z, cell)
        if closest_distance(cell, x, y, z) < TOO_CLOSE_A:
            continue
        atoms = []
        for other in cell.atoms:
            atoms.append(other)
        atoms.append((new_id, type_id, x, y, z))
        return atoms, (x, y, z)
    return None


def insertion_job(element: str, site_id: int) -> str:
    """Resume key for one inserted element at one lattice site."""
    return f"{element}:{site_id}"


def sample_insertions(bulk: vac.Cell, n_per_element: int, done: set, rng: random.Random):
    """
    200 random lattice sites for each inserted element, skipping finished jobs.

    The site element is not matched to the inserted element. A site may be
    used for more than one inserted species. Within one species the sites differ.
    """
    plans = []
    for element in vac.SPECIES:
        pool = []
        for index, atom in enumerate(bulk.atoms):
            site_id = atom[0]
            job = insertion_job(element, site_id)
            if job in done:
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


# =============================================================================
# Part 2 — One anneal
# =============================================================================
# Overlap relief, a short stay at ANNEAL_K, a linear cool to 1 K, then the
# same tight minimization used for vacancies. The box stays at the bulk volume.

def write_anneal_input(
    path: Path,
    cell: vac.Cell,
    tabgap: Path,
    eam: Path,
    temperature: float,
    md_steps: int,
    cool_steps: int,
    seed: int,
) -> None:
    """LAMMPS input for the four stages. The last minimize is the one we keep."""
    text = f"""units           metal
atom_style      atomic
boundary        p p p
atom_modify     map array

read_data       start.data

{vac.pair_block(cell, tabgap, eam)}
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


def run_anneal(work: Path, lmp: str) -> float:
    """Run in.anneal.lmp. Returns the last minimization energy, which is stage 4."""
    return vac.run_lammps(work, lmp, "in.anneal.lmp")


def lammps_seed(site_seed: int, site_id: int, element: str) -> int:
    """A positive velocity seed that stays the same if this job is repeated."""
    offset = vac.SPECIES.index(element) + 1
    value = site_seed + site_id * 10 + offset
    if value < 1:
        value = 1
    return value


# =============================================================================
# Part 3 — Classify and group
# =============================================================================
# The geometry cuts live in classify_sia_geometry.py. This part only calls
# that labeler and writes one row per anneal plus a class count.

def load_classifier():
    """The SIA geometry labeler used for the violin plots."""
    candidates = (
        vac.ALLOY_ROOT / "SIA_post_processing" / "classify_sia_geometry.py",
        vac.ALLOY_ROOT / "SIA_post_processing" / "SIA_server" / "classify_sia_geometry.py",
    )
    script = next((path for path in candidates if path.is_file()), None)
    if script is None:
        die(
            "missing geometry classifier. Looked for "
            + " and ".join(str(path) for path in candidates)
        )
    sys.path.insert(0, str(script.parent))
    import classify_sia_geometry as clf

    return clf


def label_relaxed(clf, bulk_geom: dict, relaxed: Path) -> dict:
    """
    Geometry, orientation, and chemical pair of one annealed interstitial.

    A crash in the labeler is returned as unknown rather than aborting the batch.
    """
    blank = {
        "geometry": "unknown",
        "orientation": "",
        "pair": "unknown",
        "element_a": "",
        "element_b": "",
        "note": "",
    }
    if not relaxed.is_file():
        blank["note"] = "no relaxed.data"
        return blank
    try:
        sia_geom = clf.parse_lammps_data(relaxed)
        found = clf.classify_one(bulk_geom, sia_geom)
    except Exception as exc:  # noqa: BLE001
        blank["note"] = f"classify failed: {exc}"
        return blank
    return found


def class_key(row: dict) -> str:
    """The violin bin: geometry, axis, and the relaxed chemical pair."""
    geometry = row.get("geometry", "")
    orientation = row.get("orientation", "")
    pair = row.get("pair", "")
    if geometry == "oct":
        return pair
    if orientation:
        return f"{pair}|{orientation}"
    return f"{pair}|{geometry}"


def median_of(values: list[float]) -> float:
    """Median of a non-empty list. The list is sorted here."""
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def write_class_summary(csv_path: Path, summary_path: Path) -> None:
    """
    One row per geometry and chemistry, with the count and the energy range.

    Every anneal stays in the energy CSV. This table is the deduplicated
    catalogue: repeats of the same final label are counted, not deleted.
    """
    groups: dict[str, list[float]] = {}
    labels: dict[str, dict] = {}
    if not csv_path.is_file():
        return
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            key = class_key(row)
            try:
                energy = float(row["formation_energy_SIA_eV"])
            except ValueError:
                continue
            if key not in groups:
                groups[key] = []
                labels[key] = row
            groups[key].append(energy)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "class_key",
            "geometry",
            "orientation",
            "pair",
            "count",
            "E_min_eV",
            "E_median_eV",
            "E_max_eV",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups):
            values = groups[key]
            label = labels[key]
            writer.writerow(
                {
                    "class_key": key,
                    "geometry": label.get("geometry", ""),
                    "orientation": label.get("orientation", ""),
                    "pair": label.get("pair", ""),
                    "count": len(values),
                    "E_min_eV": f"{min(values):.8f}",
                    "E_median_eV": f"{median_of(values):.8f}",
                    "E_max_eV": f"{max(values):.8f}",
                }
            )
    print(
        f"Wrote {summary_path} ({len(groups)} classes from {csv_path.name})",
        flush=True,
    )


# =============================================================================
# Part 4 — Driver
# =============================================================================
# Relax the bulk once. Place 800 random interstitials. Anneal them together.
# main only calls these steps.

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
    "geometry",
    "orientation",
    "pair",
    "element_a",
    "element_b",
    "class_key",
    "status",
    "note",
]


def energy_csv(temperature: float) -> Path:
    """Per-anneal table. The temperature is in the file name."""
    return vac.HERE / f"sia_formation_energies_{int(temperature)}K.csv"


def class_csv(temperature: float) -> Path:
    """Deduplicated class counts for this anneal temperature."""
    return vac.HERE / f"sia_classes_{int(temperature)}K.csv"


def load_done(path: Path) -> set[str]:
    """Finished job names. An older header is refused."""
    if not path.is_file():
        return set()
    done = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != SIA_FIELDS:
            die(f"{path} header does not match this script. Move the file aside and rerun.")
        for row in reader:
            done.add(row["job"])
    return done


def resolve_sia_run():
    """Read the environment. The temperature defaults to 1000 K."""
    aged = Path(os.environ.get("AGED", str(vac.DEFAULT_AGED))).expanduser()
    pot_dir = Path(os.environ.get("POT_DIR", str(vac.DEFAULT_POT))).expanduser()
    lmp = os.environ.get("LMP", "lmp")
    temperature = env_float("ANNEAL_K", DEFAULT_ANNEAL_K)
    if temperature < 1.0:
        die(f"ANNEAL_K must be at least 1 K, got {temperature}")
    n_per_element = vac.env_int("INSERTS_PER_ELEMENT", DEFAULT_INSERTS_PER_ELEMENT)
    if n_per_element < 1:
        die("INSERTS_PER_ELEMENT must be >= 1")
    md_steps = vac.env_int("MD_STEPS", DEFAULT_MD_STEPS)
    cool_steps = vac.env_int("COOL_STEPS", DEFAULT_COOL_STEPS)
    if md_steps < 1 or cool_steps < 1:
        die("MD_STEPS and COOL_STEPS must be >= 1")
    site_seed = vac.env_int("SITE_SEED", vac.DEFAULT_SITE_SEED)
    nworkers = vac.resolve_workers()
    if not aged.is_file():
        die(f"missing aged cell: {aged}")
    tabgap, eam = vac.find_potential(pot_dir)
    return {
        "aged": aged,
        "lmp": lmp,
        "temperature": temperature,
        "n_per_element": n_per_element,
        "md_steps": md_steps,
        "cool_steps": cool_steps,
        "site_seed": site_seed,
        "nworkers": nworkers,
        "tabgap": tabgap,
        "eam": eam,
    }


def blank_row(job, temperature, site_id, host, inserted, status, note) -> dict:
    """One CSV row with the energies left empty."""
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
        "geometry": "",
        "orientation": "",
        "pair": "",
        "element_a": "",
        "element_b": "",
        "class_key": "",
        "status": status,
        "note": note,
    }


def anneal_one(work, cell, bulk_geom, clf, settings, element, site_id, host, index, rng):
    """
    Place one random interstitial, anneal it, and return the CSV row.

    rng is private to this call. Worker threads do not share one Random.
    """
    temperature = settings["temperature"]
    job = insertion_job(element, site_id)
    placed = place_random_interstitial(cell, index, element, rng)
    if placed is None:
        return blank_row(job, temperature, site_id, host, element, "failed", "no open insertion point")
    atoms, point = placed
    work.mkdir(parents=True, exist_ok=True)
    vac.write_lammps_data(work / "start.data", cell, atoms)
    seed = lammps_seed(settings["site_seed"], site_id, element)
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
        e_sia = run_anneal(work, settings["lmp"])
    except RuntimeError as exc:
        row = blank_row(job, temperature, site_id, host, element, "failed", str(exc))
        row["insert_x"] = f"{point[0]:.6f}"
        row["insert_y"] = f"{point[1]:.6f}"
        row["insert_z"] = f"{point[2]:.6f}"
        return row
    label = label_relaxed(clf, bulk_geom, work / "relaxed.data")
    mu = vac.MU_EV[element]
    e_bulk = settings["e_bulk"]
    e_f = e_sia - e_bulk - mu
    note = label.get("note", "")
    row = {
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
        "geometry": label.get("geometry", ""),
        "orientation": label.get("orientation", ""),
        "pair": label.get("pair", ""),
        "element_a": label.get("element_a", ""),
        "element_b": label.get("element_b", ""),
        "class_key": "",
        "status": "ok",
        "note": note,
    }
    row["class_key"] = class_key(row)
    return row


def run_sia_jobs(bulk, bulk_geom, clf, plans, settings) -> tuple[int, int]:
    """
    Anneal every planned insertion. Returns (written, failed).

    Failures are rows in the CSV. The process still stops at the end when
    any anneal failed, after the class table has been written.
    """
    csv_path = energy_csv(settings["temperature"])
    new_file = not csv_path.is_file()
    lock = threading.Lock()
    n_done = 0
    n_fail = 0
    nworkers = settings["nworkers"]
    temperature = settings["temperature"]
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIA_FIELDS)
        if new_file:
            writer.writeheader()
            handle.flush()

        def work_one(plan):
            """Anneal one insertion on this worker."""
            element, index = plan
            atom = bulk.atoms[index]
            site_id = atom[0]
            host = bulk.masses[atom[1]]
            rng = random.Random(lammps_seed(settings["site_seed"], site_id, element))
            folder = vac.HERE / f"cases_{int(temperature)}K" / f"sia_{element}_{site_id}"
            return anneal_one(
                folder, bulk, bulk_geom, clf, settings,
                element, site_id, host, index, rng,
            )

        with ThreadPoolExecutor(max_workers=nworkers) as pool:
            futures = []
            for plan in plans:
                futures.append(pool.submit(work_one, plan))
            for fut in as_completed(futures):
                row = fut.result()
                with lock:
                    writer.writerow(row)
                    handle.flush()
                    n_done = n_done + 1
                    if row["status"] != "ok":
                        n_fail = n_fail + 1
                print(
                    f"{row['job']} {row['status']} {row['class_key']} "
                    f"Ef={row['formation_energy_SIA_eV']} ({n_done}/{len(plans)})",
                    flush=True,
                )
    return n_done, n_fail


def main() -> None:
    """Relax the bulk, anneal the random interstitials, then write the class counts."""
    settings = resolve_sia_run()
    bulk, e_bulk = vac.relax_bulk(
        settings["aged"], settings["tabgap"], settings["eam"], settings["lmp"]
    )
    settings["e_bulk"] = e_bulk
    clf = load_classifier()
    bulk_geom = clf.parse_lammps_data(vac.HERE / "bulk" / "relaxed.data")
    csv_path = energy_csv(settings["temperature"])
    done = load_done(csv_path)
    rng = random.Random(settings["site_seed"])
    plans = sample_insertions(bulk, settings["n_per_element"], done, rng)
    temperature = settings["temperature"]
    print(
        f"Annealing at {temperature:.0f} K. "
        f"{len(plans)} new insertions "
        f"({settings['n_per_element']} per element, already_done={len(done)}). "
        f"MD {settings['md_steps']} steps, cool {settings['cool_steps']} steps. "
        f"{vac.rank_line(settings['nworkers'])}.",
        flush=True,
    )
    n_done, n_fail = run_sia_jobs(bulk, bulk_geom, clf, plans, settings)
    write_class_summary(csv_path, class_csv(temperature))
    print(f"Wrote {csv_path} ({n_done} new rows, {n_fail} failed)", flush=True)
    if n_fail:
        die(f"{n_fail} anneal(s) failed. The class table still includes the rows that finished.")


if __name__ == "__main__":
    main()
