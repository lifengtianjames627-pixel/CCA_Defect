"""Shared LAMMPS cell I/O used by vacancy and SIA stages."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping


class Cell:
    """One LAMMPS cell: box, masses (type → element), and atoms."""

    def __init__(self, lo, hi, tilt, masses, atoms):
        self.lo = lo
        self.hi = hi
        self.tilt = tilt
        self.masses = masses  # type_id -> element symbol
        self.atoms = atoms  # list of (id, type, x, y, z)

    @property
    def lengths(self):
        return tuple(self.hi[i] - self.lo[i] for i in range(3))


def die(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def element_of_mass(mass: float, mass_to_element: Mapping[float, str]) -> str:
    for known, name in mass_to_element.items():
        if abs(float(known) - mass) < 1e-3:
            return name
    die(f"unrecognised mass {mass} (known={sorted(mass_to_element)})")
    return ""


def parse_lammps_data(path: Path, mass_to_element: Mapping[float, str]) -> Cell:
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
            masses[int(parts[0])] = element_of_mass(float(parts[1]), mass_to_element)
            continue
        if section == "atoms" and len(parts) >= 5 and parts[0].isdigit():
            atoms.append(
                (int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            )
    if not atoms or not masses:
        die(f"could not read atoms and masses from {path}")
    return Cell(tuple(lo), tuple(hi), tilt, masses, atoms)


def write_lammps_data(
    path: Path,
    cell: Cell,
    atoms,
    *,
    element_masses: Mapping[str, float],
    title: str = "formation-energy cell",
) -> None:
    xy, xz, yz = cell.tilt
    lines = [
        title,
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
        if element not in element_masses:
            die(f"no mass configured for element {element!r}")
        lines.append(f"{type_id} {element_masses[element]}")
    lines += ["", "Atoms # atomic", ""]
    for atom_id, type_id, x, y, z in atoms:
        lines.append(f"{atom_id} {type_id} {x:.10f} {y:.10f} {z:.10f}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def pair_block(cell: Cell, tabgap: Path, eam: Path) -> str:
    elements = [cell.masses[i] for i in sorted(cell.masses)]
    names = " ".join(elements)
    return f"""pair_style      hybrid/overlay tabgap eam/fs
pair_coeff      * * tabgap {tabgap.as_posix()} {names}
pair_coeff      * * eam/fs {eam.as_posix()} {names}
"""


def write_minimize_input(
    path: Path,
    data_name: str,
    cell: Cell,
    tabgap: Path,
    eam: Path,
    *,
    relax_box: bool,
) -> None:
    box = ""
    if relax_box:
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


def find_potential(pot_dir: Path) -> tuple[Path, Path]:
    tabgaps = sorted(Path(pot_dir).glob("*.tabgap"))
    eams = sorted(Path(pot_dir).glob("*.eam.fs"))
    if len(tabgaps) != 1 or len(eams) != 1:
        die(f"{pot_dir} needs exactly one .tabgap and one .eam.fs")
    return tabgaps[0].resolve(), eams[0].resolve()


def minimum_image(dx, dy, dz, cell: Cell):
    lx, ly, lz = cell.lengths
    xy, xz, yz = cell.tilt
    dz = dz - lz * round(dz / lz) if lz else dz
    dy = dy - yz * (dz / lz if lz else 0.0)
    dy = dy - ly * round(dy / ly) if ly else dy
    dx = dx - xz * (dz / lz if lz else 0.0) - xy * (dy / ly if ly else 0.0)
    dx = dx - lx * round(dx / lx) if lx else dx
    return dx, dy, dz


def nearest_neighbors(cell: Cell, index: int, count: int = 8) -> list[int]:
    _, _, x0, y0, z0 = cell.atoms[index]
    ranked = []
    for j, atom in enumerate(cell.atoms):
        if j == index:
            continue
        dx, dy, dz = minimum_image(atom[2] - x0, atom[3] - y0, atom[4] - z0, cell)
        ranked.append((dx * dx + dy * dy + dz * dz, j))
    ranked.sort()
    return [j for _, j in ranked[:count]]
