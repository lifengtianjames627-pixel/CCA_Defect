"""
Element-pair labeling for annealed SIAs — chemistry only.

Identifies which two species form the relaxed interstitial defect
(e.g. Cr-V, W-W). Does **not** assign geometry / orientation labels
(octahedral, <110>, <111>, <221>, crowdion, …). Those stay out of v1.
"""

from __future__ import annotations

import math
from typing import Sequence

from .lammps_cell import Cell, minimum_image


def pair_key(element_a: str, element_b: str) -> str:
    """Stable chemical-pair key: alphabetically sorted ``A-B``."""
    a, b = sorted((str(element_a), str(element_b)))
    return f"{a}-{b}"


def _site_distance(atom_xyz, site_xyz, cell: Cell) -> float:
    dx, dy, dz = minimum_image(
        atom_xyz[0] - site_xyz[0],
        atom_xyz[1] - site_xyz[1],
        atom_xyz[2] - site_xyz[2],
        cell,
    )
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _nearest_site_index(atom_xyz, bulk_positions, cell: Cell) -> tuple[int, float]:
    best_i = 0
    best_d = float("inf")
    for i, site_xyz in enumerate(bulk_positions):
        d = _site_distance(atom_xyz, site_xyz, cell)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d


def label_element_pair(bulk: Cell, sia: Cell) -> dict:
    """
    Wigner–Seitz style chemical pair from bulk vs annealed SIA cells.

    - Map each SIA atom to the nearest bulk lattice site.
    - Prefer a double-occupied site: those two atoms define ``element_a/b``.
    - Else fall back to the atom farthest from its site + its nearest neighbor.

    Returns keys: ``element_a``, ``element_b``, ``pair``, ``note``.
    Never returns geometry / orientation.
    """
    if len(sia.atoms) < 2:
        return {
            "element_a": "",
            "element_b": "",
            "pair": "unknown",
            "note": "pair failed: sia cell has <2 atoms",
        }
    if len(bulk.atoms) < 1:
        return {
            "element_a": "",
            "element_b": "",
            "pair": "unknown",
            "note": "pair failed: empty bulk cell",
        }

    bulk_positions = [(a[2], a[3], a[4]) for a in bulk.atoms]
    # sia atom index -> (site_index, distance, element)
    assignments: list[tuple[int, float, str, tuple[float, float, float]]] = []
    for atom in sia.atoms:
        xyz = (atom[2], atom[3], atom[4])
        site_i, dist = _nearest_site_index(xyz, bulk_positions, sia)
        element = sia.masses[atom[1]]
        assignments.append((site_i, dist, element, xyz))

    by_site: dict[int, list[tuple[float, str, tuple[float, float, float]]]] = {}
    for site_i, dist, element, xyz in assignments:
        by_site.setdefault(site_i, []).append((dist, element, xyz))

    doubles = [(site_i, members) for site_i, members in by_site.items() if len(members) >= 2]
    if doubles:
        # Prefer the tightest double (smallest mean distance to the site).
        def mean_dist(item):
            _site_i, members = item
            return sum(m[0] for m in members) / len(members)

        site_i, members = min(doubles, key=mean_dist)
        members_sorted = sorted(members, key=lambda m: m[0])[:2]
        el_a = members_sorted[0][1]
        el_b = members_sorted[1][1]
        a, b = sorted((el_a, el_b))
        return {
            "element_a": a,
            "element_b": b,
            "pair": pair_key(a, b),
            "note": (
                f"WS pair: double site occupancy={len(members)}; "
                f"site_distances={members_sorted[0][0]:.3f},{members_sorted[1][0]:.3f} A"
            ),
        }

    # Fallback: most displaced atom + its nearest neighbor in the SIA cell.
    far_i, far = max(enumerate(assignments), key=lambda it: it[1][1])
    _site_i, _dist, el_far, xyz_far = far
    best_j = None
    best_d = float("inf")
    for j, atom in enumerate(sia.atoms):
        if j == far_i:
            continue
        d = _site_distance(xyz_far, (atom[2], atom[3], atom[4]), sia)
        if d < best_d:
            best_d = d
            best_j = j
    if best_j is None:
        return {
            "element_a": el_far,
            "element_b": el_far,
            "pair": pair_key(el_far, el_far),
            "note": "WS pair fallback: single displaced atom only",
        }
    el_nn = sia.masses[sia.atoms[best_j][1]]
    a, b = sorted((el_far, el_nn))
    return {
        "element_a": a,
        "element_b": b,
        "pair": pair_key(a, b),
        "note": (
            f"WS pair fallback: no double site; "
            f"farthest+NN dist={best_d:.3f} A"
        ),
    }


def summarize_pairs(
    rows: Sequence[dict],
) -> list[dict]:
    """Aggregate ok rows by chemical ``pair`` (class_key == pair)."""
    buckets: dict[str, list[float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = str(row.get("pair") or row.get("class_key") or "")
        if not key:
            continue
        try:
            ef = float(row.get("formation_energy_SIA_eV", "nan"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ef):
            continue
        buckets.setdefault(key, []).append(ef)

    out = []
    for key in sorted(buckets):
        vals = sorted(buckets[key])
        n = len(vals)
        mid = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        out.append(
            {
                "pair": key,
                "class_key": key,
                "count": n,
                "E_min_eV": f"{vals[0]:.8f}",
                "E_median_eV": f"{mid:.8f}",
                "E_max_eV": f"{vals[-1]:.8f}",
            }
        )
    return out
