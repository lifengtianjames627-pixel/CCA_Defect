"""
W-Ta-Cr-V — single-process icet SQS driver.

Equiatomic bcc solid solution on one sublattice. The template geometry is the
conventional Im-3m cell of W (mp-91); Cr, Ta, V, and W then share every site.
Repeat (14, 14, 14) of that 2-atom cell is 5488 atoms (journal-scale;
comparable to the ~5600-atom Nb-Mo-Ta-W cells in Zhou et al.).

The executable code is split into three parts:
Part 1 (chemistry only), Part 2 (runtime logging + termination / MC hook),
and Part 3 (file I/O + ``main``).

Part 4 is parallel-design notes only (non-executable text).

Canonical entrypoint on the server: ``python3 W-Ta-Cr-V_SQS.py`` (this file).

Differences from the repo original ``W-Ta-Cr-V/SQS/W-Ta-Cr-V_SQS.py``
-----------------------------------------------------------------------
Chemistry (Part 1), icet ``ClusterSpace``, 5488 supercell, checkpoints, and
SIGTERM handling are the same. Executable changes in **this** copy:

1. **Q plateau early stop** (Part 2): If best ``Q`` is unchanged for
   ``SQS_PLATEAU_SEC`` seconds (default 900), the annealer runs final cost and
   cluster-vector checks (``_final_cost_and_vector_ok``). When both pass, MC
   ends with reason ``plateau_early_stop`` instead of waiting for
   ``SQS_N_MC_STEPS``. Set ``SQS_PLATEAU_SEC=0`` to disable. The repo original
   stops only at the step budget, Ctrl+C, or SIGTERM (no automatic plateau).
2. **Helpers for (1)**: ``DEFAULT_PLATEAU_IDLE_SEC``, ``PLATEAU_Q_ATOL``,
   ``_parse_plateau_idle_sec``, ``early_stop_requested``, ``request_early_stop``,
   and extra logic in ``_WTaCrVTargetClusterVectorAnnealing.generate_structure``.
3. **Cosmetic only in the repo file**: learning comments, module entry name
   ``W-Ta-Cr-V_SQS.py``, and ``validate_equiatomic_wtacrv`` error text
   ``expected …, got …`` vs ``got … expected …`` here.

``W-Ta-Cr-V_multipleseedSQS.py`` matches the repo aside from paths; neither
build includes plateau logic.

Security note: importing this module applies a **process-wide** monkeypatch to
``mchammer.ensembles.TargetClusterVectorAnnealing`` (required by icet). Do not import alongside
untrusted code in the same interpreter if that matters for your threat model.
"""
from __future__ import annotations

import logging
import math
import os
import signal
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.io import iread, read, write
from icet import ClusterSpace
from icet.input_output.logging_tools import set_log_config
from mchammer.calculators import TargetVectorCalculator, compare_cluster_vectors

def _parse_bounded_positive_int_env(name: str, default: int, *, min_v: int, max_v: int) -> int:
    """
    Read a positive integer from the environment with strict validation.

    Prevents ``ValueError`` crashes, negative / zero step counts (undefined MC behavior),
    and pathological huge values from a hostile or mistyped environment.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    if not raw.isdigit():
        raise ValueError(f"{name} must be a base-10 non-negative integer, got {raw!r}")
    v = int(raw)
    if v < min_v or v > max_v:
        raise ValueError(f"{name} must be between {min_v} and {max_v} inclusive, got {v}")
    return v


def _parse_bounded_nonnegative_int_env(name: str, default: int, *, max_v: int) -> int:
    """Read a non-negative integer env var with strict validation."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    if not raw.isdigit():
        raise ValueError(f"{name} must be a base-10 non-negative integer, got {raw!r}")
    v = int(raw)
    if v < 0 or v > max_v:
        raise ValueError(f"{name} must be between 0 and {max_v} inclusive, got {v}")
    return v


def _paths_stay_under(base: Path, paths: list[Path]) -> None:
    """Reject path escape if output basenames are ever changed to include ``..`` components."""
    base_r = base.resolve()
    for p in paths:
        try:
            p.resolve().relative_to(base_r)
        except ValueError as e:
            raise ValueError(f"Refusing to write outside script directory: {p}") from e


def assert_atom_count(atoms, expected: int, *, context: str) -> None:
    """Hard guard: SQS must always have exactly ``expected`` sites (no partial / truncated cells)."""
    n = len(atoms)
    if n != expected:
        raise RuntimeError(
            f"{context}: got {n} atoms, require exactly {expected}. "
            "Refusing to continue (avoids silent 'missing atoms' POSCARs)."
        )


# =============================================================================
# Part 1 — Chemistry (equiatomic W–Ta–Cr–V, one bcc sublattice)
# =============================================================================
PART1_CHEMISTRY_NOTES = r"""
Randomize occupants on the bcc sites of equiatomic W-Ta-Cr-V.

This follows the icet SQS entrypoint ``generate_sqs_from_supercells``:

- Read the conventional Im-3m W cell (2 sites once symmetry is applied).
- ``cs = ClusterSpace(primitive_structure, cutoffs, chemical_symbols_per_site)``.
- ``target_concentrations`` is a per-sublattice dict. There is one disordered
  sublattice; Cr, Ta, V, and W each have fraction 0.25.
- ``supercells = [primitive_structure.repeat((14, 14, 14))]`` is one fixed
  5488-site cell. Monte Carlo only swaps occupants. It does not add or
  remove sites.

Pair cutoff 4.8 Å covers the 1st-3rd bcc neighbor shells of this W cell
(3rd neighbor ≈ 4.48 Å) and stops before the 4th (≈ 5.26 Å). No triplet
cutoff: matching a long triplet list on a large quaternary cell is
unnecessary for the van de Walle pair-series SQS measure used by icet.

The cell metric is still pure W (a ≈ 3.170 Å). Relax the box with the tabGAP
before defect calculations. This script does not age the cell for short-range
order; that is a later Metropolis run driven by the potential, not by SQS.
"""

CIF_IN = "mp-91_W.cif"
# Pairs only, just past the 3rd bcc neighbor of a = 3.170 Å.
CUTOFFS = (4.8,)
# 2 * 14^3 = 5488 sites (Zhou et al. used ~5600–5760 for Nb-Mo-Ta-W).
SUPERCELL_REP = (14, 14, 14)
TARGET_TOTAL_ATOMS = 5488
SPECIES = ("Cr", "Ta", "V", "W")
# Reject accidentally huge / hostile CIF inputs before ASE allocates large structures.
MAX_CIF_BYTES = 32 * 1024 * 1024
# Cap frames read from multi-structure CIFs (malicious or broken files can enumerate without bound).
MAX_CIF_FRAMES = 2048
# Reject a single CIF frame that would allocate an enormous ASE structure.
MAX_CIF_ATOMS_PER_FRAME = 20_000


@dataclass(frozen=True)
class WTaCrVChemistry:
    """Everything needed to call ``generate_sqs_from_supercells``."""

    primitive: object
    cluster_space: ClusterSpace
    target_concentrations: dict[str, dict[str, float]]
    supercells: list[object]


def ensure_conventional_bcc(primitive):
    """
    mp-91 lists one W site plus Im-3m symmetry, including the body center.

    ASE sometimes returns that conventional cell already expanded (2 atoms) and
    sometimes returns only the origin site. A 1-atom cubic cell is not bcc.
    """
    n = len(primitive)
    if n == 2:
        return primitive
    if n != 1:
        raise ValueError(
            f"Expected 1 or 2 atoms in the W CIF frame, got {n}. "
            "Refusing to guess a supercell from an unexpected cell."
        )
    bcc = primitive.copy()
    center = np.dot([0.5, 0.5, 0.5], np.asarray(bcc.cell))
    bcc.append("W")
    bcc.positions[-1] = center
    return bcc


def build_chemical_symbols_per_site(primitive) -> list[list[str]]:
    """Every bcc site may be Cr, Ta, V, or W. There is no second sublattice."""
    out: list[list[str]] = []
    for atom in primitive:
        if atom.symbol not in SPECIES:
            raise ValueError(
                f"Unexpected species {atom.symbol!r} in the W template "
                f"(expected one of {SPECIES})."
            )
        out.append(list(SPECIES))
    return out


def target_concentrations_for_cluster_space(cluster_space: ClusterSpace) -> dict[str, dict[str, float]]:
    """Equiatomic fractions on the single disordered sublattice."""
    conc: dict[str, dict[str, float]] = {}
    prim = cluster_space.primitive_structure
    fraction = 1.0 / len(SPECIES)
    for sl in cluster_space.get_sublattices(prim):
        allowed = frozenset(sl.chemical_symbols)
        if len(allowed) == 1:
            continue
        if allowed != frozenset(SPECIES):
            raise ValueError(
                f"Unexpected disordered sublattice {sl.symbol!r} species={sorted(allowed)!r}"
            )
        conc[sl.symbol] = {el: fraction for el in SPECIES}
    for sl_sym, d in conc.items():
        s = float(sum(d.values()))
        if not math.isfinite(s) or abs(s - 1.0) > 1e-5:
            raise ValueError(
                f"target_concentrations for sublattice {sl_sym!r} must sum to 1.0 (±1e-5), got {s!r}"
            )
    if len(conc) != 1:
        raise ValueError(f"Expected exactly one disordered sublattice, found {len(conc)}.")
    return conc


def validate_equiatomic_wtacrv(atoms) -> None:
    """Cr, Ta, V, and W must each occupy exactly one quarter of the sites."""
    n = len(atoms)
    if n % len(SPECIES) != 0:
        raise RuntimeError(f"{n} sites cannot be split equally among {SPECIES}.")
    each = n // len(SPECIES)
    c = Counter(atoms.get_chemical_symbols())
    bad = [f"{el}: got {c.get(el, 0)} expected {each}" for el in SPECIES if c.get(el, 0) != each]
    extra = sorted(set(c) - set(SPECIES))
    if extra:
        bad.append(f"unexpected species {extra}")
    if bad:
        raise RuntimeError("SQS stoichiometry check failed: " + "; ".join(bad) + f" (counts={dict(c)!r})")


def prepare_wtacrv_chemistry(script_dir: Path) -> WTaCrVChemistry:
    """Read the W cell, expand 14×14×14, and build the equiatomic cluster space."""
    cif_path = script_dir / CIF_IN
    _paths_stay_under(script_dir, [cif_path])
    if cif_path.is_dir():
        raise IsADirectoryError(f"CIF path is a directory, not a file: {cif_path}")
    if not cif_path.is_file():
        raise FileNotFoundError(f"Missing {cif_path}")
    cif_size = cif_path.stat().st_size
    if cif_size > MAX_CIF_BYTES:
        raise ValueError(f"CIF is too large ({cif_size} bytes > {MAX_CIF_BYTES}); refusing to parse.")

    na, nb, nc = SUPERCELL_REP
    rep_product = int(na * nb * nc)
    if rep_product <= 0:
        raise ValueError(f"SUPERCELL_REP must have positive integers, got {SUPERCELL_REP!r}")
    if TARGET_TOTAL_ATOMS % rep_product != 0:
        raise ValueError(
            f"TARGET_TOTAL_ATOMS={TARGET_TOTAL_ATOMS} must be divisible by product(SUPERCELL_REP)={rep_product}"
        )
    expected_primitive_atoms = TARGET_TOTAL_ATOMS // rep_product
    if expected_primitive_atoms != 2:
        raise ValueError(
            f"This driver expects a 2-atom conventional bcc cell repeated to "
            f"{TARGET_TOTAL_ATOMS} atoms; product(SUPERCELL_REP) implies "
            f"{expected_primitive_atoms} atoms in the template."
        )
    frames: list[object] = []
    for i, frame in enumerate(iread(str(cif_path), format="cif")):
        if i >= MAX_CIF_FRAMES:
            raise ValueError(f"CIF has more than {MAX_CIF_FRAMES} structures; refusing unbounded parse.")
        if len(frame) > MAX_CIF_ATOMS_PER_FRAME:
            raise ValueError(f"CIF frame {i} has {len(frame)} atoms; refusing parse.")
        frames.append(frame)
    candidates = [frame for frame in frames if len(frame) in (1, 2)]
    if not candidates:
        sizes = [len(frame) for frame in frames]
        raise ValueError(f"No 1-atom or 2-atom frame in {cif_path}. Frame sizes were {sizes}.")
    primitive_structure = ensure_conventional_bcc(candidates[0])
    print(f"Conventional bcc template: {len(primitive_structure)} atoms", flush=True)
    if not all(primitive_structure.pbc):
        primitive_structure.pbc = (True, True, True)

    for c in CUTOFFS:
        if not math.isfinite(c) or c <= 0:
            raise ValueError(f"Cluster-space cutoffs must be finite and positive; got {CUTOFFS!r}")

    chemical_symbols_per_site = build_chemical_symbols_per_site(primitive_structure)
    cs = ClusterSpace(primitive_structure, list(CUTOFFS), chemical_symbols_per_site)
    print(cs, end="\n\n")

    target_concentrations = target_concentrations_for_cluster_space(cs)
    if not target_concentrations:
        raise ValueError("No disordered sublattice found; refusing SQS.")

    n_prim = len(primitive_structure)
    if n_prim != expected_primitive_atoms:
        raise ValueError(
            f"Template atom count mismatch: got {n_prim}, require exactly {expected_primitive_atoms} "
            f"for SUPERCELL_REP={SUPERCELL_REP} and TARGET_TOTAL_ATOMS={TARGET_TOTAL_ATOMS}."
        )

    supercells = [primitive_structure.repeat((na, nb, nc))]
    assert_atom_count(supercells[0], TARGET_TOTAL_ATOMS, context="Input supercell to generate_sqs_from_supercells")

    print(
        f"Supercells: one cell with {len(supercells[0])} atoms (repeat {SUPERCELL_REP}); "
        f"MC will only swap Cr/Ta/V/W on that fixed {TARGET_TOTAL_ATOMS}-site lattice.",
        flush=True,
    )

    return WTaCrVChemistry(
        primitive=primitive_structure,
        cluster_space=cs,
        target_concentrations=target_concentrations,
        supercells=supercells,
    )


def _format_float(x: float) -> str:
    if not math.isfinite(x):
        if math.isnan(x):
            return "nan"
        return "inf" if x > 0 else "-inf"
    return f"{x:.6g}"


# Driver log level configured in ``main``; defaults to WARNING.
_DRIVER_LOG_LEVEL_NAME: str = "WARNING"


def configure_driver_logging() -> None:
    """Drive icet/mchammer verbosity from ``SQS_LOG_LEVEL`` (default WARNING)."""
    global _DRIVER_LOG_LEVEL_NAME
    raw = (os.environ.get("SQS_LOG_LEVEL") or "WARNING").strip().upper()
    if raw == "":
        raw = "WARNING"
    allowed = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    if raw not in allowed:
        raise ValueError(f"SQS_LOG_LEVEL must be one of {', '.join(allowed)}; got {raw!r}")
    _DRIVER_LOG_LEVEL_NAME = raw
    set_log_config(level=raw)
    level_no = getattr(logging, raw)
    for name in ("icet", "mchammer"):
        logging.getLogger(name).setLevel(level_no)


# =============================================================================
# Part 2 — Runtime logging + termination (MC control)
# =============================================================================
# Cooperative SIGINT and hard SIGTERM flush: on ``kill <pid>`` write POSCAR +
# full cluster-vector deviation report + global Q cost, then exit(143).
# Patch ``TargetClusterVectorAnnealing`` before importing ``generate_sqs_from_supercells``.

import mchammer.ensembles as _mch_ens

Q_OPTIMALITY_WEIGHT = 1.0
Q_TOL = 1e-5
MINUTE_LOG_INTERVAL_SEC = 60.0
QUIET_CHECKPOINT_LOGS = True
# If best Q is unchanged for this long, run final cost + CV checks and maybe stop.
DEFAULT_PLATEAU_IDLE_SEC = 900  # 15 minutes
# Treat two Q values as the same plateau value within this absolute tolerance.
PLATEAU_Q_ATOL = 1e-9

_TCV = _mch_ens.TargetClusterVectorAnnealing
_interrupt_event = threading.Event()
_early_stop_event = threading.Event()

_sigterm_poscar_path: Path | None = None
_sigterm_deviations_path: Path | None = None
_sigterm_cost_path: Path | None = None
_sigterm_script_root: Path | None = None
_sigterm_annealer_ref: object | None = None
_sigterm_cluster_space: ClusterSpace | None = None
_sigterm_target_vector: np.ndarray | None = None
_sigterm_last_best_atoms: object | None = None
_sigterm_flush_armed: bool = False
_sigterm_in_progress: bool = False

_runtime_ckpt_poscar_path: Path | None = None
_runtime_ckpt_deviations_path: Path | None = None
_runtime_ckpt_cost_path: Path | None = None
_runtime_ckpt_script_root: Path | None = None
_runtime_ckpt_interval_sec: int = 30


def _sigterm_prerequisites_ready() -> tuple[bool, list[str]]:
    missing: list[str] = []
    if _sigterm_script_root is None:
        missing.append("script_root")
    if _sigterm_poscar_path is None:
        missing.append("poscar_path")
    if _sigterm_deviations_path is None:
        missing.append("deviations_path")
    if _sigterm_cost_path is None:
        missing.append("cost_path")
    if _sigterm_cluster_space is None:
        missing.append("cluster_space")
    if _sigterm_target_vector is None:
        missing.append("target_vector")
    return (len(missing) == 0, missing)


def _refresh_sigterm_arm_state() -> None:
    global _sigterm_flush_armed
    ready, _ = _sigterm_prerequisites_ready()
    _sigterm_flush_armed = ready


def _cache_sigterm_best_atoms(atoms) -> None:
    global _sigterm_last_best_atoms
    if atoms is None:
        return
    _sigterm_last_best_atoms = atoms.copy() if hasattr(atoms, "copy") else atoms


def _compute_live_q_and_worst_cv_diff(
    *,
    cluster_space: ClusterSpace,
    target_vector: np.ndarray,
    atoms,
    optimality_weight: float,
    tol: float,
) -> tuple[float, float]:
    """
    Lightweight live metrics based on official ``compare_cluster_vectors``.

    Returns ``(q_score, worst_cv_diff_excluding_zerolet)``.
    """
    cv_sys = np.asarray(cluster_space.get_cluster_vector(atoms), dtype=float)
    cv_target = np.asarray(target_vector, dtype=float)
    if cv_sys.shape != cv_target.shape:
        raise RuntimeError(f"Cluster vector length mismatch: sys {cv_sys.shape} vs target {cv_target.shape}")
    q_score = float(
        compare_cluster_vectors(
            cv_1=cv_sys,
            cv_2=cv_target,
            as_list=cluster_space.as_list,
            weights=None,
            optimality_weight=optimality_weight,
            tol=tol,
        )
    )
    abs_dev = np.abs(cv_sys - cv_target)
    if len(abs_dev) > 1:
        worst = float(np.max(abs_dev[1:]))  # skip zerolet index 0
    else:
        worst = float(np.max(abs_dev))
    return q_score, worst


def set_sigterm_output_paths(
    *,
    poscar: Path | None,
    deviations_tsv: Path | None,
    cost_txt: Path | None,
    script_root: Path | None,
) -> None:
    global _sigterm_poscar_path, _sigterm_deviations_path, _sigterm_cost_path, _sigterm_script_root
    if script_root is not None:
        to_check = [p for p in (poscar, deviations_tsv, cost_txt) if p is not None]
        if to_check:
            _paths_stay_under(script_root, to_check)
    _sigterm_poscar_path = poscar
    _sigterm_deviations_path = deviations_tsv
    _sigterm_cost_path = cost_txt
    _sigterm_script_root = script_root
    _refresh_sigterm_arm_state()


def set_sigterm_cluster_context(cluster_space: ClusterSpace, target_vector: np.ndarray) -> None:
    """Fallback context for SIGTERM writes when annealer internals are unavailable."""
    global _sigterm_cluster_space, _sigterm_target_vector
    _sigterm_cluster_space = cluster_space
    _sigterm_target_vector = np.asarray(target_vector, dtype=float).copy()
    _refresh_sigterm_arm_state()


def get_sigterm_target_vector() -> np.ndarray | None:
    """Latest annealer target vector captured at runtime (copy for safety)."""
    if _sigterm_target_vector is None:
        return None
    return np.asarray(_sigterm_target_vector, dtype=float).copy()


def set_runtime_checkpoint_config(
    *,
    poscar: Path | None,
    deviations_tsv: Path | None,
    cost_txt: Path | None,
    script_root: Path | None,
    interval_sec: int,
) -> None:
    """Configure periodic best-so-far checkpoint writes during MC."""
    global _runtime_ckpt_poscar_path, _runtime_ckpt_deviations_path, _runtime_ckpt_cost_path
    global _runtime_ckpt_script_root, _runtime_ckpt_interval_sec
    if script_root is not None:
        to_check = [p for p in (poscar, deviations_tsv, cost_txt) if p is not None]
        if to_check:
            _paths_stay_under(script_root, to_check)
    _runtime_ckpt_poscar_path = poscar
    _runtime_ckpt_deviations_path = deviations_tsv
    _runtime_ckpt_cost_path = cost_txt
    _runtime_ckpt_script_root = script_root
    _runtime_ckpt_interval_sec = int(interval_sec)


def _best_atoms_snapshot(ann: object) -> object | None:
    st = getattr(ann, "best_structure", None)
    if st is None:
        st = getattr(ann, "structure", None)
    if st is None:
        st = getattr(ann, "_structure", None)
    if st is None:
        subs = getattr(ann, "_sub_ensembles", None)
        if subs:
            ens0 = subs[0]
            st = getattr(ens0, "structure", None)
            if st is None:
                st = getattr(ens0, "_structure", None)
    if isinstance(st, list) and st:
        st = st[0]
    if st is None:
        return None
    return st.copy() if hasattr(st, "copy") else st


def _resolve_cv_context(ann: object | None) -> tuple[ClusterSpace | None, np.ndarray | None]:
    cs = getattr(ann, "_cv_cluster_space", None) if ann is not None else None
    tv = getattr(ann, "_gamma_random", None) if ann is not None else None
    if cs is None:
        cs = _sigterm_cluster_space
    if tv is None:
        tv = _sigterm_target_vector
    if tv is not None:
        tv = np.asarray(tv, dtype=float)
    return cs, tv


def _write_runtime_checkpoint(ann: object, *, reason: str) -> tuple[bool, bool, bool, float]:
    root = _runtime_ckpt_script_root
    poscar_path = _runtime_ckpt_poscar_path
    dev_path = _runtime_ckpt_deviations_path
    cost_path = _runtime_ckpt_cost_path
    if root is None or poscar_path is None or dev_path is None or cost_path is None:
        return (False, False, False, float("nan"))

    atoms = _best_atoms_snapshot(ann)
    if atoms is None:
        return (False, False, False, float("nan"))
    assert_atom_count(atoms, TARGET_TOTAL_ATOMS, context=f"checkpoint snapshot ({reason})")
    _cache_sigterm_best_atoms(atoms)

    poscar_ok = False
    tsv_ok = False
    cost_ok = False
    q_val = float("nan")
    errs: list[str] = []

    try:
        _paths_stay_under(root, [poscar_path])
        write_poscar_atomic(poscar_path, atoms)
        poscar_ok = True
    except Exception as exc:  # noqa: BLE001
        errs.append(f"poscar:{type(exc).__name__}")

    cs, tv = _resolve_cv_context(ann)
    if cs is not None and tv is not None:
        try:
            _paths_stay_under(root, [dev_path, cost_path])
            q_val = write_cluster_vector_deviations_and_cost(
                deviations_path=dev_path,
                cost_path=cost_path,
                cluster_space=cs,
                atoms=atoms,
                target_vector=tv,
                optimality_weight=Q_OPTIMALITY_WEIGHT,
                tol=Q_TOL,
            )
            tsv_ok = True
            cost_ok = True
        except Exception as exc:  # noqa: BLE001
            errs.append(f"report:{type(exc).__name__}")
    else:
        errs.append("report:missing_cv_context")

    if not QUIET_CHECKPOINT_LOGS:
        print(
            f"SQS_CHECKPOINT reason={reason} poscar_ok={poscar_ok} tsv_ok={tsv_ok} cost_ok={cost_ok} "
            f"q={_format_float(q_val)} errors={','.join(errs) if errs else 'none'}",
            flush=True,
        )
    return (poscar_ok, tsv_ok, cost_ok, q_val)


def register_sigterm_annealer(annealer: object) -> None:
    global _sigterm_annealer_ref
    _sigterm_annealer_ref = annealer


def unregister_sigterm_annealer() -> None:
    global _sigterm_annealer_ref
    _sigterm_annealer_ref = None


def interrupt_reset() -> None:
    _interrupt_event.clear()
    _early_stop_event.clear()


def interrupt_requested() -> bool:
    return _interrupt_event.is_set()


def early_stop_requested() -> bool:
    return _early_stop_event.is_set()


def request_early_stop() -> None:
    _early_stop_event.set()


def _parse_plateau_idle_sec() -> float:
    """Seconds of unchanged best-Q before attempting early stop (env SQS_PLATEAU_SEC)."""
    raw = os.environ.get("SQS_PLATEAU_SEC")
    if raw is None or raw.strip() == "":
        return float(DEFAULT_PLATEAU_IDLE_SEC)
    raw = raw.strip()
    try:
        value = float(raw)
    except ValueError as e:
        raise ValueError(f"SQS_PLATEAU_SEC must be a non-negative number, got {raw!r}") from e
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"SQS_PLATEAU_SEC must be finite and >= 0, got {value!r}")
    return value


def _final_cost_and_vector_ok(
    *,
    cluster_space: ClusterSpace,
    target_vector: np.ndarray,
    atoms,
) -> tuple[bool, float, float, str]:
    """
    Run the same final cost (Q) and cluster-vector deviation checks used at write-out.

    Returns ``(ok, q, worst_abs_dev, reason)``.
    """
    try:
        rep = compute_cluster_vector_report(
            cluster_space=cluster_space,
            atoms=atoms,
            target_vector=target_vector,
            optimality_weight=Q_OPTIMALITY_WEIGHT,
            tol=Q_TOL,
        )
    except Exception as exc:  # noqa: BLE001
        return False, float("nan"), float("nan"), f"report_failed:{type(exc).__name__}:{exc}"

    q = float(rep["q"])
    worst = float(rep["worst_abs_dev"])
    abs_dev = np.asarray(rep["abs_dev"], dtype=float)
    if not math.isfinite(q):
        return False, q, worst, "cost_check_failed:non_finite_Q"
    if not math.isfinite(worst) or not np.all(np.isfinite(abs_dev)):
        return False, q, worst, "vector_check_failed:non_finite_deviation"
    return True, q, worst, "ok"


def _interrupt_handler(_signum: int, _frame) -> None:
    _interrupt_event.set()


def _sigterm_handler(_signum: int, _frame) -> None:
    global _sigterm_in_progress
    if _sigterm_in_progress:
        os._exit(143)
    _sigterm_in_progress = True

    path = _sigterm_poscar_path
    dev_path = _sigterm_deviations_path
    cost_path = _sigterm_cost_path
    root = _sigterm_script_root
    ann = _sigterm_annealer_ref

    wrote_p = False
    wrote_t = False
    wrote_c = False
    q_val = float("nan")
    errors: list[str] = []

    ready, missing = _sigterm_prerequisites_ready()
    atoms = _best_atoms_snapshot(ann) if ann is not None else None
    if atoms is None:
        cached = _sigterm_last_best_atoms
        if cached is not None:
            atoms = cached.copy() if hasattr(cached, "copy") else cached
    if atoms is not None and hasattr(atoms, "__len__") and len(atoms) <= 0:
        atoms = None
    if atoms is not None:
        try:
            assert_atom_count(atoms, TARGET_TOTAL_ATOMS, context="SIGTERM snapshot")
        except Exception as exc:  # noqa: BLE001
            atoms = None
            errors.append(f"snapshot:{type(exc).__name__}")

    if atoms is not None and path is not None and root is not None:
        try:
            _paths_stay_under(root, [path])
            write_poscar_atomic(path, atoms)
            wrote_p = True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"poscar:{type(exc).__name__}")

    if atoms is not None and dev_path is not None and cost_path is not None and root is not None:
        cs, tv = _resolve_cv_context(ann)
        if cs is not None and tv is not None:
            try:
                _paths_stay_under(root, [dev_path, cost_path])
                q_val = write_cluster_vector_deviations_and_cost(
                    deviations_path=dev_path,
                    cost_path=cost_path,
                    cluster_space=cs,
                    atoms=atoms,
                    target_vector=tv,
                    optimality_weight=Q_OPTIMALITY_WEIGHT,
                    tol=Q_TOL,
                )
                wrote_t = True
                wrote_c = True
            except Exception as exc:  # noqa: BLE001
                errors.append(f"report:{type(exc).__name__}")
        else:
            errors.append("report:missing_cv_context")
    if atoms is None:
        errors.append("snapshot:no_atoms")
    if not ready:
        errors.append("not_armed:" + ",".join(missing))

    try:
        err_text = ",".join(errors) if errors else "none"
        sys.stderr.write(
            f"SQS_SIGTERM armed={_sigterm_flush_armed} q={_format_float(q_val)} "
            f"poscar_ok={wrote_p} tsv_ok={wrote_t} cost_ok={wrote_c} errors={err_text}\n"
        )
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(143)


def install_interrupt_handlers() -> None:
    signal.signal(signal.SIGINT, _interrupt_handler)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _sigterm_handler)
        except OSError:
            pass


class _WTaCrVTargetClusterVectorAnnealing(_TCV):
    """
    MC controller with:
    - strict best-structure copy semantics,
    - plateau early-stop: if best Q is unchanged for SQS_PLATEAU_SEC (default 15 min),
      run final cost + cluster-vector checks; if both pass, write cell and return,
    - periodic + improvement checkpoints,
    - SIGTERM-compatible best-so-far flush outputs.
    """

    def _do_trial_step(self):
        best_score_before = self.best_score
        super()._do_trial_step()
        if self.best_score < best_score_before:
            self._best_structure = self.best_structure.copy()
            _cache_sigterm_best_atoms(self._best_structure)
            self._best_score_improvement_count = int(getattr(self, "_best_score_improvement_count", 0)) + 1
            self._last_q_improve_t = time.monotonic()
            try:
                self._plateau_q_ref = float(self.best_score)
            except (TypeError, ValueError):
                self._plateau_q_ref = float("nan")
            _write_runtime_checkpoint(self, reason="best_improved")

    def _current_atoms_for_cv(self):
        st = getattr(self, "structure", None)
        if st is None:
            st = getattr(self, "_structure", None)
        if st is None:
            subs = getattr(self, "_sub_ensembles", None)
            if subs:
                ens0 = subs[0]
                st = getattr(ens0, "structure", None)
                if st is None:
                    st = getattr(ens0, "_structure", None)
        if isinstance(st, list):
            if not st:
                raise RuntimeError("TargetClusterVectorAnnealing current structure list is empty")
            return st[0]
        if st is None:
            st = _best_atoms_snapshot(self)
        if st is None:
            raise RuntimeError("TargetClusterVectorAnnealing cannot resolve current structure")
        return st

    def _worst_abs_cluster_deviation(self) -> float:
        if interrupt_requested():
            return float("inf")
        atoms = self._current_atoms_for_cv()
        cv = np.asarray(self._cv_cluster_space.get_cluster_vector(atoms), dtype=float)
        tgt = self._gamma_random
        diff = np.abs(cv - tgt)
        if len(diff) <= 1:
            worst = float(np.max(diff))
        else:
            worst = float(np.max(diff[1:]))  # ignore zerolet element index 0
        self._last_worst_dev = worst
        return worst

    def generate_structure(self, number_of_trial_steps=None):
        sub_ensembles = getattr(self, "_sub_ensembles", None)
        n_sub = len(sub_ensembles) if sub_ensembles is not None else 1
        if n_sub <= 0:
            n_sub = 1
        if number_of_trial_steps is None:
            max_trials = int(3000 * n_sub)
        else:
            max_trials = int(number_of_trial_steps)

        self._n_steps = max_trials
        try:
            self.n_steps = max_trials
        except (AttributeError, TypeError):
            pass

        self._temperature = self._T_start
        self._total_trials = 0
        self._accepted_trials = 0

        calculators = getattr(self, "calculators", None)
        if calculators is None:
            calculators = getattr(self, "_calculators", None)
        if not calculators and sub_ensembles is not None:
            recovered: list[object] = []
            for ens in sub_ensembles:
                calc = getattr(ens, "calculator", None)
                if calc is None:
                    calc = getattr(ens, "_calculator", None)
                if calc is not None:
                    recovered.append(calc)
            calculators = recovered
        if not calculators:
            raise RuntimeError(
                "TargetClusterVectorAnnealing: no calculators found on "
                "attributes calculators/_calculators/_sub_ensembles"
            )
        calc0 = calculators[0]
        self._cv_cluster_space = calc0.cluster_space
        self._gamma_random = np.asarray(calc0.target_vector, dtype=float)
        set_sigterm_cluster_context(self._cv_cluster_space, self._gamma_random)

        interrupt_reset()
        self._last_worst_dev = float("inf")
        self._best_score_improvement_count = 0
        plateau_idle_sec = _parse_plateau_idle_sec()
        self._last_q_improve_t = time.monotonic()
        try:
            self._plateau_q_ref = float(self.best_score)
        except (TypeError, ValueError):
            self._plateau_q_ref = float("nan")

        ann = logging.getLogger("icet.target_cluster_vector_annealing")
        ann.setLevel(getattr(logging, _DRIVER_LOG_LEVEL_NAME))

        interrupted = False
        early_stopped = False
        minute_idx = 0
        next_minute_t = time.monotonic()
        last_q = float("nan")
        last_worst = float("inf")

        try:
            register_sigterm_annealer(self)
            _cache_sigterm_best_atoms(_best_atoms_snapshot(self))
            _write_runtime_checkpoint(self, reason="mc_start")
            try:
                bq0 = float(self.best_score)
            except (TypeError, ValueError):
                bq0 = float("nan")
            print(
                f"SQS_MC start | max_trials={max_trials} | best_Q={_format_float(bq0)} | "
                f"worst_dev={_format_float(float(getattr(self, '_last_worst_dev', float('inf'))))} | "
                f"plateau_idle_sec={_format_float(plateau_idle_sec)}",
                flush=True,
            )

            trials_done = 0
            while trials_done < max_trials:
                if interrupt_requested() or early_stop_requested():
                    interrupted = interrupt_requested()
                    early_stopped = early_stop_requested()
                    break
                self._do_trial_step()
                trials_done += 1
                if interrupt_requested() or early_stop_requested():
                    interrupted = interrupt_requested()
                    early_stopped = early_stop_requested()
                    break

                worst = self._worst_abs_cluster_deviation()

                now = time.monotonic()
                if now >= next_minute_t:
                    minute_idx += 1
                    atoms_for_log = _best_atoms_snapshot(self)
                    if atoms_for_log is not None:
                        try:
                            q_now, worst_now = _compute_live_q_and_worst_cv_diff(
                                cluster_space=self._cv_cluster_space,
                                target_vector=self._gamma_random,
                                atoms=atoms_for_log,
                                optimality_weight=Q_OPTIMALITY_WEIGHT,
                                tol=Q_TOL,
                            )
                            last_q = q_now
                            last_worst = worst_now
                        except Exception:
                            pass
                    # Keep plateau clock in sync with logged best-Q (lower is better).
                    try:
                        best_q_now = float(self.best_score)
                    except (TypeError, ValueError):
                        best_q_now = last_q
                    if math.isfinite(best_q_now):
                        ref = float(getattr(self, "_plateau_q_ref", float("nan")))
                        if (not math.isfinite(ref)) or abs(best_q_now - ref) > PLATEAU_Q_ATOL:
                            self._plateau_q_ref = best_q_now
                            self._last_q_improve_t = now
                    idle = now - float(getattr(self, "_last_q_improve_t", now))
                    status = "Running"
                    if plateau_idle_sec > 0 and idle >= plateau_idle_sec:
                        status = f"Plateau({_format_float(idle)}s)"
                    print(
                        f"MINUTE: {minute_idx} | Q SCORE: {_format_float(last_q)} | "
                        f"WORST CV DIFF: {_format_float(last_worst)} | STATUS: {status}",
                        flush=True,
                    )
                    while next_minute_t <= now:
                        next_minute_t += MINUTE_LOG_INTERVAL_SEC

                    # After Q is flat for plateau_idle_sec, run final cost + vector checks.
                    if plateau_idle_sec > 0 and idle >= plateau_idle_sec and atoms_for_log is not None:
                        ok, q_chk, worst_chk, reason = _final_cost_and_vector_ok(
                            cluster_space=self._cv_cluster_space,
                            target_vector=self._gamma_random,
                            atoms=atoms_for_log,
                        )
                        print(
                            f"SQS_PLATEAU_CHECK idle_sec={_format_float(idle)} "
                            f"cost_ok={ok} vector_ok={ok} Q={_format_float(q_chk)} "
                            f"worst_cv={_format_float(worst_chk)} reason={reason}",
                            flush=True,
                        )
                        if ok:
                            _write_runtime_checkpoint(self, reason="plateau_early_stop")
                            request_early_stop()
                            early_stopped = True
                            print(
                                "SQS_EARLY_STOP: Q unchanged for "
                                f"{_format_float(idle)}s; final cost + cluster-vector "
                                "checks passed — writing best cell and exiting MC.",
                                flush=True,
                            )
                            break
                        # Failed checks: keep searching; require a fresh plateau window.
                        self._last_q_improve_t = now
                        print(
                            "SQS_PLATEAU_CHECK failed; continuing MC and resetting plateau timer.",
                            flush=True,
                        )

                ckpt_iv = int(_runtime_ckpt_interval_sec)
                if ckpt_iv > 0:
                    now = time.monotonic()
                    last = float(getattr(self, "_last_runtime_ckpt_t", 0.0))
                    if now - last >= float(ckpt_iv):
                        _write_runtime_checkpoint(self, reason="periodic")
                        self._last_runtime_ckpt_t = now

        finally:
            atoms_for_log = _best_atoms_snapshot(self)
            if atoms_for_log is not None:
                try:
                    q_now, worst_now = _compute_live_q_and_worst_cv_diff(
                        cluster_space=self._cv_cluster_space,
                        target_vector=self._gamma_random,
                        atoms=atoms_for_log,
                        optimality_weight=Q_OPTIMALITY_WEIGHT,
                        tol=Q_TOL,
                    )
                    last_q = q_now
                    last_worst = worst_now
                except Exception:
                    pass
            minute_idx += 1
            end_status = "Sleeping"
            if early_stopped or early_stop_requested():
                end_status = "EarlyStop"
            elif interrupted or interrupt_requested():
                end_status = "Interrupted"
            print(
                f"MINUTE: {minute_idx} | Q SCORE: {_format_float(last_q)} | "
                f"WORST CV DIFF: {_format_float(last_worst)} | STATUS: {end_status}",
                flush=True,
            )
            _write_runtime_checkpoint(self, reason="mc_exit")
            unregister_sigterm_annealer()

        out = _best_atoms_snapshot(self)
        if out is None:
            raise RuntimeError("TargetClusterVectorAnnealing did not produce a best structure")
        return out


_mch_ens.TargetClusterVectorAnnealing = _WTaCrVTargetClusterVectorAnnealing

from icet.tools.structure_generation import generate_sqs_from_supercells  # noqa: E402


# =============================================================================
# Part 3 — Output + driver
# =============================================================================
# VASP ``poscar_out`` / ``poscar_out_interrupt`` + full cluster-vector diff TSV + global cost (Q).
# Wires Part 1 chemistry → Part 2 interrupt hooks → ``generate_sqs_from_supercells``.

POSCAR_OUT = "poscar_out"
POSCAR_OUT_INTERRUPT = "poscar_out_interrupt"
FINAL_CLUSTER_VECTOR_DEVIATIONS_TSV = "final_cluster_vector_deviations.tsv"
FINAL_COST_FUNCTION_TXT = "final_cost_function.txt"
# Larger cell needs more swap trials; override with SQS_N_MC_STEPS if needed.
DEFAULT_N_MC_STEPS = 2_000_000
MAX_N_MC_STEPS = 50_000_000
# Bound TSV metadata column size (orbit repr can be enormous on pathological cluster spaces).
ORBIT_META_MAX_CHARS = 400


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text via a temp file in the same directory, then ``os.replace`` (reduces torn files on crash)."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".sqs_", suffix=".tmp", dir=str(parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(text, encoding=encoding)
        os.replace(str(tmp_path), str(path))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_poscar_atomic(path: Path, atoms) -> None:
    """Write a viewer-friendly VASP POSCAR via temp file + atomic replace."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".sqs_poscar_", suffix=".tmp", dir=str(parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        atoms_to_write = atoms.copy() if hasattr(atoms, "copy") else atoms
        if hasattr(atoms_to_write, "pbc") and not all(atoms_to_write.pbc):
            atoms_to_write.pbc = (True, True, True)
        if hasattr(atoms_to_write, "wrap"):
            atoms_to_write.wrap()
        if not hasattr(atoms_to_write, "cell"):
            raise RuntimeError("POSCAR write failed: atoms object has no cell.")
        if abs(float(atoms_to_write.cell.volume)) <= 1e-12:
            raise RuntimeError("POSCAR write failed: non-physical zero-volume cell.")

        # Cartesian + sorted species tends to be accepted by more viewers (including VESTA).
        write(str(tmp_path), atoms_to_write, format="vasp", vasp5=True, direct=False, sort=True)
        chk = read(str(tmp_path), format="vasp")
        assert_atom_count(chk, len(atoms_to_write), context="POSCAR temp read-back")
        os.replace(str(tmp_path), str(path))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def orbit_row_metadata(cluster_space: ClusterSpace, index: int) -> str:
    orbits = cluster_space.as_list
    if index >= len(orbits):
        return ""
    o = orbits[index]
    if isinstance(o, dict):
        parts = [f"{k}={o[k]!r}" for k in ("order", "radius", "multiplicity") if k in o]
        s = "; ".join(parts) if parts else repr(o)
    else:
        s = repr(o)
    if len(s) > ORBIT_META_MAX_CHARS:
        return s[: ORBIT_META_MAX_CHARS - 3] + "..."
    return s


def compute_cluster_vector_report(
    *,
    cluster_space: ClusterSpace,
    atoms,
    target_vector: np.ndarray,
    optimality_weight: float,
    tol: float,
) -> dict[str, object]:
    """
    Compute full per-component cluster-vector deviations and global Q using official mchammer APIs.

    Q is evaluated via both ``compare_cluster_vectors`` and ``TargetVectorCalculator`` and must agree.
    """
    cv_sys = np.asarray(cluster_space.get_cluster_vector(atoms), dtype=float)
    cv_target = np.asarray(target_vector, dtype=float)
    if cv_sys.shape != cv_target.shape:
        raise RuntimeError(f"Cluster vector length mismatch: sys {cv_sys.shape} vs target {cv_target.shape}")

    abs_dev = np.abs(cv_sys - cv_target)
    q_compare = float(
        compare_cluster_vectors(
            cv_1=cv_sys,
            cv_2=cv_target,
            as_list=cluster_space.as_list,
            weights=None,
            optimality_weight=optimality_weight,
            tol=tol,
        )
    )
    tv_calc = TargetVectorCalculator(
        structure=atoms.copy(),
        cluster_space=cluster_space,
        target_vector=cv_target.tolist(),
        weights=None,
        optimality_weight=optimality_weight,
        optimality_tol=tol,
        name="Target vector calculator",
    )
    q_calc = float(tv_calc.calculate_total(occupations=atoms.get_atomic_numbers()))
    if not math.isfinite(q_compare) or not math.isfinite(q_calc):
        raise RuntimeError(f"Non-finite Q detected (compare={q_compare!r}, calculator={q_calc!r})")
    if abs(q_compare - q_calc) > 1e-9:
        raise RuntimeError(
            f"Q mismatch between compare_cluster_vectors ({q_compare}) and "
            f"TargetVectorCalculator ({q_calc})"
        )

    # Zerolet is index 0; worst deviation for stopping/reporting excludes it when possible.
    if len(abs_dev) > 1:
        worst_abs_dev = float(np.max(abs_dev[1:]))
    else:
        worst_abs_dev = float(np.max(abs_dev))

    return {
        "cv_sys": cv_sys,
        "cv_target": cv_target,
        "abs_dev": abs_dev,
        "q": q_compare,
        "worst_abs_dev": worst_abs_dev,
    }


def write_cluster_vector_deviations_and_cost(
    *,
    deviations_path: Path,
    cost_path: Path,
    cluster_space: ClusterSpace,
    atoms,
    target_vector: np.ndarray,
    optimality_weight: float,
    tol: float,
) -> float:
    """Write all cluster-vector components (e.g., all 51 entries in this setup) plus global Q."""
    rep = compute_cluster_vector_report(
        cluster_space=cluster_space,
        atoms=atoms,
        target_vector=target_vector,
        optimality_weight=optimality_weight,
        tol=tol,
    )
    cv_sys = rep["cv_sys"]
    cv_target = rep["cv_target"]
    abs_dev = rep["abs_dev"]
    q = float(rep["q"])
    worst_abs_dev = float(rep["worst_abs_dev"])

    lines = ["index\torbit_meta\tgamma_sys\tgamma_target\tabs_deviation\n"]
    for i in range(len(cv_sys)):
        meta = orbit_row_metadata(cluster_space, i).replace("\t", " ")
        lines.append(
            f"{i}\t{meta}\t{cv_sys[i]:.16g}\t{cv_target[i]:.16g}\t{abs_dev[i]:.16g}\n"
        )
    write_text_atomic(deviations_path, "".join(lines), encoding="utf-8")

    cost_text = (
        f"Q={q:.16g}\n"
        f"worst_abs_deviation_excluding_zerolet={worst_abs_dev:.16g}\n"
        f"n_cluster_vector_components={len(cv_sys)}\n"
        f"optimality_weight={optimality_weight:.16g}\n"
        f"optimality_tol={tol:.16g}\n"
    )
    write_text_atomic(cost_path, cost_text, encoding="utf-8")
    return q


def main() -> None:
    root = Path(__file__).resolve().parent

    configure_driver_logging()
    print(
        f"Logging: stdout print mode (SQS_LOG_LEVEL={_DRIVER_LOG_LEVEL_NAME}).",
        flush=True,
    )
    print("Progress: checkpoints are written during MC and on termination.", flush=True)

    print(
        "SQS note: `poscar_out_interrupt` is written when MC returns after Ctrl+C (SIGINT). "
        "`poscar_out` is written on normal completion or after SIGINT return to `main`. "
        "`kill <pid>` (SIGTERM) best-effort writes `poscar_out`, "
        "`final_cluster_vector_deviations.tsv` (all components), and `final_cost_function.txt` (Q), "
        "from the current best structure, then exits. "
        "SIGKILL (`kill -9`) cannot run Python cleanup.",
        flush=True,
    )
    print(f"Output directory: {root}", flush=True)

    _sig_poscar = root / POSCAR_OUT
    _sig_dev = root / FINAL_CLUSTER_VECTOR_DEVIATIONS_TSV
    _sig_cost = root / FINAL_COST_FUNCTION_TXT
    set_sigterm_output_paths(
        poscar=_sig_poscar,
        deviations_tsv=_sig_dev,
        cost_txt=_sig_cost,
        script_root=root,
    )

    print("SQS: preparing chemistry (CIF → cluster space, may take a while) …", flush=True)
    chem = prepare_wtacrv_chemistry(root)
    install_interrupt_handlers()
    print("SQS: chemistry ready; starting SQS Monte Carlo …", flush=True)

    n_mc_steps = _parse_bounded_positive_int_env(
        "SQS_N_MC_STEPS",
        DEFAULT_N_MC_STEPS,
        min_v=1,
        max_v=MAX_N_MC_STEPS,
    )
    ckpt_sec = _parse_bounded_nonnegative_int_env("SQS_CHECKPOINT_SEC", 30, max_v=3600)
    set_runtime_checkpoint_config(
        poscar=_sig_poscar,
        deviations_tsv=_sig_dev,
        cost_txt=_sig_cost,
        script_root=root,
        interval_sec=ckpt_sec,
    )
    plateau_sec = _parse_plateau_idle_sec()
    print(
        "MC stops after SQS_N_MC_STEPS trials "
        f"({n_mc_steps}), or earlier on Ctrl+C / SIGTERM, or on Q plateau "
        f"({_format_float(plateau_sec)}s unchanged best-Q + passing final cost/CV checks; "
        "env SQS_PLATEAU_SEC, 0 disables).",
        flush=True,
    )
    print(
        f"Checkpoint: best-so-far poscar/tsv/cost written on improvements and every {ckpt_sec}s "
        "(env SQS_CHECKPOINT_SEC, 0 disables periodic checkpoints).",
        flush=True,
    )

    # ``icet_script.py`` lines 28–31: explicit supercell list + ``generate_sqs_from_supercells``.
    print(
        "SQS: calling generate_sqs_from_supercells (icet may spend time here before MC starts) …",
        flush=True,
    )
    sqs = generate_sqs_from_supercells(
        chem.cluster_space,
        chem.supercells,
        chem.target_concentrations,
        n_steps=n_mc_steps,
        random_seed=42,
    )

    assert_atom_count(sqs, TARGET_TOTAL_ATOMS, context="generate_sqs_from_supercells return value")
    assert_atom_count(
        chem.supercells[0],
        TARGET_TOTAL_ATOMS,
        context="Reference supercell (must match returned SQS size)",
    )
    if len(sqs) != len(chem.supercells[0]):
        raise RuntimeError(
            f"SQS atom count {len(sqs)} != input supercell {len(chem.supercells[0])}; "
            "cluster_space / supercell mismatch."
        )
    sqs.wrap()
    assert_atom_count(sqs, TARGET_TOTAL_ATOMS, context="After sqs.wrap()")
    validate_equiatomic_wtacrv(sqs)

    out_poscar = root / POSCAR_OUT
    dev_path = root / FINAL_CLUSTER_VECTOR_DEVIATIONS_TSV
    cost_path = root / FINAL_COST_FUNCTION_TXT
    if interrupt_requested():
        out_poscar = root / POSCAR_OUT_INTERRUPT

    _paths_stay_under(root, [out_poscar, dev_path, cost_path])

    write_poscar_atomic(out_poscar, sqs)
    target_vector = get_sigterm_target_vector()
    if target_vector is None:
        raise RuntimeError(
            "Missing runtime SQS target vector from annealer; cannot compute final Q/deviations."
        )
    cv_components = int(len(target_vector))
    print(
        f"Cluster-vector components in this run: {cv_components}.",
        flush=True,
    )
    q_final = write_cluster_vector_deviations_and_cost(
        deviations_path=dev_path,
        cost_path=cost_path,
        cluster_space=chem.cluster_space,
        atoms=sqs,
        target_vector=target_vector,
        optimality_weight=Q_OPTIMALITY_WEIGHT,
        tol=Q_TOL,
    )

    chk = read(out_poscar, format="vasp")
    assert_atom_count(chk, TARGET_TOTAL_ATOMS, context="POSCAR read-back")
    if len(chk) != len(sqs):
        raise RuntimeError(f"POSCAR read-back length {len(chk)} != {len(sqs)}")

    if interrupt_requested():
        print(
            f"Interrupt: wrote best-so-far → {out_poscar.name}, {dev_path.name}, {cost_path.name} "
            f"(Q={_format_float(q_final)})",
            flush=True,
        )
        sys.exit(130)

    print(
        f"Done → {out_poscar.name}, deviations → {dev_path.name}, cost → {cost_path.name} "
        f"(Q={_format_float(q_final)})",
        flush=True,
    )


if __name__ == "__main__":
    main()


# =============================================================================
# Part 4 — Parallel design (documentation only)
# =============================================================================
PART4_PARALLEL_NOTES = r"""
The previous ``sqs_generator_final.py`` supported optional parallel SQS chains:

- ``Config.n_parallel`` (env ``SQS_N_PARALLEL``): spawn ``N`` independent MC workers via
  ``concurrent.futures.ProcessPoolExecutor``.
- Each worker ran its own seed and output basename suffix ``_wid``, with optional per-worker CSV /
  ``bestcorr`` stems.
- ``multiprocessing.Manager().dict()`` held shared keys (step / Q / best / status) so the parent
  could print aggregate minute summaries across workers (``run_parent_minute_summary``).
- ``finalize_parallel_bestsqs`` compared worker bests and optionally copied the winner.

That pool is omitted here: single-process MC keeps logs simple, preserves a single seed, and
avoids multiplying expensive per-step cluster-vector deviation work across processes.
"""
