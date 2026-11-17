"""
Canonical Metropolis swaps for equiatomic W-Ta-Cr-V at room temperature.

5488-atom branch (14×14×14 conventional bcc). The 128-atom run lives in
../MC_Metropolis_128/.

Pipeline
1. Read the SQS POSCAR (default: local poscar_out.vasp, else ../../SQS/_5488/).
2. Relax the cell once with tabGAP (writes relaxed.data).
3. Freeze coordinates and run binary fix atom/swap at 300 K, cycling all
   element pairs. Progress every LOG_EVERY swap attempts goes to the driver
   log and updates mc_progress.png (no separate .dat). One "round" is
   N = n_atoms attempts. After >= MIN_ROUNDS, stop when |Δ(pe/N)| stays
   below ~1 meV/atom over several rounds and frac_interval is steady;
   write aged.data. Do not wait for a fixed attempts-per-atom budget.

MPI: default 4 ranks. Do not use 16 — the ghost layer is too thick for this
cell. Use BENCH=1 to time ~1000 steps at 1, 4, and 8 ranks and pick the
fastest time/step.

Optional environment variables:
    POT_DIR, POSCAR, LMP, MPIRUN, NPROCS, SEED
    BENCH=1          rank timing only (1/4/8), then exit
    SKIP_RELAX=1     reuse existing relaxed.data
    LOG_EVERY        progress row spacing (default 1000)
    MIN_ROUNDS       default 10
    STABLE_ROUNDS    consecutive quiet rounds needed (default 3)
    MAX_ROUNDS       safety cap (default 200)
    PE_TOL_MEV       |Δ(pe/N)| threshold in meV/atom (default 1.0)
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TEMPERATURE_K = 300.0
TARGET_NATOMS = 5488
DEFAULT_SEED = 12345
DEFAULT_NPROCS = 4
FORBIDDEN_NPROCS = {16}
BENCH_RANKS = (1, 4, 8) # if BENCH = 1 it will compare, else it doesn't move
BENCH_STEPS = 1000

LOG_EVERY_DEFAULT = 1000
MIN_ROUNDS_DEFAULT = 10
STABLE_ROUNDS_DEFAULT = 3
MAX_ROUNDS_DEFAULT = 200
PE_TOL_MEV_DEFAULT = 1.0  # meV/atom
# Round-averaged frac_interval is "steady" if |Δ mean_frac| between successive
# rounds stays below this absolute tolerance. Do NOT use per-chunk rel_std:
# binary pair cycling makes frac_interval jump 0 ↔ ~0.1 every 1000 steps even
# when the alloy is equilibrated. Cumulative accept/attempt is not a stop signal.
FRAC_ROUND_DELTA_MAX = 0.05

MASSES = {"Cr": 51.9961, "Ta": 180.94788, "V": 50.9415, "W": 183.84}

HERE = Path(__file__).resolve().parent
ALLOY_ROOT = HERE.parents[1]
DEFAULT_POT_DIR = Path.home() / "HEA_Irr_ML_study/ML_potentials/tabGAP/W-Ta-Cr-V"
PLOT_PATH = HERE / "mc_progress.png"


def default_poscar() -> Path:
    local = HERE / "poscar_out.vasp"
    if local.is_file():
        return local
    sqs = ALLOY_ROOT / "SQS" / "_5488"
    for name in ("poscar_out.vasp", "poscar_out", "poscar_out_interrupt"):
        path = sqs / name
        if path.is_file():
            return path
    return local


DEFAULT_POSCAR = default_poscar()


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def env_int(name: str, default: int, *, lo: int = 1, hi: int = 10_000_000) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    if not raw.isdigit():
        die(f"{name} must be a non-negative integer, got {raw!r}")
    value = int(raw)
    if value < lo or value > hi:
        die(f"{name} must be in [{lo}, {hi}], got {value}")
    return value


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


def env_flag(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}

# parsing the poscar file
def read_poscar(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 8:
        die(f"{path} is not a POSCAR")
    scale = float(lines[1].split()[0])
    cell = []
    for row in lines[2:5]:
        cell.append([scale * float(x) for x in row.split()[:3]])
    tokens = lines[5].split()
    if tokens[0].isdigit():
        die("POSCAR has no element symbols on line 6")
    symbols = tokens
    counts = [int(x) for x in lines[6].split()]
    if len(symbols) != len(counts):
        die("element list and count list differ in length")
    cursor = 7
    mode = lines[cursor].strip().lower()
    if mode.startswith("s"):
        cursor += 1
        mode = lines[cursor].strip().lower()
    cursor += 1
    n_atoms = sum(counts)
    coord_lines = lines[cursor : cursor + n_atoms]
    if len(coord_lines) != n_atoms:
        die(f"expected {n_atoms} coordinates, found {len(coord_lines)}")
    direct = mode.startswith("d")
    positions = []
    species = []
    for symbol, count in zip(symbols, counts):
        if symbol not in MASSES:
            die(f"unexpected element {symbol}")
        species.extend([symbol] * count)
    for symbol, line in zip(species, coord_lines):
        frac_or_cart = [float(x) for x in line.split()[:3]]
        if direct:
            x = sum(frac_or_cart[k] * cell[k][0] for k in range(3))
            y = sum(frac_or_cart[k] * cell[k][1] for k in range(3))
            z = sum(frac_or_cart[k] * cell[k][2] for k in range(3))
        else:
            x, y, z = (scale * v for v in frac_or_cart)
        positions.append((symbol, x, y, z))
    return cell, positions

# Writing the lammps data file with POSCAR read results
def write_lammps_data(path: Path, cell, positions) -> list[str]:
    order = []
    for symbol, _, _, _ in positions:
        if symbol not in order:
            order.append(symbol)
    type_of = {symbol: i + 1 for i, symbol in enumerate(order)}
    lengths = [cell[i][i] for i in range(3)]
    xy = cell[1][0]
    xz = cell[2][0]
    yz = cell[2][1]
    lines = [
        "W-Ta-Cr-V SQS, types follow the POSCAR species order",
        "",
        f"{len(positions)} atoms",
        f"{len(order)} atom types",
        "",
        f"0.0 {lengths[0]:.10f} xlo xhi",
        f"0.0 {lengths[1]:.10f} ylo yhi",
        f"0.0 {lengths[2]:.10f} zlo zhi",
        f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz",
        "",
        "Masses",
        "",
    ]

    for symbol in order:
        lines.append(f"{type_of[symbol]} {MASSES[symbol]}")
    lines += ["", "Atoms # atomic", ""]
    for i, (symbol, x, y, z) in enumerate(positions, start=1):
        lines.append(f"{i} {type_of[symbol]} {x:.10f} {y:.10f} {z:.10f}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return order

# the tabGap potential dir on server environment
def find_potential(pot_dir: Path) -> tuple[Path, Path]:
    tabgaps = sorted(pot_dir.glob("*.tabgap"))
    eams = sorted(pot_dir.glob("*.eam.fs"))
    if len(tabgaps) != 1 or len(eams) != 1:
        die(
            f"{pot_dir} must contain exactly one .tabgap and one .eam.fs "
            f"(found {len(tabgaps)} and {len(eams)})"
        )
    return tabgaps[0], eams[0]

# check if lammps environment can support the run
def lammps_has_atom_swap(lmp: str) -> bool:
    try:
        proc = subprocess.run([lmp, "-help"], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        die(f"cannot run {lmp!r}")
    text = (proc.stdout or "") + (proc.stderr or "")
    if "tabgap" not in text:
        die(f"{lmp} has no tabgap pair style; use the lammps-ml binary")
    return "atom/swap" in text


def binary_type_pairs(n_types: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(1, n_types + 1) for j in range(i + 1, n_types + 1)]


def build_lammps_cmd(lmp: str, nprocs: int, mpirun: str, in_name: str, log_name: str) -> list[str]:
    base = [lmp, "-in", in_name, "-log", log_name, "-screen", "none"]
    if nprocs == 1:
        return base
    launcher = shutil.which(mpirun) or mpirun
    return [launcher, "-np", str(nprocs), *base]


def pair_header(elements: list[str], tabgap: Path, eam: Path) -> str:
    elem = " ".join(elements)
    return f"""pair_style      hybrid/overlay tabgap eam/fs
pair_coeff      * * tabgap {tabgap} {elem}
pair_coeff      * * eam/fs {eam} {elem}
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
"""

# build lmp input for relaxation
def write_relax_input(path: Path, *, data_name: str, elements: list[str], tabgap: Path, eam: Path) -> None:
    text = f"""# Stage 1: relax pure-W SQS volume for W-Ta-Cr-V tabGAP.
units           metal
atom_style      atomic
boundary        p p p
atom_modify     map array
read_data       {data_name}
{pair_header(elements, tabgap, eam)}
thermo          50
thermo_style    custom step pe
fix             box all box/relax iso 0.0 vmax 0.001
minimize        1.0e-8 1.0e-6 5000 50000
unfix           box
write_data      relaxed.data
print           "RELAX_DONE pe=$(pe)"
"""
    path.write_text(text, encoding="utf-8")


def write_swap_chunk_input(
    path: Path,
    *,
    data_name: str,
    elements: list[str],
    tabgap: Path,
    eam: Path,
    t1: int,
    t2: int,
    n_steps: int,
    seed: int,
    out_data: str,
) -> None:
    text = f"""# Frozen-coordinate binary atom/swap chunk ({n_steps} attempts).
units           metal
atom_style      atomic
boundary        p p p
atom_modify     map array
read_data       {data_name}
{pair_header(elements, tabgap, eam)}
velocity        all set 0.0 0.0 0.0
fix             freeze all setforce 0.0 0.0 0.0
fix             integrate all nve
timestep        1.0
thermo          {max(n_steps, 1)} 
thermo_style    custom step pe
fix             mc all atom/swap 1 1 {seed} {TEMPERATURE_K} ke no types {t1} {t2}
run             {n_steps}
print           "MC_CHUNK pe=$(pe) att=$(f_mc[1]) suc=$(f_mc[2])"
unfix           mc
write_data      {out_data}
"""
    path.write_text(text, encoding="utf-8")

# logging info stored through parsing mc_chunk
def parse_mc_chunk(log_text: str) -> tuple[float, int, int]:
    """Parse the evaluated MC_CHUNK print line (skip the echoed input command)."""
    pe = None
    att = None
    suc = None
    # Match only evaluated output, e.g.  MC_CHUNK pe=-59230.43 att=1000 suc=17
    # Ignore echoed input: print "MC_CHUNK pe=$(pe) ..."
    chunk_pat = re.compile(
        r"^\s*MC_CHUNK\s+pe=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"\s+att=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"\s+suc=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
    )
    for line in log_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("print") or "$(" in stripped:
            continue
        m = chunk_pat.match(stripped)
        if not m:
            continue
        pe = float(m.group(1))
        att = int(float(m.group(2)))
        suc = int(float(m.group(3)))
    if pe is None or att is None or suc is None: # no return energy means lmp dies
        die("LAMMPS log missing evaluated MC_CHUNK pe/att/suc line")
    return pe, att, suc


def run_cmd(cmd: list[str], *, log_path: Path) -> str:
    proc = subprocess.run(cmd, cwd=HERE)
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    if proc.returncode != 0:
        die(f"LAMMPS exited {proc.returncode}. See {log_path}")
    return text

# matplotlib driver here
def plot_progress(progress_rows: list[dict], out_png: Path) -> None:
    """Write mc_progress.png from in-memory rows (driver log already has the text)."""
    if len(progress_rows) < 2:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped ({exc})", flush=True)
        return
    steps = [r["step"] for r in progress_rows]
    pe_n = [r["pe_per_atom"] for r in progress_rows]
    frac = [r["frac_interval"] for r in progress_rows]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    ax0.plot(steps, pe_n, color="#1f4e79", lw=1.4)
    ax0.set_ylabel("pe / atom (eV)")
    ax0.set_title("Metropolis MC progress (W-Ta-Cr-V, 5488 atoms)")
    ax0.grid(True, alpha=0.3)
    ax1.plot(steps, frac, color="#b85c38", lw=1.4)
    ax1.set_xlabel("swap attempts")
    ax1.set_ylabel("frac_interval")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

# printing logging info per round
def round_end_metrics(progress_rows: list[dict], n_atoms: int) -> list[dict]:
    """
    One entry per finished round k >= 1.

    Round k is the attempts in [k*N, (k+1)*N). The entry keeps the last logged
    sample inside that window, the same rows _round_mean_frac averages. The
    round counts as finished only after a later sample has crossed (k+1)*N.
    """
    if n_atoms < 1 or not progress_rows:
        return []
    last_attempt = progress_rows[-1]["n_attempt"]
    finished_through = last_attempt // n_atoms
    last_row = {}
    for row in progress_rows:
        rid = row["n_attempt"] // n_atoms
        if rid < 1:
            continue
        last_row[rid] = row
    out = []
    round_id = 1
    while round_id < finished_through:
        if round_id not in last_row:
            break
        row = last_row[round_id]
        out.append(
            {
                "round": round_id,
                "pe_per_atom": row["pe_per_atom"],
                "frac_interval": row["frac_interval"],
                "step": row["step"],
            }
        )
        round_id += 1
    return out


def _round_mean_frac(progress_rows: list[dict], round_id: int) -> float | None:
    vals = [
        row["frac_interval"]
        for row in progress_rows
        if row["round"] == round_id
        and math.isfinite(row["frac_interval"])
        and row["frac_interval"] >= 0.0
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)

# condition for converging
def converged(
    progress_rows: list[dict],
    *,
    n_atoms: int,
    min_rounds: int,
    stable_rounds: int,
    pe_tol_ev: float,
) -> tuple[bool, str]:
    ends = round_end_metrics(progress_rows, n_atoms)
    if len(ends) < min_rounds:
        return False, f"rounds={len(ends)}<{min_rounds}"
    if len(ends) < stable_rounds + 1:
        return False, f"need>={stable_rounds} round deltas"
    window = ends[-(stable_rounds + 1) :]
    pe_ok = True
    max_dpe = 0.0
    for a, b in zip(window, window[1:]):
        dpe = abs(b["pe_per_atom"] - a["pe_per_atom"])
        max_dpe = max(max_dpe, dpe) # change in pe > pe_tol_ev
        if dpe > pe_tol_ev:
            pe_ok = False

    # Steady frac_interval: compare round-averaged values, not raw 1000-step chunks.
    round_fracs: list[float] = []
    for end in window:
        mean_f = _round_mean_frac(progress_rows, end["round"])
        if mean_f is None:
            return False, f"missing frac samples for round {end['round']}"
        round_fracs.append(mean_f)
    frac_ok = True
    max_dfrac = 0.0
    for a, b in zip(round_fracs, round_fracs[1:]):
        dfrac = abs(b - a)
        max_dfrac = max(max_dfrac, dfrac)
        if dfrac > FRAC_ROUND_DELTA_MAX:
            frac_ok = False

    if pe_ok and frac_ok:
        return True, (
            f"stable: |Δ(pe/N)|≤{pe_tol_ev * 1000:.2f} meV/atom over "
            f"{stable_rounds} rounds (max |Δ|={max_dpe * 1000:.3f} meV/atom); "
            f"round-mean frac_interval |Δ|≤{FRAC_ROUND_DELTA_MAX} "
            f"(max |Δ|={max_dfrac:.4f})"
        )
    return False, (
        f"pe_ok={pe_ok} frac_ok={frac_ok} "
        f"max_dpe_meV={max_dpe * 1000:.3f} max_dfrac={max_dfrac:.4f}"
    )


def check_nprocs(nprocs: int) -> None:
    if nprocs in FORBIDDEN_NPROCS:
        die(f"NPROCS={nprocs} is forbidden for this 5488-atom cell (ghost layer too thick)")

# used for comparison, skipped
def bench_ranks(
    *,
    lmp: str,
    mpirun: str,
    elements: list[str],
    tabgap: Path,
    eam: Path,
    seed: int,
) -> int:
    pairs = binary_type_pairs(len(elements))
    t1, t2 = pairs[0]
    results = []
    for nprocs in BENCH_RANKS:
        check_nprocs(nprocs)
        in_name = f"in.bench_{nprocs}.lmp"
        log_name = f"log.bench_{nprocs}.lammps"
        out_data = f"bench_{nprocs}.data"
        write_swap_chunk_input(
            HERE / in_name,
            data_name="relaxed.data",
            elements=elements,
            tabgap=tabgap,
            eam=eam,
            t1=t1,
            t2=t2,
            n_steps=BENCH_STEPS,
            seed=seed + nprocs,
            out_data=out_data,
        )
        cmd = build_lammps_cmd(lmp, nprocs, mpirun, in_name, log_name)
        print(f"BENCH nprocs={nprocs} steps={BENCH_STEPS} cmd={' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        run_cmd(cmd, log_path=HERE / log_name)
        dt = time.perf_counter() - t0
        tps = dt / BENCH_STEPS
        results.append((nprocs, dt, tps))
        print(f"BENCH nprocs={nprocs} wall={dt:.3f}s time_per_step={tps:.6f}s", flush=True)
    results.sort(key=lambda row: row[2])
    best = results[0][0]
    print("BENCH ranking (best = lowest time/step):", flush=True)
    for nprocs, dt, tps in results:
        mark = " <-- selected" if nprocs == best else ""
        print(f"  ranks={nprocs} wall={dt:.3f}s s/step={tps:.6f}{mark}", flush=True)
    (HERE / "bench_ranks.txt").write_text(
        "\n".join(f"{n} {dt:.6f} {tps:.8f}" for n, dt, tps in results) + f"\nbest {best}\n",
        encoding="utf-8",
    )
    return best


def main() -> None:
    poscar = Path(os.environ.get("POSCAR", str(DEFAULT_POSCAR))).expanduser()
    pot_dir = Path(os.environ.get("POT_DIR", str(DEFAULT_POT_DIR))).expanduser()
    lmp = os.environ.get("LMP", "lmp")
    mpirun = os.environ.get("MPIRUN", "mpirun")
    seed = env_int("SEED", DEFAULT_SEED)
    nprocs = env_int("NPROCS", DEFAULT_NPROCS, lo=1, hi=8)
    log_every = env_int("LOG_EVERY", LOG_EVERY_DEFAULT)
    min_rounds = env_int("MIN_ROUNDS", MIN_ROUNDS_DEFAULT)
    stable_rounds = env_int("STABLE_ROUNDS", STABLE_ROUNDS_DEFAULT)
    max_rounds = env_int("MAX_ROUNDS", MAX_ROUNDS_DEFAULT)
    pe_tol_ev = env_float("PE_TOL_MEV", PE_TOL_MEV_DEFAULT) / 1000.0
    do_bench = env_flag("BENCH")
    skip_relax = env_flag("SKIP_RELAX")

    check_nprocs(nprocs)
    if not poscar.is_file():
        die(f"missing POSCAR: {poscar}")
    if not lammps_has_atom_swap(lmp):
        die(f"{lmp} has no fix atom/swap")
    tabgap, eam = find_potential(pot_dir)
    tabgap = tabgap.resolve()
    eam = eam.resolve()

    cell, positions = read_poscar(poscar)
    n_atoms = len(positions)
    if n_atoms != TARGET_NATOMS:
        die(f"POSCAR has {n_atoms} atoms, expected {TARGET_NATOMS}")
    elements = write_lammps_data(HERE / "start.data", cell, positions)
    pairs = binary_type_pairs(len(elements))
    if not pairs:
        die("need at least two element types")

    print(
        f"Room temperature {TEMPERATURE_K:.0f} K | N={n_atoms} | "
        f"log_every={log_every} | round=N attempts | "
        f"min_rounds={min_rounds} stable_rounds={stable_rounds} "
        f"pe_tol={pe_tol_ev * 1000:.2f} meV/atom",
        flush=True,
    )
    print(f"POSCAR: {poscar}", flush=True)
    print(f"tabgap: {tabgap}", flush=True)
    print(f"eam   : {eam}", flush=True)

    if not skip_relax or not (HERE / "relaxed.data").is_file():
        write_relax_input(
            HERE / "in.relax.lmp",
            data_name="start.data",
            elements=elements,
            tabgap=tabgap,
            eam=eam,
        )
        cmd = build_lammps_cmd(lmp, nprocs, mpirun, "in.relax.lmp", "log.relax.lammps")
        print(f"relax: {' '.join(cmd)}", flush=True)
        run_cmd(cmd, log_path=HERE / "log.relax.lammps")
        if not (HERE / "relaxed.data").is_file():
            die("relax finished without relaxed.data")
    else:
        print("SKIP_RELAX=1 — reusing relaxed.data", flush=True)

    if do_bench:
        best = bench_ranks(
            lmp=lmp,
            mpirun=mpirun,
            elements=elements,
            tabgap=tabgap,
            eam=eam,
            seed=seed,
        )
        print(f"BENCH done. Recommended NPROCS={best}", flush=True)
        return

    # Production MC: chunked swaps with live progress + early stop.
    current = HERE / "mc_current.data"
    shutil.copyfile(HERE / "relaxed.data", current)

    cum_att = 0
    cum_suc = 0
    prev_att = 0
    prev_suc = 0
    progress_rows: list[dict] = []
    pair_idx = 0
    chunk_i = 0
    max_attempts = max_rounds * n_atoms

    print(
        f"MC start with NPROCS={nprocs} (forbidden ranks: {sorted(FORBIDDEN_NPROCS)}). "
        f"Safety cap {max_rounds} rounds ({max_attempts} attempts).",
        flush=True,
    )

    # logging data
    while cum_att < max_attempts:
        remaining_to_log = log_every - (cum_att % log_every)
        if remaining_to_log <= 0:
            remaining_to_log = log_every
        remaining_to_cap = max_attempts - cum_att
        n_steps = min(remaining_to_log, remaining_to_cap)
        t1, t2 = pairs[pair_idx % len(pairs)]
        pair_idx += 1
        chunk_i += 1
        in_name = "in.swap_chunk.lmp"
        log_name = "log.swap_chunk.lammps"
        write_swap_chunk_input(
            HERE / in_name,
            data_name=current.name,
            elements=elements,
            tabgap=tabgap,
            eam=eam,
            t1=t1,
            t2=t2,
            n_steps=n_steps,
            seed=seed + chunk_i,
            out_data=current.name,
        )
        cmd = build_lammps_cmd(lmp, nprocs, mpirun, in_name, log_name)
        print(
            f"chunk {chunk_i}: types {t1}-{t2} ({elements[t1 - 1]}-{elements[t2 - 1]}) "
            f"n_steps={n_steps} cum_att={cum_att} -> {cum_att + n_steps}",
            flush=True,
        )
        log_text = run_cmd(cmd, log_path=HERE / log_name)
        pe, att, suc = parse_mc_chunk(log_text)
        cum_att += att
        cum_suc += suc

        # Emit a progress row whenever we hit a log_every boundary (or the end).
        if cum_att % log_every == 0 or cum_att >= max_attempts:
            d_att = cum_att - prev_att
            d_suc = cum_suc - prev_suc
            frac_interval = (d_suc / d_att) if d_att > 0 else float("nan")
            pe_n = pe / n_atoms
            round_id = cum_att // n_atoms
            row = {
                "step": cum_att,
                "pe": pe,
                "n_accept": cum_suc,
                "n_attempt": cum_att,
                "frac_interval": frac_interval,
                "pe_per_atom": pe_n,
                "round": round_id,
            }
            progress_rows.append(row)
            prev_att = cum_att
            prev_suc = cum_suc
            print(
                f"progress step={cum_att} pe/N={pe_n:.6f} "
                f"accept={cum_suc}/{cum_att} frac_interval={frac_interval:.4f} "
                f"round={round_id}",
                flush=True,
            )
            plot_progress(progress_rows, PLOT_PATH)

            ok, reason = converged(
                progress_rows,
                n_atoms=n_atoms,
                min_rounds=min_rounds,
                stable_rounds=stable_rounds,
                pe_tol_ev=pe_tol_ev,
            )
            if ok:
                shutil.copyfile(current, HERE / "aged.data")
                print(f"CONVERGED after {round_id} rounds: {reason}", flush=True)
                print(f"Wrote {HERE / 'aged.data'}", flush=True)
                print(f"Plot: {PLOT_PATH}", flush=True)
                return

    shutil.copyfile(current, HERE / "aged.data")
    print(
        f"Reached MAX_ROUNDS={max_rounds} without formal convergence; wrote aged.data anyway.",
        flush=True,
    )
    plot_progress(progress_rows, PLOT_PATH)


if __name__ == "__main__":
    main()
