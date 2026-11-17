# Metropolis aging — method contract

Method contract for `run_metropolis.py` in this directory. Do not append progress screenshots, accepted-swap traces, or host timing benches here. Those belong in a separate run note.

Canonical Metropolis swaps for equiatomic W–Ta–Cr–V at room temperature on the 5488-atom cell (14×14×14 conventional bcc). The 128-atom branch, if present, lives beside this folder and is not this contract.

## Pipeline

1. Read the SQS POSCAR (default: local `poscar_out.vasp`, else `../../SQS/_5488/poscar_out` / `poscar_out.vasp` / `poscar_out_interrupt`).
2. Relax the cell once with tabGAP → `relaxed.data`.
3. Freeze coordinates and run binary `fix atom/swap` at 300 K, cycling all element pairs.
4. Progress every `LOG_EVERY` swap attempts goes to the driver log and updates `mc_progress.png`. One “round” is \(N = n_{\mathrm{atoms}}\) attempts.
5. After `MIN_ROUNDS`, stop when \(|\Delta(pe/N)|\) stays below about 1 meV/atom over several rounds and the round-averaged `frac_interval` is steady → write `aged.data`. Do not wait for a fixed attempts-per-atom budget.

This step ages short-range order on the potential. It is not SQS: SQS only matched cluster vectors on a fixed W metric. Defect calculations use `aged.data`, not the raw SQS POSCAR.

## MPI

Default 4 ranks. Do not use 16 — the ghost layer is too thick for this cell. `BENCH=1` times about 1000 steps at 1, 4, and 8 ranks and exits; pick the fastest time/step.

## Stop signals

| Signal | Role |
|--------|------|
| `MIN_ROUNDS` | Do not stop before this many rounds (default: 10) |
| `STABLE_ROUNDS` | Consecutive quiet rounds needed (default: 3) |
| `MAX_ROUNDS` | Safety cap (default: 200) |
| `PE_TOL_MEV` | \(\|\Delta(pe/N)\|\) threshold in meV/atom (default: 1.0) |
| Round-averaged `frac_interval` | Steady if successive-round \(\|\Delta\|\) stays below 0.05 |

Do not use per-chunk relative std of `frac_interval` as a stop: binary pair cycling makes that quantity jump even when the alloy is equilibrated. Cumulative accept/attempt is not a stop signal either.

## Run

```bash
cd ~/HEA_Irr_ML_study/W-Ta-Cr-V/Metropolis_MC/MC_Metropolis_5488
python3 run_metropolis.py
```

| Variable | Role |
|----------|------|
| `POT_DIR` | tabGAP directory |
| `POSCAR` | Input POSCAR path |
| `LMP`, `MPIRUN` | LAMMPS binary and MPI launcher |
| `NPROCS` | MPI ranks (default: 4; 16 is refused) |
| `SEED` | RNG seed (default: 12345) |
| `BENCH` | `1` = rank timing only, then exit |
| `SKIP_RELAX` | `1` = reuse existing `relaxed.data` |
| `LOG_EVERY` | Progress row spacing (default: 1000) |

## Downstream

Vacancy and SIA formation read `aged.data` from this directory. Migration uses the same aged cell unless `AGED` is overridden.
