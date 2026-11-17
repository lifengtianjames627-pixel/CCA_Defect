# W–Ta–Cr–V icet SQS — method contract

Method contract for the scripts in this directory. Do not append Q-vs-time plots, interrupt POSCAR names from one run, or host wall-clock notes here. Those belong in a separate run note.

Copy this when the alloy changes. Do not change the Monte Carlo. It only swaps species on a fixed bcc supercell so the pair correlations match a fully random alloy. It does not relax the box and it does not age short-range order.

## Files

| File | What it runs |
|------|----------------|
| `W-Ta-Cr-V_SQS.py` | One chain, seed 42, checkpoints, Ctrl+C / SIGTERM keep the best structure so far. **This server copy** also stops early when best Q is unchanged for `SQS_PLATEAU_SEC` seconds (default 900) and the final cost / cluster-vector checks pass. Set `SQS_PLATEAU_SEC=0` to disable. |
| `W-Ta-Cr-V_multipleseedSQS.py` | One chain per seed, in parallel. Parent keeps the lowest Q. Do not import the single-chain file. No plateau logic. |

Both need `mp-91_W.cif` in this directory (conventional Im-3m W, a = 3.170 Å, used only as the lattice).

## Chemistry block (the only part that changes for a new alloy)

```text
CIF_IN = "mp-91_W.cif"
CUTOFFS = (4.8,)          # one float: pairs only. Second float would add triplets.
SUPERCELL_REP = (14, 14, 14)
TARGET_TOTAL_ATOMS = 5488  # 2 sites × 14 × 14 × 14
SPECIES = ("Cr", "Ta", "V", "W")   # one sublattice, equiatomic
```

Neighbor distances at a = 3.170 Å: 2.75, 3.17, 4.48, 5.26 Å. Cutoff 4.8 sits between shell 3 and shell 4. Put a new cutoff between the last shell you want and the next one. On 128 atoms, do not add a triplet cutoff near 4 Å.

Build, in this order:

1. Read the CIF. If ASE returns 1 atom, append the body center `(0.5, 0.5, 0.5)`.
2. `ClusterSpace(primitive, CUTOFFS, [SPECIES] * n_sites)`.
3. One disordered sublattice. `target_concentrations = {symbol: {el: 1/len(SPECIES)}}`.
4. `supercell = primitive.repeat(SUPERCELL_REP)`. Require `len == TARGET_TOTAL_ATOMS`.

## Cost function Q

MC minimizes Q, not energy. Lower is better; Q may be negative when the optimality term is large.

\[
Q = -\,\omega L + \sum_{\alpha} \left| \Gamma_{\alpha} - \Gamma_{\alpha}^{\mathrm{target}} \right|
\]

Score with `compare_cluster_vectors` and check it against `TargetVectorCalculator`. Outputs: `final_cost_function.txt` (global Q) and `final_cluster_vector_deviations.tsv` (per orbit).

## The call that must stay

```text
generate_sqs_from_supercells(
    cluster_space, [supercell], target_concentrations,
    n_steps, random_seed,
)
```

`n_steps` is the number of species swaps (default `SQS_N_MC_STEPS` = 2_000_000 on this 5488 cell). The anneal temperature inside icet is not a physical temperature.

Multi-seed: seeds `SQS_BASE_SEED + i` (default 42, 43, …). `SQS_N_SEEDS` defaults to min(4, CPUs). Each chain is still one core. Keep `min(Q)`.

## Stop reasons (single-chain server copy)

| Reason | Behaviour |
|--------|-----------|
| Step budget | `trials_done >= SQS_N_MC_STEPS` |
| Q plateau | Best Q unchanged for `SQS_PLATEAU_SEC` (default 900 s) and final checks pass |
| Ctrl+C / SIGTERM | Best-so-far kept |
| SIGKILL | No Python cleanup |

There is no automatic plateau stop in the multipleseed driver.

## Done when

- 5488 atoms, Cr/Ta/V/W exactly 1372 each.
- `final_cluster_vector_deviations.tsv`: `order = 2` shells at radius 1.3728, 1.5852, 2.2418 (icet prints half the pair distance) have absolute deviation ≲ 0.01.
- Winner is `poscar_out`. Multi-seed also writes `seed_scores.tsv` and `poscar_seed_<seed>`.

## Run

```bash
cd ~/HEA_Irr_ML_study/W-Ta-Cr-V/SQS/_5488
python3 W-Ta-Cr-V_SQS.py
```

| Variable | Role |
|----------|------|
| `SQS_N_MC_STEPS` | MC trial cap (default: 2_000_000) |
| `SQS_PLATEAU_SEC` | Idle seconds before plateau early stop (`0` disables) |
| `SQS_CHECKPOINT_SEC` | Periodic checkpoint interval |
| `SQS_LOG_LEVEL` | Logger level |

Lattice constant is still pure W. Relax the cell with tabGAP / Metropolis aging before any defect calculation.
