# Self-interstitial formation energies — method contract

Method contract for `SIA_form_e.py` in this directory. Do not append run logs, class histograms from one campaign, or host-specific paths here. Those belong in a separate run note.

Both defects use the room-temperature aged cell and the same equiatomic chemical potentials as the vacancy script.

| element | \(\mu\) (eV) |
|---------|----------------|
| Cr | −9.388 |
| Ta | −11.724 |
| V | −9.033 |
| W | −13.029 |

## Why SIA anneals

An interstitial adds an extra atom. The stable product can be a split dumbbell along ⟨100⟩, ⟨110⟩, or ⟨111⟩, an octahedral-like configuration, or a crowdion; the dumbbell chemistry is often Cr–Cr, Cr–V, or V–V, not the element that was inserted. The inserted atom can leave the starting neighbourhood. That is a high-dimensional landscape with many local minima. A single 0 K minimization from a random or preset start often lands in a wrong basin.

The anneal uses a short hot segment so the cluster can cross rearrangement barriers, then a cool and a tight minimize to read a classified basin. Vacancy formation does not use the same hot MD because a removed atom does not need axis or partner choice.

Byggmästar, Sobieraj, Wróbel et al., Acta Mater. 296 (2025) 121276, already treat a walk away from the insertion point as normal for random alloys. A preset same-species ⟨111⟩ dumbbell is retired.

| | Anneal here | NEB in `migration_energy.py` |
| --- | --- | --- |
| Goal | Find a stable interstitial basin after depositing one atom from the reservoir | Barrier \(E_{\mathrm{m}}\) between two already classified basins |
| Quantity stored | Stage-4 minimize energy → \(E_{\mathrm{f}}^{\mathrm{SIA}}\) | Saddle minus initial basin on a guessed hop path |
| Dynamics | Default 5 ps NVT at `ANNEAL_K` plus 20 ps linear cool, fixed volume | 0 K; no thermal MD in the barrier step |

`site_id` means “we inserted species \(A\) near this lattice site,” not “the interstitial must remain at this site.”

## Protocol

Follows Zhou, Shi, Samolyuk, Gao et al., Comput. Mater. Sci. 234 (2024) 112765: put one extra atom at a random place and relax it by simulated annealing. Production count here is 200 insertions of each species (800 total), larger than the paper’s 100 per species because the cell is larger.

The lattice site is random. The inserted element is the species being counted. It is not forced to equal the element that already sits on that site. The extra atom is placed 0.8–1.4 Å from the site in a random direction (inside the first neighbor shell, not a dumbbell axis). A draw closer than 0.7 Å to any atom is thrown out and drawn again.

Each anneal is one LAMMPS run, at fixed bulk volume:

1. A short minimization, only to separate an overlap.
2. 5 ps of NVT molecular dynamics at `ANNEAL_K` (default 1000 K; 5000 steps of 1 fs).
3. A linear cool from that temperature to 1 K over 20 ps (20000 steps).
4. A full minimization, same tolerances as the vacancy run.

\[
E_{\mathrm{f}}^{\mathrm{SIA}}(A) = E_{\mathrm{SIA}} - E_{\mathrm{bulk}} - \mu_{A}
\]

\(A\) is the atom that was inserted. The CSV also records the relaxed pair, which may be different.

Every finished anneal is kept in `sia_formation_energies_<T>K.csv`. `sia_classes_<T>K.csv` counts distinct final labels. Repeats are counted there and kept in the energy table for the violin distribution.

## Higher temperatures

1000 K is the run to do first. If 1200 K or 1400 K finds labels that 1000 K never produces, the lower anneal was not long enough or not hot enough, and the class table should be read from the hotter run. That is a completeness test for the class table, not a substitute for migration-energy NEB.

## Run

`SIA_form_e.py` imports `V_form_e` for the cell reader and shared potentials. Keep that module importable from this process (same directory or `PYTHONPATH`).

```bash
cd ~/HEA_Irr_ML_study/W-Ta-Cr-V/Formation_energy/sia_5488_800
python3 SIA_form_e.py
ANNEAL_K=1200 python3 SIA_form_e.py
ANNEAL_K=1400 python3 SIA_form_e.py
INSERTS_PER_ELEMENT=2 python3 SIA_form_e.py
```

| Variable | Role |
|----------|------|
| `ANNEAL_K` | Hot thermostat in K (default: 1000) |
| `INSERTS_PER_ELEMENT` | Random sites per inserted element (default: 200) |
| `MD_STEPS` | Steps at `ANNEAL_K`, 1 fs each (default: 5000) |
| `COOL_STEPS` | Cool to 1 K (default: 20000) |
| `SITE_SEED` | RNG seed (default: 12345) |
| `NWORKERS` | Concurrent LAMMPS ranks, capped at 13 |
| `AGED`, `POT_DIR`, `LMP` | Same meaning as in `V_form_e.py` |

Each temperature has its own CSV and its own `cases_<T>K` directory. OpenMP stays at one thread so a rank cannot spread onto idle cores.

## Downstream

Geometry labels: `../../SIA_post_processing/classify_sia_geometry.py`. Migration barriers: `../../Migration_energy/`.
