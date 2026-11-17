# Vacancy formation energies — method contract

Method contract for `V_form_e.py` in this directory. Do not append run logs, failure tables, or host-specific paths here. Those belong in a separate run note.

0 K vacancy formation on the room-temperature aged W–Ta–Cr–V cell, with the equiatomic chemical potentials of this tabGAP.

| element | \(\mu\) (eV) |
|---------|----------------|
| Cr | −9.388 |
| Ta | −11.724 |
| V | −9.033 |
| W | −13.029 |

Interstitials are not computed here. They are a separate anneal in `../SIA/SIA_form_e.py`. The retired campaign that placed a same-species ⟨111⟩ dumbbell and minimized once is not used: that start fixes geometry and chemistry before the potential has a say (Byggmästar, Sobieraj, Wróbel et al., Acta Mater. 296 (2025) 121276).

## Why vacancy only minimizes

A vacancy removes a known atom at a known lattice site. Neighbours relax into the void. For formation energy there is essentially one basin topology. Chemical environment widens \(P(E_{\mathrm{f}}^{\mathrm{V}})\), but the defect identity does not fork into dumbbell axes or mixed pairs. Vacancy work therefore stays at 0 K: delete, minimize, record \(E_{\mathrm{f}}^{\mathrm{V}}\).

## Protocol

Delete the atom of element \(A\). Keep the cell at the relaxed bulk volume and minimize.

\[
E_{\mathrm{f}}^{\mathrm{V}}(A) = E_{\mathrm{vac}} - E_{\mathrm{bulk}} + \mu_{A}
\]

Default sample: 125 sites of each element (Cr, Ta, V, W). At most 13 serial `lmp` processes run at once, one core each; three cores on a 16-core host stay idle. Each rank minimizes a different vacancy. Do not wrap these in `mpirun -np 1`: that binds every process to core 0. Bulk relaxation stays one process at the start.

Results go to `v_form_e.csv`. The old combined `formation_energies.csv` interstitial columns are not kept.

## Run

```bash
cd ~/HEA_Irr_ML_study/W-Ta-Cr-V/form_e_5488
python3 V_form_e.py
```

| Variable | Role |
|----------|------|
| `AGED` | Aged LAMMPS data (default: `Metropolis_MC/MC_Metropolis_5488/aged.data`) |
| `POT_DIR` | Directory with one `.tabgap` and one `.eam.fs` |
| `LMP` | LAMMPS binary (default: `lmp`) |
| `SAMPLES_PER_ELEMENT` | Vacancy trials per species (default: 125) |
| `SITE_SEED` | RNG seed for site sampling (default: 12345) |
| `NWORKERS` | Concurrent serial `lmp` processes, capped at 13 |

## Downstream

SIA anneals: `../SIA/SIA_METHOD.md`. Geometry labels for the annealed cells: `../../SIA_post_processing/SIA_pp_METHOD.md`. Migration barriers start from those labels, not from a fresh ⟨111⟩ insertion.
