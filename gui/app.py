"""
RCCA Workflow — Streamlit GUI (local job-pack configurator).

Run from the repo root:

    streamlit run gui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcca_workflow.family import DEFAULT_WTACRV, ElementSpec, Family  # noqa: E402
from rcca_workflow.pack import PackRequest, emit_pack, zip_pack  # noqa: E402
from rcca_workflow.potential import PotentialRef, resolve_potential  # noqa: E402
from rcca_workflow.stages.sqs import default_cif_path  # noqa: E402


st.set_page_config(page_title="RCCA Workflow", page_icon="⚗️", layout="wide")


def _init_state() -> None:
    if "family_rows" not in st.session_state:
        st.session_state.family_rows = [
            {
                "symbol": e.symbol,
                "mass": e.mass,
                "mu_eV": e.mu_eV,
                "fraction": e.fraction,
            }
            for e in DEFAULT_WTACRV.elements
        ]
    defaults = {
        "family_name": DEFAULT_WTACRV.name,
        "potential_id": DEFAULT_WTACRV.potential_id,
        "pot_dir": "",
        "lmp": "lmp",
        "mpirun": "mpirun",
        "pack_name": "rcca_pack",
        "include_sqs": True,
        "include_metro": True,
        "include_vac": True,
        "include_sia": True,
        # Stage 1
        "sqs_rep": "2,2,2",
        "sqs_cutoffs": "4.8",
        "sqs_steps": 5000,
        "sqs_seed": 42,
        "sqs_plateau": 0.0,
        "sqs_ckpt": 30,
        # Stage 2
        "metro_T": 300.0,
        "metro_nprocs": 4,
        "metro_seed": 12345,
        "metro_min_rounds": 10,
        "metro_stable": 3,
        "metro_max_rounds": 200,
        "metro_pe_tol": 1.0,
        "metro_expected": 16,
        # Stage 3
        "vac_samples": 2,
        "vac_seed": 12345,
        "vac_workers": 1,
        "vac_expected": 16,
        "sia_anneal": 1000.0,
        "sia_inserts": 2,
        "sia_md": 500,
        "sia_cool": 1000,
        "sia_seed": 12345,
        "sia_workers": 1,
        "sia_expected": 16,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _parse_rep(text: str) -> list[int]:
    parts = [p.strip() for p in text.replace("×", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("supercell_rep must be three integers, e.g. 2,2,2 or 14,14,14")
    return [int(p) for p in parts]


def _parse_cutoffs(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("cutoffs required")
    return [float(p) for p in parts]


def build_family_from_ui() -> Family:
    rows = st.session_state.family_rows
    elements = []
    for row in rows:
        sym = str(row.get("symbol", "")).strip()
        if not sym:
            continue
        elements.append(
            ElementSpec(
                symbol=sym,
                mass=float(row["mass"]),
                mu_eV=float(row["mu_eV"]),
                fraction=float(row["fraction"]),
            )
        )
    if not elements:
        raise ValueError("Add at least one element")
    total = sum(e.fraction for e in elements)
    if abs(total - 1.0) > 1e-6:
        # Soft-normalize for UX when fractions are slightly off.
        elements = [
            ElementSpec(e.symbol, e.mass, e.mu_eV, e.fraction / total) for e in elements
        ]
    return Family(
        name=str(st.session_state.family_name).strip() or "alloy",
        elements=tuple(elements),
        potential_id=str(st.session_state.potential_id).strip() or "local",
    )


def page_family() -> None:
    st.subheader("Family & runtime")
    st.caption(
        "Family = element set + potential identity. Composition = molar fractions inside that family."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Family name", key="family_name")
        st.text_input("Potential id / label", key="potential_id")
        st.text_input(
            "POT_DIR (one .tabgap + one .eam.fs)",
            key="pot_dir",
            help="Required to emit Metropolis / Vacancy / SIA packs.",
        )
    with c2:
        st.text_input("LAMMPS binary (LMP)", key="lmp")
        st.text_input("MPI launcher (MPIRUN)", key="mpirun")
        if st.button("Load demo W–Ta–Cr–V", use_container_width=True):
            st.session_state.family_rows = [
                {
                    "symbol": e.symbol,
                    "mass": e.mass,
                    "mu_eV": e.mu_eV,
                    "fraction": e.fraction,
                }
                for e in DEFAULT_WTACRV.elements
            ]
            st.session_state.family_name = DEFAULT_WTACRV.name
            st.session_state.potential_id = DEFAULT_WTACRV.potential_id
            st.rerun()

    st.markdown("**Elements** (fractions should sum to 1; slight drift is auto-normalized)")
    edited = st.data_editor(
        st.session_state.family_rows,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", required=True),
            "mass": st.column_config.NumberColumn("Mass", format="%.6f"),
            "mu_eV": st.column_config.NumberColumn("μ (eV)", format="%.6f"),
            "fraction": st.column_config.NumberColumn("Fraction", format="%.6f", min_value=0.0),
        },
        key="family_editor",
    )
    st.session_state.family_rows = edited

    pot_dir = str(st.session_state.pot_dir).strip()
    if pot_dir:
        try:
            resolved = resolve_potential(PotentialRef(pot_dir=Path(pot_dir)))
            st.success(
                f"Potential OK — tabgap=`{resolved.tabgap.name}`, "
                f"eam=`{resolved.eam_fs.name}`, hash=`{resolved.content_hash[:12]}…`"
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"POT_DIR not ready: {exc}")


def page_stage1() -> None:
    st.subheader("Stage 1 — SQS")
    st.caption(f"Default CIF template: `{default_cif_path().name}` (bcc W lattice).")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Supercell rep (nx,ny,nz)", key="sqs_rep")
        st.text_input("Cutoffs (Å)", key="sqs_cutoffs")
    with c2:
        st.number_input("MC steps", min_value=1, key="sqs_steps")
        st.number_input("Seed", min_value=0, key="sqs_seed")
    with c3:
        st.number_input("Plateau idle (s, 0=off)", min_value=0.0, key="sqs_plateau")
        st.number_input("Checkpoint (s)", min_value=0, key="sqs_ckpt")
    st.info(
        "Emit writes a runnable `01_sqs/` pack. Full `run` needs `icet` on the compute host "
        "(often Linux/WSL, not bare Windows)."
    )


def page_stage2() -> None:
    st.subheader("Stage 2 — Metropolis aging")
    st.caption("Auto-wires Stage-1 `poscar_out` when the pack is chained.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Temperature (K)", min_value=1.0, key="metro_T")
        st.number_input("NPROCS (16 forbidden)", min_value=1, max_value=8, key="metro_nprocs")
    with c2:
        st.number_input("Seed", min_value=0, key="metro_seed")
        st.number_input("Expected atoms", min_value=2, key="metro_expected")
    with c3:
        st.number_input("MIN_ROUNDS", min_value=1, key="metro_min_rounds")
        st.number_input("STABLE_ROUNDS", min_value=1, key="metro_stable")
        st.number_input("MAX_ROUNDS", min_value=1, key="metro_max_rounds")
        st.number_input("PE_TOL_MEV", min_value=0.0, key="metro_pe_tol")


def page_stage3() -> None:
    st.subheader("Stage 3 — Vacancy & SIA")
    st.caption(
        "Both stages chain from Metropolis `aged.data`. "
        "SIA labels **element pairs only** (no octahedral / ⟨hkl⟩ geometry)."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Vacancy**")
        st.number_input("Samples / element", min_value=1, key="vac_samples")
        st.number_input("SITE_SEED", min_value=0, key="vac_seed")
        st.number_input("NWORKERS", min_value=1, key="vac_workers")
        st.number_input("Expected atoms", min_value=2, key="vac_expected")
    with right:
        st.markdown("**SIA anneal**")
        st.number_input("ANNEAL_K", min_value=1.0, key="sia_anneal")
        st.number_input("Inserts / element", min_value=1, key="sia_inserts")
        st.number_input("MD_STEPS", min_value=1, key="sia_md")
        st.number_input("COOL_STEPS", min_value=1, key="sia_cool")
        st.number_input("SITE_SEED (SIA)", min_value=0, key="sia_seed")
        st.number_input("NWORKERS (SIA)", min_value=1, key="sia_workers")
        st.number_input("Expected atoms (SIA)", min_value=2, key="sia_expected")


def _sqs_params() -> dict:
    rep = _parse_rep(st.session_state.sqs_rep)
    return {
        "supercell_rep": rep,
        "target_total_atoms": 2 * rep[0] * rep[1] * rep[2],
        "cutoffs": _parse_cutoffs(st.session_state.sqs_cutoffs),
        "n_mc_steps": int(st.session_state.sqs_steps),
        "seed": int(st.session_state.sqs_seed),
        "plateau_idle_sec": float(st.session_state.sqs_plateau),
        "checkpoint_sec": int(st.session_state.sqs_ckpt),
    }


def _metro_params() -> dict:
    return {
        "temperature_k": float(st.session_state.metro_T),
        "nprocs": int(st.session_state.metro_nprocs),
        "seed": int(st.session_state.metro_seed),
        "min_rounds": int(st.session_state.metro_min_rounds),
        "stable_rounds": int(st.session_state.metro_stable),
        "max_rounds": int(st.session_state.metro_max_rounds),
        "pe_tol_mev": float(st.session_state.metro_pe_tol),
        "expected_natoms": int(st.session_state.metro_expected),
    }


def _vac_params() -> dict:
    return {
        "samples_per_element": int(st.session_state.vac_samples),
        "site_seed": int(st.session_state.vac_seed),
        "nworkers": int(st.session_state.vac_workers),
        "expected_natoms": int(st.session_state.vac_expected),
    }


def _sia_params() -> dict:
    return {
        "anneal_k": float(st.session_state.sia_anneal),
        "inserts_per_element": int(st.session_state.sia_inserts),
        "md_steps": int(st.session_state.sia_md),
        "cool_steps": int(st.session_state.sia_cool),
        "site_seed": int(st.session_state.sia_seed),
        "nworkers": int(st.session_state.sia_workers),
        "expected_natoms": int(st.session_state.sia_expected),
    }


def page_export() -> None:
    st.subheader("Pack export")
    st.caption("Writes a chained folder you can copy to a cluster. Optional zip download.")

    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Pack folder name", key="pack_name")
        st.checkbox("Include Stage 1 SQS", key="include_sqs")
        st.checkbox("Include Stage 2 Metropolis", key="include_metro")
    with c2:
        st.checkbox("Include Stage 3-1 Vacancy", key="include_vac")
        st.checkbox("Include Stage 3-2 SIA (pair chemistry only)", key="include_sia")

    out_root = ROOT / "packs"
    st.write(f"Packs are written under `{out_root}`")

    if st.button("Emit job pack", type="primary", use_container_width=True):
        try:
            family = build_family_from_ui()
            pot_dir = str(st.session_state.pot_dir).strip() or None
            needs_pot = (
                st.session_state.include_metro
                or st.session_state.include_vac
                or st.session_state.include_sia
            )
            if needs_pot and not pot_dir:
                st.error("Set POT_DIR on the Family tab before emitting Metropolis/Vacancy/SIA.")
                return

            pack_dir = out_root / str(st.session_state.pack_name).strip()
            req = PackRequest(
                family=family,
                pack_dir=pack_dir,
                pot_dir=Path(pot_dir) if pot_dir else None,
                lmp=str(st.session_state.lmp).strip() or "lmp",
                mpirun=str(st.session_state.mpirun).strip() or "mpirun",
                include_sqs=bool(st.session_state.include_sqs),
                include_metropolis=bool(st.session_state.include_metro),
                include_vacancy=bool(st.session_state.include_vac),
                include_sia=bool(st.session_state.include_sia),
                sqs_params=_sqs_params(),
                metro_params=_metro_params(),
                vacancy_params=_vac_params(),
                sia_params=_sia_params(),
            )
            result = emit_pack(req)
            st.session_state["last_pack"] = str(result.pack_dir)
            st.success(f"Wrote pack → `{result.pack_dir}`")
            st.json(
                {
                    "stages": {k: v.name for k, v in result.stage_dirs.items()},
                    "readme": result.readme.name,
                }
            )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    last = st.session_state.get("last_pack")
    if last and Path(last).is_dir():
        st.markdown(f"Last pack: `{last}`")
        if st.button("Build zip for download", use_container_width=True):
            zpath = zip_pack(Path(last))
            data = zpath.read_bytes()
            st.download_button(
                "Download zip",
                data=data,
                file_name=zpath.name,
                mime="application/zip",
                use_container_width=True,
            )


def main() -> None:
    _init_state()
    st.title("RCCA Workflow")
    st.caption(
        "Configure family → stages → emit a chained local job pack "
        "(SQS → Metropolis → Vacancy / SIA). SIA: element-pair labels only."
    )

    tabs = st.tabs(["Family", "Stage 1 SQS", "Stage 2 Metropolis", "Stage 3 Defects", "Pack export"])
    with tabs[0]:
        page_family()
    with tabs[1]:
        page_stage1()
    with tabs[2]:
        page_stage2()
    with tabs[3]:
        page_stage3()
    with tabs[4]:
        page_export()


if __name__ == "__main__":
    main()
