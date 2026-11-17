"""Stage 3-2 — SIA anneal only (no geometry classification)."""

from __future__ import annotations

from pathlib import Path

from ..manifest import Manifest, StageIO, StageName, previous_artifact
from .base import StageConfig, StageContext, StageResult, StageRunner

# Explicitly excluded from this stage (ex_scripts Part 3).
EXCLUDED_CLASSIFIER_SYMBOLS = (
    "load_classifier",
    "label_relaxed",
    "class_key",
    "write_class_summary",
    "sia_classes_*.csv",
    "geometry",
    "orientation",
    "pair",
    "element_a",
    "element_b",
)

ENERGY_CSV_COLUMNS = (
    "job",
    "site_id",
    "host_element",
    "inserted_element",
    "x",
    "y",
    "z",
    "E_sia",
    "E_bulk",
    "mu",
    "E_f",
    "status",
    "note",
)


class SiaStage(StageRunner):
    name = StageName.SIA

    def emit(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        work = Path(cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        if ctx.potential is None:
            raise ValueError("SIA stage requires a resolved potential")

        aged = None
        if ctx.prev_dir is not None:
            aged = previous_artifact(self.name, Path(ctx.prev_dir))

        params = {
            "anneal_k": cfg.params.get("anneal_k", 1000.0),
            "inserts_per_element": cfg.params.get("inserts_per_element", 200),
            "md_steps": cfg.params.get("md_steps", 5000),
            "cool_steps": cfg.params.get("cool_steps", 20000),
            "site_seed": cfg.params.get("site_seed", 12345),
            "mu_eV": dict(ctx.family.mu_eV),
            "masses": dict(ctx.family.masses),
            "species": list(ctx.family.symbols),
            "energy_csv_columns": list(ENERGY_CSV_COLUMNS),
            "excluded_classifier": list(EXCLUDED_CLASSIFIER_SYMBOLS),
        }
        # Output CSV basename follows anneal T when non-default, like the example.
        anneal_k = float(params["anneal_k"])
        csv_name = (
            "sia_formation_energies.csv"
            if anneal_k == 1000.0
            else f"sia_formation_energies_{int(anneal_k)}K.csv"
        )
        manifest = Manifest(
            stage=self.name,
            family_name=ctx.family.name,
            composition=dict(ctx.family.fractions),
            seeds={"site_seed": params["site_seed"]},
            potential_hash=ctx.potential.content_hash,
            potential=ctx.potential.to_dict(relative_to=work),
            io=StageIO(
                inputs={"aged": str(aged) if aged else cfg.env.get("AGED", "")},
                outputs={
                    "sia_formation_energies": csv_name,
                    # No sia_classes_*.csv — classification is out of scope for v1.
                },
            ),
            params=params,
            env_echo={
                "POT_DIR": str(ctx.potential.pot_dir),
                "LMP": ctx.lmp,
                "ANNEAL_K": str(params["anneal_k"]),
                "INSERTS_PER_ELEMENT": str(params["inserts_per_element"]),
                "MD_STEPS": str(params["md_steps"]),
                "COOL_STEPS": str(params["cool_steps"]),
                "SITE_SEED": str(params["site_seed"]),
                **dict(cfg.env),
            },
            notes=(
                "Stub emit; driver port is implementation step 5. "
                "Parts 1–2 + driver only — no Part 3 classification."
            ),
        )
        manifest.write(work)
        return StageResult(stage=self.name, work_dir=work, manifest=manifest)
