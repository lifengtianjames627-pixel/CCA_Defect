"""Stage 1 — SQS (to be ported from ex_scripts/Step1_SQS)."""

from __future__ import annotations

from pathlib import Path

from ..manifest import Manifest, StageIO, StageName
from .base import StageConfig, StageContext, StageResult, StageRunner


class SqsStage(StageRunner):
    name = StageName.SQS

    def emit(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        work = Path(cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        params = {
            "cutoffs": cfg.params.get("cutoffs", (4.8,)),
            "supercell_rep": cfg.params.get("supercell_rep", (14, 14, 14)),
            "target_total_atoms": cfg.params.get("target_total_atoms", 5488),
            "cif_in": cfg.params.get("cif_in", "mp-91_W.cif"),
            "species": list(ctx.family.symbols),
            "target_concentrations": ctx.family.fractions,
        }
        manifest = Manifest(
            stage=self.name,
            family_name=ctx.family.name,
            composition=dict(ctx.family.fractions),
            seeds={"sqs_seed": cfg.params.get("seed")},
            potential_hash=None,
            io=StageIO(
                inputs={"cif": str(params["cif_in"])},
                outputs={
                    "poscar_out": "poscar_out",
                    "final_cost_function": "final_cost_function.txt",
                    "final_cluster_vector_deviations": "final_cluster_vector_deviations.tsv",
                },
            ),
            params=params,
            env_echo=dict(cfg.env),
            notes="Stub emit; driver port is implementation step 2.",
        )
        manifest.write(work)
        return StageResult(stage=self.name, work_dir=work, manifest=manifest)
