"""Stage 2 — Metropolis aging (to be ported from ex_scripts/Step2_MetroMC)."""

from __future__ import annotations

from pathlib import Path

from ..manifest import Manifest, StageIO, StageName, previous_artifact
from .base import StageConfig, StageContext, StageResult, StageRunner

FORBIDDEN_NPROCS = {16}


class MetropolisStage(StageRunner):
    name = StageName.METROPOLIS

    def emit(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        work = Path(cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)

        nprocs = int(cfg.params.get("nprocs", cfg.env.get("NPROCS", 4)))
        if nprocs in FORBIDDEN_NPROCS:
            raise ValueError(
                f"NPROCS={nprocs} is unsafe for this cell (ghost layer too thick); "
                f"forbidden set={sorted(FORBIDDEN_NPROCS)}"
            )
        if ctx.potential is None:
            raise ValueError("Metropolis stage requires a resolved potential")

        poscar = None
        if ctx.prev_dir is not None:
            poscar = previous_artifact(self.name, Path(ctx.prev_dir))

        params = {
            "temperature_k": cfg.params.get("temperature_k", 300.0),
            "nprocs": nprocs,
            "seed": cfg.params.get("seed", 12345),
            "min_rounds": cfg.params.get("min_rounds", 10),
            "stable_rounds": cfg.params.get("stable_rounds", 3),
            "max_rounds": cfg.params.get("max_rounds", 200),
            "pe_tol_mev": cfg.params.get("pe_tol_mev", 1.0),
        }
        manifest = Manifest(
            stage=self.name,
            family_name=ctx.family.name,
            composition=dict(ctx.family.fractions),
            seeds={"seed": params["seed"]},
            potential_hash=ctx.potential.content_hash,
            potential=ctx.potential.to_dict(relative_to=work),
            io=StageIO(
                inputs={"poscar": str(poscar) if poscar else cfg.env.get("POSCAR", "")},
                outputs={
                    "aged": "aged.data",
                    "relaxed": "relaxed.data",
                    "progress_plot": "mc_progress.png",
                },
            ),
            params=params,
            env_echo={
                "POT_DIR": str(ctx.potential.pot_dir),
                "LMP": ctx.lmp,
                "MPIRUN": ctx.mpirun,
                "NPROCS": str(nprocs),
                **dict(cfg.env),
            },
            notes="Stub emit; driver port is implementation step 3.",
        )
        manifest.write(work)
        return StageResult(stage=self.name, work_dir=work, manifest=manifest)
