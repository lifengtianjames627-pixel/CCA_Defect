"""Stage 3-1 — Vacancy formation (to be ported from ex_scripts/Step3_FormE/Step3-1_V)."""

from __future__ import annotations

from pathlib import Path

from ..manifest import Manifest, StageIO, StageName, previous_artifact
from .base import StageConfig, StageContext, StageResult, StageRunner


class VacancyStage(StageRunner):
    name = StageName.VACANCY

    def emit(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        work = Path(cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        if ctx.potential is None:
            raise ValueError("Vacancy stage requires a resolved potential")

        aged = None
        if ctx.prev_dir is not None:
            aged = previous_artifact(self.name, Path(ctx.prev_dir))

        params = {
            "samples_per_element": cfg.params.get("samples_per_element", 125),
            "site_seed": cfg.params.get("site_seed", 12345),
            "nworkers": cfg.params.get("nworkers", 13),
            "mu_eV": dict(ctx.family.mu_eV),
            "masses": dict(ctx.family.masses),
            "species": list(ctx.family.symbols),
        }
        manifest = Manifest(
            stage=self.name,
            family_name=ctx.family.name,
            composition=dict(ctx.family.fractions),
            seeds={"site_seed": params["site_seed"]},
            potential_hash=ctx.potential.content_hash,
            potential=ctx.potential.to_dict(relative_to=work),
            io=StageIO(
                inputs={"aged": str(aged) if aged else cfg.env.get("AGED", "")},
                outputs={"v_form_e": "v_form_e.csv", "cases_dir": "v_cases"},
            ),
            params=params,
            env_echo={
                "POT_DIR": str(ctx.potential.pot_dir),
                "LMP": ctx.lmp,
                "SAMPLES_PER_ELEMENT": str(params["samples_per_element"]),
                "SITE_SEED": str(params["site_seed"]),
                "NWORKERS": str(params["nworkers"]),
                **dict(cfg.env),
            },
            notes="Stub emit; driver port is implementation step 4. mu/masses from Family.",
        )
        manifest.write(work)
        return StageResult(stage=self.name, work_dir=work, manifest=manifest)
