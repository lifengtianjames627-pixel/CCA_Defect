"""Stage 3-1 — Vacancy formation (parameterized port of ex_scripts Step3-1_V)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Optional

from ..manifest import Manifest, StageIO, StageName, previous_artifact
from ..potential import PotentialRef, resolve_potential
from .base import StageConfig, StageContext, StageResult, StageRunner
from .vacancy_driver import CASES_DIRNAME, CSV_NAME, csv_fields

AGED_IN = "aged.data"
CONFIG_OUT = "vacancy_config.json"

# Golden CSV columns from ex_scripts W–Ta–Cr–V run (demo family).
GOLDEN_CSV_FIELDS = [
    "site_id",
    "removed_element",
    "element_nearest_neighbors",
    "nn_Cr",
    "nn_Ta",
    "nn_V",
    "nn_W",
    "E_bulk_eV",
    "E_vacancy_eV",
    "mu_eV",
    "formation_energy_vacancy_eV",
]


def resolve_aged(ctx: StageContext, cfg: StageConfig, work: Path) -> Path:
    raw = cfg.params.get("aged") or cfg.env.get("AGED")
    if raw:
        path = Path(str(raw)).expanduser()
        if path.is_file():
            return path.resolve()
        cand = work / path.name
        if cand.is_file():
            return cand.resolve()
        raise FileNotFoundError(f"aged.data not found: {path}")

    if ctx.prev_dir is not None:
        return previous_artifact(StageName.VACANCY, Path(ctx.prev_dir)).resolve()

    local = work / AGED_IN
    if local.is_file():
        return local.resolve()
    raise FileNotFoundError(
        "No aged.data: set params['aged']/AGED, StageContext.prev_dir, "
        f"or place {AGED_IN} under the work directory"
    )


def build_vacancy_params(ctx: StageContext, cfg: StageConfig) -> dict[str, Any]:
    expected = cfg.params.get("expected_natoms", cfg.params.get("target_total_atoms"))
    return {
        "samples_per_element": int(
            cfg.params.get(
                "samples_per_element",
                cfg.env.get("SAMPLES_PER_ELEMENT", 125),
            )
        ),
        "site_seed": int(cfg.params.get("site_seed", cfg.env.get("SITE_SEED", 12345))),
        "nworkers": (
            None
            if cfg.params.get("nworkers") in (None, "auto")
            else int(cfg.params.get("nworkers", cfg.env.get("NWORKERS", 0)) or 0) or None
        ),
        "expected_natoms": None if expected in (None, "any") else int(expected),
        "skip_bulk": bool(cfg.params.get("skip_bulk", False)),
        "mu_eV": dict(ctx.family.mu_eV),
        "masses": dict(ctx.family.masses),
        "species": list(ctx.family.symbols),
    }


def assert_vacancy_csv_shape(
    path: Path, *, species: Optional[list[str]] = None
) -> dict[str, Any]:
    """Golden-check v_form_e.csv header/rows against the ex_scripts contract."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} has no header")
        fields = list(reader.fieldnames)
        expected = GOLDEN_CSV_FIELDS if species is None else csv_fields(species)
        if fields != expected:
            raise ValueError(
                f"{path.name} header mismatch:\n  got: {fields}\n  want: {expected}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} has header but no data rows")
    required = ("site_id", "removed_element", "formation_energy_vacancy_eV", "mu_eV")
    for key in required:
        if key not in rows[0]:
            raise ValueError(f"{path.name} row missing {key}")
    return {"n_rows": len(rows), "fields": fields}


def assert_vacancy_output_shape(
    work_dir: Path, *, species: Optional[list[str]] = None
) -> dict[str, Path]:
    work = Path(work_dir)
    csv_path = work / CSV_NAME
    assert_vacancy_csv_shape(csv_path, species=species)
    cases = work / CASES_DIRNAME
    # cases/ may be absent in partial golden CSVs; prefer presence when directory exists.
    return {"csv": csv_path, "cases": cases}


class VacancyStage(StageRunner):
    name = StageName.VACANCY

    def emit(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        work = Path(cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        params = build_vacancy_params(ctx, cfg)

        pot = ctx.potential
        if pot is None:
            pot_dir = cfg.params.get("pot_dir") or cfg.env.get("POT_DIR")
            if pot_dir:
                pot = resolve_potential(PotentialRef(pot_dir=Path(str(pot_dir))))
            else:
                raise ValueError("Vacancy stage requires ctx.potential or POT_DIR")

        aged: Optional[Path] = None
        try:
            aged = resolve_aged(ctx, cfg, work)
        except FileNotFoundError:
            aged = None
        if aged is not None:
            dst = work / AGED_IN
            if aged.resolve() != dst.resolve():
                shutil.copy2(aged, dst)
            params["aged"] = AGED_IN
        else:
            params["aged"] = cfg.params.get("aged") or cfg.env.get("AGED") or ""

        params_with_pot = dict(params)
        params_with_pot["pot_dir"] = str(pot.pot_dir)
        (work / CONFIG_OUT).write_text(
            json.dumps(params_with_pot, indent=2) + "\n", encoding="utf-8"
        )
        (work / "family.json").write_text(
            json.dumps(ctx.family.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        (work / "run_vacancy.py").write_text(
            "\n".join(
                [
                    "#!/usr/bin/env python3",
                    '"""Auto-generated vacancy launcher for this job pack."""',
                    "from pathlib import Path",
                    "import json",
                    "from rcca_workflow.family import Family",
                    "from rcca_workflow.potential import PotentialRef, resolve_potential",
                    "from rcca_workflow.stages.base import StageConfig, StageContext",
                    "from rcca_workflow.stages.vacancy import VacancyStage",
                    "",
                    "HERE = Path(__file__).resolve().parent",
                    "family = Family.from_dict(json.loads((HERE / 'family.json').read_text()))",
                    "params = json.loads((HERE / 'vacancy_config.json').read_text())",
                    "man = json.loads((HERE / 'manifest.json').read_text())",
                    "pot_dir = params.get('pot_dir') or (man.get('potential') or {}).get('pot_dir')",
                    "pot = resolve_potential(PotentialRef(pot_dir=Path(pot_dir)))",
                    "ctx = StageContext(family=family, potential=pot)",
                    "cfg = StageConfig(work_dir=HERE, params=params)",
                    "VacancyStage().run(ctx, cfg)",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        manifest = Manifest(
            stage=self.name,
            family_name=ctx.family.name,
            composition=dict(ctx.family.fractions),
            seeds={"site_seed": params["site_seed"]},
            potential_hash=pot.content_hash,
            potential=pot.to_dict(relative_to=work),
            io=StageIO(
                inputs={"aged": params.get("aged") or ""},
                outputs={"v_form_e": CSV_NAME, "cases_dir": CASES_DIRNAME},
            ),
            params=params_with_pot,
            env_echo={
                "POT_DIR": str(pot.pot_dir),
                "LMP": ctx.lmp,
                "SAMPLES_PER_ELEMENT": str(params["samples_per_element"]),
                "SITE_SEED": str(params["site_seed"]),
                **dict(cfg.env),
            },
            notes="Stage 3-1 vacancy pack; mu/masses from Family; chain input is aged.data.",
        )
        manifest.write(work)
        return StageResult(stage=self.name, work_dir=work, manifest=manifest)

    def run(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        emitted = self.emit(ctx, cfg)
        work = Path(emitted.work_dir)
        params = build_vacancy_params(ctx, cfg)

        pot = ctx.potential
        if pot is None:
            pot_dir = emitted.manifest.params.get("pot_dir") or cfg.env.get("POT_DIR")
            pot = resolve_potential(PotentialRef(pot_dir=Path(str(pot_dir))))

        aged = resolve_aged(ctx, cfg, work)
        from .vacancy_driver import VacancyRunConfig, run_vacancy

        nworkers = params["nworkers"]
        result = run_vacancy(
            work,
            VacancyRunConfig(
                aged=aged,
                pot_dir=pot.pot_dir,
                species=tuple(ctx.family.symbols),
                masses=ctx.family.masses,
                mu_eV=ctx.family.mu_eV,
                lmp=ctx.lmp,
                samples_per_element=int(params["samples_per_element"]),
                site_seed=int(params["site_seed"]),
                nworkers=nworkers,
                expected_natoms=params["expected_natoms"],
                family_name=ctx.family.name,
                skip_bulk=bool(params["skip_bulk"]),
            ),
        )
        paths = assert_vacancy_output_shape(work, species=list(ctx.family.symbols))
        emitted.manifest.io.outputs["v_form_e"] = paths["csv"].name
        emitted.manifest.params["run_result"] = {
            k: result[k]
            for k in ("status", "e_bulk", "n_atoms", "n_samples", "n_done", "nworkers")
            if k in result
        }
        emitted.manifest.write(work)
        return StageResult(
            stage=self.name,
            work_dir=work,
            manifest=emitted.manifest,
            success=result.get("status") == "ok",
            message=f"wrote {result.get('n_done')} vacancy rows",
        )
