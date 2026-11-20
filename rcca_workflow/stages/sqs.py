"""Stage 1 — SQS (parameterized port of ex_scripts/Step1_SQS)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

from ..family import Family
from ..manifest import Manifest, StageIO, StageName
from .base import StageConfig, StageContext, StageResult, StageRunner

DEFAULT_CIF_NAME = "mp-91_W.cif"
BUNDLED_CIF = Path(__file__).resolve().parents[1] / "templates" / DEFAULT_CIF_NAME

POSCAR_OUT = "poscar_out"
COST_OUT = "final_cost_function.txt"
DEVIATIONS_OUT = "final_cluster_vector_deviations.tsv"
CONFIG_OUT = "sqs_config.json"

# Golden output shape from ex_scripts/Step1_SQS
GOLDEN_COST_KEYS = (
    "Q=",
    "worst_abs_deviation_excluding_zerolet=",
    "n_cluster_vector_components=",
    "optimality_weight=",
    "optimality_tol=",
)
GOLDEN_DEVIATIONS_HEADER = "index\torbit_meta\tgamma_sys\tgamma_target\tabs_deviation"


def default_cif_path() -> Path:
    return BUNDLED_CIF


def resolve_cif(cfg: StageConfig, work: Path) -> Path:
    raw = cfg.params.get("cif_in") or cfg.env.get("SQS_CIF")
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            # Allow basename relative to work dir / templates.
            cand = work / path.name
            if cand.is_file():
                return cand.resolve()
            bundled = BUNDLED_CIF if path.name == DEFAULT_CIF_NAME else None
            if bundled and bundled.is_file():
                return bundled.resolve()
            raise FileNotFoundError(f"CIF not found: {path}")
        return path.resolve()
    local = work / DEFAULT_CIF_NAME
    if local.is_file():
        return local.resolve()
    if BUNDLED_CIF.is_file():
        return BUNDLED_CIF.resolve()
    raise FileNotFoundError(f"Missing lattice CIF ({DEFAULT_CIF_NAME})")


def _as_rep(value: Any, default: tuple[int, int, int] = (14, 14, 14)) -> tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(f"supercell_rep must be a length-3 sequence, got {value!r}")


def _as_cutoffs(value: Any, default: tuple[float, ...] = (4.8,)) -> tuple[float, ...]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, (list, tuple)) and value:
        return tuple(float(v) for v in value)
    raise ValueError(f"cutoffs must be a float or sequence of floats, got {value!r}")


def build_sqs_params(family: Family, cfg: StageConfig) -> dict[str, Any]:
    rep = _as_rep(cfg.params.get("supercell_rep"))
    target = cfg.params.get("target_total_atoms")
    if target is None:
        target = 2 * rep[0] * rep[1] * rep[2]
    return {
        "cif_in": str(cfg.params.get("cif_in", DEFAULT_CIF_NAME)),
        "cutoffs": list(_as_cutoffs(cfg.params.get("cutoffs"))),
        "supercell_rep": list(rep),
        "target_total_atoms": int(target),
        "species": list(family.symbols),
        "target_concentrations": dict(family.fractions),
        "n_mc_steps": int(cfg.params.get("n_mc_steps", cfg.env.get("SQS_N_MC_STEPS", 2_000_000))),
        "seed": int(cfg.params.get("seed", cfg.env.get("SQS_SEED", 42))),
        "plateau_idle_sec": float(
            cfg.params.get("plateau_idle_sec", cfg.env.get("SQS_PLATEAU_SEC", 900))
        ),
        "checkpoint_sec": int(
            cfg.params.get("checkpoint_sec", cfg.env.get("SQS_CHECKPOINT_SEC", 30))
        ),
        "log_level": str(cfg.params.get("log_level", cfg.env.get("SQS_LOG_LEVEL", "WARNING"))),
    }


def assert_sqs_output_shape(work_dir: Path, *, require_poscar: bool = True) -> dict[str, Path]:
    """
    Golden-check artifact names/schemas against ex_scripts/Step1_SQS.

    Does not compare numeric Q (seed/MC path dependent); checks file presence and headers.
    """
    work = Path(work_dir)
    poscar = work / POSCAR_OUT
    if not poscar.is_file():
        alt = work / "poscar_out.vasp"
        poscar = alt if alt.is_file() else poscar
    cost = work / COST_OUT
    dev = work / DEVIATIONS_OUT
    missing = []
    if require_poscar and not poscar.is_file():
        missing.append(POSCAR_OUT)
    if not cost.is_file():
        missing.append(COST_OUT)
    if not dev.is_file():
        missing.append(DEVIATIONS_OUT)
    if missing:
        raise FileNotFoundError(f"SQS outputs missing under {work}: {missing}")

    cost_text = cost.read_text(encoding="utf-8")
    for key in GOLDEN_COST_KEYS:
        if key not in cost_text:
            raise ValueError(f"{COST_OUT} missing required field prefix {key!r}")

    header = dev.read_text(encoding="utf-8").splitlines()[0].strip()
    if header != GOLDEN_DEVIATIONS_HEADER:
        raise ValueError(
            f"{DEVIATIONS_OUT} header mismatch:\n  got: {header!r}\n  want: {GOLDEN_DEVIATIONS_HEADER!r}"
        )
    return {"poscar": poscar, "cost": cost, "deviations": dev}


class SqsStage(StageRunner):
    name = StageName.SQS

    def emit(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        work = Path(cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        params = build_sqs_params(ctx.family, cfg)

        cif_src = resolve_cif(cfg, work)
        cif_dst = work / Path(params["cif_in"]).name
        if cif_src.resolve() != cif_dst.resolve():
            shutil.copy2(cif_src, cif_dst)
        params["cif_in"] = cif_dst.name

        (work / CONFIG_OUT).write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

        run_py = work / "run_sqs.py"
        run_py.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "\"\"\"Auto-generated SQS launcher for this job pack.\"\"\"",
                    "from pathlib import Path",
                    "from rcca_workflow.stages.sqs import SqsStage",
                    "from rcca_workflow.stages.base import StageConfig, StageContext",
                    "from rcca_workflow.family import Family",
                    "import json",
                    "",
                    "HERE = Path(__file__).resolve().parent",
                    "family = Family.from_dict(json.loads((HERE / 'family.json').read_text()))",
                    "params = json.loads((HERE / 'sqs_config.json').read_text())",
                    "ctx = StageContext(family=family, potential=None)",
                    "cfg = StageConfig(work_dir=HERE, params=params)",
                    "SqsStage().run(ctx, cfg)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (work / "family.json").write_text(
            json.dumps(ctx.family.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

        manifest = Manifest(
            stage=self.name,
            family_name=ctx.family.name,
            composition=dict(ctx.family.fractions),
            seeds={"sqs_seed": params["seed"]},
            potential_hash=None,
            io=StageIO(
                inputs={"cif": params["cif_in"]},
                outputs={
                    "poscar_out": POSCAR_OUT,
                    "final_cost_function": COST_OUT,
                    "final_cluster_vector_deviations": DEVIATIONS_OUT,
                },
            ),
            params=params,
            env_echo=dict(cfg.env),
            notes="Stage 1 SQS pack; run via run_sqs.py or SqsStage.run().",
        )
        manifest.write(work)
        return StageResult(stage=self.name, work_dir=work, manifest=manifest)

    def run(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        emitted = self.emit(ctx, cfg)
        work = Path(emitted.work_dir)
        params = build_sqs_params(ctx.family, cfg)
        # Prefer CIF copied into the work directory.
        cif_path = resolve_cif(
            StageConfig(work_dir=work, params={**cfg.params, "cif_in": params["cif_in"]}, env=cfg.env),
            work,
        )

        # Lazy import: driver monkeypatches mchammer on import and needs icet.
        from .sqs_driver import SqsRunConfig, run_sqs

        result = run_sqs(
            work,
            SqsRunConfig(
                cif_path=cif_path,
                cutoffs=_as_cutoffs(params["cutoffs"]),
                supercell_rep=_as_rep(params["supercell_rep"]),
                target_total_atoms=int(params["target_total_atoms"]),
                species=tuple(params["species"]),
                fractions=dict(params["target_concentrations"]),
                n_mc_steps=int(params["n_mc_steps"]),
                random_seed=int(params["seed"]),
                plateau_idle_sec=float(params["plateau_idle_sec"]),
                checkpoint_sec=int(params["checkpoint_sec"]),
                log_level=str(params["log_level"]),
            ),
        )
        paths = assert_sqs_output_shape(work, require_poscar=True)
        emitted.manifest.io.outputs["poscar_out"] = paths["poscar"].name
        emitted.manifest.params["run_result"] = {
            k: result[k]
            for k in ("status", "q", "n_atoms", "n_cluster_vector_components")
            if k in result
        }
        emitted.manifest.write(work)
        return StageResult(
            stage=self.name,
            work_dir=work,
            manifest=emitted.manifest,
            success=result.get("status") == "ok",
            message=f"Q={result.get('q')}",
        )


def config_from_mapping(family: Family, data: Mapping[str, Any], work_dir: Path) -> StageConfig:
    """Helper for tests / CLI: build StageConfig from a plain dict."""
    return StageConfig(work_dir=work_dir, params=dict(data))
