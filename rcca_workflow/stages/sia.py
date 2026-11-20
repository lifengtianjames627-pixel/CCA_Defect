"""Stage 3-2 — SIA anneal + element-pair chemistry (no geometry classification)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Optional

from ..manifest import Manifest, StageIO, StageName, previous_artifact
from ..potential import PotentialRef, resolve_potential
from .base import StageConfig, StageContext, StageResult, StageRunner
from .sia_driver import (
    EXCLUDED_GEOMETRY_FIELDS,
    SIA_FIELDS,
    energy_csv_name,
    pair_csv_name,
)

AGED_IN = "aged.data"
CONFIG_OUT = "sia_config.json"


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
        return previous_artifact(StageName.SIA, Path(ctx.prev_dir)).resolve()
    local = work / AGED_IN
    if local.is_file():
        return local.resolve()
    raise FileNotFoundError(
        "No aged.data: set params['aged']/AGED, StageContext.prev_dir, "
        f"or place {AGED_IN} under the work directory"
    )


def build_sia_params(ctx: StageContext, cfg: StageConfig) -> dict[str, Any]:
    expected = cfg.params.get("expected_natoms", cfg.params.get("target_total_atoms"))
    nworkers_raw = cfg.params.get("nworkers", cfg.env.get("NWORKERS"))
    return {
        "anneal_k": float(cfg.params.get("anneal_k", cfg.env.get("ANNEAL_K", 1000.0))),
        "inserts_per_element": int(
            cfg.params.get(
                "inserts_per_element",
                cfg.env.get("INSERTS_PER_ELEMENT", 200),
            )
        ),
        "md_steps": int(cfg.params.get("md_steps", cfg.env.get("MD_STEPS", 5000))),
        "cool_steps": int(cfg.params.get("cool_steps", cfg.env.get("COOL_STEPS", 20000))),
        "site_seed": int(cfg.params.get("site_seed", cfg.env.get("SITE_SEED", 12345))),
        "nworkers": None if nworkers_raw in (None, "", "auto") else int(nworkers_raw),
        "expected_natoms": None if expected in (None, "any") else int(expected),
        "skip_bulk": bool(cfg.params.get("skip_bulk", False)),
        "mu_eV": dict(ctx.family.mu_eV),
        "masses": dict(ctx.family.masses),
        "species": list(ctx.family.symbols),
        "energy_csv_columns": list(SIA_FIELDS),
        "excluded_geometry": list(EXCLUDED_GEOMETRY_FIELDS),
    }


def assert_sia_csv_shape(path: Path) -> dict[str, Any]:
    """Energy CSV must match chemistry-only schema (no geometry/orientation)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if fields != SIA_FIELDS:
            raise ValueError(
                f"{path.name} header mismatch:\n  got: {fields}\n  want: {SIA_FIELDS}"
            )
        if "geometry" in fields or "orientation" in fields:
            raise ValueError(f"{path.name} must not contain geometry/orientation columns")
        rows = list(reader)
    return {"n_rows": len(rows), "fields": fields}


def assert_sia_output_shape(work_dir: Path, *, anneal_k: float = 1000.0) -> dict[str, Path]:
    work = Path(work_dir)
    energy = work / energy_csv_name(anneal_k)
    pairs = work / pair_csv_name(anneal_k)
    assert_sia_csv_shape(energy)
    if pairs.is_file():
        with pairs.open(newline="", encoding="utf-8") as handle:
            fields = list(csv.DictReader(handle).fieldnames or [])
        for banned in ("geometry", "orientation"):
            if banned in fields:
                raise ValueError(f"{pairs.name} must not contain {banned}")
    return {"energy_csv": energy, "pair_csv": pairs}


class SiaStage(StageRunner):
    name = StageName.SIA

    def emit(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        work = Path(cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        params = build_sia_params(ctx, cfg)

        pot = ctx.potential
        if pot is None:
            pot_dir = cfg.params.get("pot_dir") or cfg.env.get("POT_DIR")
            if pot_dir:
                pot = resolve_potential(PotentialRef(pot_dir=Path(str(pot_dir))))
            else:
                raise ValueError("SIA stage requires ctx.potential or POT_DIR")

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

        anneal_k = float(params["anneal_k"])
        energy_name = energy_csv_name(anneal_k)
        pair_name = pair_csv_name(anneal_k)

        params_with_pot = dict(params)
        params_with_pot["pot_dir"] = str(pot.pot_dir)
        (work / CONFIG_OUT).write_text(
            json.dumps(params_with_pot, indent=2) + "\n", encoding="utf-8"
        )
        (work / "family.json").write_text(
            json.dumps(ctx.family.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        (work / "run_sia.py").write_text(
            "\n".join(
                [
                    "#!/usr/bin/env python3",
                    '"""Auto-generated SIA launcher (anneal + element-pair only)."""',
                    "from pathlib import Path",
                    "import json",
                    "from rcca_workflow.family import Family",
                    "from rcca_workflow.potential import PotentialRef, resolve_potential",
                    "from rcca_workflow.stages.base import StageConfig, StageContext",
                    "from rcca_workflow.stages.sia import SiaStage",
                    "",
                    "HERE = Path(__file__).resolve().parent",
                    "family = Family.from_dict(json.loads((HERE / 'family.json').read_text()))",
                    "params = json.loads((HERE / 'sia_config.json').read_text())",
                    "man = json.loads((HERE / 'manifest.json').read_text())",
                    "pot_dir = params.get('pot_dir') or (man.get('potential') or {}).get('pot_dir')",
                    "pot = resolve_potential(PotentialRef(pot_dir=Path(pot_dir)))",
                    "ctx = StageContext(family=family, potential=pot)",
                    "cfg = StageConfig(work_dir=HERE, params=params)",
                    "SiaStage().run(ctx, cfg)",
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
                outputs={
                    "sia_formation_energies": energy_name,
                    "sia_pairs": pair_name,
                },
            ),
            params=params_with_pot,
            env_echo={
                "POT_DIR": str(pot.pot_dir),
                "LMP": ctx.lmp,
                "ANNEAL_K": str(params["anneal_k"]),
                "INSERTS_PER_ELEMENT": str(params["inserts_per_element"]),
                "MD_STEPS": str(params["md_steps"]),
                "COOL_STEPS": str(params["cool_steps"]),
                "SITE_SEED": str(params["site_seed"]),
                **dict(cfg.env),
            },
            notes=(
                "Stage 3-2 SIA: anneal + Ef + element-pair chemistry only. "
                "No geometry/orientation classifier (oct / <110> / <111> / <221>)."
            ),
        )
        manifest.write(work)
        return StageResult(stage=self.name, work_dir=work, manifest=manifest)

    def run(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        emitted = self.emit(ctx, cfg)
        work = Path(emitted.work_dir)
        params = build_sia_params(ctx, cfg)

        pot = ctx.potential
        if pot is None:
            pot_dir = emitted.manifest.params.get("pot_dir") or cfg.env.get("POT_DIR")
            pot = resolve_potential(PotentialRef(pot_dir=Path(str(pot_dir))))

        aged = resolve_aged(ctx, cfg, work)
        from .sia_driver import SiaRunConfig, run_sia

        result = run_sia(
            work,
            SiaRunConfig(
                aged=aged,
                pot_dir=pot.pot_dir,
                species=tuple(ctx.family.symbols),
                masses=ctx.family.masses,
                mu_eV=ctx.family.mu_eV,
                lmp=ctx.lmp,
                anneal_k=float(params["anneal_k"]),
                inserts_per_element=int(params["inserts_per_element"]),
                md_steps=int(params["md_steps"]),
                cool_steps=int(params["cool_steps"]),
                site_seed=int(params["site_seed"]),
                nworkers=params["nworkers"],
                expected_natoms=params["expected_natoms"],
                family_name=ctx.family.name,
                skip_bulk=bool(params["skip_bulk"]),
            ),
        )
        paths = assert_sia_output_shape(work, anneal_k=float(result["anneal_k"]))
        emitted.manifest.io.outputs["sia_formation_energies"] = paths["energy_csv"].name
        if paths["pair_csv"].is_file():
            emitted.manifest.io.outputs["sia_pairs"] = paths["pair_csv"].name
        emitted.manifest.params["run_result"] = {
            k: result[k]
            for k in ("status", "e_bulk", "n_atoms", "n_done", "n_fail", "anneal_k")
            if k in result
        }
        emitted.manifest.write(work)
        return StageResult(
            stage=self.name,
            work_dir=work,
            manifest=emitted.manifest,
            success=result.get("status") == "ok",
            message=f"wrote {result.get('n_done')} SIA rows (pair chemistry only)",
        )
