"""Stage 2 — Metropolis aging (parameterized port of ex_scripts/Step2_MetroMC)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from ..manifest import Manifest, StageIO, StageName, previous_artifact
from ..potential import PotentialRef, resolve_potential
from .base import StageConfig, StageContext, StageResult, StageRunner
from .metropolis_driver import FORBIDDEN_NPROCS

AGED_OUT = "aged.data"
RELAXED_OUT = "relaxed.data"
PLOT_OUT = "mc_progress.png"
CONFIG_OUT = "metro_config.json"
POSCAR_IN = "poscar_in"

_ATOMS_LINE = re.compile(r"^(\d+)\s+atoms\s*$", re.IGNORECASE)
_TYPES_LINE = re.compile(r"^(\d+)\s+atom types\s*$", re.IGNORECASE)


def resolve_poscar(ctx: StageContext, cfg: StageConfig, work: Path) -> Path:
    """Resolve Stage-1 POSCAR: explicit param/env, prev_dir chain, or local copy."""
    raw = cfg.params.get("poscar") or cfg.env.get("POSCAR")
    if raw:
        path = Path(str(raw)).expanduser()
        if path.is_file():
            return path.resolve()
        cand = work / path.name
        if cand.is_file():
            return cand.resolve()
        raise FileNotFoundError(f"POSCAR not found: {path}")

    if ctx.prev_dir is not None:
        return previous_artifact(StageName.METROPOLIS, Path(ctx.prev_dir)).resolve()

    for name in (POSCAR_IN, "poscar_out", "poscar_out.vasp", "poscar_out_interrupt"):
        cand = work / name
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        "No POSCAR: set params['poscar']/POSCAR, StageContext.prev_dir, "
        f"or place {POSCAR_IN} under the work directory"
    )


def build_metro_params(ctx: StageContext, cfg: StageConfig) -> dict[str, Any]:
    nprocs = int(cfg.params.get("nprocs", cfg.env.get("NPROCS", 4)))
    if nprocs in FORBIDDEN_NPROCS:
        raise ValueError(
            f"NPROCS={nprocs} is unsafe for this cell (ghost layer too thick); "
            f"forbidden set={sorted(FORBIDDEN_NPROCS)}"
        )
    expected = cfg.params.get("expected_natoms", cfg.params.get("target_total_atoms"))
    return {
        "temperature_k": float(cfg.params.get("temperature_k", 300.0)),
        "nprocs": nprocs,
        "seed": int(cfg.params.get("seed", cfg.env.get("SEED", 12345))),
        "log_every": int(cfg.params.get("log_every", cfg.env.get("LOG_EVERY", 1000))),
        "min_rounds": int(cfg.params.get("min_rounds", cfg.env.get("MIN_ROUNDS", 10))),
        "stable_rounds": int(
            cfg.params.get("stable_rounds", cfg.env.get("STABLE_ROUNDS", 3))
        ),
        "max_rounds": int(cfg.params.get("max_rounds", cfg.env.get("MAX_ROUNDS", 200))),
        "pe_tol_mev": float(cfg.params.get("pe_tol_mev", cfg.env.get("PE_TOL_MEV", 1.0))),
        "skip_relax": bool(cfg.params.get("skip_relax", False)),
        "bench": bool(cfg.params.get("bench", False)),
        "expected_natoms": None if expected in (None, "any") else int(expected),
        "masses": dict(ctx.family.masses),
        "species": list(ctx.family.symbols),
    }


def assert_lammps_data_shape(path: Path, *, min_atoms: int = 2) -> dict[str, int]:
    """Golden-check a LAMMPS data file has atoms / atom types headers."""
    if not path.is_file():
        raise FileNotFoundError(path)
    n_atoms = None
    n_types = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
        m = _ATOMS_LINE.match(line.strip())
        if m:
            n_atoms = int(m.group(1))
        m = _TYPES_LINE.match(line.strip())
        if m:
            n_types = int(m.group(1))
    if n_atoms is None or n_types is None:
        raise ValueError(f"{path.name} missing 'N atoms' / 'N atom types' header lines")
    if n_atoms < min_atoms:
        raise ValueError(f"{path.name} has too few atoms: {n_atoms}")
    if n_types < 2:
        raise ValueError(f"{path.name} needs >=2 atom types for swaps, got {n_types}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if "Masses" not in text or "Atoms" not in text:
        raise ValueError(f"{path.name} missing Masses/Atoms sections")
    return {"n_atoms": n_atoms, "n_types": n_types}


def assert_metro_output_shape(work_dir: Path, *, require_plot: bool = False) -> dict[str, Path]:
    """Golden-check Stage-2 artifacts against ex_scripts/Step2_MetroMC."""
    work = Path(work_dir)
    aged = work / AGED_OUT
    relaxed = work / RELAXED_OUT
    plot = work / PLOT_OUT
    missing = [p.name for p in (aged, relaxed) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Metropolis outputs missing under {work}: {missing}")
    assert_lammps_data_shape(aged)
    assert_lammps_data_shape(relaxed)
    if require_plot and not plot.is_file():
        raise FileNotFoundError(f"missing {PLOT_OUT}")
    return {"aged": aged, "relaxed": relaxed, "plot": plot}


class MetropolisStage(StageRunner):
    name = StageName.METROPOLIS

    def emit(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        work = Path(cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        params = build_metro_params(ctx, cfg)

        pot = ctx.potential
        if pot is None:
            pot_dir = cfg.params.get("pot_dir") or cfg.env.get("POT_DIR")
            if pot_dir:
                pot = resolve_potential(PotentialRef(pot_dir=Path(str(pot_dir))))
            else:
                raise ValueError(
                    "Metropolis stage requires ctx.potential or params/env POT_DIR"
                )

        poscar: Optional[Path] = None
        try:
            poscar = resolve_poscar(ctx, cfg, work)
        except FileNotFoundError:
            poscar = None

        if poscar is not None:
            dst = work / POSCAR_IN
            if poscar.resolve() != dst.resolve():
                shutil.copy2(poscar, dst)
            params["poscar"] = POSCAR_IN
        else:
            params["poscar"] = cfg.params.get("poscar") or cfg.env.get("POSCAR") or ""

        (work / CONFIG_OUT).write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
        (work / "family.json").write_text(
            json.dumps(ctx.family.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        (work / "run_metropolis.py").write_text(
            "\n".join(
                [
                    "#!/usr/bin/env python3",
                    '"""Auto-generated Metropolis launcher for this job pack."""',
                    "from pathlib import Path",
                    "import json",
                    "from rcca_workflow.family import Family",
                    "from rcca_workflow.potential import PotentialRef, resolve_potential",
                    "from rcca_workflow.stages.base import StageConfig, StageContext",
                    "from rcca_workflow.stages.metropolis import MetropolisStage",
                    "",
                    "HERE = Path(__file__).resolve().parent",
                    "family = Family.from_dict(json.loads((HERE / 'family.json').read_text()))",
                    "params = json.loads((HERE / 'metro_config.json').read_text())",
                    "man = json.loads((HERE / 'manifest.json').read_text())",
                    "pot_dir = params.get('pot_dir') or (man.get('potential') or {}).get('pot_dir')",
                    "pot = resolve_potential(PotentialRef(pot_dir=Path(pot_dir)))",
                    "ctx = StageContext(family=family, potential=pot)",
                    "cfg = StageConfig(work_dir=HERE, params=params)",
                    "MetropolisStage().run(ctx, cfg)",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        # Store pot_dir in params for the generated launcher.
        params_with_pot = dict(params)
        params_with_pot["pot_dir"] = str(pot.pot_dir)
        (work / CONFIG_OUT).write_text(
            json.dumps(params_with_pot, indent=2) + "\n", encoding="utf-8"
        )

        manifest = Manifest(
            stage=self.name,
            family_name=ctx.family.name,
            composition=dict(ctx.family.fractions),
            seeds={"seed": params["seed"]},
            potential_hash=pot.content_hash,
            potential=pot.to_dict(relative_to=work),
            io=StageIO(
                inputs={"poscar": params.get("poscar") or ""},
                outputs={
                    "aged": AGED_OUT,
                    "relaxed": RELAXED_OUT,
                    "progress_plot": PLOT_OUT,
                },
            ),
            params=params_with_pot,
            env_echo={
                "POT_DIR": str(pot.pot_dir),
                "LMP": ctx.lmp,
                "MPIRUN": ctx.mpirun,
                "NPROCS": str(params["nprocs"]),
                **dict(cfg.env),
            },
            notes="Stage 2 Metropolis pack; chain input is Stage-1 poscar_out.",
        )
        manifest.write(work)
        return StageResult(stage=self.name, work_dir=work, manifest=manifest)

    def run(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        emitted = self.emit(ctx, cfg)
        work = Path(emitted.work_dir)
        params = build_metro_params(ctx, cfg)

        pot = ctx.potential
        if pot is None:
            pot_dir = emitted.manifest.params.get("pot_dir") or cfg.env.get("POT_DIR")
            pot = resolve_potential(PotentialRef(pot_dir=Path(str(pot_dir))))

        poscar = resolve_poscar(ctx, cfg, work)
        from .metropolis_driver import MetroRunConfig, run_metropolis

        result = run_metropolis(
            work,
            MetroRunConfig(
                poscar=poscar,
                pot_dir=pot.pot_dir,
                masses=ctx.family.masses,
                temperature_k=float(params["temperature_k"]),
                expected_natoms=params["expected_natoms"],
                lmp=ctx.lmp,
                mpirun=ctx.mpirun,
                nprocs=int(params["nprocs"]),
                seed=int(params["seed"]),
                log_every=int(params["log_every"]),
                min_rounds=int(params["min_rounds"]),
                stable_rounds=int(params["stable_rounds"]),
                max_rounds=int(params["max_rounds"]),
                pe_tol_mev=float(params["pe_tol_mev"]),
                skip_relax=bool(params["skip_relax"]),
                bench=bool(params["bench"]),
                family_name=ctx.family.name,
            ),
        )
        if result.get("status") != "bench":
            paths = assert_metro_output_shape(work, require_plot=False)
            emitted.manifest.io.outputs["aged"] = paths["aged"].name
            emitted.manifest.io.outputs["relaxed"] = paths["relaxed"].name
        emitted.manifest.params["run_result"] = {
            k: result[k]
            for k in ("status", "n_atoms", "rounds", "stop_reason")
            if k in result
        }
        emitted.manifest.write(work)
        return StageResult(
            stage=self.name,
            work_dir=work,
            manifest=emitted.manifest,
            success=result.get("status") in {"converged", "max_rounds", "bench"},
            message=str(result.get("stop_reason") or result.get("status")),
        )
