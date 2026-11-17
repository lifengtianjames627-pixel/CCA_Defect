"""Shared stage interfaces for local runners and job-pack emission."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..family import Family
from ..manifest import Manifest, StageName
from ..potential import ResolvedPotential


@dataclass
class StageConfig:
    """Stage parameters + env-var style overrides (mirrors ex_scripts)."""

    work_dir: Path
    params: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class StageContext:
    """Runtime context shared across stages in one pack."""

    family: Family
    potential: Optional[ResolvedPotential]
    lmp: str = "lmp"
    mpirun: str = "mpirun"
    prev_dir: Optional[Path] = None


@dataclass
class StageResult:
    """Outcome of emit or run."""

    stage: StageName
    work_dir: Path
    manifest: Manifest
    success: bool = True
    message: str = ""


class StageRunner(ABC):
    """One pipeline stage: emit a runnable folder and optionally execute it."""

    name: StageName

    @abstractmethod
    def emit(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        """Write configs / scripts / manifest into ``cfg.work_dir``."""

    def run(self, ctx: StageContext, cfg: StageConfig) -> StageResult:
        """Default: emit then raise until the concrete port is implemented."""
        result = self.emit(ctx, cfg)
        raise NotImplementedError(
            f"Stage {self.name.value} runner is not implemented yet "
            f"(pack written under {result.work_dir})"
        )
