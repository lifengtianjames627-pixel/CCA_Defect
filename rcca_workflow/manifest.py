"""Stage I/O contracts and chain links (one manifest.json per stage folder)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional


class StageName(str, Enum):
    SQS = "sqs"
    METROPOLIS = "metropolis"
    VACANCY = "vacancy"
    SIA = "sia"


# Primary chain artifacts (mirrors ex_scripts outputs).
CHAIN_OUTPUTS: dict[StageName, tuple[str, ...]] = {
    StageName.SQS: ("poscar_out", "final_cost_function.txt", "final_cluster_vector_deviations.tsv"),
    StageName.METROPOLIS: ("aged.data", "relaxed.data", "mc_progress.png"),
    StageName.VACANCY: ("v_form_e.csv",),
    StageName.SIA: ("sia_formation_energies.csv",),
}

CHAIN_LINKS: dict[StageName, Optional[tuple[StageName, str]]] = {
    StageName.SQS: None,
    StageName.METROPOLIS: (StageName.SQS, "poscar_out"),
    StageName.VACANCY: (StageName.METROPOLIS, "aged.data"),
    StageName.SIA: (StageName.METROPOLIS, "aged.data"),
}


@dataclass
class StageIO:
    """Declared inputs consumed and outputs produced by one stage run."""

    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)


@dataclass
class Manifest:
    """
    Small JSON sidecar written into each stage work directory so the next
    stage only needs the previous folder + family config.
    """

    stage: StageName
    family_name: str
    composition: dict[str, float]
    seeds: dict[str, Any] = field(default_factory=dict)
    potential_hash: Optional[str] = None
    potential: Optional[dict[str, Any]] = None
    io: StageIO = field(default_factory=StageIO)
    params: dict[str, Any] = field(default_factory=dict)
    env_echo: dict[str, str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "family_name": self.family_name,
            "composition": self.composition,
            "seeds": self.seeds,
            "potential_hash": self.potential_hash,
            "potential": self.potential,
            "io": asdict(self.io),
            "params": self.params,
            "env_echo": self.env_echo,
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        io_raw = data.get("io") or {}
        return cls(
            stage=StageName(data["stage"]),
            family_name=str(data["family_name"]),
            composition={k: float(v) for k, v in dict(data.get("composition") or {}).items()},
            seeds=dict(data.get("seeds") or {}),
            potential_hash=data.get("potential_hash"),
            potential=data.get("potential"),
            io=StageIO(
                inputs=dict(io_raw.get("inputs") or {}),
                outputs=dict(io_raw.get("outputs") or {}),
            ),
            params=dict(data.get("params") or {}),
            env_echo={str(k): str(v) for k, v in dict(data.get("env_echo") or {}).items()},
            created_at=str(data.get("created_at") or ""),
            notes=str(data.get("notes") or ""),
        )

    def write(self, directory: Path, filename: str = "manifest.json") -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Manifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def previous_artifact(stage: StageName, prev_dir: Path) -> Path:
    """Resolve the chained input file from a previous stage folder."""
    link = CHAIN_LINKS.get(stage)
    if link is None:
        raise ValueError(f"Stage {stage.value} has no upstream chain artifact")
    _prev_stage, artifact = link
    # Prefer exact name; also accept common POSCAR aliases from Step1.
    candidates = [prev_dir / artifact]
    if artifact == "poscar_out":
        candidates.extend(
            [
                prev_dir / "poscar_out.vasp",
                prev_dir / "poscar_out_interrupt",
            ]
        )
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Missing chain artifact {artifact!r} under {prev_dir} "
        f"(needed by stage {stage.value})"
    )
