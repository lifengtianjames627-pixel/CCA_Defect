"""Resolve tabGAP + eam/fs hybrid potential files (same contract as ex_scripts)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PotentialRef:
    """How to find the potential: local POT_DIR now; API download later."""

    pot_dir: Optional[Path] = None
    api_id: Optional[str] = None
    family_name: str = ""


@dataclass(frozen=True)
class ResolvedPotential:
    """Concrete files locked for a job pack / stage run."""

    pot_dir: Path
    tabgap: Path
    eam_fs: Path
    content_hash: str
    pair_style: str = "hybrid/overlay tabgap eam/fs"

    def to_dict(self, *, relative_to: Optional[Path] = None) -> dict:
        def rel(p: Path) -> str:
            if relative_to is None:
                return str(p)
            try:
                return str(p.resolve().relative_to(relative_to.resolve()))
            except ValueError:
                return str(p.resolve())

        return {
            "pot_dir": rel(self.pot_dir),
            "tabgap": rel(self.tabgap),
            "eam_fs": rel(self.eam_fs),
            "content_hash": self.content_hash,
            "pair_style": self.pair_style,
        }

    def pair_coeff_block(self, *, types_comment: str = "") -> str:
        """LAMMPS pair_style / pair_coeff fragment used by Stages 2–3."""
        comment = f"  # {types_comment}" if types_comment else ""
        return (
            f"pair_style      {self.pair_style}\n"
            f"pair_coeff      * * tabgap {self.tabgap.as_posix()}{comment}\n"
            f"pair_coeff      * * eam/fs  {self.eam_fs.as_posix()}{comment}\n"
        )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_potential(ref: PotentialRef) -> ResolvedPotential:
    """
    Resolve potential files.

    1. local ``POT_DIR`` with exactly one ``.tabgap`` and one ``.eam.fs``, or
    2. later: API download into ``potentials/<family>/<hash>/`` + lockfile.
    """
    if ref.pot_dir is None and ref.api_id:
        raise NotImplementedError(
            "API potential download is not implemented in v1; set pot_dir "
            "(same as ex_scripts POT_DIR)."
        )
    if ref.pot_dir is None:
        raise ValueError("PotentialRef.pot_dir is required (env POT_DIR)")

    pot_dir = Path(ref.pot_dir).expanduser().resolve()
    if not pot_dir.is_dir():
        raise FileNotFoundError(f"POT_DIR does not exist: {pot_dir}")

    tabgaps = sorted(pot_dir.glob("*.tabgap"))
    eams = sorted(pot_dir.glob("*.eam.fs"))
    if len(tabgaps) != 1 or len(eams) != 1:
        raise FileNotFoundError(
            f"{pot_dir} must contain exactly one .tabgap and one .eam.fs "
            f"(found {len(tabgaps)} tabgap, {len(eams)} eam/fs)"
        )

    tabgap, eam_fs = tabgaps[0], eams[0]
    content_hash = hashlib.sha256(
        (_file_sha256(tabgap) + _file_sha256(eam_fs)).encode()
    ).hexdigest()

    return ResolvedPotential(
        pot_dir=pot_dir,
        tabgap=tabgap,
        eam_fs=eam_fs,
        content_hash=content_hash,
    )


def write_lockfile(resolved: ResolvedPotential, path: Path) -> None:
    """Persist resolved paths + hash next to a job pack."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(resolved.to_dict(relative_to=path.parent), indent=2) + "\n",
        encoding="utf-8",
    )
