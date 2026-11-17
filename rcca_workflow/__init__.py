"""RCCA workflow: SQS → Metropolis aging → vacancy/SIA formation job packs."""

from .family import DEFAULT_WTACRV, ElementSpec, Family
from .manifest import Manifest, StageIO, StageName
from .potential import PotentialRef, ResolvedPotential, resolve_potential

__all__ = [
    "Family",
    "ElementSpec",
    "DEFAULT_WTACRV",
    "Manifest",
    "StageIO",
    "StageName",
    "PotentialRef",
    "ResolvedPotential",
    "resolve_potential",
]

__version__ = "0.1.0"
