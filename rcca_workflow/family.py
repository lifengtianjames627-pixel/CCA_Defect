"""Alloy family: elements, composition, masses, chemical potentials, potential identity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ElementSpec:
    """One species in the family."""

    symbol: str
    mass: float
    mu_eV: float
    fraction: float


@dataclass(frozen=True)
class Family:
    """
    Family = element set + potential identity.
    Composition = molar fractions inside that family (need not be equiatomic).
    """

    name: str
    elements: tuple[ElementSpec, ...]
    potential_id: str = "local"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.elements:
            raise ValueError("Family must contain at least one element")
        symbols = [e.symbol for e in self.elements]
        if len(symbols) != len(set(symbols)):
            raise ValueError(f"Duplicate element symbols in family: {symbols}")
        total = sum(e.fraction for e in self.elements)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Molar fractions must sum to 1.0, got {total}")
        for e in self.elements:
            if e.fraction < 0.0:
                raise ValueError(f"Negative fraction for {e.symbol}")
            if e.mass <= 0.0:
                raise ValueError(f"Mass for {e.symbol} must be positive")

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(e.symbol for e in self.elements)

    @property
    def masses(self) -> dict[str, float]:
        return {e.symbol: e.mass for e in self.elements}

    @property
    def mu_eV(self) -> dict[str, float]:
        return {e.symbol: e.mu_eV for e in self.elements}

    @property
    def fractions(self) -> dict[str, float]:
        return {e.symbol: e.fraction for e in self.elements}

    @property
    def mass_to_element(self) -> dict[float, str]:
        """LAMMPS mass → symbol map used by vacancy/SIA cell I/O."""
        return {e.mass: e.symbol for e in self.elements}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "potential_id": self.potential_id,
            "notes": self.notes,
            "elements": [
                {
                    "symbol": e.symbol,
                    "mass": e.mass,
                    "mu_eV": e.mu_eV,
                    "fraction": e.fraction,
                }
                for e in self.elements
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> Family:
        elements = tuple(
            ElementSpec(
                symbol=str(item["symbol"]),
                mass=float(item["mass"]),
                mu_eV=float(item["mu_eV"]),
                fraction=float(item["fraction"]),
            )
            for item in data["elements"]
        )
        return cls(
            name=str(data["name"]),
            elements=elements,
            potential_id=str(data.get("potential_id", "local")),
            notes=str(data.get("notes", "")),
        )


# Demo default matching ex_scripts W–Ta–Cr–V (equiatomic tabGAP mu).
DEFAULT_WTACRV = Family(
    name="W-Ta-Cr-V",
    potential_id="tabgap-eam/fs:W-Ta-Cr-V",
    notes="Equiatomic demo family from ex_scripts; mu from Byggmästar et al. tabGAP fit.",
    elements=(
        ElementSpec("Cr", mass=51.9961, mu_eV=-9.388, fraction=0.25),
        ElementSpec("Ta", mass=180.94788, mu_eV=-11.724, fraction=0.25),
        ElementSpec("V", mass=50.9415, mu_eV=-9.033, fraction=0.25),
        ElementSpec("W", mass=183.84, mu_eV=-13.029, fraction=0.25),
    ),
)
