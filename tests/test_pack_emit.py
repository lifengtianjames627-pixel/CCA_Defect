"""Job-pack emitter used by the Streamlit GUI."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rcca_workflow import DEFAULT_WTACRV
from rcca_workflow.pack import PackRequest, emit_pack, zip_pack
from rcca_workflow.potential import ResolvedPotential


def _fake_potential(tmp: Path) -> ResolvedPotential:
    pot_dir = tmp / "pot"
    pot_dir.mkdir()
    tab = pot_dir / "demo.tabgap"
    eam = pot_dir / "demo.eam.fs"
    tab.write_text("x\n", encoding="utf-8")
    eam.write_text("y\n", encoding="utf-8")
    return ResolvedPotential(
        pot_dir=pot_dir, tabgap=tab, eam_fs=eam, content_hash="packbeef"
    )


class TestPackEmit(unittest.TestCase):
    def test_emit_chained_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pot = _fake_potential(root)
            pack_dir = root / "demo_pack"
            result = emit_pack(
                PackRequest(
                    family=DEFAULT_WTACRV,
                    pack_dir=pack_dir,
                    potential=pot,
                    include_sqs=True,
                    include_metropolis=True,
                    include_vacancy=True,
                    include_sia=True,
                    sqs_params={
                        "supercell_rep": [2, 2, 2],
                        "target_total_atoms": 16,
                        "n_mc_steps": 100,
                        "plateau_idle_sec": 0,
                    },
                    metro_params={"expected_natoms": 16, "nprocs": 4},
                    vacancy_params={"samples_per_element": 1, "expected_natoms": 16},
                    sia_params={
                        "anneal_k": 1000,
                        "inserts_per_element": 1,
                        "expected_natoms": 16,
                    },
                )
            )
            self.assertTrue((pack_dir / "README.md").is_file())
            self.assertTrue((pack_dir / "family.json").is_file())
            self.assertTrue((pack_dir / "01_sqs" / "manifest.json").is_file())
            self.assertTrue((pack_dir / "02_metropolis" / "manifest.json").is_file())
            self.assertTrue((pack_dir / "03_vacancy" / "manifest.json").is_file())
            self.assertTrue((pack_dir / "03_sia" / "manifest.json").is_file())
            self.assertTrue((pack_dir / "01_sqs" / "mp-91_W.cif").is_file())
            z = zip_pack(result.pack_dir, root / "demo_pack.zip")
            self.assertTrue(z.is_file())
            self.assertGreater(z.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
