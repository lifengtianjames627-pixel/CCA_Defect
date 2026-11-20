"""Stage 3-2 SIA: emit/chain, element-pair only (no geometry fields)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from rcca_workflow import DEFAULT_WTACRV
from rcca_workflow.potential import ResolvedPotential
from rcca_workflow.stages.base import StageConfig, StageContext
from rcca_workflow.stages.lammps_cell import parse_lammps_data
from rcca_workflow.stages.sia import SiaStage, assert_sia_csv_shape
from rcca_workflow.stages.sia_driver import SIA_FIELDS, EXCLUDED_GEOMETRY_FIELDS
from rcca_workflow.stages.sia_pair import label_element_pair, pair_key, summarize_pairs

ROOT = Path(__file__).resolve().parents[1]
EX_SIA = ROOT / "ex_scripts" / "Step3_FormE" / "Step3-2_SIA"
EX_BULK = ROOT / "ex_scripts" / "Step3_FormE" / "bulk" / "relaxed.data"
EX_AGED = ROOT / "ex_scripts" / "Step2_MetroMC" / "aged.data"
EX_CASE = EX_SIA / "cases_1000K" / "sia_Cr_356" / "relaxed.data"
EX_ENERGY = EX_SIA / "sia_formation_energies_1000K.csv"


def _fake_potential(tmp: Path) -> ResolvedPotential:
    pot_dir = tmp / "pot"
    pot_dir.mkdir()
    tab = pot_dir / "demo.tabgap"
    eam = pot_dir / "demo.eam.fs"
    tab.write_text("tabgap-placeholder\n", encoding="utf-8")
    eam.write_text("eam-placeholder\n", encoding="utf-8")
    return ResolvedPotential(
        pot_dir=pot_dir, tabgap=tab, eam_fs=eam, content_hash="siabeef"
    )


class TestSiaEmitAndSchema(unittest.TestCase):
    def test_emit_pack_and_chain_aged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage2 = root / "stage2"
            stage3 = root / "stage3"
            stage2.mkdir()
            (stage2 / "aged.data").write_bytes(EX_AGED.read_bytes())
            pot = _fake_potential(root)
            ctx = StageContext(family=DEFAULT_WTACRV, potential=pot, prev_dir=stage2)
            cfg = StageConfig(
                work_dir=stage3,
                params={"anneal_k": 1000, "inserts_per_element": 1, "site_seed": 3},
            )
            result = SiaStage().emit(ctx, cfg)
            self.assertTrue((stage3 / "manifest.json").is_file())
            self.assertTrue((stage3 / "sia_config.json").is_file())
            self.assertTrue((stage3 / "run_sia.py").is_file())
            self.assertTrue((stage3 / "aged.data").is_file())
            self.assertEqual(result.manifest.stage.value, "sia")
            self.assertEqual(
                result.manifest.io.outputs["sia_formation_energies"],
                "sia_formation_energies_1000K.csv",
            )
            self.assertEqual(
                result.manifest.io.outputs["sia_pairs"], "sia_pairs_1000K.csv"
            )
            self.assertIn("geometry", result.manifest.params["excluded_geometry"])

    def test_schema_excludes_geometry(self) -> None:
        self.assertNotIn("geometry", SIA_FIELDS)
        self.assertNotIn("orientation", SIA_FIELDS)
        self.assertIn("pair", SIA_FIELDS)
        self.assertIn("element_a", SIA_FIELDS)
        self.assertIn("class_key", SIA_FIELDS)
        self.assertIn("geometry", EXCLUDED_GEOMETRY_FIELDS)
        self.assertIn("orientation", EXCLUDED_GEOMETRY_FIELDS)


class TestElementPairOnly(unittest.TestCase):
    def test_pair_key_sorted(self) -> None:
        self.assertEqual(pair_key("V", "Cr"), "Cr-V")
        self.assertEqual(pair_key("W", "W"), "W-W")

    def test_label_real_annealed_case(self) -> None:
        self.assertTrue(EX_BULK.is_file(), f"missing {EX_BULK}")
        self.assertTrue(EX_CASE.is_file(), f"missing {EX_CASE}")
        m2e = DEFAULT_WTACRV.mass_to_element
        bulk = parse_lammps_data(EX_BULK, m2e)
        sia = parse_lammps_data(EX_CASE, m2e)
        self.assertEqual(len(bulk.atoms) + 1, len(sia.atoms))
        label = label_element_pair(bulk, sia)
        self.assertIn(label["pair"], {"Cr-Cr", "Cr-Ta", "Cr-V", "Cr-W", "Ta-Ta", "Ta-V", "Ta-W", "V-V", "V-W", "W-W"})
        self.assertEqual(label["pair"], pair_key(label["element_a"], label["element_b"]))
        self.assertNotIn("geometry", label)
        self.assertNotIn("orientation", label)
        self.assertTrue(label["note"])

    def test_summarize_pairs_no_geometry(self) -> None:
        rows = [
            {"status": "ok", "pair": "Cr-V", "formation_energy_SIA_eV": "1.0"},
            {"status": "ok", "pair": "Cr-V", "formation_energy_SIA_eV": "2.0"},
            {"status": "ok", "pair": "V-V", "formation_energy_SIA_eV": "1.5"},
            {"status": "failed", "pair": "Cr-Cr", "formation_energy_SIA_eV": "9.0"},
        ]
        summary = summarize_pairs(rows)
        self.assertEqual([r["pair"] for r in summary], ["Cr-V", "V-V"])
        self.assertEqual(summary[0]["count"], 2)
        self.assertEqual(summary[0]["class_key"], "Cr-V")
        for row in summary:
            self.assertNotIn("geometry", row)
            self.assertNotIn("orientation", row)

    def test_template_csv_has_geometry_we_drop(self) -> None:
        """Document that ex_scripts still carries geometry; our schema does not."""
        self.assertTrue(EX_ENERGY.is_file())
        with EX_ENERGY.open(newline="", encoding="utf-8") as handle:
            fields = list(csv.DictReader(handle).fieldnames or [])
        self.assertIn("geometry", fields)
        self.assertIn("orientation", fields)
        # Our ported schema must not match the geometry-laden template header.
        self.assertNotEqual(fields, SIA_FIELDS)


class TestSiaCsvRoundtrip(unittest.TestCase):
    def test_assert_accepts_chemistry_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sia_formation_energies_1000K.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=SIA_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "job": "Cr:1",
                        "anneal_K": "1000.0",
                        "site_id": "1",
                        "host_element": "W",
                        "inserted_element": "Cr",
                        "insert_x": "0",
                        "insert_y": "0",
                        "insert_z": "0",
                        "E_bulk_eV": "0",
                        "E_SIA_eV": "0",
                        "mu_eV": "-9.388",
                        "formation_energy_SIA_eV": "1.0",
                        "pair": "Cr-V",
                        "element_a": "Cr",
                        "element_b": "V",
                        "class_key": "Cr-V",
                        "status": "ok",
                        "note": "WS pair test",
                    }
                )
            meta = assert_sia_csv_shape(path)
            self.assertEqual(meta["n_rows"], 1)


if __name__ == "__main__":
    unittest.main()
