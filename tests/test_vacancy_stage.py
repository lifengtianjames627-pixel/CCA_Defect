"""Stage 3-1 vacancy emit, chain wiring, and golden CSV-shape checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rcca_workflow import DEFAULT_WTACRV
from rcca_workflow.potential import ResolvedPotential
from rcca_workflow.stages.base import StageConfig, StageContext
from rcca_workflow.stages.lammps_cell import parse_lammps_data
from rcca_workflow.stages.vacancy import (
    GOLDEN_CSV_FIELDS,
    VacancyStage,
    assert_vacancy_csv_shape,
    resolve_aged,
)
from rcca_workflow.stages.vacancy_driver import csv_fields, nn_counts, nn_string
from rcca_workflow.stages.lammps_cell import nearest_neighbors

ROOT = Path(__file__).resolve().parents[1]
EX_VAC = ROOT / "ex_scripts" / "Step3_FormE" / "Step3-1_V"
EX_AGED = ROOT / "ex_scripts" / "Step2_MetroMC" / "aged.data"
EX_CSV = EX_VAC / "v_form_e.csv"


def _fake_potential(tmp: Path) -> ResolvedPotential:
    pot_dir = tmp / "pot"
    pot_dir.mkdir()
    tab = pot_dir / "demo.tabgap"
    eam = pot_dir / "demo.eam.fs"
    tab.write_text("tabgap-placeholder\n", encoding="utf-8")
    eam.write_text("eam-placeholder\n", encoding="utf-8")
    return ResolvedPotential(
        pot_dir=pot_dir, tabgap=tab, eam_fs=eam, content_hash="vacbeef"
    )


class TestVacancyEmitAndChain(unittest.TestCase):
    def test_emit_pack_and_chain_aged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage2 = root / "stage2"
            stage3 = root / "stage3"
            stage2.mkdir()
            (stage2 / "aged.data").write_bytes(EX_AGED.read_bytes())

            pot = _fake_potential(root)
            ctx = StageContext(
                family=DEFAULT_WTACRV, potential=pot, prev_dir=stage2, lmp="lmp"
            )
            cfg = StageConfig(
                work_dir=stage3,
                params={"samples_per_element": 2, "site_seed": 1, "nworkers": 1},
            )
            result = VacancyStage().emit(ctx, cfg)
            self.assertTrue((stage3 / "manifest.json").is_file())
            self.assertTrue((stage3 / "vacancy_config.json").is_file())
            self.assertTrue((stage3 / "family.json").is_file())
            self.assertTrue((stage3 / "run_vacancy.py").is_file())
            self.assertTrue((stage3 / "aged.data").is_file())
            self.assertEqual(result.manifest.stage.value, "vacancy")
            self.assertEqual(result.manifest.io.outputs["v_form_e"], "v_form_e.csv")
            self.assertEqual(result.manifest.params["mu_eV"]["W"], -13.029)
            self.assertEqual(resolve_aged(ctx, cfg, stage3).name, "aged.data")

    def test_csv_fields_follow_family(self) -> None:
        self.assertEqual(csv_fields(DEFAULT_WTACRV.symbols), GOLDEN_CSV_FIELDS)


class TestVacancyCellHelpers(unittest.TestCase):
    def test_parse_aged_and_nn(self) -> None:
        self.assertTrue(EX_AGED.is_file())
        cell = parse_lammps_data(EX_AGED, DEFAULT_WTACRV.mass_to_element)
        self.assertEqual(len(cell.atoms), 5488)
        self.assertEqual(len(cell.masses), 4)
        idxs = nearest_neighbors(cell, 0, count=8)
        self.assertEqual(len(idxs), 8)
        counts = nn_counts(cell, idxs, DEFAULT_WTACRV.symbols)
        self.assertEqual(sum(counts.values()), 8)
        text = nn_string(counts, DEFAULT_WTACRV.symbols)
        self.assertIn("Cr:", text)
        self.assertIn("W:", text)


class TestExScriptsVacancyGoldenShape(unittest.TestCase):
    def test_example_csv_matches_contract(self) -> None:
        self.assertTrue(EX_CSV.is_file())
        meta = assert_vacancy_csv_shape(EX_CSV, species=None)
        self.assertGreaterEqual(meta["n_rows"], 1)
        self.assertEqual(meta["fields"], GOLDEN_CSV_FIELDS)


if __name__ == "__main__":
    unittest.main()
