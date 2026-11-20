"""Stage 2 Metropolis emit, chain wiring, and golden output-shape checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rcca_workflow import DEFAULT_WTACRV
from rcca_workflow.potential import ResolvedPotential
from rcca_workflow.stages.base import StageConfig, StageContext
from rcca_workflow.stages.metropolis import (
    MetropolisStage,
    assert_lammps_data_shape,
    assert_metro_output_shape,
    resolve_poscar,
)
from rcca_workflow.stages.metropolis_driver import (
    FORBIDDEN_NPROCS,
    read_poscar,
    write_lammps_data,
)

ROOT = Path(__file__).resolve().parents[1]
EX_METRO = ROOT / "ex_scripts" / "Step2_MetroMC"
EX_POSCAR = EX_METRO / "poscar_out.vasp"


def _fake_potential(tmp: Path) -> ResolvedPotential:
    pot_dir = tmp / "pot"
    pot_dir.mkdir()
    tab = pot_dir / "demo.tabgap"
    eam = pot_dir / "demo.eam.fs"
    tab.write_text("tabgap-placeholder\n", encoding="utf-8")
    eam.write_text("eam-placeholder\n", encoding="utf-8")
    return ResolvedPotential(
        pot_dir=pot_dir,
        tabgap=tab,
        eam_fs=eam,
        content_hash="deadbeef",
    )


class TestMetropolisEmitAndChain(unittest.TestCase):
    def test_emit_pack_and_chain_poscar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage1 = root / "stage1"
            stage2 = root / "stage2"
            stage1.mkdir()
            # Stage-1 style artifact name preferred by chain helper.
            shutil_poscar = stage1 / "poscar_out"
            shutil_poscar.write_text(EX_POSCAR.read_text(encoding="utf-8"), encoding="utf-8")

            pot = _fake_potential(root)
            ctx = StageContext(
                family=DEFAULT_WTACRV,
                potential=pot,
                prev_dir=stage1,
                lmp="lmp",
                mpirun="mpirun",
            )
            cfg = StageConfig(
                work_dir=stage2,
                params={"expected_natoms": 5488, "nprocs": 4, "seed": 99},
            )
            result = MetropolisStage().emit(ctx, cfg)
            self.assertTrue((stage2 / "manifest.json").is_file())
            self.assertTrue((stage2 / "metro_config.json").is_file())
            self.assertTrue((stage2 / "family.json").is_file())
            self.assertTrue((stage2 / "run_metropolis.py").is_file())
            self.assertTrue((stage2 / "poscar_in").is_file())
            self.assertEqual(result.manifest.stage.value, "metropolis")
            self.assertEqual(result.manifest.io.outputs["aged"], "aged.data")
            self.assertEqual(result.manifest.potential_hash, "deadbeef")

            resolved = resolve_poscar(ctx, cfg, stage2)
            self.assertTrue(resolved.is_file())

    def test_forbidden_nprocs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pot = _fake_potential(root)
            ctx = StageContext(family=DEFAULT_WTACRV, potential=pot)
            cfg = StageConfig(work_dir=root / "s2", params={"nprocs": 16})
            with self.assertRaises(ValueError):
                MetropolisStage().emit(ctx, cfg)
            self.assertIn(16, FORBIDDEN_NPROCS)

    def test_poscar_to_start_data(self) -> None:
        self.assertTrue(EX_POSCAR.is_file())
        cell, positions = read_poscar(EX_POSCAR, DEFAULT_WTACRV.masses)
        self.assertEqual(len(positions), 5488)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "start.data"
            elements = write_lammps_data(
                out, cell, positions, DEFAULT_WTACRV.masses, title="test"
            )
            self.assertEqual(len(elements), 4)
            meta = assert_lammps_data_shape(out)
            self.assertEqual(meta["n_atoms"], 5488)
            self.assertEqual(meta["n_types"], 4)


class TestExScriptsMetroGoldenShape(unittest.TestCase):
    def test_example_outputs_match_contract(self) -> None:
        self.assertTrue(EX_METRO.is_dir())
        paths = assert_metro_output_shape(EX_METRO, require_plot=True)
        aged = assert_lammps_data_shape(paths["aged"])
        relaxed = assert_lammps_data_shape(paths["relaxed"])
        self.assertEqual(aged["n_atoms"], 5488)
        self.assertEqual(relaxed["n_atoms"], 5488)
        self.assertEqual(aged["n_types"], 4)


if __name__ == "__main__":
    unittest.main()
