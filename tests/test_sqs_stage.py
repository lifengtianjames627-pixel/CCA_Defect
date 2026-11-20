"""Stage 1 SQS emit + golden output-shape checks (no full 5488 MC required)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rcca_workflow import DEFAULT_WTACRV
from rcca_workflow.stages.base import StageConfig, StageContext
from rcca_workflow.stages.sqs import (
    GOLDEN_COST_KEYS,
    GOLDEN_DEVIATIONS_HEADER,
    SqsStage,
    assert_sqs_output_shape,
    build_sqs_params,
)

ROOT = Path(__file__).resolve().parents[1]
EX_SQS = ROOT / "ex_scripts" / "Step1_SQS"


class TestSqsEmit(unittest.TestCase):
    def test_emit_writes_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            ctx = StageContext(family=DEFAULT_WTACRV, potential=None)
            cfg = StageConfig(
                work_dir=work,
                params={
                    "supercell_rep": [2, 2, 2],
                    "target_total_atoms": 16,
                    "n_mc_steps": 100,
                    "seed": 7,
                    "plateau_idle_sec": 0,
                },
            )
            result = SqsStage().emit(ctx, cfg)
            self.assertTrue((work / "manifest.json").is_file())
            self.assertTrue((work / "sqs_config.json").is_file())
            self.assertTrue((work / "family.json").is_file())
            self.assertTrue((work / "mp-91_W.cif").is_file())
            self.assertTrue((work / "run_sqs.py").is_file())
            self.assertEqual(result.manifest.stage.value, "sqs")
            self.assertEqual(result.manifest.io.outputs["poscar_out"], "poscar_out")
            params = build_sqs_params(DEFAULT_WTACRV, cfg)
            self.assertEqual(params["target_total_atoms"], 16)
            self.assertEqual(params["species"], ["Cr", "Ta", "V", "W"])


class TestExScriptsGoldenShape(unittest.TestCase):
    def test_example_outputs_match_contract(self) -> None:
        self.assertTrue(EX_SQS.is_dir(), f"missing {EX_SQS}")
        paths = assert_sqs_output_shape(EX_SQS, require_poscar=True)
        cost = paths["cost"].read_text(encoding="utf-8")
        for key in GOLDEN_COST_KEYS:
            self.assertIn(key, cost)
        header = paths["deviations"].read_text(encoding="utf-8").splitlines()[0].strip()
        self.assertEqual(header, GOLDEN_DEVIATIONS_HEADER)
        # Example used .vasp suffix; assert_sqs_output_shape accepts either.
        self.assertTrue(paths["poscar"].is_file())


if __name__ == "__main__":
    unittest.main()
