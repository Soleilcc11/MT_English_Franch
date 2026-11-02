import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from enfr.common import ROOT, load_config, load_npz, read_json
from enfr.evaluation import resolve_checkpoint
from enfr.training import train


class PipelineTests(unittest.TestCase):
    def check(self, framework):
        data = ROOT / "cache/smoke"
        if not (data / "audit.json").exists():
            self.skipTest("Run scripts/run.py smoke-data to enable end-to-end tests")
        c = load_config(ROOT / "scripts/configs/smoke.json", "A3" if framework == "tensorflow" else "A0")
        c["training"]["epochs"] = 1
        c["model"]["dropout"] = .2
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            interrupted, whole = root / "resumed", root / "whole"
            train(c, data, interrupted, framework, "cpu", max_updates=2)
            self.assertEqual(read_json(interrupted / "status.json")["status"], "paused_at_update_limit")
            train(c, data, interrupted, framework, "cpu", resume=interrupted / "latest.json")
            train(c, data, whole, framework, "cpu")
            a = load_npz(resolve_checkpoint(interrupted / "latest.json") / "weights.npz")
            b = load_npz(resolve_checkpoint(whole / "latest.json") / "weights.npz")
            for name in a:
                np.testing.assert_allclose(a[name], b[name], atol=1e-7, rtol=1e-6, err_msg=name)
            status = read_json(interrupted / "status.json")
            self.assertEqual(status["status"], "complete")
            self.assertFalse(status["formal_training_complete"])
            meta = read_json(resolve_checkpoint(interrupted) / "metadata.json")
            self.assertEqual(meta["state"]["epoch"], 1)
            self.assertTrue((interrupted / "validation/epoch-01/predictions.jsonl").exists())

    def test_torch_full_pipeline_resume(self):
        self.check("torch")

    def test_tensorflow_full_pipeline_resume(self):
        try:
            import tensorflow
        except ImportError:
            self.skipTest("TensorFlow is in its own environment")
        self.check("tensorflow")


if __name__ == "__main__":
    unittest.main()
