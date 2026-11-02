import tempfile
import unittest
from pathlib import Path

import numpy as np

from enfr.common import load_config
from enfr.training import TorchTrainer


class ResumeTests(unittest.TestCase):
    def config(self):
        c = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.json")
        c["model"]["dropout"] = .2
        return c

    def rows(self):
        return [{"src_ids": [2, 5, 6, 3], "tgt_ids": [2, 7, 8, 3]},
                {"src_ids": [2, 9, 3], "tgt_ids": [2, 8, 3]}]

    def check(self, cls):
        with tempfile.TemporaryDirectory() as tmp:
            a = cls(self.config(), "cpu")
            a.update(self.rows(), .001)
            a.save(Path(tmp))
            a.update(self.rows(), .0008)
            expected = a.model.canonical()
            b = cls(self.config(), "cpu")
            b.restore(Path(tmp))
            b.update(self.rows(), .0008)
            for key, value in b.model.canonical().items():
                np.testing.assert_allclose(value, expected[key], rtol=1e-6, atol=1e-7, err_msg=key)

    def test_torch_optimizer_and_dropout_rng_resume(self):
        self.check(TorchTrainer)

    def test_tensorflow_optimizer_and_dropout_rng_resume(self):
        try:
            import tensorflow
        except ImportError:
            self.skipTest("TensorFlow tested in its separate environment")
        from enfr.training import TensorFlowTrainer
        self.check(TensorFlowTrainer)


if __name__ == "__main__":
    unittest.main()
