import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from enfr.common import ROOT, load_config, read_json, sha256, source_manifest, write_json
from enfr.data import build_prepared
from enfr.evaluation import resolve_checkpoint
from enfr.extensions import run_extensions, select_alpha
from enfr.reporting import check_frozen
from enfr.training import train
from run import parser


class ExtensionQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prepared, self.old, self.out = (self.root / n for n in ("prepared", "old", "new"))
        raw = {}
        for split, start, count in (("train", 0, 32), ("validation", 32, 4), ("test", 36, 4)):
            raw[split] = [{"id": f"sample-{i}", "year": 2010, "doc_id": "toy",
                           "src_raw": f"The red bird sings number {i}.",
                           "tgt_raw": f"L'oiseau rouge chante le nombre {i}."}
                          for i in range(start, start + count)]
        self.audit = build_prepared(raw, self.prepared, vocab_size=128, provenance={"synthetic_fixture": True})
        cfg = load_config(ROOT / "scripts/configs/smoke.json")
        cfg["model"].update(d_model=16, d_ff=32, dropout=.1)
        cfg["training"].update(epochs=1, effective_batch=16, micro_batch=8)
        cfg["evaluation"].update(max_new_tokens=3, batch_size=2)
        self.config = self.root / "config.json"
        write_json(self.config, cfg)
        entries = []
        with contextlib.redirect_stdout(io.StringIO()):
            for seed in (42, 43, 44):
                c = load_config(self.config, "A0", seed)
                run = self.old / "runs" / f"A0-torch-seed{seed}"
                train(c, self.prepared, run)
                cp = resolve_checkpoint(run)
                meta = read_json(cp / "metadata.json")
                meta["code_id"] = "legacy-source"
                write_json(cp / "metadata.json", meta)
                entries.append({"weights_sha256": sha256(cp / "weights.npz"), "config": c, "code_id": "legacy-source"})
        self.freeze = self.old / "freeze.json"
        write_json(self.freeze, {"data_id": self.audit["data_id"], "tokenizer_sha256": self.audit["tokenizer_sha256"],
                                "runs": entries})

    def queue(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_extensions(self.old / "runs", self.freeze, self.prepared, self.out,
                                  self.config, "cpu", sys.executable, smoke=True, **kwargs)

    def snapshot(self, root):
        return {str(p): (sha256(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
                and p.name != ".compare.lock"}

    def test_dry_run_cli_and_original_output_guard(self):
        before = self.snapshot(self.old)
        result = self.queue(dry_run=True, finalize=True)
        self.assertEqual(result["new_training_runs"], 8)
        self.assertEqual([r["alpha"] for r in result["plan"][3:6]], [1, 3, 5])
        self.assertEqual(result["plan"][-1]["alpha"], "validation-selected")
        self.assertFalse(self.out.exists())
        cmd = [sys.executable, str(ROOT / "scripts/run.py"), "run-extensions",
               "--baseline-runs", str(self.old / "runs"), "--baseline-freeze", str(self.freeze),
               "--prepared", str(self.prepared), "--output", str(self.out), "--config", str(self.config),
               "--device", "cpu", "--smoke", "--dry-run", "--finalize"]
        p = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(p.stdout)["new_training_runs"], 8)
        self.assertEqual(before, self.snapshot(self.old))
        self.out = self.old / "new"
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.queue(dry_run=True)

    def test_full_eight_run_queue_then_finalize_resume_and_freeze_guards(self):
        before = self.snapshot(self.old)
        # Real training subprocesses exercise CLI config/alpha propagation and
        # release each trainer before starting the next GPU run on the server.
        with (self.root / "training.log").open("w") as log:
            real_run = subprocess.run
            def quiet(command, **kw):
                return real_run(command, **kw, stdout=log, stderr=subprocess.STDOUT)
            with patch("enfr.extensions.subprocess.run", side_effect=quiet):
                result = self.queue()
        self.assertEqual(result["status"], "training_complete")
        self.assertEqual(len(list((self.out / "runs").iterdir())), 8)
        self.assertFalse((self.out / "extension-freeze.json").exists())
        self.assertFalse((self.out / "test").exists())
        selection = read_json(self.out / "a5-selection.json")
        self.assertFalse(selection["uses_test"])
        self.assertEqual(selection["alpha"], select_alpha(selection["candidates"])["alpha"])
        for p in (self.out / "runs").iterdir():
            c = read_json(p / "config.json")
            if c["variant"] == "A5" and c["seed"] in (43, 44):
                self.assertEqual(c["training"]["rdrop_alpha"], selection["alpha"])
        # Failure before first test output; completed training must not repeat.
        with patch("enfr.extensions.subprocess.run", side_effect=AssertionError("must skip training")), \
             patch("enfr.extensions.evaluate_model", side_effect=RuntimeError("injected evaluation failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.queue(finalize=True)
        self.assertEqual(read_json(self.out / "status.json")["status"], "interrupted")
        with patch("enfr.extensions.subprocess.run", side_effect=AssertionError("must skip training")):
            self.assertEqual(self.queue(finalize=True)["status"], "complete")
        self.assertEqual(before, self.snapshot(self.old))
        summary = read_json(self.out / "report/summary.json")
        self.assertFalse(summary["formal"])
        self.assertEqual(len(summary["experiments"]), 9)
        self.assertEqual(set(summary["aggregate"]), {"A0", "A4", "A5"})
        self.assertEqual(len(list((self.out / "test").iterdir())), 9)
        self.assertTrue((self.out / "report/bootstrap-A0-A4.json").is_file())
        frozen_path = self.out / "extension-freeze.json"
        frozen = read_json(frozen_path)
        cp = Path(frozen["runs"][-1]["checkpoint"])
        protocol = frozen["protocol"]
        check_frozen(cp, self.audit["data_id"], frozen_path, protocol)
        for altered in ({**protocol, "cache": True}, {**protocol, "decode": "beam"}, {**protocol, "precision": "bf16"}):
            with self.assertRaises(ValueError):
                check_frozen(cp, self.audit["data_id"], frozen_path, altered)
        with self.assertRaises(ValueError):
            check_frozen(cp, "changed-data", frozen_path, protocol)
        with patch("enfr.extensions.source_manifest", return_value={"code_id": "changed"}):
            with self.assertRaises(ValueError):
                check_frozen(cp, self.audit["data_id"], frozen_path, protocol)
        before_complete = self.snapshot(self.out)
        with patch("enfr.extensions.load_model", side_effect=AssertionError("must skip evaluation")), \
             patch("enfr.extensions.subprocess.run", side_effect=AssertionError("must skip training")):
            self.assertTrue(self.queue(finalize=True)["resumed"])
        self.assertEqual(before_complete, self.snapshot(self.out))
        for filename in ("metadata.json", "weights.npz"):
            path = cp / filename
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.assertRaises(ValueError):
                check_frozen(cp, self.audit["data_id"], frozen_path, protocol)
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                self.queue(finalize=True)
            path.write_bytes(original)
        changed = self.out / "report/experiments.csv"
        changed.write_text(changed.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.queue(finalize=True)

    def test_selection_does_not_use_test_and_cli_accepts_new_variants(self):
        c = [{"alpha": a, "validation_bleu": 20, "validation_nll": 2, "test_bleu": a * 10}
             for a in (5, 3, 1)]
        self.assertEqual(select_alpha(c)["alpha"], 1)
        c[0]["validation_nll"] = 1
        self.assertEqual(select_alpha(c)["alpha"], 5)
        c[1]["validation_bleu"] = 21
        self.assertEqual(select_alpha(c)["alpha"], 3)
        args = parser().parse_args(["train", "--variant", "A5", "--rdrop-alpha", "3"])
        self.assertEqual(args.rdrop_alpha, 3)
        args = parser().parse_args(["profile", "--variant", "A4", "--output", "tmp.json"])
        self.assertEqual(args.variant, "A4")

    def test_baseline_config_change_and_checkpoint_tampering_rejected(self):
        cp = resolve_checkpoint(self.old / "runs/A0-torch-seed42")
        path = cp / "weights.npz"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaises(ValueError):
            self.queue(dry_run=True)
        path.write_bytes(original)
        cfg = read_json(self.config)
        cfg["training"]["lr"] *= 2
        write_json(self.config, cfg)
        with self.assertRaisesRegex(ValueError, "config differs"):
            self.queue(dry_run=True)


if __name__ == "__main__":
    unittest.main()
