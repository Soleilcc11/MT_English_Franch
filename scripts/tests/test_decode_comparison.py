import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sentencepiece as spm
import torch

from enfr.common import ROOT, ModelConfig, load_config, read_json, save_npz, sha256, write_json
from enfr.data import build_prepared
from enfr.decode_comparison import (compare_decode, directory_lock, protect_output,
                                    select_beam)
from enfr.evaluation import evaluate_model, generation_protocol
from enfr.reporting import check_frozen
from enfr.torch_model import Transformer
from run import parser


class DecodeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.data, self.old, self.out = root / "prepared", root / "old", root / "new"
        raw = {}
        for split, start, count in (("train", 0, 32), ("validation", 32, 4), ("test", 36, 4)):
            raw[split] = [{"id": f"sample-{i}", "year": 2010, "doc_id": "toy",
                           "src_raw": f"The red bird sings number {i}.",
                           "tgt_raw": f"L'oiseau rouge chante le nombre {i}."}
                          for i in range(start, start + count)]
        audit = build_prepared(raw, self.data, vocab_size=128, provenance={"synthetic_fixture": True})
        self.audit = audit
        config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.json", "A0", 42)
        config["evaluation"].update(max_new_tokens=3, batch_size=2)
        config["model"].update(d_model=16, d_ff=32)
        self.run = self.old / "runs/A0-torch-seed42"
        self.cp = self.run / "checkpoints/step-00000001"
        self.cp.mkdir(parents=True)
        model = Transformer(ModelConfig(**config["model"])).eval()
        with torch.no_grad():
            model.output_bias[3] = 5.0
        save_npz(self.cp / "weights.npz", model.canonical())
        meta = {"framework": "torch", "config": config, "data_id": audit["data_id"],
                "tokenizer_sha256": audit["tokenizer_sha256"], "code_id": "original-training-code"}
        write_json(self.cp / "metadata.json", meta)
        for name in ("best.json", "latest.json"):
            write_json(self.run / name, {"checkpoint": "checkpoints/step-00000001"})
        write_json(self.run / "status.json", {"status": "complete", "epochs": 1})
        write_json(self.run / "config.json", config)
        self.freeze = self.old / "freeze.json"
        write_json(self.freeze, {"data_id": audit["data_id"], "tokenizer_sha256": audit["tokenizer_sha256"],
            "runs": [{"weights_sha256": sha256(self.cp / "weights.npz"), "config": config,
                      "code_id": meta["code_id"]}]})
        from enfr.common import read_jsonl
        sp = spm.SentencePieceProcessor(model_file=str(self.data / "tokenizer.model"))
        evaluate_model(model, "torch", read_jsonl(self.data / "test.jsonl"), sp,
                       **config["evaluation"], output=self.run / "test")

    def compare(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return compare_decode(self.run, self.data, self.freeze, self.out,
                                  device="cpu", precision="fp32", smoke=True, **kwargs)

    def old_hashes(self):
        return {str(p.relative_to(self.old)): sha256(p) for p in self.old.rglob("*") if p.is_file()}

    def test_full_comparison_and_completed_restart_preserve_old_results(self):
        before = self.old_hashes()
        self.assertEqual(self.compare()["status"], "complete")
        self.assertEqual(self.old_hashes(), before)
        summary = read_json(self.out / "report/comparison.json")
        self.assertFalse(summary["formal"])
        self.assertEqual(summary["metrics"]["greedy"]["nll"], summary["metrics"]["beam"]["nll"])
        self.assertTrue(summary["historical_greedy"]["available"])
        self.assertEqual(len(summary["validation_candidates"]), 7)
        state = read_json(self.out / "status.json")
        self.assertEqual(len(state["completed"]), 14)
        self.assertTrue(state["old_results_unchanged"])
        times = {str(p): p.stat().st_mtime_ns for p in self.out.rglob("*") if p.is_file() and p.name != ".compare.lock"}
        with patch("enfr.decode_comparison.load_model", side_effect=AssertionError("Should skip completed comparison")):
            self.assertTrue(self.compare()["resumed"])
        self.assertEqual(times, {str(p): p.stat().st_mtime_ns for p in self.out.rglob("*") if p.is_file() and p.name != ".compare.lock"})

    def test_interrupted_stage_restarts_but_completed_validation_is_not_overwritten(self):
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected interruption")
            return evaluate_model(*args, **kwargs)
        with patch("enfr.decode_comparison.evaluate_model", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.compare()
        state = read_json(self.out / "status.json")
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(list(state["completed"]), ["validation/greedy"])
        first = self.out / "validation/greedy/metrics.json"
        timestamp = first.stat().st_mtime_ns
        self.assertEqual(self.compare()["status"], "complete")
        self.assertEqual(timestamp, first.stat().st_mtime_ns)

    def test_freeze_rejects_parameter_weights_metadata_and_data_changes(self):
        self.compare()
        frozen_path = self.out / "decode-freeze.json"
        protocol = read_json(frozen_path)["systems"]["beam"]
        check_frozen(self.cp, self.audit["data_id"], frozen_path, protocol)
        for altered in ({**protocol, "length_penalty": 7.0}, {**protocol, "cache": False},
                        {**protocol, "precision": "bf16"}):
            with self.assertRaises(ValueError):
                check_frozen(self.cp, self.audit["data_id"], frozen_path, altered)
        with self.assertRaises(ValueError):
            check_frozen(self.cp, "wrong-data", frozen_path, protocol)
        for filename in ("metadata.json", "weights.npz"):
            path = self.cp / filename
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.assertRaises(ValueError):
                check_frozen(self.cp, self.audit["data_id"], frozen_path, protocol)
            path.write_bytes(original)
        with patch("enfr.decode_comparison.source_manifest", return_value={"code_id": "changed"}):
            with self.assertRaises(ValueError):
                check_frozen(self.cp, self.audit["data_id"], frozen_path, protocol)

    def test_completed_artifact_tampering_is_not_overwritten(self):
        self.compare()
        path = self.out / "test/beam/metrics.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.compare()

    def test_dry_run_writes_nothing(self):
        result = self.compare(dry_run=True)
        self.assertEqual(result["training_runs"], 0)
        self.assertEqual(len(result["stages"]), 14)
        self.assertFalse(self.out.exists())

    def test_cli_comparison_evaluation_and_translation(self):
        base = [sys.executable, str(ROOT / "scripts/run.py")]
        compare = base + ["compare-decode", "--checkpoint", str(self.run), "--prepared", str(self.data),
                          "--freeze", str(self.freeze), "--output", str(self.out),
                          "--device", "cpu", "--precision", "fp32", "--smoke"]
        dry = subprocess.run(compare + ["--dry-run"], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(dry.stdout)["training_runs"], 0)
        self.assertFalse(self.out.exists())
        subprocess.run(compare, capture_output=True, text=True, check=True)
        settings = read_json(self.out / "decode-freeze.json")["systems"]["beam"]
        options = ["--checkpoint", str(self.run), "--prepared", str(self.data), "--device", "cpu",
                   "--decode", "beam", "--beam-size", str(settings["beam_size"]),
                   "--length-penalty", str(settings["length_penalty"]), "--cache"]
        manual = self.out / "manual"
        subprocess.run(base + ["evaluate", *options, "--split", "test", "--freeze",
                              str(self.out / "decode-freeze.json"), "--output", str(manual)],
                       capture_output=True, text=True, check=True)
        self.assertFalse(read_json(manual / "metrics.json")["formal"])
        translated = subprocess.run(base + ["translate", *options, "--text", "The red bird."],
                                    capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(translated.stdout)["decode"], "beam")

    def test_original_freeze_is_greedy_only_and_new_output_cannot_overlap(self):
        check_frozen(self.cp, self.audit["data_id"], self.freeze)
        with self.assertRaisesRegex(ValueError, "uncached greedy"):
            check_frozen(self.cp, self.audit["data_id"], self.freeze, generation_protocol("beam"))
        for path in (self.old, self.run / "test", self.old.parent):
            with self.assertRaises(ValueError):
                protect_output(path, (self.old, self.data))
        self.out.mkdir()
        (self.out / "alias").symlink_to(self.old, target_is_directory=True)
        with self.assertRaises(ValueError):
            protect_output(self.out, (self.old, self.data))

    def test_exclusive_directory_lock(self):
        with directory_lock(self.out):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with directory_lock(self.out):
                    self.fail("Acquired a competing lock")
        with directory_lock(self.out):
            pass

    def test_validation_selection_ties_and_cli_defaults(self):
        def row(k, alpha, bleu=30, chrf=60):
            return {"settings": {"decode": "beam", "beam_size": k, "length_penalty": alpha},
                    "metrics": {"bleu": bleu, "chrf2": chrf}}
        candidates = [row(8, 0), row(4, 1), row(4, .6), row(4, 0, bleu=29)]
        self.assertEqual(select_beam(candidates), candidates[2])
        candidates.append(row(8, 1, chrf=61))
        self.assertEqual(select_beam(candidates), candidates[-1])
        p = parser()
        self.assertFalse(p.parse_args(["evaluate", "--checkpoint", "x", "--output", "y"]).cache)
        self.assertTrue(p.parse_args(["translate", "--checkpoint", "x", "--text", "hi"]).cache)
        self.assertFalse(p.parse_args(["translate", "--checkpoint", "x", "--text", "hi", "--no-cache"]).cache)


if __name__ == "__main__":
    unittest.main()
