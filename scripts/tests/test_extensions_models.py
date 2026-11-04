import contextlib
import copy
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from enfr.common import ROOT, ModelConfig, canonical_weights, load_config, load_npz, read_json
from enfr.data import collate
from enfr.evaluation import resolve_checkpoint
from enfr.torch_model import Attention, Transformer, loss_sums, rdrop_loss_sums
from enfr.training import TorchTrainer, TensorFlowTrainer, train

torch.set_num_threads(1)


class RelativeAttentionTests(unittest.TestCase):
    def setUp(self):
        self.base = ModelConfig(vocab_size=32, d_model=16, heads=4, d_ff=32,
                                encoder_layers=2, decoder_layers=2, dropout=0)
        self.c = replace(self.base, encoder_positions=False, encoder_relative_positions=True, relative_max_distance=2)
        self.src = torch.tensor([[2, 4, 5, 3, 0], [2, 6, 3, 0, 0]])
        self.tgt = torch.tensor([[2, 7, 8], [2, 9, 0]])

    def test_common_initial_weights_and_parameter_counts(self):
        a, b = canonical_weights(self.base, 43), canonical_weights(self.c, 43)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(set(b) - set(a), {f"encoder.{i}.self_attn.relative_{kind}"
                                         for i in range(2) for kind in ("k", "v")})
        self.assertEqual(sum(x.size for x in b.values()) - sum(x.size for x in a.values()), 2 * 2 * 5 * 4)
        actual = load_config(variant="A4")
        self.assertFalse(actual["model"]["encoder_positions"])
        self.assertTrue(actual["model"]["encoder_relative_positions"])
        self.assertEqual(actual["model"]["relative_max_distance"], 16)

    def test_shaw_formula_against_explicit_reference_and_gradients(self):
        model = Transformer(self.c)
        attn = model.encoder[0].self_attn
        x = torch.randn(2, 5, 16)
        allowed = (self.src != 0)[:, None, None, :]
        actual = attn(x, x, allowed)
        q, (k, v) = attn.split(attn.q(x)), attn.kv(x)
        # Explicit [B,H,I,J,D] reference, used only for this small numerical test.
        delta = torch.tensor([[max(-2, min(2, j-i)) + 2 for j in range(5)] for i in range(5)])
        keys = k[:, :, None] + attn.relative_k[delta][None, None]
        values = v[:, :, None] + attn.relative_v[delta][None, None]
        scores = (q[:, :, :, None] * keys).sum(-1) / 2.0
        probs = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        expected = (probs[..., None] * values).sum(-2).transpose(1, 2).reshape(2, 5, 16)
        expected = attn.out(expected)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        actual.square().sum().backward()
        for table in (attn.relative_k, attn.relative_v):
            self.assertTrue(torch.isfinite(table.grad).all())
            self.assertGreater(float(table.grad.abs().sum()), 0)

    def test_padding_causality_permutation_and_mask(self):
        m = Transformer(self.c).eval()
        m.set_capture(True)
        logits = m(self.src, self.tgt)
        torch.testing.assert_close(logits, m(F.pad(self.src, (0, 3)), F.pad(self.tgt, (0, 2)))[:, :3],
                                   atol=1e-5, rtol=1e-4)
        p = m.encoder[0].self_attn.last["probs"]
        self.assertTrue((p[1, :, :, 3:] == 0).all())
        torch.testing.assert_close(p.sum(-1), torch.ones_like(p.sum(-1)))
        future = self.tgt.clone()
        future[:, 2] = 15
        torch.testing.assert_close(logits[:, :2], m(self.src, future)[:, :2])
        permuted = self.src[:, [0, 2, 1, 3, 4]]
        self.assertGreater(float((logits - m(permuted, self.tgt)).abs().max()), 1e-4)

    def test_zero_relative_tables_match_no_encoder_positions(self):
        a = Transformer(replace(self.base, encoder_positions=False)).eval()
        b = Transformer(self.c).eval()
        with torch.no_grad():
            for layer in b.encoder:
                layer.self_attn.relative_k.zero_()
                layer.self_attn.relative_v.zero_()
        torch.testing.assert_close(a(self.src, self.tgt), b(self.src, self.tgt), atol=1e-6, rtol=1e-6)

    def test_cache_beam_save_load_and_backend_guard(self):
        m = Transformer(self.c).eval()
        memory, cache = m.encode(self.src), None
        for i in range(3):
            step, cache = m.decode_step(self.tgt[:, i], memory, cache)
            torch.testing.assert_close(step, m.decode(self.tgt[:, :i+1], memory)[:, -1], atol=1e-5, rtol=1e-4)
        for beam in (1, 3):
            a, fa = m.beam_search(self.src, 5, beam, use_cache=True)
            b, fb = m.beam_search(self.src, 5, beam, use_cache=False)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(fa, fb))
        loaded = Transformer(self.c, seed=9).eval()
        loaded.load_canonical(m.canonical())
        torch.testing.assert_close(m(self.src, self.tgt), loaded(self.src, self.tgt), atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "SDPA"):
            m.set_backend("sdpa")
        with self.assertRaises(ValueError):
            ModelConfig(encoder_relative_positions=True)


class RDropTests(unittest.TestCase):
    def config(self, variant="A5"):
        c = load_config(ROOT / "scripts/configs/smoke.json", variant, rdrop_alpha=3 if variant == "A5" else None)
        c["model"].update(d_model=16, d_ff=32, dropout=.2)
        c["training"].update(micro_batch=1, label_smoothing=.1)
        return c

    def rows(self):
        return [{"src_ids": [2, 5, 6, 3], "tgt_ids": [2, 7, 8, 3]},
                {"src_ids": [2, 9, 3], "tgt_ids": [2, 8, 3]}]

    def test_symmetric_kl_formula_pad_positions_and_both_gradients(self):
        torch.manual_seed(13)
        a, b = torch.randn(2, 3, 7, requires_grad=True), torch.randn(2, 3, 7, requires_grad=True)
        labels = torch.tensor([[4, 5, 3], [6, 3, 0]])
        loss, nll, count, ce, kl = rdrop_loss_sums(a, b, labels, .1, 3)
        pa, pb = a.double().softmax(-1), b.double().softmax(-1)
        ref = .5 * (pa * (pa.log() - pb.log()) + pb * (pb.log() - pa.log())).sum(-1)
        torch.testing.assert_close(kl.double(), ref[labels != 0].sum(), atol=1e-6, rtol=1e-5)
        self.assertEqual(int(count), 5)
        self.assertGreaterEqual(float(kl), 0)
        torch.testing.assert_close(loss, ce + 3 * kl)
        expected_nll = .5 * (loss_sums(a, labels)[1] + loss_sums(b, labels)[1])
        torch.testing.assert_close(nll, expected_nll)
        loss.backward()
        for x in (a, b):
            self.assertTrue((x.grad[1, 2] == 0).all())
            self.assertGreater(float(x.grad[labels != 0].abs().sum()), 0)
        identical = rdrop_loss_sums(a, a, labels)[4]
        self.assertEqual(float(identical), 0.0)

    def test_alpha_zero_preserves_single_forward_and_rng(self):
        c = self.config("A0")
        a = TorchTrainer(c, "cpu")
        calls = []
        a.model.register_forward_hook(lambda *args: calls.append(1))
        a.update(self.rows(), .001)
        expected, rng = a.model.canonical(), torch.get_rng_state()
        c["training"]["rdrop_alpha"] = 0
        b = TorchTrainer(c, "cpu")
        b.update(self.rows(), .001)
        self.assertEqual(len(calls), 2)  # two micro-batches, one pass each
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in expected.items():
            np.testing.assert_array_equal(v, b.model.canonical()[k])

    def test_independent_dropout_and_logged_objective(self):
        t = TorchTrainer(self.config(), "cpu")
        outputs = []
        t.model.register_forward_hook(lambda m, i, o: outputs.append(o.detach().clone()))
        total, nll, count, _ = t.update(self.rows(), .001)
        self.assertEqual(len(outputs), 4)
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        self.assertGreater(t.last_metrics["symmetric_kl"], 0)
        self.assertAlmostEqual(total / count, t.last_metrics["ce"] + 3 * t.last_metrics["symmetric_kl"], places=5)
        self.assertTrue(np.isfinite(nll))

    def test_full_and_accumulated_gradients_with_nonzero_kl(self):
        # Deterministic, input-dependent second-path perturbation gives nonzero KL
        # without making the comparison depend on RNG batching order.
        base = Transformer(ModelConfig(vocab_size=128, d_model=16, d_ff=32, encoder_layers=1,
                                       decoder_layers=1, dropout=0))
        rows = self.rows()
        denom = sum(len(r["tgt_ids"])-1 for r in rows)
        def gradients(micro):
            base.zero_grad()
            for start in range(0, len(rows), micro):
                src, tgt, lab = map(torch.from_numpy, collate(rows[start:start+micro]))
                a = base(src, tgt)
                perturb = torch.linspace(-.3, .3, 128)[None, None]
                objective = rdrop_loss_sums(a, a + perturb, lab, .1, 3)[0]
                (objective / denom).backward()
            return {n: p.grad.clone() for n, p in base.named_parameters()}
        full, accumulated = gradients(2), gradients(1)
        for name in full:
            torch.testing.assert_close(full[name], accumulated[name], atol=2e-6, rtol=1e-4)

    def test_resume_optimizer_rng_for_both_new_variants(self):
        for variant in ("A4", "A5"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                c = self.config(variant)
                a = TorchTrainer(c, "cpu")
                a.update(self.rows(), .001)
                a.save(Path(tmp))
                a.update(self.rows(), .0008)
                expected = a.model.canonical()
                b = TorchTrainer(c, "cpu")
                b.restore(Path(tmp))
                b.update(self.rows(), .0008)
                for name in expected:
                    np.testing.assert_array_equal(expected[name], b.model.canonical()[name])

    def test_pipeline_resume_and_epoch_logging_for_both_variants(self):
        for variant in ("A4", "A5"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                c = self.config(variant)
                c["training"]["epochs"] = 1
                c["evaluation"]["max_new_tokens"] = 3
                resumed, whole = Path(tmp) / "resumed", Path(tmp) / "whole"
                train(c, ROOT / "cache/smoke", resumed, max_updates=2)
                train(c, ROOT / "cache/smoke", resumed, resume=resumed / "latest.json")
                train(c, ROOT / "cache/smoke", whole)
                a = load_npz(resolve_checkpoint(resumed / "latest.json") / "weights.npz")
                b = load_npz(resolve_checkpoint(whole / "latest.json") / "weights.npz")
                for name in a:
                    np.testing.assert_array_equal(a[name], b[name])
                if variant == "A5":
                    from enfr.common import read_jsonl
                    epoch = [r for r in read_jsonl(resumed / "log.jsonl") if r["event"] == "epoch"][0]
                    self.assertAlmostEqual(epoch["train_loss"], epoch["train_ce"] + 3 * epoch["train_symmetric_kl"], places=5)
                    self.assertIn("nll", epoch["validation"])

    def test_unsupported_training_and_bad_alpha_rejected(self):
        for alpha in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                load_config(variant="A5", rdrop_alpha=alpha)
        with self.assertRaises(ValueError):
            load_config(variant="A4", rdrop_alpha=1)
        with self.assertRaisesRegex(ValueError, "PyTorch-only"):
            TensorFlowTrainer(self.config(), "cpu")


if __name__ == "__main__":
    unittest.main()
