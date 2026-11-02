import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from enfr.common import ModelConfig, canonical_weights, learning_rate
from enfr.data import batch_manifest, clean_rows, collate, normalize, parse_archive
from enfr.diagnostics import permutation_diagnostic
from enfr.torch_model import Attention, Transformer, loss_sums

torch.set_num_threads(1)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.c = ModelConfig(vocab_size=32, d_model=16, heads=4, d_ff=32,
                             encoder_layers=2, decoder_layers=2, dropout=0)
        self.model = Transformer(self.c).eval()
        self.src = torch.tensor([[2, 4, 5, 3, 0], [2, 6, 3, 0, 0]])
        self.tgt = torch.tensor([[2, 7, 8, 3], [2, 9, 3, 0]])
        self.labels = torch.tensor([[7, 8, 3, 0], [9, 3, 0, 0]])

    def test_causal_future_does_not_affect_prefix(self):
        modified = self.tgt.clone(); modified[:, 2:] = 15
        torch.testing.assert_close(self.model(self.src, self.tgt)[:, :2], self.model(self.src, modified)[:, :2])

    def test_right_padding_invariance(self):
        padded_src = torch.nn.functional.pad(self.src, (0, 3))
        padded_tgt = torch.nn.functional.pad(self.tgt, (0, 2))
        torch.testing.assert_close(self.model(self.src, self.tgt), self.model(padded_src, padded_tgt)[:, :4])

    def test_attention_probabilities_and_masks(self):
        self.model.set_capture(True)
        self.model(self.src, self.tgt)
        for layer in self.model.encoder:
            p = layer.self_attn.last["probs"]
            torch.testing.assert_close(p.sum(-1), torch.ones_like(p.sum(-1)))
            self.assertTrue(bool((p[1, :, :, 3:] == 0).all()))
        for layer in self.model.decoder:
            p = layer.self_attn.last["probs"]
            torch.testing.assert_close(p.sum(-1), torch.ones_like(p.sum(-1)))
            self.assertTrue(bool((p.triu(1) == 0).all()))
            self.assertTrue(bool(torch.isfinite(p).all()))

    def test_cache_matches_every_prefix_with_padding(self):
        memory = self.model.encode(self.src)
        cache = None
        for i in range(self.tgt.shape[1]):
            expected = self.model.decode(self.tgt[:, :i+1], memory)[:, -1]
            actual, cache = self.model.decode_step(self.tgt[:, i], memory, cache)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            self.assertEqual(cache["layers"][0]["k"].shape[2], i+1)

    def test_sdpa_forward_and_cached_equivalence(self):
        expected = self.model(self.src, self.tgt)
        self.model.set_backend("sdpa")
        actual = self.model(self.src, self.tgt)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        self.test_cache_matches_every_prefix_with_padding()

    def test_eval_sdpa_dropout_is_disabled(self):
        m = Transformer(replace(self.c, dropout=.6)).eval(); m.set_backend("sdpa")
        torch.testing.assert_close(m(self.src, self.tgt), m(self.src, self.tgt), atol=0, rtol=0)

    def test_gradient_accumulation_token_weighting(self):
        self.model.zero_grad()
        loss, _, count = loss_sums(self.model(self.src, self.tgt), self.labels, .1)
        (loss / count).backward()
        expected = {n: p.grad.clone() for n, p in self.model.named_parameters()}
        self.model.zero_grad()
        for i in range(2):
            micro_loss, _, _ = loss_sums(self.model(self.src[i:i+1], self.tgt[i:i+1]), self.labels[i:i+1], .1)
            (micro_loss / count).backward()
        for name, p in self.model.named_parameters():
            torch.testing.assert_close(p.grad, expected[name], atol=1e-6, rtol=1e-4)

    def test_loss_ignores_pad_and_uses_nonpad_smoothing(self):
        logits = torch.zeros(1, 2, 4)
        labels = torch.tensor([[3, 0]])
        loss, nll, count = loss_sums(logits, labels, .1)
        self.assertEqual(int(count), 1)
        self.assertAlmostEqual(float(loss), np.log(4), places=6)
        self.assertAlmostEqual(float(nll), np.log(4), places=6)
        logits[:, 1] = torch.tensor([10., -7., 1., 2.])
        torch.testing.assert_close(loss_sums(logits, labels, .1)[0], loss)

    def test_position_free_encoder_symmetry(self):
        report = permutation_diagnostic(Transformer(replace(self.c, encoder_positions=False)))
        self.assertTrue(report["encoder_equivariant"])
        self.assertTrue(report["logits_invariant"])
        self.assertFalse(permutation_diagnostic(self.model)["logits_invariant"])

    def test_single_head_preserves_parameters_and_initial_weights(self):
        one = Transformer(replace(self.c, heads=1))
        self.assertEqual(sum(p.numel() for p in one.parameters()), sum(p.numel() for p in self.model.parameters()))
        for name, array in one.canonical().items():
            np.testing.assert_array_equal(array, self.model.canonical()[name])

    def test_save_load_and_unique_tied_embedding(self):
        arrays = self.model.canonical()
        clone = Transformer(self.c, seed=45); clone.load_canonical(arrays); clone.eval()
        torch.testing.assert_close(clone(self.src, self.tgt), self.model(self.src, self.tgt), atol=0, rtol=0)
        self.assertEqual([n for n in arrays if "embedding" in n], ["embedding"])

    def test_mixed_batch_generation_equals_individual(self):
        batched, flags = self.model.greedy(self.src, 12)
        for i in range(2):
            length = int((self.src[i] != 0).sum())
            single, flag = self.model.greedy(self.src[i:i+1, :length], 12)
            a = [x for x in batched[i].tolist() if x != 0]
            b = [x for x in single[0].tolist() if x != 0]
            self.assertEqual(a, b)
            self.assertEqual(bool(flags[i]), bool(flag[0]))

    def test_eos_rows_stop_without_changing_live_rows(self):
        def fake(last, memory, cache=None):
            step = 0 if cache is None else cache
            logits = torch.full((2, self.c.vocab_size), -100.)
            logits[0, 3] = 100.
            logits[1, 3 if step == 2 else 5] = 100.
            return logits, step + 1
        with patch.object(self.model, "decode_step", side_effect=fake):
            tokens, truncated = self.model.greedy(self.src, 10)
        self.assertEqual(tokens.tolist(), [[3, 0, 0], [5, 5, 3]])
        self.assertFalse(bool(truncated.any()))


class DataTests(unittest.TestCase):
    def test_normalization_preserves_french_and_case(self):
        self.assertEqual(normalize("  E\u0301cole\u2028  l'oiseau  "), "École l'oiseau")

    def test_shift_and_padding(self):
        rows = [{"src_ids": [2, 4, 3], "tgt_ids": [2, 5, 6, 3]}, {"src_ids": [2, 3], "tgt_ids": [2, 7, 3]}]
        src, target, labels = collate(rows)
        np.testing.assert_array_equal(target, [[2, 5, 6], [2, 7, 3]])
        np.testing.assert_array_equal(labels, [[5, 6, 3], [7, 3, 0]])

    def test_dedup_and_heldout_source_removal(self):
        def row(i, s, t): return {"id": i, "src_raw": s, "tgt_raw": t}
        raw = {"train": [row("a", "same", "a"), row("b", "same", "a"), row("c", "test", "b"), row("d", "", "d")],
               "validation": [row("v", "test", "other")], "test": []}
        rows, counts = clean_rows(raw)
        self.assertEqual(len(rows["train"]), 1)
        self.assertEqual(counts, {"empty": 1, "duplicate_pair": 1, "heldout_source_overlap": 1})

    def test_unicode_line_separator_does_not_split_sentence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("train.tags.en-fr.en", "<doc>\nHello.\n</doc>\n")
                z.writestr("train.tags.en-fr.fr", "<doc>\nBon\u2028jour.\n</doc>\n")
                for split in ["dev2010"] + [f"tst{y}" for y in range(2010, 2016)]:
                    for lang in ("en", "fr"):
                        z.writestr(f"IWSLT17.TED.{split}.en-fr.{lang}.xml", '<mteval><doc docid="1"><seg id="1">Hello &amp; bonjour.</seg></doc></mteval>')
            with patch("enfr.data.EXPECTED", {"train": 1, "validation": 1, "test": 6}):
                parsed = parse_archive(path)
            self.assertEqual(parsed["train"][0]["tgt_raw"], "Bon\u2028jour.")
            self.assertEqual(parsed["test"][0]["src_raw"], "Hello & bonjour.")

    def test_manifests_are_shared_deterministic_and_cover_all_examples(self):
        rows = [{"id": str(i), "src_ids": [2] * (i % 6 + 2), "tgt_ids": [2, 3]} for i in range(19)]
        with tempfile.TemporaryDirectory() as tmp:
            a = batch_manifest(rows, tmp, "data", 42, 0, 4, 8)
            b = batch_manifest(rows, tmp, "data", 42, 0, 4, 8)
            self.assertEqual(a, b)
            self.assertEqual(sorted(i for batch in a for i in batch), list(range(19)))
            with self.assertRaises(ValueError):
                batch_manifest(rows, tmp, "other", 42, 0, 4, 8)

    def test_learning_rate_update_index(self):
        c = {"warmup_steps": 10, "lr": .001, "min_lr": .0001}
        self.assertAlmostEqual(learning_rate(1, 100, c), .0001)
        self.assertAlmostEqual(learning_rate(10, 100, c), .001)
        self.assertAlmostEqual(learning_rate(100, 100, c), .0001)


if __name__ == "__main__":
    unittest.main()
