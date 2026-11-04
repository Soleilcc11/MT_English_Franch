import itertools
import math
import unittest
from unittest.mock import patch

import torch

from enfr.common import BOS, EOS, PAD, ModelConfig
from enfr.evaluation import decode_settings, generate
from enfr.torch_model import Transformer

torch.set_num_threads(1)


class BeamTests(unittest.TestCase):
    def setUp(self):
        self.model = Transformer(ModelConfig(vocab_size=32, d_model=16, heads=4, d_ff=32,
                                            encoder_layers=2, decoder_layers=2, dropout=0)).eval()
        self.src = torch.tensor([[BOS, 4, 5, EOS, PAD], [BOS, 6, EOS, PAD, PAD]])

    def test_beam_one_is_exactly_greedy(self):
        for cache in (True, False):
            expected = self.model.greedy(self.src, 8, cache)
            actual = self.model.beam_search(self.src, 8, 1, 1.0, cache)
            for a, b in zip(actual, expected):
                self.assertTrue(torch.equal(a, b))

    def test_cache_and_full_prefix_beam_agree(self):
        a = self.model.beam_search(self.src, 8, 4, .6, True)
        b = self.model.beam_search(self.src, 8, 4, .6, False)
        for left, right in zip(a, b):
            self.assertTrue(torch.equal(left, right))

    def test_encode_and_cross_projection_run_once_before_expansion(self):
        cross = self.model.decoder[0].cross_attn.k
        with patch.object(self.model, "encode", wraps=self.model.encode) as encode, \
                patch.object(cross, "forward", wraps=cross.forward) as project:
            self.model.beam_search(self.src, 8, 4)
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(project.call_count, 1)
        self.assertEqual(project.call_args.args[0].shape[0], self.src.shape[0])

    def test_cache_gather_reorders_every_tensor_and_duplicates_parents(self):
        memory = self.model.encode(self.src)
        _, cache = self.model.decode_step(torch.tensor([BOS, BOS]), memory)
        indices = torch.tensor([1, 0, 1])
        new = self.model.reorder_cache(cache, indices)
        self.assertEqual(new["position"], cache["position"])
        torch.testing.assert_close(new["valid"], cache["valid"][indices])
        for a, b in zip(cache["layers"], new["layers"]):
            for key in ("k", "v"):
                torch.testing.assert_close(b[key], a[key][indices])
            for old, gathered in zip(a["cross"], b["cross"]):
                torch.testing.assert_close(gathered, old[indices])

    def test_mixed_lengths_match_individual_and_pad_is_never_generated(self):
        batched, flags = self.model.beam_search(self.src, 8, 4)
        for i, row in enumerate(self.src):
            single, capped = self.model.beam_search(row[row != PAD][None], 8, 4)
            self.assertEqual(batched[i][batched[i] != PAD].tolist(), single[0][single[0] != PAD].tolist())
            self.assertEqual(bool(flags[i]), bool(capped[0]))
            self.assertNotIn(BOS, single[0].tolist())

    def test_branched_cache_logits_match_full_prefix_at_every_step(self):
        memory = self.model.encode(self.src)
        prefix = torch.full((2, 1), BOS, dtype=torch.long)
        _, cache = self.model.decode_step(prefix[:, -1], memory)
        for parents, nxt in (([1, 0, 1, 0], [4, 5, 7, 8]),
                             ([2, 0, 3, 3], [9, 10, 11, 12])):
            indices, last = torch.tensor(parents), torch.tensor(nxt)
            memory = tuple(x[indices] for x in memory)
            cache = self.model.reorder_cache(cache, indices)
            prefix = torch.cat((prefix[indices], last[:, None]), 1)
            expected = self.model.decode(prefix, memory)[:, -1]
            actual, cache = self.model.decode_step(last, memory, cache)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_invalid_parameters_and_tensorflow_rejection(self):
        for kwargs in ({"beam_size": 0}, {"beam_size": 31}, {"length_penalty": -1},
                       {"length_penalty": float("nan")}, {"max_new_tokens": 0}):
            with self.assertRaises(ValueError):
                self.model.beam_search(self.src, **kwargs)
        with self.assertRaises(ValueError):
            generate(None, "tensorflow", None, decode="beam")
        self.assertEqual(decode_settings(), {"decode": "greedy", "beam_size": 1, "length_penalty": 0.0})

    @staticmethod
    def toy_logits(prefix, source=0):
        logits = torch.full((32,), -float("inf"))
        if source == 1:  # The second example finishes immediately.
            logits[EOS] = 0
            return logits
        if len(prefix) == 1:
            weights = {PAD: .05, BOS: .05, EOS: .03, 1: .02, 4: .55, 5: .30}
        elif prefix[-1] == 4:
            # Forbidden mass changes with the prefix: post-mask renormalization
            # would wrongly favor this path and fail the exhaustive reference.
            weights = {PAD: .40, BOS: .40, EOS: .08, 1: .02, 4: .06, 5: .04}
        else:
            weights = {PAD: .01, BOS: .01, EOS: .45, 1: .03, 4: .20, 5: .30}
        for token, p in weights.items():
            logits[token] = math.log(p)
        return logits

    def toy_encode(self, src):
        return src[:, :1, None].float(), torch.ones(src.shape[0], 1, 1, 1, dtype=torch.bool)

    def toy_decode(self, prefix, memory):
        return torch.stack([torch.stack([self.toy_logits(row[:i+1].tolist(), int(memory[0][j, 0, 0]))
                                        for i in range(prefix.shape[1])]) for j, row in enumerate(prefix)])

    def toy_step(self, last, memory, cache=None):
        if cache is None:
            prefix = last[:, None]
        else:
            prefix = torch.cat((cache["layers"][0]["k"][:, 0, :, 0].long(), last[:, None]), 1)
        cross = (memory[0][:, None], memory[0][:, None] + 100)
        state = {"position": prefix.shape[1], "valid": prefix != PAD,
                 "layers": [{"k": prefix[:, None, :, None].float(),
                             "v": prefix[:, None, :, None].float() * 2, "cross": cross}]}
        return self.toy_decode(prefix, memory)[:, -1], state

    def exhaustive(self, max_len, alpha):
        candidates = []
        def visit(prefix, score):
            if len(prefix) > 1 and (prefix[-1] == EOS or len(prefix) - 1 == max_len):
                length = len(prefix) - 1
                candidates.append((score / ((5 + length) / 6) ** alpha, prefix[1:]))
                return
            probabilities = torch.log_softmax(self.toy_logits(prefix), -1)
            for token in (1, EOS, 4, 5):
                visit(prefix + [token], score + float(probabilities[token]))
        visit([BOS], 0.0)
        return max(candidates, key=lambda x: x[0])[1]

    def test_exhaustive_reference_eos_reordering_and_no_renormalization(self):
        with patch.object(self.model, "encode", side_effect=self.toy_encode), \
                patch.object(self.model, "decode", side_effect=self.toy_decode), \
                patch.object(self.model, "decode_step", side_effect=self.toy_step):
            for cache, alpha in itertools.product((False, True), (0.0, .6, 1.0)):
                tokens, flags = self.model.beam_search(torch.tensor([[0], [1]]), 3, 30, alpha, cache)
                result = tokens[0][tokens[0] != PAD].tolist()
                self.assertEqual(result, self.exhaustive(3, alpha))
                self.assertEqual(bool(flags[0]), result[-1] != EOS)
                self.assertEqual(tokens[1][tokens[1] != PAD].tolist(), [EOS])
                self.assertFalse(bool(flags[1]))

    def test_length_cap_and_all_eos(self):
        with patch.object(self.model, "encode", side_effect=self.toy_encode), \
                patch.object(self.model, "decode_step", side_effect=self.toy_step):
            tokens, flags = self.model.beam_search(torch.tensor([[0], [1]]), 1, 4)
            self.assertEqual(tokens.tolist(), [[4], [EOS]])
            self.assertEqual(flags.tolist(), [True, False])
            tokens, flags = self.model.beam_search(torch.tensor([[1], [1]]), 10, 4)
            self.assertEqual(tokens.tolist(), [[EOS], [EOS]])
            self.assertEqual(flags.tolist(), [False, False])

    def test_length_normalization_changes_selection(self):
        def logits(prefix, source=0):
            out = torch.full((32,), -float("inf"))
            if len(prefix) == 1:
                out[EOS], out[4] = math.log(.51), math.log(.49)
            else:
                out[EOS] = 0.0
            return out
        with patch.object(self, "toy_logits", side_effect=logits), \
                patch.object(self.model, "encode", side_effect=self.toy_encode), \
                patch.object(self.model, "decode_step", side_effect=self.toy_step):
            short, _ = self.model.beam_search(torch.tensor([[0]]), 3, 4, 0.0)
            long, _ = self.model.beam_search(torch.tensor([[0]]), 3, 4, 1.0)
            self.assertEqual(short.tolist(), [[EOS]])
            self.assertEqual(long.tolist(), [[4, EOS]])


if __name__ == "__main__":
    unittest.main()
