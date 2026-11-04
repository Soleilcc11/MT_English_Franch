from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import BOS, EOS, PAD, ModelConfig, canonical_weights


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape, self.weight.float(),
                            self.bias.float(), self.eps).to(x.dtype)


def positions(length, dim, device, offset=0):
    pos = torch.arange(offset, offset + length, device=device, dtype=torch.float32)[:, None]
    inv = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000) / dim))
    angles = pos * inv[None, :]
    return torch.stack((angles.sin(), angles.cos()), dim=-1).reshape(length, dim)


class Attention(nn.Module):
    def __init__(self, c, relative=False):
        super().__init__()
        self.heads, self.dk, self.dropout = c.heads, c.d_model // c.heads, c.dropout
        self.q, self.k, self.v, self.out = [nn.Linear(c.d_model, c.d_model) for _ in range(4)]
        self.backend = "manual"
        self.capture = False
        self.last = {}
        self.relative = relative
        self.max_distance = c.relative_max_distance
        if relative:
            # Shaw K/V tables: shared across heads, distinct for each layer.
            self.relative_k = nn.Parameter(torch.empty(2 * self.max_distance + 1, self.dk))
            self.relative_v = nn.Parameter(torch.empty(2 * self.max_distance + 1, self.dk))

    def split(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.heads, self.dk).transpose(1, 2)

    def kv(self, context):
        return self.split(self.k(context)), self.split(self.v(context))

    def attend(self, x, k, v, allowed):
        q = self.split(self.q(x))
        if self.relative and self.backend != "manual":
            raise ValueError("Shaw relative K/V attention requires the manual backend")
        if self.backend == "sdpa" and not self.capture:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        else:
            scores = (q @ k.transpose(-2, -1)).float() / math.sqrt(self.dk)
            if self.relative:
                if q.shape[2] != k.shape[2]:
                    raise ValueError("Relative attention is restricted to full encoder self-attention")
                pos = torch.arange(q.shape[2], device=q.device)
                distance = (pos[None, :] - pos[:, None]).clamp(-self.max_distance, self.max_distance) + self.max_distance
                rk, rv = self.relative_k[distance], self.relative_v[distance]
                scores = scores + torch.einsum("bhid,ijd->bhij", q, rk.to(q.dtype)).float() / math.sqrt(self.dk)
            scores = scores.masked_fill(~allowed, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            if self.capture:
                self.last = {"scores": scores.detach(), "probs": probs.detach()}
            probs = F.dropout(probs, self.dropout, self.training)
            out = probs.to(v.dtype) @ v
            if self.relative:
                out = out + torch.einsum("bhij,ijd->bhid", probs.to(v.dtype), rv.to(v.dtype))
        out = out.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[1], -1)
        out = self.out(out)
        if self.capture:
            self.last["output"] = out.detach()
        return out

    def forward(self, x, context, allowed):
        k, v = self.kv(context)
        return self.attend(x, k, v, allowed)


class EncoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = Attention(c, relative=c.encoder_relative_positions)
        self.norm1 = FP32LayerNorm(c.d_model, eps=c.norm_eps)
        self.norm2 = FP32LayerNorm(c.d_model, eps=c.norm_eps)
        self.ff1, self.ff2 = nn.Linear(c.d_model, c.d_ff), nn.Linear(c.d_ff, c.d_model)
        self.dropout = c.dropout

    def drop(self, x):
        return F.dropout(x, self.dropout, self.training)

    def forward(self, x, allowed):
        z = self.norm1(x)
        x = x + self.drop(self.self_attn(z, z, allowed))
        return x + self.drop(self.ff2(self.drop(F.relu(self.ff1(self.norm2(x))))))


class DecoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(c), Attention(c)
        self.norm1 = FP32LayerNorm(c.d_model, eps=c.norm_eps)
        self.norm2 = FP32LayerNorm(c.d_model, eps=c.norm_eps)
        self.norm3 = FP32LayerNorm(c.d_model, eps=c.norm_eps)
        self.ff1, self.ff2 = nn.Linear(c.d_model, c.d_ff), nn.Linear(c.d_ff, c.d_model)
        self.dropout = c.dropout

    def drop(self, x):
        return F.dropout(x, self.dropout, self.training)

    def finish(self, x, memory, src_mask, cross_kv=None):
        z = self.norm2(x)
        if cross_kv is None:
            cross = self.cross_attn(z, memory, src_mask)
        else:
            cross = self.cross_attn.attend(z, *cross_kv, src_mask)
        x = x + self.drop(cross)
        return x + self.drop(self.ff2(self.drop(F.relu(self.ff1(self.norm3(x))))))

    def forward(self, x, memory, src_mask, target_mask):
        z = self.norm1(x)
        x = x + self.drop(self.self_attn(z, z, target_mask))
        return self.finish(x, memory, src_mask)

    def step(self, x, memory, src_mask, allowed, previous):
        z = self.norm1(x)
        k, v = self.self_attn.kv(z)
        if previous is not None:
            k, v = torch.cat((previous["k"], k), dim=2), torch.cat((previous["v"], v), dim=2)
            cross = previous["cross"]
        else:
            cross = self.cross_attn.kv(memory)
        x = x + self.drop(self.self_attn.attend(z, k, v, allowed))
        return self.finish(x, memory, src_mask, cross), {"k": k, "v": v, "cross": cross}


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, seed=42):
        super().__init__()
        self.config = config
        self.embedding = nn.Parameter(torch.empty(config.vocab_size, config.d_model))
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.encoder = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.encoder_norm = FP32LayerNorm(config.d_model, eps=config.norm_eps)
        self.decoder_norm = FP32LayerNorm(config.d_model, eps=config.norm_eps)
        self.capture, self.trace = False, {}
        self.load_canonical(canonical_weights(config, seed))

    def load_canonical(self, weights):
        self.load_state_dict({k: torch.as_tensor(v.copy()) for k, v in weights.items()}, strict=True)

    def canonical(self):
        return {k: v.detach().cpu().float().numpy().copy() for k, v in self.state_dict().items()}

    def set_backend(self, backend):
        if backend not in ("manual", "sdpa"):
            raise ValueError(backend)
        if backend == "sdpa" and self.config.encoder_relative_positions:
            raise ValueError("A4 Shaw relative K/V attention is not supported by the SDPA benchmark")
        for m in self.modules():
            if isinstance(m, Attention):
                m.backend = backend

    def set_capture(self, capture):
        self.capture = capture
        self.trace = {}
        for m in self.modules():
            if isinstance(m, Attention):
                m.capture, m.last = capture, {}

    def embed(self, ids, use_positions=True, offset=0):
        x = F.embedding(ids, self.embedding) * math.sqrt(self.config.d_model)
        if use_positions:
            x = x + positions(ids.shape[1], self.config.d_model, ids.device, offset)[None]
        return F.dropout(x, self.config.dropout, self.training)

    def record(self, key, value):
        if self.capture:
            self.trace[key] = value.detach()

    def encode(self, src_ids):
        mask = (src_ids != PAD)[:, None, None, :]
        x = self.embed(src_ids, self.config.encoder_positions)
        self.record("src_embedding", x)
        for i, layer in enumerate(self.encoder):
            x = layer(x, mask)
            self.record(f"encoder.{i}", x)
        x = self.encoder_norm(x)
        self.record("memory", x)
        return x, mask

    def decode(self, target, memory):
        enc, src_mask = memory
        n = target.shape[1]
        causal = torch.ones(n, n, dtype=torch.bool, device=target.device).tril()[None, None]
        allowed = causal & (target != PAD)[:, None, None, :]
        x = self.embed(target)
        self.record("tgt_embedding", x)
        for i, layer in enumerate(self.decoder):
            x = layer(x, enc, src_mask, allowed)
            self.record(f"decoder.{i}", x)
        x = self.decoder_norm(x)
        self.record("decoder_norm", x)
        logits = F.linear(x, self.embedding, self.output_bias).float()
        self.record("logits", logits)
        return logits

    def forward(self, src_ids, tgt_input_ids):
        return self.decode(tgt_input_ids, self.encode(src_ids))

    def decode_step(self, last_token, memory, cache=None):
        if last_token.ndim == 1:
            last_token = last_token[:, None]
        if last_token.shape[1] != 1:
            raise ValueError("decode_step accepts exactly one token per example")
        offset = 0 if cache is None else cache["position"]
        valid = last_token != PAD
        if cache is not None:
            valid = torch.cat((cache["valid"], valid), dim=1)
        x = self.embed(last_token, offset=offset)
        new_layers = []
        for i, layer in enumerate(self.decoder):
            previous = None if cache is None else cache["layers"][i]
            x, new = layer.step(x, *memory, valid[:, None, None, :], previous)
            new_layers.append(new)
        logits = F.linear(self.decoder_norm(x), self.embedding, self.output_bias).float()[:, 0]
        return logits, {"position": offset + 1, "valid": valid, "layers": new_layers}

    @torch.no_grad()
    def greedy(self, src_ids, max_new_tokens=256, use_cache=True):
        if self.training:
            raise RuntimeError("Call model.eval() before decoding")
        memory = self.encode(src_ids)
        tokens = torch.full((src_ids.shape[0], 1), BOS, device=src_ids.device, dtype=torch.long)
        finished = torch.zeros(src_ids.shape[0], device=src_ids.device, dtype=torch.bool)
        cache = None
        for _ in range(max_new_tokens):
            if use_cache:
                logits, cache = self.decode_step(tokens[:, -1], memory, cache)
            else:
                logits = self.decode(tokens, memory)[:, -1]
            logits[:, [PAD, BOS]] = -float("inf")
            nxt = logits.argmax(-1)
            nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
            tokens = torch.cat((tokens, nxt[:, None]), 1)
            finished = finished | (nxt == EOS)
            if bool(finished.all()):
                break
        return tokens[:, 1:], ~finished

    @staticmethod
    def reorder_cache(cache, indices):
        """Gather beam parents, including duplicated parents, on the batch axis."""
        if cache is None:
            return None
        return {"position": cache["position"],
                "valid": cache["valid"].index_select(0, indices),
                "layers": [{"k": layer["k"].index_select(0, indices),
                            "v": layer["v"].index_select(0, indices),
                            "cross": tuple(x.index_select(0, indices) for x in layer["cross"])}
                           for layer in cache["layers"]]}

    @torch.no_grad()
    def beam_search(self, src_ids, max_new_tokens=256, beam_size=4,
                    length_penalty=0.6, use_cache=True):
        """Length-normalized beam search with absorbing finished hypotheses.

        Scores use the original vocabulary probabilities (no renormalization
        after masking PAD/BOS). Length includes EOS but excludes BOS and PAD.
        Equal scores prefer the earlier parent, then the smaller token ID.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before decoding")
        if not isinstance(beam_size, int) or not 1 <= beam_size <= self.config.vocab_size - 2:
            raise ValueError("beam_size must be an integer in [1, vocab_size - 2]")
        if not math.isfinite(length_penalty) or length_penalty < 0:
            raise ValueError("length_penalty must be finite and nonnegative")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if beam_size == 1:
            return self.greedy(src_ids, max_new_tokens, use_cache)

        batch, vocab = src_ids.shape[0], self.config.vocab_size
        memory = self.encode(src_ids)
        tokens = torch.full((batch, 1), BOS, device=src_ids.device, dtype=torch.long)
        scores = torch.zeros(batch, 1, device=src_ids.device, dtype=torch.float32)
        lengths = torch.zeros(batch, 1, device=src_ids.device, dtype=torch.long)
        finished = torch.zeros(batch, 1, device=src_ids.device, dtype=torch.bool)
        cache, width = None, 1
        for _ in range(max_new_tokens):
            # The first BOS step runs at batch B, so cross K/V is projected once
            # per source before the beam expansion to B*K.
            if use_cache:
                logits, cache = self.decode_step(tokens[:, -1], memory, cache)
            else:
                logits = self.decode(tokens, memory)[:, -1]
            log_probs = F.log_softmax(logits.float(), dim=-1).reshape(batch, width, vocab)
            log_probs[:, :, [PAD, BOS]] = -float("inf")
            log_probs = log_probs.masked_fill(finished[:, :, None], -float("inf"))
            log_probs[:, :, PAD] = torch.where(finished, 0.0, -float("inf"))
            candidate_scores = scores[:, :, None] + log_probs
            candidate_lengths = lengths + (~finished).long()
            ranks = candidate_scores / ((5.0 + candidate_lengths[:, :, None]) / 6.0).pow(length_penalty)
            # Stable sort makes tie handling explicit, including across batches.
            chosen = torch.argsort(ranks.flatten(1), dim=-1, descending=True, stable=True)[:, :beam_size]
            parents, nxt = chosen // vocab, chosen % vocab
            indices = (torch.arange(batch, device=src_ids.device)[:, None] * width + parents).flatten()
            scores = candidate_scores.flatten(1).gather(1, chosen)
            if not bool(torch.isfinite(scores[:, 0]).all()):
                raise FloatingPointError("Beam search found no finite best hypothesis")
            lengths = candidate_lengths.gather(1, parents)
            finished = finished.gather(1, parents) | (nxt == EOS) | ~torch.isfinite(scores)
            tokens = torch.cat((tokens.index_select(0, indices), nxt.reshape(-1, 1)), dim=1)
            memory = tuple(x.index_select(0, indices) for x in memory)
            cache = self.reorder_cache(cache, indices)
            width = beam_size
            if bool(finished.all()):
                break

        ranks = scores / ((5.0 + lengths) / 6.0).pow(length_penalty)
        best = ranks.argmax(-1)
        indices = torch.arange(batch, device=src_ids.device) * width + best
        result = tokens.index_select(0, indices)[:, 1:]
        # Other beams may outlive the winning hypothesis; omit trailing all-PAD
        # columns while retaining right padding between examples in the batch.
        result = result[:, :int(lengths.gather(1, best[:, None]).max())]
        return result, ~finished.gather(1, best[:, None]).squeeze(1)


def loss_sums(logits, labels, smoothing=0.1):
    log_probs = F.log_softmax(logits.float(), -1)
    valid = labels != PAD
    nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    uniform = -log_probs[..., 1:].mean(-1)
    losses = (1 - smoothing) * nll + smoothing * uniform
    return (losses * valid).sum(), (nll * valid).sum(), valid.sum()


def rdrop_loss_sums(first, second, labels, smoothing=0.1, alpha=1.0):
    """Token SUMS: mean CE + alpha * mean bidirectional KL; both paths backprop.

    KL covers the full predictive vocabulary, including PAD/BOS, but excludes
    PAD target positions. This is separate from non-PAD label smoothing.
    """
    a, na, count = loss_sums(first, labels, smoothing)
    b, nb, _ = loss_sums(second, labels, smoothing)
    logp, logq = F.log_softmax(first.float(), -1), F.log_softmax(second.float(), -1)
    divergence = 0.5 * ((logp.exp() - logq.exp()) * (logp - logq)).sum(-1)
    kl = divergence.masked_select(labels != PAD).sum()
    ce, nll = (a + b) * 0.5, (na + nb) * 0.5
    return ce + alpha * kl, nll, count, ce, kl


def make_optimizer(model, config):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim == 1 else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": config["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
        lr=config["lr"], betas=tuple(config["betas"]), eps=config["eps"])
