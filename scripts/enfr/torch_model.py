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
    def __init__(self, c):
        super().__init__()
        self.heads, self.dk, self.dropout = c.heads, c.d_model // c.heads, c.dropout
        self.q, self.k, self.v, self.out = [nn.Linear(c.d_model, c.d_model) for _ in range(4)]
        self.backend = "manual"
        self.capture = False
        self.last = {}

    def split(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.heads, self.dk).transpose(1, 2)

    def kv(self, context):
        return self.split(self.k(context)), self.split(self.v(context))

    def attend(self, x, k, v, allowed):
        q = self.split(self.q(x))
        if self.backend == "sdpa" and not self.capture:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        else:
            scores = (q @ k.transpose(-2, -1)).float() / math.sqrt(self.dk)
            scores = scores.masked_fill(~allowed, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            if self.capture:
                self.last = {"scores": scores.detach(), "probs": probs.detach()}
            probs = F.dropout(probs, self.dropout, self.training)
            out = probs.to(v.dtype) @ v
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
        self.self_attn = Attention(c)
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


def loss_sums(logits, labels, smoothing=0.1):
    log_probs = F.log_softmax(logits.float(), -1)
    valid = labels != PAD
    nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    uniform = -log_probs[..., 1:].mean(-1)
    losses = (1 - smoothing) * nll + smoothing * uniform
    return (losses * valid).sum(), (nll * valid).sum(), valid.sum()


def make_optimizer(model, config):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim == 1 else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": config["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
        lr=config["lr"], betas=tuple(config["betas"]), eps=config["eps"])
