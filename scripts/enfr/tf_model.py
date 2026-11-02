from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from .common import BOS, EOS, PAD, ModelConfig, canonical_weights


def positions(length, dim, offset=0):
    pos = tf.cast(tf.range(offset, offset + length), tf.float32)[:, None]
    inv = tf.exp(tf.cast(tf.range(0, dim, 2), tf.float32) * (-math.log(10000) / dim))
    angles = pos * inv[None, :]
    return tf.reshape(tf.stack([tf.sin(angles), tf.cos(angles)], -1), [length, dim])


class Norm(tf.keras.layers.Layer):
    def __init__(self, c):
        super().__init__(dtype="float32")
        self.eps = c.norm_eps
        self.gamma = self.add_weight(name="gamma", shape=(c.d_model,), initializer="ones")
        self.beta = self.add_weight(name="beta", shape=(c.d_model,), initializer="zeros")

    def call(self, x):
        x = tf.cast(x, tf.float32)
        mean, var = tf.nn.moments(x, [-1], keepdims=True)
        return (x - mean) * tf.math.rsqrt(var + self.eps) * self.gamma + self.beta


class TrackedDropout(tf.keras.layers.Layer):
    """Generator state is checkpointed, unlike legacy global TF RNG counters."""
    def __init__(self, rate, rng):
        super().__init__(dtype="float32")
        self.rate, self.rng = rate, rng

    def call(self, x, training=False):
        if training and self.rate:
            return tf.nn.experimental.stateless_dropout(x, rate=self.rate, seed=self.rng.make_seeds(1)[:, 0])
        return x


class Attention(tf.keras.layers.Layer):
    def __init__(self, c, rng):
        super().__init__(dtype="float32")
        self.heads, self.dk = c.heads, c.d_model // c.heads
        self.q, self.k, self.v, self.out = [tf.keras.layers.Dense(c.d_model) for _ in range(4)]
        self.dropout = TrackedDropout(c.dropout, rng)
        self.capture, self.last = False, {}

    def split(self, x):
        shape = tf.shape(x)
        return tf.transpose(tf.reshape(x, [shape[0], shape[1], self.heads, self.dk]), [0, 2, 1, 3])

    def call(self, x, context, allowed, training=False):
        q, k, v = self.split(self.q(x)), self.split(self.k(context)), self.split(self.v(context))
        scores = tf.cast(tf.matmul(q, k, transpose_b=True), tf.float32) / math.sqrt(self.dk)
        scores = tf.where(allowed, scores, tf.constant(-float("inf"), tf.float32))
        probs = tf.nn.softmax(scores, -1)
        if self.capture:
            self.last = {"scores": scores, "probs": probs}
        dropped = self.dropout(probs, training=training)
        result = tf.matmul(tf.cast(dropped, v.dtype), v)
        result = tf.reshape(tf.transpose(result, [0, 2, 1, 3]), [tf.shape(x)[0], tf.shape(x)[1], self.heads * self.dk])
        result = self.out(result)
        if self.capture:
            self.last["output"] = result
        return result


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, c, rng):
        super().__init__(dtype="float32")
        self.self_attn = Attention(c, rng)
        self.norm1, self.norm2 = Norm(c), Norm(c)
        self.ff1, self.ff2 = tf.keras.layers.Dense(c.d_ff), tf.keras.layers.Dense(c.d_model)
        self.dropout = TrackedDropout(c.dropout, rng)

    def call(self, x, allowed, training=False):
        z = self.norm1(x)
        x = x + self.dropout(self.self_attn(z, z, allowed, training=training), training=training)
        ff = self.ff2(self.dropout(tf.nn.relu(self.ff1(self.norm2(x))), training=training))
        return x + self.dropout(ff, training=training)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, c, rng):
        super().__init__(dtype="float32")
        self.self_attn, self.cross_attn = Attention(c, rng), Attention(c, rng)
        self.norm1, self.norm2, self.norm3 = Norm(c), Norm(c), Norm(c)
        self.ff1, self.ff2 = tf.keras.layers.Dense(c.d_ff), tf.keras.layers.Dense(c.d_model)
        self.dropout = TrackedDropout(c.dropout, rng)

    def call(self, x, memory, src_mask, target_mask, training=False):
        z = self.norm1(x)
        x = x + self.dropout(self.self_attn(z, z, target_mask, training=training), training=training)
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, src_mask, training=training), training=training)
        ff = self.ff2(self.dropout(tf.nn.relu(self.ff1(self.norm3(x))), training=training))
        return x + self.dropout(ff, training=training)


class Transformer(tf.keras.Model):
    def __init__(self, config: ModelConfig, seed=42):
        super().__init__(dtype="float32")
        self.config = config
        self.rng = tf.random.Generator.from_seed(seed)
        self.embedding = self.add_weight(name="embedding", shape=(config.vocab_size, config.d_model), initializer="zeros")
        self.output_bias = self.add_weight(name="output_bias", shape=(config.vocab_size,), initializer="zeros")
        self.encoder = [EncoderLayer(config, self.rng) for _ in range(config.encoder_layers)]
        self.decoder = [DecoderLayer(config, self.rng) for _ in range(config.decoder_layers)]
        self.encoder_norm, self.decoder_norm = Norm(config), Norm(config)
        self.dropout = TrackedDropout(config.dropout, self.rng)
        self.capture, self.trace = False, {}
        dummy = tf.constant([[BOS, EOS]], tf.int64)
        self(dummy, dummy, training=False)
        self.load_canonical(canonical_weights(config, seed))

    def parameter_map(self):
        """name -> (variable, transpose required to canonical [out,in])."""
        out = {"embedding": (self.embedding, False), "output_bias": (self.output_bias, False)}

        def linear(name, layer):
            out[name + ".weight"] = (layer.kernel, True)
            out[name + ".bias"] = (layer.bias, False)

        def norm(name, layer):
            out[name + ".weight"] = (layer.gamma, False)
            out[name + ".bias"] = (layer.beta, False)

        for side in ("encoder", "decoder"):
            for i, layer in enumerate(getattr(self, side)):
                p = f"{side}.{i}"
                for name in (["self_attn"] if side == "encoder" else ["self_attn", "cross_attn"]):
                    attn = getattr(layer, name)
                    for proj in ("q", "k", "v", "out"):
                        linear(f"{p}.{name}.{proj}", getattr(attn, proj))
                for f in ("ff1", "ff2"):
                    linear(f"{p}.{f}", getattr(layer, f))
                for n in range(1, 3 if side == "encoder" else 4):
                    norm(f"{p}.norm{n}", getattr(layer, f"norm{n}"))
            norm(side + "_norm", getattr(self, side + "_norm"))
        return out

    def load_canonical(self, weights):
        mapping = self.parameter_map()
        if mapping.keys() != weights.keys():
            raise ValueError("Canonical weight names do not match")
        for name, (var, transpose) in mapping.items():
            var.assign(weights[name].T if transpose else weights[name])

    def canonical(self):
        return {name: var.numpy().T.copy() if transpose else var.numpy().copy()
                for name, (var, transpose) in self.parameter_map().items()}

    def set_capture(self, capture):
        self.capture, self.trace = capture, {}
        for layer in self.encoder + self.decoder:
            for name in ("self_attn", "cross_attn"):
                if hasattr(layer, name):
                    a = getattr(layer, name)
                    a.capture, a.last = capture, {}

    def record(self, name, value):
        if self.capture:
            self.trace[name] = value

    def embed(self, ids, use_positions=True, training=False):
        x = tf.gather(self.embedding, ids) * math.sqrt(self.config.d_model)
        if use_positions:
            x = x + positions(tf.shape(ids)[1], self.config.d_model)[None]
        return self.dropout(x, training=training)

    def encode(self, src, training=False):
        allowed = (src != PAD)[:, None, None, :]
        x = self.embed(src, self.config.encoder_positions, training)
        self.record("src_embedding", x)
        for i, layer in enumerate(self.encoder):
            x = layer(x, allowed, training=training)
            self.record(f"encoder.{i}", x)
        x = self.encoder_norm(x)
        self.record("memory", x)
        return x, allowed

    def decode(self, target, memory, training=False):
        enc, src_mask = memory
        n = tf.shape(target)[1]
        causal = tf.range(n)[None, :] <= tf.range(n)[:, None]
        allowed = causal[None, None] & (target != PAD)[:, None, None, :]
        x = self.embed(target, training=training)
        self.record("tgt_embedding", x)
        for i, layer in enumerate(self.decoder):
            x = layer(x, enc, src_mask, allowed, training=training)
            self.record(f"decoder.{i}", x)
        x = self.decoder_norm(x)
        self.record("decoder_norm", x)
        # Match the mixed-precision Linear projection used by the PyTorch model.
        compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        logits = tf.cast(tf.matmul(tf.cast(x, compute_dtype),
                                  tf.cast(self.embedding, compute_dtype), transpose_b=True), tf.float32)
        logits = logits + self.output_bias
        self.record("logits", logits)
        return logits

    def call(self, src_ids, tgt_input_ids, training=False):
        return self.decode(tgt_input_ids, self.encode(src_ids, training), training)

    def forward(self, src_ids, tgt_input_ids):
        return self(src_ids, tgt_input_ids, training=False)

    def greedy(self, src_ids, max_new_tokens=256):
        # tf.while_loop avoids one Python/GPU round trip per generated token.
        return self._greedy_graph(src_ids, tf.constant(max_new_tokens, tf.int32))

    @tf.function(reduce_retracing=True)
    def _greedy_graph(self, src_ids, max_new_tokens):
        memory = self.encode(src_ids, training=False)
        tokens = tf.fill([tf.shape(src_ids)[0], 1], tf.cast(BOS, tf.int64))
        finished = tf.zeros([tf.shape(src_ids)[0]], tf.bool)
        blocked = tf.reduce_any(tf.equal(tf.range(self.config.vocab_size)[:, None], [PAD, BOS]), axis=1)

        def cond(step, tokens, finished):
            return (step < max_new_tokens) & ~tf.reduce_all(finished)

        def body(step, tokens, finished):
            logits = self.decode(tokens, memory, training=False)[:, -1]
            logits = tf.where(blocked[None], tf.constant(-float("inf")), logits)
            nxt = tf.argmax(logits, -1, output_type=tf.int64)
            nxt = tf.where(finished, tf.zeros_like(nxt), nxt)
            return step + 1, tf.concat([tokens, nxt[:, None]], 1), finished | (nxt == EOS)

        _, tokens, finished = tf.while_loop(cond, body, (tf.constant(0), tokens, finished),
            shape_invariants=(tf.TensorShape([]), tf.TensorShape([None, None]), tf.TensorShape([None])))
        return tokens[:, 1:], ~finished


def loss_sums(logits, labels, smoothing=0.1):
    log_probs = tf.nn.log_softmax(tf.cast(logits, tf.float32), -1)
    nll = -tf.gather(log_probs, labels, axis=2, batch_dims=2)
    uniform = -tf.reduce_mean(log_probs[..., 1:], -1)
    valid = tf.cast(labels != PAD, tf.float32)
    loss = (1 - smoothing) * nll + smoothing * uniform
    return tf.reduce_sum(loss * valid), tf.reduce_sum(nll * valid), tf.reduce_sum(valid)
