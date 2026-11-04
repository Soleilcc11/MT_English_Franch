from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAD, UNK, BOS, EOS = 0, 1, 2, 3


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    d_model: int = 256
    heads: int = 4
    d_ff: int = 1024
    encoder_layers: int = 4
    decoder_layers: int = 4
    dropout: float = 0.1
    norm_eps: float = 1e-5
    encoder_positions: bool = True
    encoder_relative_positions: bool = False
    relative_max_distance: int = 16

    def __post_init__(self):
        if self.d_model % self.heads or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by heads")
        if min(self.encoder_layers, self.decoder_layers) < 1:
            raise ValueError("At least one encoder and decoder layer is required")
        if self.relative_max_distance < 1 or not isinstance(self.relative_max_distance, int):
            raise ValueError("relative_max_distance must be a positive integer")
        if self.encoder_relative_positions and self.encoder_positions:
            raise ValueError("A4 replaces encoder absolute positions with relative positions")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def source_manifest():
    files = sorted((ROOT / "scripts/enfr").glob("*.py")) + [ROOT / "scripts/run.py"]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    return {"files": hashes, "code_id": fingerprint(hashes)}


def load_config(path=None, variant="A0", seed=42, rdrop_alpha=None):
    c = read_json(path or ROOT / "scripts/configs/base.json")
    c["variant"], c["seed"] = variant, int(seed)
    if variant == "A1":
        c["model"]["heads"] = 1
    elif variant == "A2":
        c["model"]["encoder_positions"] = False
    elif variant == "A4":
        c["model"].update(encoder_positions=False, encoder_relative_positions=True, relative_max_distance=16)
    elif variant == "A5":
        c["training"]["rdrop_alpha"] = float(rdrop_alpha if rdrop_alpha is not None else c["training"].get("rdrop_alpha", 1.0))
        if not math.isfinite(c["training"]["rdrop_alpha"]) or c["training"]["rdrop_alpha"] <= 0:
            raise ValueError("A5 requires a finite positive rdrop_alpha")
    elif variant not in ("A0", "A3"):
        raise ValueError(f"Unknown variant: {variant}")
    if rdrop_alpha is not None and variant != "A5":
        raise ValueError("--rdrop-alpha is only valid for A5")
    ModelConfig(**c["model"])
    if c["training"]["effective_batch"] % c["training"]["micro_batch"]:
        raise ValueError("effective_batch must be divisible by micro_batch")
    return c


def learning_rate(step, total_steps, cfg):
    """step is the 1-based optimizer update number."""
    warmup = cfg["warmup_steps"]
    if step <= warmup:
        return cfg["lr"] * step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress))


def canonical_weights(cfg: ModelConfig, seed=42):
    """Framework-independent initialization; Linear kernels use [out, in]."""
    rng = np.random.default_rng(seed)
    d, ff = cfg.d_model, cfg.d_ff
    w = {"embedding": rng.normal(0, d ** -0.5, (cfg.vocab_size, d)).astype("float32"),
         "output_bias": np.zeros(cfg.vocab_size, "float32")}

    def linear(name, inp, out):
        bound = math.sqrt(6 / (inp + out))
        w[name + ".weight"] = rng.uniform(-bound, bound, (out, inp)).astype("float32")
        w[name + ".bias"] = np.zeros(out, "float32")

    def norm(name):
        w[name + ".weight"] = np.ones(d, "float32")
        w[name + ".bias"] = np.zeros(d, "float32")

    for side, count in (("encoder", cfg.encoder_layers), ("decoder", cfg.decoder_layers)):
        for i in range(count):
            p = f"{side}.{i}"
            for attn in (["self_attn"] if side == "encoder" else ["self_attn", "cross_attn"]):
                for proj in ("q", "k", "v", "out"):
                    linear(f"{p}.{attn}.{proj}", d, d)
            linear(p + ".ff1", d, ff)
            linear(p + ".ff2", ff, d)
            for j in range(2 if side == "encoder" else 3):
                norm(f"{p}.norm{j + 1}")
        norm(side + "_norm")
    # Append new draws only AFTER all baseline weights. Matching seeds retain
    # bit-identical shared parameters for A0/A4/A5.
    if cfg.encoder_relative_positions:
        dk = cfg.d_model // cfg.heads
        for i in range(cfg.encoder_layers):
            for kind in ("relative_k", "relative_v"):
                w[f"encoder.{i}.self_attn.{kind}"] = rng.normal(
                    0, dk ** -0.5, (2 * cfg.relative_max_distance + 1, dk)).astype("float32")
    return w


def save_npz(path, weights):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def environment():
    import importlib.metadata as md
    versions = {}
    for name in ("torch", "tensorflow", "tensorflow-macos", "numpy", "sentencepiece", "sacrebleu"):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def length_bucket(n):
    for upper in (16, 32, 64, 128):
        if n <= upper:
            return {16: "01_<=16", 32: "02_17-32", 64: "03_33-64", 128: "04_65-128"}[upper]
    return "05_>128"
