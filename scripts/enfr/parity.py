from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import ROOT, ModelConfig, load_npz, save_npz, write_json


def fixture():
    # Use differentiable test points. The original source token 6 placed one FFN
    # preactivation at 0.0 (TF) / 1.7e-7 (Torch), straddling the ReLU kink.
    # Keep the strict tolerances; do not claim gradient equality at a kink.
    return (np.array([[2, 8, 9, 3, 0], [2, 13, 7, 10, 3]], np.int64),
            np.array([[2, 11, 12, 3], [2, 7, 3, 0]], np.int64),
            np.array([[11, 12, 3, 0], [7, 3, 0, 0]], np.int64))


def export_torch(config, seed=42):
    import torch
    from .torch_model import Attention, Transformer, loss_sums
    torch.set_num_threads(1)
    model = Transformer(config, seed).cpu().eval()
    model.set_capture(True)
    src, target, labels = [torch.from_numpy(x) for x in fixture()]
    logits = model(src, target)
    loss, nll, count = loss_sums(logits, labels, 0.1)
    (loss / count).backward()
    out = {"forward/" + k: v.cpu().numpy() for k, v in model.trace.items()}
    for name, layer in model.named_modules():
        if isinstance(layer, Attention):
            out.update({"forward/" + name + "." + k: v.cpu().numpy() for k, v in layer.last.items()})
    out["forward/loss"] = np.array(float(loss / count))
    out["forward/nll"] = np.array(float(nll / count))
    for name, param in model.named_parameters():
        if param.grad is None:
            raise AssertionError(f"No gradient: {name}")
        out["gradient/" + name] = param.grad.detach().numpy().copy()
        out["sgd/" + name] = (param.detach() - 0.001 * param.grad).numpy()
    return out


def export_tensorflow(config, seed=42):
    import tensorflow as tf
    tf.keras.mixed_precision.set_global_policy("float32")
    from .tf_model import Transformer, loss_sums
    with tf.device("/CPU:0"):
        model = Transformer(config, seed)
        model.set_capture(True)
        src, target, labels = [tf.constant(x) for x in fixture()]
        mapping = model.parameter_map()
        variables = [v for v, _ in mapping.values()]
        with tf.GradientTape() as tape:
            logits = model(src, target, training=False)
            loss, nll, count = loss_sums(logits, labels, 0.1)
            objective = loss / count
        grads = tape.gradient(objective, variables)
        out = {"forward/" + k: v.numpy() for k, v in model.trace.items()}
        for side in ("encoder", "decoder"):
            for i, layer in enumerate(getattr(model, side)):
                for name in ("self_attn", "cross_attn"):
                    if hasattr(layer, name):
                        a = getattr(layer, name)
                        out.update({f"forward/{side}.{i}.{name}.{k}": v.numpy() for k, v in a.last.items()})
        out["forward/loss"] = objective.numpy()
        out["forward/nll"] = (nll / count).numpy()
        for (name, (variable, transpose)), gradient in zip(mapping.items(), grads):
            if gradient is None:
                raise AssertionError(f"No gradient: {name}")
            gradient = tf.convert_to_tensor(gradient).numpy()
            updated = variable.numpy() - 0.001 * gradient
            out["gradient/" + name] = gradient.T if transpose else gradient
            out["sgd/" + name] = updated.T if transpose else updated
        return out


def compare(left, right):
    if left.keys() != right.keys():
        raise AssertionError(f"Different stage names: {left.keys() ^ right.keys()}")
    items = []
    for name in left:
        a, b = left[name], right[name]
        if a.shape != b.shape:
            raise AssertionError(f"Shape mismatch at {name}: {a.shape} vs {b.shape}")
        rtol, atol = (1e-3, 1e-5) if name.startswith("gradient/") else (1e-4, 1e-5)
        finite = np.isfinite(a) & np.isfinite(b)
        same_nonfinite = np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(np.isposinf(a), np.isposinf(b)) and np.array_equal(np.isneginf(a), np.isneginf(b))
        ok = a.shape == b.shape and same_nonfinite and not np.isnan(a).any() and np.allclose(a, b, rtol=rtol, atol=atol)
        maximum = float(np.max(np.abs(a[finite] - b[finite]))) if np.any(finite) else 0.0
        items.append({"name": name, "shape": list(a.shape), "max_abs_error": maximum,
                      "atol": atol, "rtol": rtol, "passed": bool(ok)})
    return {"passed": all(x["passed"] for x in items), "checks": len(items), "stages": items,
            "protocol": "CPU FP32, shared canonical initialization, dropout disabled, identical smoothed token loss, SGD lr=0.001"}


def check_parity(config_path, output, seed=42, torch_python=None, tf_python=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
    for framework, python in (("torch", torch_python or sys.executable), ("tensorflow", tf_python or sys.executable)):
        subprocess.run([python, str(ROOT / "scripts/run.py"), "_parity-export", "--framework", framework,
            "--config", str(Path(config_path).resolve()), "--seed", str(seed), "--output", str(output / f"{framework}.npz")],
            check=True, env=env)
    report = compare(load_npz(output / "torch.npz"), load_npz(output / "tensorflow.npz"))
    write_json(output / "report.json", report)
    if not report["passed"]:
        bad = [x["name"] for x in report["stages"] if not x["passed"]]
        raise AssertionError(f"Cross-framework parity failed: {bad}")
    return report
