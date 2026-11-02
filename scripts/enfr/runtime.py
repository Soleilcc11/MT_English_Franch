from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import environment, read_jsonl, write_json
from .data import batch_manifest, verify_prepared


def doctor(framework, require_gpu=False):
    result = environment()
    result["framework"] = framework
    if framework == "torch":
        import torch
        result.update(cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda)
        if torch.cuda.is_available():
            result.update(gpu=torch.cuda.get_device_name(), bf16=torch.cuda.is_bf16_supported(),
                          vram_bytes=torch.cuda.get_device_properties(0).total_memory)
        ready = result["cuda_available"] and result.get("bf16", False)
    else:
        import tensorflow as tf
        result.update(gpu_devices=[str(x) for x in tf.config.list_physical_devices("GPU")],
                      build=tf.sysconfig.get_build_info())
        ready = bool(result["gpu_devices"])
    result["gpu_ready"] = ready
    if require_gpu and not ready:
        raise RuntimeError(f"{framework}: no suitable GPU visible. Diagnostic: {result}")
    return result


def profile_training(config, prepared, framework, device, output, warmup=20, measured=200):
    from .training import TorchTrainer, TensorFlowTrainer
    info = doctor(framework, require_gpu=True)
    audit = verify_prepared(prepared)
    rows = read_jsonl(Path(prepared) / "train.jsonl")
    tc = config["training"]
    manifest = batch_manifest(rows, Path(prepared) / "manifests", audit["data_id"], config["seed"], 0,
                              tc["effective_batch"], config["data"]["bucket_window"])
    if warmup + measured > len(manifest):
        raise ValueError("Not enough distinct batches for this profiling protocol")
    trainer = (TorchTrainer if framework == "torch" else TensorFlowTrainer)(config, device)
    records = []
    for i, batch in enumerate(trainer.iter_batches(rows, manifest[:warmup + measured])):
        start = time.perf_counter()
        loss, nll, count, norm = trainer.update(batch, tc["lr"])
        if framework == "torch":
            trainer.torch.cuda.synchronize()
        # TF update returns .numpy()-materialized scalar gradients and loss.
        elapsed = time.perf_counter() - start
        if i >= warmup:
            records.append({"seconds": elapsed, "tokens": count})
    mean = float(np.mean([r["seconds"] for r in records]))
    result = {"environment": info, "data_id": audit["data_id"], "config": config,
        "warmup_updates": warmup, "measured_updates": measured, "mean_update_seconds": mean,
        "target_tokens_per_second": sum(r["tokens"] for r in records) / sum(r["seconds"] for r in records),
        "peak_memory_bytes": trainer.memory(), "estimated_training_only_hours_per_run": mean * len(manifest) * tc["epochs"] / 3600,
        "excludes": ["validation decoding", "checkpoint IO", "test evaluation", "other framework/architecture timings"],
        "records": records}
    write_json(output, result)
    return result
