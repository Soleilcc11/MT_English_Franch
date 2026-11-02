from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np

from .common import BOS, environment, write_json
from .evaluation import load_model, torch_context, translate


def benchmark(checkpoint, output, device="cuda", smoke=False, prepared=None):
    import torch
    model, metadata, _ = load_model(checkpoint, "torch", device)
    if not smoke and (device == "cpu" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("The formal BF16 benchmark requires a CUDA GPU. --smoke labels CPU results as non-formal.")
    if not smoke and (metadata["config"]["variant"] != "A0" or metadata["config"]["seed"] != 42):
        raise ValueError("Formal benchmark requires the A0 seed 42 checkpoint")
    precision = "fp32" if smoke else "bf16"
    warmups, repeats = (1, 3) if smoke else (20, 50)
    shapes = [(1, 8, 8)] if smoke else list(itertools.product((1, 8), (32, 128), (32, 128)))
    report = {"formal": not smoke, "environment": environment(), "device": device, "precision": precision,
              "checkpoint": str(Path(checkpoint).resolve()), "data_id": metadata["data_id"],
              "warmups": warmups, "repeats": repeats, "fixed_trajectory": True, "results": []}
    if device.startswith("cuda"):
        report["gpu"] = torch.cuda.get_device_name()
        report["cuda"] = torch.version.cuda
    rng = np.random.default_rng(2026)

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    @torch.no_grad()
    def run(src, target, use_cache):
        with torch_context(model, precision):
            sync(); begin = time.perf_counter()
            memory = model.encode(src)
            sync(); encoded = time.perf_counter()
            if use_cache:
                logits, cache = model.decode_step(target[:, 0], memory)
            else:
                logits, cache = model.decode(target[:, :1], memory)[:, -1], None
            sync(); first = time.perf_counter()
            for step in range(1, target.shape[1]):
                if use_cache:
                    logits, cache = model.decode_step(target[:, step], memory, cache)
                else:
                    logits = model.decode(target[:, :step + 1], memory)[:, -1]
            sync(); end = time.perf_counter()
        return {"encoder_ms": (encoded - begin) * 1000,
                "first_decode_ms": (first - encoded) * 1000,
                "ttft_ms": (first - begin) * 1000,
                "remaining_decode_ms": (end - first) * 1000,
                "total_ms": (end - begin) * 1000,
                "tokens_per_second": src.shape[0] * target.shape[1] / (end - encoded)}, logits

    for batch, source_len, target_len in shapes:
        src = torch.from_numpy(rng.integers(4, model.config.vocab_size, (batch, source_len))).to(device)
        target = torch.from_numpy(rng.integers(4, model.config.vocab_size, (batch, target_len))).to(device)
        src[:, 0], src[:, -1], target[:, 0] = BOS, 3, BOS
        reference = None
        for backend, cached in itertools.product(("manual", "sdpa"), (False, True)):
            model.set_backend(backend)
            for _ in range(warmups):
                run(src, target, cached)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            records = []
            for _ in range(repeats):
                timings, logits = run(src, target, cached)
                records.append(timings)
            array = logits.float().cpu().numpy()
            if reference is None:
                reference = array
            item = {"backend": backend, "cache": cached, "batch": batch,
                    "source_tokens": source_len, "generated_tokens": target_len,
                    "max_abs_last_logits_diff": float(np.abs(reference - array).max()),
                    "last_token_agreement": float(np.mean(reference.argmax(-1) == array.argmax(-1))),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None}
            for key in records[0]:
                values = [r[key] for r in records]
                item[key] = {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
            item["sdpa_operators"] = []
            if backend == "sdpa":
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    run(src, target, cached)
                item["sdpa_operators"] = sorted({e.key for e in prof.key_averages()
                    if any(term in e.key.lower() for term in ("scaled_dot", "flash_attention", "efficient_attention"))})
            report["results"].append(item)
            write_json(output, report)

    # Natural, EOS-terminated translations are reported separately from replay latency.
    if prepared:
        import sentencepiece as spm
        from .common import read_jsonl
        data = Path(prepared)
        sp = spm.SentencePieceProcessor(model_file=str(data / "tokenizer.model"))
        rows = read_jsonl(data / "validation.jsonl")[:(2 if smoke else 20)]
        natural, reference_text = [], None
        for backend, cached in itertools.product(("manual", "sdpa"), (False, True)):
            model.set_backend(backend)
            outputs, times = [], []
            for row in rows:
                # Warm actual shape, then time; tokenizer cost is excluded from the replay matrix above.
                translate(model, "torch", sp, row["src"], 24 if smoke else 256, precision, cached)
                sync(); start = time.perf_counter()
                out = translate(model, "torch", sp, row["src"], 24 if smoke else 256, precision, cached)
                sync(); times.append((time.perf_counter() - start) * 1000)
                outputs.append(out)
            texts = [x["translation"] for x in outputs]
            if reference_text is None:
                reference_text = texts
            natural.append({"backend": backend, "cache": cached, "count": len(rows),
                "text_match_fraction": float(np.mean([a == b for a, b in zip(reference_text, texts)])),
                "latency_ms_median": float(np.median(times)), "latency_ms_p95": float(np.percentile(times, 95)),
                "includes_tokenization": True, "outputs": outputs})
        report["natural_generation"] = natural
        write_json(output, report)
    return report
