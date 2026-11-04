"""Read-only reuse of a frozen A0 checkpoint for a new decoding experiment."""
from __future__ import annotations

import contextlib
import csv
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np

from .common import (BOS, PAD, environment, read_json, read_jsonl,
                     sha256, source_manifest, write_json)
from .data import collate, verify_prepared
from .evaluation import (evaluate_model, generate, generation_protocol, load_model,
                         resolve_checkpoint, torch_context)


SCHEMA = "enfr.decode-freeze.v1"
EVALUATION_FILES = ("metrics.json", "predictions.jsonl", "hypotheses.fr.txt", "references.fr.txt")


def protect_output(output, protected):
    """Refuse overlapping roots and symlink aliases of protected input files."""
    output = Path(output).resolve()
    for path in protected:
        path = Path(path).resolve()
        if output == path or output in path.parents or path in output.parents:
            raise ValueError(f"Output overlaps protected input: {path}. Use a separate output directory.")
    if output.exists() and any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("Comparison output must not contain symlinks")


@contextlib.contextmanager
def directory_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".compare.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another comparison is already running in this output directory") from error
        try:
            lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class Stages:
    """Only fully published, checksummed stages can be skipped on restart."""
    def __init__(self, output, identity):
        self.output = Path(output)
        path = self.output / "experiment.json"
        if path.exists():
            if read_json(path) != identity:
                raise ValueError("Comparison code, inputs or settings changed; use a new output directory")
        else:
            if (self.output / "status.json").exists():
                raise ValueError("Missing experiment identity for existing status")
            write_json(path, identity)
        self.path = self.output / "status.json"
        self.state = read_json(self.path) if self.path.exists() else {"completed": {}}
        for name, record in self.state["completed"].items():
            self.verify(name, record)

    def verify(self, name, record):
        for filename, digest in record["files"].items():
            path = self.output / filename
            if self.output.resolve() not in path.resolve().parents:
                raise ValueError("Invalid stage artifact path")
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"Completed stage artifact changed: {name}: {filename}")

    def publish(self, **fields):
        self.state.update(fields, updated_at_unix=time.time())
        write_json(self.path, self.state)

    def progress(self, processed, total):
        self.publish(processed=processed, total=total)
        if processed == total or processed % 160 == 0:
            print(json.dumps({"event": "progress", "stage": self.state["stage"],
                              "processed": processed, "total": total}), flush=True)

    def run(self, name, files, action):
        if name in self.state["completed"]:
            self.verify(name, self.state["completed"][name])
            print(json.dumps({"event": "skip_complete", "stage": name}), flush=True)
            return
        self.publish(status="running", stage=name, processed=0, total=None, error=None)
        print(json.dumps({"event": "stage_start", "stage": name}), flush=True)
        try:
            action()
            hashes = {str(Path(p).relative_to(self.output)): sha256(p) for p in files}
            self.state["completed"][name] = {"files": hashes}
            self.publish(status="running")
            print(json.dumps({"event": "stage_complete", "stage": name}), flush=True)
        except BaseException as error:
            self.publish(status="interrupted", error=repr(error))
            raise


def select_beam(candidates):
    beams = [c for c in candidates if c["settings"]["decode"] == "beam"]
    return min(beams, key=lambda c: (-c["metrics"]["bleu"], -c["metrics"]["chrf2"],
                                     c["settings"]["beam_size"], c["settings"]["length_penalty"]))


def check_decode_frozen(checkpoint, data_id, manifest, protocol):
    cp = resolve_checkpoint(checkpoint)
    if (manifest["data_id"] != data_id or
            manifest["weights_sha256"] != sha256(cp / "weights.npz") or
            manifest["metadata_sha256"] != sha256(cp / "metadata.json")):
        raise ValueError("Checkpoint or dataset differs from the decoding freeze")
    if manifest["evaluation_code_id"] != source_manifest()["code_id"]:
        raise ValueError("Evaluation code differs from the decoding freeze")
    if protocol is None or protocol not in manifest["systems"].values():
        raise ValueError("Decoding parameters differ from the decoding freeze")


def benchmark_generation(model, rows, systems, precision, output, smoke=False):
    """Natural generation timing, including encoder, excluding tokenization/NLL/IO."""
    import torch
    model.eval()
    model.set_backend("manual")
    device = next(model.parameters()).device
    selected = rows[:2 if smoke else 20]
    warmups, repeats = (1, 2) if smoke else (3, 10)
    report = {"formal": not smoke, "sample_ids": [r["id"] for r in selected],
              "warmups": warmups, "repeats": repeats, "precision": precision,
              "includes": ["encoder", "autoregressive generation", "beam bookkeeping"],
              "excludes": ["tokenization", "teacher forcing", "model loading", "disk IO"],
              "token_count": "returned tokens including EOS, excluding BOS and PAD; not internal beam expansions",
              "environment": environment(), "results": []}
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for batch_size in (1, 8):
        batches = [torch.from_numpy(collate(selected[i:i + batch_size])[0]).to(device)
                   for i in range(0, len(selected), batch_size)]
        for name, settings in systems.items():
            def run():
                with torch.no_grad(), torch_context(model, precision):
                    return [generate(model, "torch", src, settings["max_new_tokens"], settings["cache"],
                                     settings["decode"], settings["beam_size"], settings["length_penalty"])
                            for src in batches]
            for _ in range(warmups):
                run()
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            records = []
            for _ in range(repeats):
                sync(); start = time.perf_counter()
                generated = run()
                sync(); elapsed = time.perf_counter() - start
                lengths, capped = [], 0
                for tokens, flags in generated:
                    lengths.extend(((tokens != PAD) & (tokens != BOS)).sum(-1).cpu().tolist())
                    capped += int(flags.sum())
                records.append({"seconds": elapsed, "generated_tokens": sum(lengths),
                                "lengths": lengths, "truncated": capped,
                                "tokens_per_second": sum(lengths) / elapsed})
                del generated
            latencies = [r["seconds"] * 1000 for r in records]
            report["results"].append({"system": name, "settings": settings, "batch_size": batch_size,
                "samples": len(selected), "latency_scope": "complete fixed sample set, not one sentence",
                "latency_ms_median": float(np.median(latencies)), "latency_ms_p95": float(np.percentile(latencies, 95)),
                "tokens_per_second": sum(r["generated_tokens"] for r in records) / sum(r["seconds"] for r in records),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "records": records})
    write_json(output, report)


def historical_difference(historical, current, historical_metrics, current_metrics):
    if not historical.is_file():
        return {"available": False, "reason": "Historical predictions were not found", "path": str(historical)}
    old, new = read_jsonl(historical), read_jsonl(current)
    if [(r["id"], r["reference"]) for r in old] != [(r["id"], r["reference"]) for r in new]:
        raise ValueError("Historical and current test sample IDs/references differ")
    changed = [a["id"] for a, b in zip(old, new) if a["hypothesis"] != b["hypothesis"]]
    result = {"available": True, "count": len(old), "changed_texts": len(changed),
              "text_match_fraction": 1 - len(changed) / max(1, len(old)), "changed_ids": changed,
              "changed_truncation_flags": sum(a["truncated"] != b["truncated"] for a, b in zip(old, new)),
              "note": "Historical greedy used no cache; current comparison uses cache for both systems."}
    if historical_metrics.is_file():
        before = read_json(historical_metrics)
        result["metrics"] = {k: {"historical": before[k], "current": current_metrics[k],
                                  "delta": current_metrics[k] - before[k]} for k in ("bleu", "chrf2", "nll", "ppl")}
    return result


def make_report(output, frozen, metrics, performance, history, bootstrap_report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ["system", "beam_size", "length_penalty", "bleu", "chrf2", "nll", "ppl",
              "word_length_ratio", "empty_fraction", "length_cap_fraction"]
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, m in metrics.items():
            writer.writerow({"system": name, **{k: m[k] for k in fields[1:]}})
    delta = metrics["beam"]["bleu"] - metrics["greedy"]["bleu"]
    summary = {"formal": frozen["formal"], "systems": frozen["systems"], "metrics": metrics,
               "bleu_delta_beam_minus_greedy": delta, "bootstrap": bootstrap_report,
               "historical_greedy": history, "performance": performance,
               "validation_candidates": frozen["validation_candidates"]}
    write_json(output / "comparison.json", summary)
    choice = frozen["systems"]["beam"]
    lines = ["# A0 seed 42：Greedy 与 Beam Search", "",
             f"正式实验：{frozen['formal']}；同一最佳验证 checkpoint，无新增训练。",
             "测试集：IWSLT2017 英法数据包，tst2010–2015 合集。",
             f"验证集选参：beam={choice['beam_size']}，length_penalty={choice['length_penalty']}。",
             "两种解码均使用手写 attention 和 KV Cache；最大生成长度与精度见冻结清单。", "",
             "| 系统 | BLEU | chrF2 | NLL | PPL | 长度比 | 达到上限比例 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['bleu']:.4f} | {m['chrf2']:.4f} | {m['nll']:.6f} | {m['ppl']:.4f} | {m['word_length_ratio']:.4f} | {m['length_cap_fraction']:.4f} |")
    lines += ["", f"Beam − greedy：{delta:+.4f} BLEU；95% paired-bootstrap 区间：{bootstrap_report['delta_ci95']}。",
              "区间反映测试句子抽样不确定性，不代表多随机种子波动。NLL/PPL 来自相同 teacher forcing，不是解码收益。",
              "Beam 未提升时照实保留结果。旧测试集成绩已经查看，本轮属于扩展实验，参数选择只使用验证集。", ""]
    for group in ("by_year", "by_length"):
        lines += [f"## {group}", "", "| 分组 | 数量 | Greedy BLEU | Beam BLEU | Greedy chrF2 | Beam chrF2 |", "|---|---:|---:|---:|---:|---:|"]
        for key, a in metrics["greedy"][group].items():
            b = metrics["beam"][group][key]
            lines.append(f"| {key} | {a['count']} | {a.get('bleu', 'NA')} | {b.get('bleu', 'NA')} | {a.get('chrf2', 'NA')} | {b.get('chrf2', 'NA')} |")
        lines.append("")
    lines += ["## 自然生成性能", "", "延迟为固定样本集合完整生成耗时，包含 Encoder，不是单句延迟；两种策略可能生成不同长度。", "",
              "| 系统 | Batch | 中位数 ms | P95 ms | 输出 token/s | 峰值显存 bytes |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in performance["results"]:
        lines.append(f"| {r['system']} | {r['batch_size']} | {r['latency_ms_median']:.2f} | {r['latency_ms_p95']:.2f} | {r['tokens_per_second']:.2f} | {r['peak_memory_bytes']} |")
    lines += ["", "## 与历史 greedy 的区别", "", json.dumps(history, ensure_ascii=False), "",
              "完整预测、指标签名、分桶长度比和异常输出比例保存在 test/*/metrics.json 与 predictions.jsonl。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_decode(checkpoint, prepared, baseline_freeze, output, device="cuda", precision="bf16",
                   dry_run=False, smoke=False):
    from .reporting import check_frozen, bootstrap
    checkpoint, prepared, baseline_freeze, output = map(Path, (checkpoint, prepared, baseline_freeze, output))
    cp = resolve_checkpoint(checkpoint).resolve()
    run = cp.parent.parent if cp.parent.name == "checkpoints" else cp.parent
    protect_output(output, (baseline_freeze.parent, run, prepared))
    audit = verify_prepared(prepared)
    meta = read_json(cp / "metadata.json")
    base = read_json(baseline_freeze)
    if base.get("schema") == SCHEMA:
        raise ValueError("--freeze must reference the original training-suite freeze")
    check_frozen(cp, audit["data_id"], baseline_freeze)
    if meta["data_id"] != audit["data_id"] or meta["tokenizer_sha256"] != audit["tokenizer_sha256"]:
        raise ValueError("Checkpoint data/tokenizer identity mismatch")
    if base["tokenizer_sha256"] != audit["tokenizer_sha256"]:
        raise ValueError("Original freeze tokenizer mismatch")
    matching = [r for r in base["runs"] if r["weights_sha256"] == sha256(cp / "weights.npz")]
    if not any(r["config"] == meta["config"] and r["code_id"] == meta["code_id"] for r in matching):
        raise ValueError("Checkpoint metadata differs from original freeze")
    if meta["framework"] != "torch" or meta["config"]["variant"] != "A0" or meta["config"]["seed"] != 42:
        raise ValueError("Comparison requires the PyTorch A0 seed 42 checkpoint")
    if smoke:
        if audit["provenance"].get("synthetic_fixture") is not True:
            raise ValueError("--smoke is restricted to synthetic fixture data")
    else:
        status = read_json(run / "status.json")
        if status["status"] != "complete" or status.get("epochs") != 30 or resolve_checkpoint(run).resolve() != cp:
            raise ValueError("Use the completed 30-epoch run's best validation checkpoint")
        if (audit["provenance"].get("dataset") != "IWSLT/iwslt2017" or audit["vocab_size"] != 8000 or
                audit["splits"]["validation"]["count"] != 890 or audit["splits"]["test"]["count"] != 8597):
            raise ValueError("Formal comparison requires the audited IWSLT2017 dataset")
        if precision != "bf16" or not device.startswith("cuda"):
            raise ValueError("Formal comparison requires CUDA BF16; CPU smoke runs are non-formal")
    evaluation = meta["config"]["evaluation"]
    if not smoke and evaluation["max_new_tokens"] != 256:
        raise ValueError("Formal comparison requires the frozen 256-token generation limit")
    common = {**evaluation, "precision": precision, "use_cache": True}
    grid = [{"name": "greedy", "settings": generation_protocol(**common)}]
    grid += [{"name": f"beam{k}-lp{alpha:g}", "settings": generation_protocol(
        "beam", k, alpha, **common)} for k in (4, 8) for alpha in (0.0, 0.6, 1.0)]
    historical = run / "test/predictions.jsonl"
    protected_files = [cp / "weights.npz", cp / "metadata.json", baseline_freeze,
                       run / "best.json", run / "latest.json", run / "status.json", run / "config.json",
                       historical, run / "test/metrics.json", baseline_freeze.parent / "report/experiments.csv"]
    snapshot = {str(p.resolve()): sha256(p) for p in protected_files if p.is_file()}
    identity = {"schema": "enfr.decode-comparison.v1", "formal": not smoke, "checkpoint": str(cp),
                "data_id": audit["data_id"], "tokenizer_sha256": audit["tokenizer_sha256"],
                "splits": audit["splits"], "metadata_sha256": sha256(cp / "metadata.json"),
                "weights_sha256": sha256(cp / "weights.npz"), "baseline_freeze_sha256": sha256(baseline_freeze),
                "evaluation_code_id": source_manifest()["code_id"], "device": device, "grid": grid,
                "attention_backend": "manual", "protected_files": snapshot,
                "benchmark": {"samples": 2 if smoke else 20, "batches": [1, 8],
                              "warmups": 1 if smoke else 3, "repeats": 2 if smoke else 10},
                "bootstrap_repetitions": 20 if smoke else 1000}
    if dry_run:
        return {"dry_run": True, "experiment": identity,
                "stages": [f"validation/{r['name']}" for r in grid] +
                          ["freeze", "test/greedy", "test/beam", "bootstrap", "benchmark", "history", "report"],
                "output": str(output), "training_runs": 0}
    if not smoke:
        from .runtime import doctor
        doctor("torch", require_gpu=True)
    with directory_lock(output):
        stages = Stages(output, identity)
        if stages.state.get("status") == "complete":
            return {"status": "complete", "resumed": True, "output": str(output)}
        import sentencepiece as spm
        model, _, _ = load_model(cp, "torch", device, precision)
        model.set_backend("manual")
        sp = spm.SentencePieceProcessor(model_file=str(prepared / "tokenizer.model"))
        validation = read_jsonl(prepared / "validation.jsonl")

        def evaluate(split, name, settings, rows):
            dest = output / split / name
            def action():
                m, _ = evaluate_model(model, "torch", rows, sp, batch_size=settings["batch_size"],
                    max_new_tokens=settings["max_new_tokens"], precision=settings["precision"],
                    use_cache=settings["cache"], decode=settings["decode"], beam_size=settings["beam_size"],
                    length_penalty=settings["length_penalty"], output=dest, progress=stages.progress)
                m.update(checkpoint=str(cp), data_id=audit["data_id"], formal=not smoke,
                         weights_sha256=identity["weights_sha256"], evaluation_code_id=identity["evaluation_code_id"])
                write_json(dest / "metrics.json", m)
            stages.run(f"{split}/{name}", [dest / f for f in EVALUATION_FILES], action)
            return read_json(dest / "metrics.json")

        candidates = [{**r, "metrics": evaluate("validation", r["name"], r["settings"], validation)} for r in grid]
        best = select_beam(candidates)
        systems = {"greedy": grid[0]["settings"], "beam": best["settings"]}
        frozen = {"schema": SCHEMA, "formal": not smoke, "data_id": audit["data_id"],
                  "tokenizer_sha256": audit["tokenizer_sha256"], "weights_sha256": identity["weights_sha256"],
                  "metadata_sha256": identity["metadata_sha256"], "baseline_freeze_sha256": identity["baseline_freeze_sha256"],
                  "training_code_id": meta["code_id"], "evaluation_code_id": identity["evaluation_code_id"],
                  "systems": systems, "attention_backend": "manual", "validation_candidates": candidates,
                  "validation_sample_ids": [r["id"] for r in validation],
                  "selection": "validation BLEU desc, chrF2 desc, beam size asc, length penalty asc",
                  "test_protocol": "tst2010-2015 pooled; BLEU 13a mixed case; chrF2; no test-based selection"}
        freeze_path = output / "decode-freeze.json"
        stages.run("freeze", [freeze_path], lambda: write_json(freeze_path, frozen))
        if read_json(freeze_path) != frozen:
            raise ValueError("Frozen selection differs from validation artifacts")
        test = read_jsonl(prepared / "test.jsonl")
        metrics = {}
        for name, settings in systems.items():
            check_frozen(cp, audit["data_id"], freeze_path, settings)
            metrics[name] = evaluate("test", name, settings, test)
        for key in ("nll", "ppl", "target_tokens"):
            if not np.isclose(metrics["greedy"][key], metrics["beam"][key], atol=1e-7, rtol=1e-7):
                raise AssertionError(f"Teacher-forcing {key} differs between decoding systems")
        report = output / "report"
        boot_path = report / "bootstrap-greedy-beam.json"
        stages.run("bootstrap", [boot_path], lambda: bootstrap(output / "test/greedy/predictions.jsonl",
            output / "test/beam/predictions.jsonl", boot_path, identity["bootstrap_repetitions"]))
        perf_path = output / "benchmark.json"
        stages.run("benchmark", [perf_path], lambda: benchmark_generation(model, validation, systems, precision, perf_path, smoke))
        history_path = output / "historical-greedy.json"
        stages.run("history", [history_path], lambda: write_json(history_path, historical_difference(
            historical, output / "test/greedy/predictions.jsonl", run / "test/metrics.json", metrics["greedy"])))
        stages.run("report", [report / f for f in ("comparison.csv", "comparison.json", "report.md")],
                   lambda: make_report(report, frozen, metrics, read_json(perf_path), read_json(history_path), read_json(boot_path)))
        if any(not Path(p).is_file() or sha256(p) != digest for p, digest in snapshot.items()):
            raise RuntimeError("A protected baseline input changed during evaluation")
        stages.publish(status="complete", stage="complete", old_results_unchanged=True)
        return {"status": "complete", "output": str(output), "selected_beam": systems["beam"],
                "bleu_delta_beam_minus_greedy": metrics["beam"]["bleu"] - metrics["greedy"]["bleu"]}
