"""A4/A5 extension queue. Legacy suites are read-only and never resumed here."""
from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .common import (ROOT, ModelConfig, load_config, read_json, read_jsonl, sha256,
                     source_manifest, write_json)
from .data import verify_prepared
from .decode_comparison import EVALUATION_FILES, Stages, directory_lock, protect_output
from .evaluation import evaluate_model, generation_protocol, load_model, resolve_checkpoint

SCHEMA = "enfr.extension-freeze.v1"
ALPHAS = (1.0, 3.0, 5.0)
SEEDS = (42, 43, 44)


def extension_name(variant, seed, alpha=None):
    return f"{variant}-torch-seed{seed}" if variant == "A4" else f"A5-alpha{alpha:g}-torch-seed{seed}"


def select_alpha(candidates):
    """Only validation scores; a tie favors lower NLL, then smaller alpha."""
    return min(candidates, key=lambda r: (-r["validation_bleu"], r["validation_nll"], r["alpha"]))


def comparable(config, variant_adjustments=True):
    c = copy.deepcopy(config)
    c.pop("variant", None)
    c.pop("seed", None)
    c["model"] = asdict(ModelConfig(**c["model"]))
    if not variant_adjustments:
        c["training"].setdefault("rdrop_alpha", 0.0)
        return c
    c["model"].update(encoder_positions=True, encoder_relative_positions=False, relative_max_distance=16)
    c["training"].pop("rdrop_alpha", None)
    return c


def file_snapshot(paths):
    return {str(Path(p).resolve()): sha256(p) for p in paths if Path(p).is_file()}


def verify_snapshot(snapshot):
    if any(not Path(p).is_file() or sha256(p) != digest for p, digest in snapshot.items()):
        raise ValueError("A checksummed input or completed training artifact changed")


def check_extension_frozen(checkpoint, data_id, manifest, protocol):
    cp = resolve_checkpoint(checkpoint)
    entry = next((r for r in manifest["runs"] if r["weights_sha256"] == sha256(cp / "weights.npz")
                  and r["metadata_sha256"] == sha256(cp / "metadata.json")), None)
    if entry is None or manifest["data_id"] != data_id:
        raise ValueError("Checkpoint/metadata/data differs from extension freeze")
    if manifest["evaluation_code_id"] != source_manifest()["code_id"] or protocol != manifest["protocol"]:
        raise ValueError("Evaluation source or decoding protocol differs from extension freeze")


def validate_run(run, config, audit, code_id):
    status = read_json(run / "status.json")
    cp = resolve_checkpoint(run)
    meta = read_json(cp / "metadata.json")
    if (status.get("status") != "complete" or status.get("epochs") != config["training"]["epochs"]
            or meta["framework"] != "torch" or meta["config"] != config
            or meta["data_id"] != audit["data_id"] or meta["tokenizer_sha256"] != audit["tokenizer_sha256"]
            or meta.get("code_id") != code_id):
        raise ValueError(f"Completed run identity/config/code mismatch: {run}")
    return cp, meta


def checkpoint_entry(name, run, config, audit, code_id):
    cp, meta = validate_run(run, config, audit, code_id)
    return {"run": name, "checkpoint": str(cp.resolve()), "config": config,
            "weights_sha256": sha256(cp / "weights.npz"), "metadata_sha256": sha256(cp / "metadata.json"),
            "training_code_id": meta["code_id"], "parameter_count": meta["parameter_count"],
            "validation_bleu": meta["state"]["best_bleu"], "validation_nll": meta["state"]["best_nll"],
            "status": read_json(run / "status.json")}


def write_report(output, frozen, metrics, candidates, history):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in frozen["runs"]:
        c, name = entry["config"], entry["run"]
        m, status = metrics[name], entry["status"]
        rows.append({"run": name, "variant": c["variant"], "seed": c["seed"],
                     "rdrop_alpha": c["training"].get("rdrop_alpha", 0),
                     "parameter_count": entry["parameter_count"], "training_seconds": status.get("training_seconds"),
                     "peak_memory_bytes": status.get("peak_memory_bytes"),
                     **{k: m[k] for k in ("bleu", "chrf2", "nll", "ppl", "word_length_ratio",
                                          "empty_fraction", "length_cap_fraction")}})
    fields = list(rows[0])
    with (output / "experiments.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    for variant in ("A0", "A4", "A5"):
        group = [r for r in rows if r["variant"] == variant]
        aggregate[variant] = {k: {"mean": float(np.mean([r[k] for r in group])),
                                  "std": float(np.std([r[k] for r in group], ddof=1))}
                              for k in ("bleu", "chrf2", "nll", "ppl")}
    summary = {"formal": frozen["formal"], "experiments": rows, "aggregate": aggregate,
               "a5_selection": frozen["selection"], "a5_candidates": candidates,
               "test_metrics": metrics, "historical_a0": history,
               "new_training_runs": 8, "a5_total_training_runs": 5,
               "a5_candidate_training_seconds": sum(r["status"]["training_seconds"] for r in candidates)}
    write_json(output / "summary.json", summary)
    with (output / "a5-tuning.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "validation_bleu", "validation_nll", "selected"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in ("alpha", "validation_bleu", "validation_nll")} |
                         {"selected": r["alpha"] == frozen["selection"]["alpha"]} for r in candidates)
    lines = ["# A4 相对位置注意力 / A5 R-Drop 扩展实验", "",
             f"正式实验：{frozen['formal']}。A5 选中 α={frozen['selection']['alpha']:g}。",
             "新增训练共 8 次：A4 三种子；A5 seed 42 三个候选，选定后补跑 seed 43、44。",
             "A0 只复用原最佳 checkpoint，在当前评估代码下重算 greedy；旧结果未覆盖。",
             "IWSLT2017 英法 tst2010–2015 合集；完整年份、长度桶和指标签名见 summary.json/test_metrics。",
             "本轮在已查看旧测试成绩之后设计，属于扩展实验；选参只使用验证集，不能当作全新盲测。",
             "", "| 实验 | BLEU 均值 ± 样本标准差 | chrF2 均值 ± 标准差 |",
             "|---|---:|---:|"]
    for variant, a in aggregate.items():
        lines.append(f"| {variant} | {a['bleu']['mean']:.4f} ± {a['bleu']['std']:.4f} | "
                     f"{a['chrf2']['mean']:.4f} ± {a['chrf2']['std']:.4f} |")
    for variant in ("A4", "A5"):
        b = read_json(output / f"bootstrap-A0-{variant}.json")
        lines += ["", f"{variant} − A0（seed 42）：{b['delta']:+.4f} BLEU，95% 区间 {b['delta_ci95']}。"]
    lines += ["", "区间为句子配对抽样，不能代替训练种子不确定性或 TED 文档聚类抽样。",
              "A4 仅替换 Encoder 的位置机制，增加相对 K/V 参数；不声称参数量完全一致。",
              "A5 使用两次 dropout 前向、平均交叉熵 + α × 对称 KL；NLL/PPL 仅来自单次推理 teacher forcing。",
              "A5 搜索的额外训练成本单独保留，不能据此声称与 A0 等计算预算。",
              "不预设 BLEU 提升；error_review.jsonl 中的语言学错误需要人工标注。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_extensions(baseline_runs, baseline_freeze, prepared, output, config_path=None,
                   device="cuda", torch_python=None, dry_run=False, finalize=False, smoke=False):
    from .reporting import check_frozen, bootstrap, error_review
    baseline_runs, baseline_freeze, prepared, output = map(
        lambda p: Path(p).resolve(), (baseline_runs, baseline_freeze, prepared, output))
    # Do not resolve the venv Python symlink: that would launch the base Python
    # and silently lose the selected environment and its installed packages.
    python = str(Path(torch_python or sys.executable).expanduser().absolute())
    protect_output(output, (baseline_runs.parent, baseline_freeze.parent, prepared))
    audit = verify_prepared(prepared)
    base = read_json(baseline_freeze)
    if base.get("schema") or base["data_id"] != audit["data_id"] or base["tokenizer_sha256"] != audit["tokenizer_sha256"]:
        raise ValueError("Use the original A0-A3 freeze and its audited dataset")
    if smoke:
        if audit["provenance"].get("synthetic_fixture") is not True:
            raise ValueError("--smoke is restricted to synthetic fixture data")
    elif (audit["provenance"].get("dataset") != "IWSLT/iwslt2017" or audit["vocab_size"] != 8000
          or audit["splits"]["validation"]["count"] != 890 or audit["splits"]["test"]["count"] != 8597):
        raise ValueError("Formal extensions require the audited IWSLT2017 splits and 8000-token vocabulary")
    reference = load_config(config_path, "A0", 42)
    if not smoke and (reference["training"]["epochs"] != 30 or reference["training"]["precision"] != "bf16"
                      or reference["evaluation"]["max_new_tokens"] != 256 or not device.startswith("cuda")):
        raise ValueError("Formal extensions require 30 epochs, CUDA BF16, and 256-token generation")
    baselines = []
    for seed in SEEDS:
        run = baseline_runs / f"A0-torch-seed{seed}"
        cp = resolve_checkpoint(run)
        meta = read_json(cp / "metadata.json")
        check_frozen(cp, audit["data_id"], baseline_freeze)
        if (comparable(meta["config"], False) != comparable(reference, False) or meta["config"]["variant"] != "A0"
                or meta["config"]["seed"] != seed):
            raise ValueError("A0 baseline config differs; use the original base config, including micro-batch")
        if not any(e["weights_sha256"] == sha256(cp / "weights.npz") and e["config"] == meta["config"]
                   and e["code_id"] == meta["code_id"] for e in base["runs"]):
            raise ValueError("Baseline metadata does not match original freeze")
        # Exact legacy config is accepted, including omitted newly added defaults.
        baselines.append(checkpoint_entry(f"A0-torch-seed{seed}", run, meta["config"], audit, meta["code_id"]))
    protected = [baseline_freeze]
    for pattern in ("runs/*/*.json", "runs/*/test/*", "runs/*/checkpoints/*/metadata.json",
                    "runs/*/checkpoints/*/weights.npz", "report/*"):
        protected.extend(p for p in baseline_runs.parent.glob(pattern) if p.is_file())
    snapshot = file_snapshot(protected)
    code_id = source_manifest()["code_id"]
    plan = [{"variant": "A4", "seed": s, "alpha": None} for s in SEEDS]
    plan += [{"variant": "A5", "seed": 42, "alpha": a} for a in ALPHAS]
    plan += [{"variant": "A5", "seed": s, "alpha": "validation-selected"} for s in (43, 44)]
    identity = {"schema": "enfr.extensions.v1", "formal": not smoke, "data_id": audit["data_id"],
                "tokenizer_sha256": audit["tokenizer_sha256"], "splits": audit["splits"],
                "baseline_freeze_sha256": sha256(baseline_freeze), "protected_files": snapshot,
                "source_code_id": code_id, "config": reference, "plan": plan, "device": device,
                "selection": "validation BLEU desc, NLL asc, alpha asc", "torch_python": python}
    if dry_run:
        return {"dry_run": True, "new_training_runs": 8, "plan": plan, "finalize": finalize,
                "output": str(output), "baseline_runs": [r["run"] for r in baselines]}
    if not smoke:
        from .runtime import doctor
        doctor("torch", require_gpu=True)
    with directory_lock(output):
        stages = Stages(output, identity)
        if stages.state.get("status") == "complete":
            for receipt in (output / "receipts").glob("*.json"):
                verify_snapshot(read_json(receipt)["files"])
            return {"status": "complete", "resumed": True, "output": str(output),
                    "selected_alpha": read_json(output / "a5-selection.json")["alpha"]}

        def train_one(variant, seed, alpha=None):
            c = load_config(config_path, variant, seed, alpha)
            if comparable(c) != comparable(reference):
                raise ValueError("Base configuration changed during the queue")
            name = extension_name(variant, seed, alpha)
            run = output / "runs" / name
            receipt = output / "receipts" / (name + ".json")

            def action():
                if not (run / "status.json").exists() or read_json(run / "status.json").get("status") != "complete":
                    command = [python, str(ROOT / "scripts/run.py"), "train", "--framework", "torch",
                               "--variant", variant, "--seed", str(seed), "--prepared", str(prepared),
                               "--run", str(run), "--device", device]
                    if config_path:
                        command += ["--config", str(Path(config_path).resolve())]
                    if alpha is not None:
                        command += ["--rdrop-alpha", str(alpha)]
                    if (run / "latest.json").exists():
                        command += ["--resume", str(run / "latest.json")]
                    print(json.dumps({"event": "train_command", "command": command}), flush=True)
                    subprocess.run(command, check=True)
                validate_run(run, c, audit, code_id)
                write_json(receipt, {"files": file_snapshot(p for p in run.rglob("*") if p.is_file())})
            stages.run("train/" + name, [receipt], action)
            verify_snapshot(read_json(receipt)["files"])
            return checkpoint_entry(name, run, c, audit, code_id)

        a4 = [train_one("A4", s) for s in SEEDS]
        candidates = [{**train_one("A5", 42, a), "alpha": a} for a in ALPHAS]
        selected = select_alpha(candidates)
        selection_path = output / "a5-selection.json"
        selection = {"alpha": selected["alpha"], "seed42_run": selected["run"], "candidates": candidates,
                     "rule": identity["selection"], "uses_test": False}
        stages.run("select-alpha", [selection_path], lambda: write_json(selection_path, selection))
        if read_json(selection_path) != selection:
            raise ValueError("Selected alpha differs from completed validation artifacts")
        a5 = [selected] + [train_one("A5", s, selected["alpha"]) for s in (43, 44)]
        if not finalize:
            verify_snapshot(snapshot)
            stages.publish(status="training_complete", stage="awaiting_finalize", training_runs=8)
            return {"status": "training_complete", "selected_alpha": selected["alpha"], "output": str(output)}
        protocol = generation_protocol(**reference["evaluation"], precision=reference["training"]["precision"], use_cache=False)
        frozen = {"schema": SCHEMA, "formal": not smoke, "data_id": audit["data_id"],
                  "tokenizer_sha256": audit["tokenizer_sha256"], "evaluation_code_id": code_id,
                  "baseline_freeze_sha256": sha256(baseline_freeze), "protocol": protocol,
                  "runs": baselines + a4 + a5, "selection": selection,
                  "test_protocol": "IWSLT2017 en-fr; tst2010-2015 pooled; greedy; BLEU 13a mixed case; chrF2"}
        freeze_path = output / "extension-freeze.json"
        stages.run("freeze", [freeze_path], lambda: write_json(freeze_path, frozen))
        if read_json(freeze_path) != frozen:
            raise ValueError("Extension freeze differs from completed training and selection")
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(prepared / "tokenizer.model"))
        test = read_jsonl(prepared / "test.jsonl")
        metrics = {}
        for entry in frozen["runs"]:
            name, cp = entry["run"], Path(entry["checkpoint"])
            check_extension_frozen(cp, audit["data_id"], frozen, protocol)
            dest = output / "test" / name
            def evaluate():
                model, _, _ = load_model(cp, "torch", device, protocol["precision"])
                model.set_backend("manual")
                m, _ = evaluate_model(model, "torch", test, sp, **reference["evaluation"],
                    precision=protocol["precision"], use_cache=False, output=dest, progress=stages.progress)
                m.update(formal=not smoke, checkpoint=str(cp), data_id=audit["data_id"],
                         weights_sha256=entry["weights_sha256"], evaluation_code_id=code_id)
                write_json(dest / "metrics.json", m)
            stages.run("test/" + name, [dest / f for f in EVALUATION_FILES], evaluate)
            metrics[name] = read_json(dest / "metrics.json")
        report = output / "report"
        paths = {"A0": output / "test/A0-torch-seed42/predictions.jsonl",
                 "A4": output / "test/A4-torch-seed42/predictions.jsonl",
                 "A5": output / "test" / selected["run"] / "predictions.jsonl"}
        for variant in ("A4", "A5"):
            dest = report / f"bootstrap-A0-{variant}.json"
            stages.run("bootstrap/" + variant, [dest],
                       lambda v=variant, d=dest: bootstrap(paths["A0"], paths[v], d, 20 if smoke else 1000))
        review = report / "error_review.jsonl"
        def review_action():
            # An interrupted stage may have published the JSONL before its receipt.
            # Reuse existing labels; never replace manually reviewed content.
            if not review.exists():
                error_review(paths, review)
            elif not review.with_suffix(".sampling.json").exists():
                raise ValueError("Incomplete error review output; preserve it and inspect before retry")
        stages.run("error-review", [review.with_suffix(".sampling.json")], review_action)
        history_path = output / "historical-a0.json"
        def compare_history():
            history = {}
            for s in SEEDS:
                name = f"A0-torch-seed{s}"
                old = baseline_runs / name / "test/predictions.jsonl"
                current = read_jsonl(output / "test" / name / "predictions.jsonl")
                if old.exists():
                    previous = read_jsonl(old)
                    if [(r["id"], r["reference"]) for r in previous] != [(r["id"], r["reference"]) for r in current]:
                        raise ValueError("Historical A0 test IDs/references differ")
                    history[name] = {"available": True, "count": len(current),
                                     "changed_texts": sum(a["hypothesis"] != b["hypothesis"] for a, b in zip(previous, current))}
                else:
                    history[name] = {"available": False}
            write_json(history_path, history)
        stages.run("history", [history_path], compare_history)
        stages.run("report", [report / f for f in ("experiments.csv", "summary.json", "a5-tuning.csv", "report.md")],
                   lambda: write_report(report, frozen, metrics, candidates, read_json(history_path)))
        verify_snapshot(snapshot)
        stages.publish(status="complete", stage="complete", old_results_unchanged=True, training_runs=8)
        return {"status": "complete", "selected_alpha": selected["alpha"], "output": str(output)}
