from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .common import read_json, read_jsonl, sha256, source_manifest, write_json, write_jsonl
from .data import verify_prepared
from .evaluation import resolve_checkpoint

EXPERIMENTS = [(v, seed, "torch") for v in ("A0", "A1", "A2") for seed in (42, 43, 44)] + [("A3", 42, "tensorflow")]


def run_name(variant, seed, framework):
    return f"{variant}-{framework}-seed{seed}"


def freeze(runs, prepared, output):
    runs = Path(runs)
    audit = verify_prepared(prepared)
    if audit["provenance"].get("dataset") != "IWSLT/iwslt2017" or audit["vocab_size"] != 8000:
        raise ValueError("A formal freeze requires the audited IWSLT dataset and 8000-token vocabulary")
    entries = []
    reference = None
    for variant, seed, framework in EXPERIMENTS:
        run = runs / run_name(variant, seed, framework)
        status = read_json(run / "status.json")
        if status["status"] != "complete" or status.get("epochs") != 30:
            raise ValueError(f"All 30 epochs must finish before test freeze: {run}")
        cp = resolve_checkpoint(run)
        meta = read_json(cp / "metadata.json")
        c = meta["config"]
        if meta.get("code_id") != source_manifest()["code_id"]:
            raise ValueError("Freeze requires the same source code for every completed run and evaluation")
        if meta["data_id"] != audit["data_id"] or c["variant"] != variant or c["seed"] != seed or meta["framework"] != framework:
            raise ValueError(f"Wrong experiment identity: {run}")
        comparable = {"training": c["training"], "data": c["data"], "evaluation": c["evaluation"], "model": dict(c["model"])}
        comparable["model"].update(heads=4, encoder_positions=True)
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ValueError("Uncontrolled configuration differences across experiments")
        entries.append({"run": run.name, "checkpoint": str(cp), "weights_sha256": sha256(cp / "weights.npz"),
                        "config": c, "code_id": meta["code_id"], "validation_bleu": meta["state"]["best_bleu"]})
    manifest = {"data_id": audit["data_id"], "tokenizer_sha256": audit["tokenizer_sha256"],
                "test_protocol": "tst2010-2015 pooled; greedy; BLEU 13a mixed case; chrF2", "runs": entries}
    if Path(output).exists():
        if read_json(output) != manifest:
            raise FileExistsError("A different test freeze already exists; preserve it and use another output")
    else:
        write_json(output, manifest)
    return manifest


def check_frozen(checkpoint, data_id, manifest_path):
    if not manifest_path:
        raise ValueError("Test evaluation requires --freeze after all planned training and model selection")
    manifest = read_json(manifest_path)
    cp = resolve_checkpoint(checkpoint)
    if manifest["data_id"] != data_id or sha256(cp / "weights.npz") not in {r["weights_sha256"] for r in manifest["runs"]}:
        raise ValueError("Checkpoint or dataset was not included in the frozen experiment")


def bootstrap(left_path, right_path, output, repetitions=1000):
    from sacrebleu.metrics import BLEU
    left, right = read_jsonl(left_path), read_jsonl(right_path)
    if [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("Paired bootstrap requires identical sample IDs and order")
    if [r["reference"] for r in left] != [r["reference"] for r in right]:
        raise ValueError("Reference mismatch")
    metric = BLEU(tokenize="13a", lowercase=False, smooth_method="exp", effective_order=False)
    refs = [[r["reference"] for r in left]]
    # SacreBLEU 2.6.0 sufficient statistics avoid retokenizing 8,597 sentences 1000 times.
    a = np.asarray(metric._extract_corpus_statistics([r["hypothesis"] for r in left], refs))
    b = np.asarray(metric._extract_corpus_statistics([r["hypothesis"] for r in right], refs))
    score = lambda stats: metric._compute_score_from_stats(stats.sum(axis=0).tolist()).score
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(repetitions):
        ids = rng.integers(0, len(left), len(left))
        deltas.append(score(b[ids]) - score(a[ids]))
    report = {"metric": "SacreBLEU", "direction": "right minus left", "left": str(left_path), "right": str(right_path),
              "samples": len(left), "repetitions": repetitions, "seed": 42,
              "delta": score(b) - score(a), "delta_ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
              "scope": "Paired sentence resampling; does not measure training-seed uncertainty or TED-document clustering"}
    write_json(output, report)
    return report


def error_review(prediction_paths, output):
    if Path(output).exists():
        raise FileExistsError("Preserve existing human review labels; use a new output file")
    systems = {name: read_jsonl(path) for name, path in prediction_paths.items()}
    first = next(iter(systems.values()))
    ids = [r["id"] for r in first]
    if any([r["id"] for r in rows] != ids for rows in systems.values()):
        raise ValueError("Prediction files must have the same order")
    rng = np.random.default_rng(42)
    selected_indices, shortages = [], {}
    buckets = ("01_<=16", "02_17-32", "03_33-64", "04_65-128", "05_>128")
    for bucket in buckets:
        candidates = [i for i, r in enumerate(first) if r["length_bucket"] == bucket]
        chosen = rng.choice(candidates, size=min(20, len(candidates)), replace=False).tolist()
        if len(chosen) < 20:
            shortages[bucket] = 20 - len(chosen)
        selected_indices.extend(chosen)
    # The real corpus has only one >128-token test source. Fill the deficit from
    # other strata without duplication, and disclose the realized allocation.
    while len(selected_indices) < min(100, len(first)):
        for bucket in buckets:
            available = [i for i, r in enumerate(first) if r["length_bucket"] == bucket and i not in selected_indices]
            if available:
                selected_indices.append(int(rng.choice(available)))
            if len(selected_indices) >= min(100, len(first)):
                break
    selected = []
    for i in selected_indices:
            r = first[i]
            selected.append({"id": r["id"], "length_bucket": r["length_bucket"], "source": r["source"], "reference": r["reference"],
                "outputs": {name: rows[i]["hypothesis"] for name, rows in systems.items()},
                "review_status": "unreviewed", "omission_or_addition": None, "repetition": None,
                "numbers_or_names": None, "negation_or_roles": None, "french_agreement": None,
                "confidence": "uncertain", "notes": ""})
    write_jsonl(output, selected)
    write_json(Path(output).with_suffix(".sampling.json"), {"seed": 42, "count": len(selected), "shortages": shortages,
               "realized_bucket_counts": {b: sum(r["length_bucket"] == b for r in selected) for b in buckets},
               "note": "No fabricated human labels; complete the review fields after inspecting references and hypotheses."})
    return selected


def summarize(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs, output = Path(runs), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = []
    for variant, seed, framework in EXPERIMENTS:
        name = run_name(variant, seed, framework)
        run = runs / name
        row = {"run": name, "variant": variant, "seed": seed, "framework": framework, "status": "not_started"}
        if (run / "status.json").exists():
            row.update(read_json(run / "status.json"))
        if (run / "test/metrics.json").exists():
            metrics = read_json(run / "test/metrics.json")
            row.update({k: metrics[k] for k in ("bleu", "chrf2", "nll", "ppl")})
        table.append(row)
    aggregated = {}
    for variant in ("A0", "A1", "A2"):
        rows = [r for r in table if r["variant"] == variant and "bleu" in r]
        if rows:
            aggregated[variant] = {"runs": len(rows), **{metric: {"mean": float(np.mean([r[metric] for r in rows])),
                "std": float(np.std([r[metric] for r in rows], ddof=1)) if len(rows) > 1 else None}
                for metric in ("bleu", "chrf2", "nll")}}
    write_json(output / "summary.json", {"experiments": table, "aggregate": aggregated})
    fields = ("run", "status", "bleu", "chrf2", "nll", "ppl", "training_seconds", "peak_memory_bytes")
    with (output / "experiments.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(table)
    for metric in ("train_nll", "validation_bleu"):
        fig, ax = plt.subplots(figsize=(9, 5))
        lines = 0
        for row in table:
            log = runs / row["run"] / "log.jsonl"
            if not log.exists():
                continue
            epochs = [r for r in read_jsonl(log) if r["event"] == "epoch"]
            if not epochs:
                continue
            ax.plot([r["epoch"] for r in epochs], [r["validation"]["bleu"] if metric == "validation_bleu" else r[metric] for r in epochs], label=row["run"])
            lines += 1
        ax.set(xlabel="Epoch", ylabel=metric)
        if lines:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, "No completed experiment epochs yet", ha="center", transform=ax.transAxes)
        fig.tight_layout(); fig.savefig(output / f"{metric}.png", dpi=160); plt.close(fig)
    text = ["# 英法 Transformer 实验报告", "", "测试协议：IWSLT2017 英法数据包，tst2010–2015 合集；贪心解码。", "",
            "| 实验 | 状态 | BLEU | chrF2 |", "|---|---|---:|---:|"]
    for row in table:
        text.append(f"| {row['run']} | {row['status']} | {row.get('bleu', '未测量')} | {row.get('chrf2', '未测量')} |")
    text += ["", "未完成训练、未测量的指标保留为未完成，不从冒烟实验或其他数据集推测正式结果。",
             "", "待补充：消融结论、bootstrap 区间、推理性能、人工错误分析及局限。"]
    (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return {"experiments": table, "aggregate": aggregated}
