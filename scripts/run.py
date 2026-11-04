#!/usr/bin/env python3
"""Single entry point; framework imports are lazy so training environments stay separate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from enfr.common import ROOT, ModelConfig, load_config, read_json, read_jsonl, save_npz, write_json


def parser():
    p = argparse.ArgumentParser(description="English-to-French Transformer experiment toolkit")
    sub = p.add_subparsers(dest="command", required=True)
    prepared = ROOT / "cache/prepared"

    def cfg(q):
        q.add_argument("--config", type=Path, default=ROOT / "scripts/configs/base.json")
    def data(q):
        q.add_argument("--prepared", type=Path, default=prepared)
    def decoding(q, cache):
        q.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
        q.add_argument("--beam-size", type=int, default=4)
        q.add_argument("--length-penalty", type=float, default=0.6)
        q.add_argument("--cache", action=argparse.BooleanOptionalAction, default=cache)

    q = sub.add_parser("prepare"); cfg(q)
    q.add_argument("--archive", type=Path); q.add_argument("--revision")
    q.add_argument("--output", type=Path, default=prepared)
    q = sub.add_parser("train"); cfg(q); data(q)
    q.add_argument("--framework", choices=["torch", "tensorflow"], default="torch")
    q.add_argument("--variant", choices=["A0", "A1", "A2", "A3", "A4", "A5"], default="A0")
    q.add_argument("--rdrop-alpha", type=float, help="A5 only; mean CE + alpha * symmetric KL")
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--device", default="cuda")
    q.add_argument("--run", type=Path); q.add_argument("--resume", type=Path)
    q.add_argument("--max-updates", type=int, help="Stop at this global update count; never labels a partial run complete")
    q = sub.add_parser("evaluate"); data(q); decoding(q, False)
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--split", choices=["validation", "test"], default="validation")
    q.add_argument("--device", default="cpu")
    q.add_argument("--framework", choices=["torch", "tensorflow"])
    q.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    q.add_argument("--output", type=Path, required=True); q.add_argument("--freeze", type=Path)
    q.add_argument("--smoke", action="store_true", help="Non-formal evaluation for synthetic fixtures only")
    q = sub.add_parser("translate"); data(q); decoding(q, True)
    q.add_argument("--checkpoint", type=Path, required=True); q.add_argument("--text", required=True)
    q.add_argument("--device", default="cpu"); q.add_argument("--framework", choices=["torch", "tensorflow"])
    q.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    q = sub.add_parser("compare-decode"); data(q)
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--freeze", type=Path, required=True, help="Original training-suite freeze, read-only")
    q.add_argument("--output", type=Path, default=ROOT / "output/host4090-torch27-decode/A0-seed42")
    q.add_argument("--device", default="cuda")
    q.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--smoke", action="store_true", help="Synthetic fixtures only; never formal benchmark results")
    q = sub.add_parser("check-parity"); cfg(q)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--output", type=Path, default=ROOT / "output/parity")
    q.add_argument("--torch-python"); q.add_argument("--tf-python")
    q = sub.add_parser("_parity-export", help=argparse.SUPPRESS); cfg(q)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--framework", choices=["torch", "tensorflow"], required=True)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("benchmark"); data(q)
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--device", default="cuda"); q.add_argument("--smoke", action="store_true")
    q.add_argument("--output", type=Path, default=ROOT / "output/benchmark.json")
    q = sub.add_parser("diagnose"); data(q)
    q.add_argument("--kind", choices=["scaling", "attention"], required=True)
    q.add_argument("--checkpoint", type=Path)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--count", type=int, default=20)
    q = sub.add_parser("freeze"); data(q)
    q.add_argument("--runs", type=Path, default=ROOT / "output/runs")
    q.add_argument("--output", type=Path, default=ROOT / "output/freeze.json")
    q = sub.add_parser("summarize")
    q.add_argument("--runs", type=Path, default=ROOT / "output/runs")
    q.add_argument("--output", type=Path, default=ROOT / "output/report")
    q = sub.add_parser("bootstrap")
    q.add_argument("--left", type=Path, required=True); q.add_argument("--right", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True); q.add_argument("--repetitions", type=int, default=1000)
    q = sub.add_parser("error-review")
    for name in ("a0", "a1", "a2"):
        q.add_argument("--" + name, type=Path, required=True)
    q.add_argument("--output", type=Path, default=ROOT / "output/report/error_review.jsonl")
    q = sub.add_parser("run-suite"); cfg(q); data(q)
    q.add_argument("--runs", type=Path, default=ROOT / "output/runs")
    q.add_argument("--torch-python", default=sys.executable); q.add_argument("--tf-python", default=sys.executable)
    q.add_argument("--device", default="cuda")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--finalize", action="store_true", help="After all runs: freeze, test, bootstrap, benchmark, diagnostics and report")
    q = sub.add_parser("run-extensions"); cfg(q); data(q)
    q.add_argument("--baseline-runs", type=Path, default=ROOT / "output/host4090-torch27/runs")
    q.add_argument("--baseline-freeze", type=Path, default=ROOT / "output/host4090-torch27/freeze.json")
    q.add_argument("--output", type=Path, default=ROOT / "output/host4090-torch27-extensions")
    q.add_argument("--torch-python", default=sys.executable)
    q.add_argument("--device", default="cuda")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--finalize", action="store_true", help="After 8 new runs: freeze, evaluate A0/A4/A5, bootstrap and report")
    q.add_argument("--smoke", action="store_true", help="Synthetic fixture data only")
    q = sub.add_parser("doctor")
    q.add_argument("--framework", choices=["torch", "tensorflow"], default="torch")
    q.add_argument("--require-gpu", action="store_true")
    q.add_argument("--output", type=Path)
    q = sub.add_parser("profile"); cfg(q); data(q)
    q.add_argument("--framework", choices=["torch", "tensorflow"], default="torch")
    q.add_argument("--variant", choices=["A0", "A3", "A4", "A5"])
    q.add_argument("--rdrop-alpha", type=float)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--device", default="cuda")
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("smoke-data")
    q.add_argument("--output", type=Path, default=ROOT / "cache/smoke")
    q = sub.add_parser("overfit"); data(q)
    q.add_argument("--device", default="cpu"); q.add_argument("--steps", type=int, default=2000)
    q.add_argument("--output", type=Path, default=ROOT / "output/overfit")
    return p


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        from enfr.data import prepare
        c = load_config(args.config)
        out = prepare(ROOT / "cache", args.output, args.archive, args.revision,
                      c["model"]["vocab_size"], c["data"]["max_train_tokens"])
    elif args.command == "train":
        from enfr.training import train
        from enfr.reporting import run_name
        c = load_config(args.config, args.variant, args.seed, args.rdrop_alpha)
        if (args.variant == "A3") != (args.framework == "tensorflow"):
            raise ValueError("A3 is the TensorFlow baseline; A0/A1/A2/A4/A5 use PyTorch")
        run = args.run or ROOT / "output/runs" / run_name(args.variant, args.seed, args.framework)
        out = {"run": str(train(c, args.prepared, run, args.framework, args.device, args.resume, args.max_updates))}
    elif args.command in ("evaluate", "translate"):
        import sentencepiece as spm
        from enfr.data import verify_prepared
        from enfr.evaluation import evaluate_model, load_model, translate, decode_settings, generation_protocol, resolve_checkpoint
        audit = verify_prepared(args.prepared)
        model, meta, framework = load_model(args.checkpoint, args.framework, args.device, args.precision)
        if meta["data_id"] != audit["data_id"] or meta["tokenizer_sha256"] != audit["tokenizer_sha256"]:
            raise ValueError("Checkpoint was trained on a different prepared dataset")
        settings = decode_settings(args.decode, args.beam_size, args.length_penalty)
        if framework == "tensorflow" and args.decode == "beam":
            raise ValueError("Beam search is implemented only for PyTorch")
        sp = spm.SentencePieceProcessor(model_file=str(args.prepared / "tokenizer.model"))
        if args.command == "translate":
            out = translate(model, framework, sp, args.text, meta["config"]["evaluation"]["max_new_tokens"],
                            args.precision, args.cache, **settings)
        else:
            protocol = generation_protocol(**settings, **meta["config"]["evaluation"], precision=args.precision,
                                           use_cache=bool(args.cache and framework == "torch"))
            if framework == "torch" and (args.decode == "beam" or args.cache):
                from enfr.decode_comparison import protect_output
                cp = resolve_checkpoint(args.checkpoint).resolve()
                run = cp.parent.parent if cp.parent.name == "checkpoints" else cp.parent
                protected = [run, args.prepared]
                if run.parent.name == "runs":
                    protected.append(run.parent.parent)
                protect_output(args.output, protected)
                if (args.output / "metrics.json").exists():
                    raise FileExistsError("Preserve completed evaluation; use another --output")
            if args.split == "test":
                from enfr.reporting import check_frozen
                if args.smoke and audit["provenance"].get("synthetic_fixture") is True:
                    pass
                else:
                    check_frozen(args.checkpoint, audit["data_id"], args.freeze, protocol)
            out, _ = evaluate_model(model, framework, read_jsonl(args.prepared / f"{args.split}.jsonl"), sp,
                **meta["config"]["evaluation"], precision=args.precision, use_cache=args.cache, output=args.output, **settings)
            out.update(checkpoint=str(args.checkpoint), data_id=audit["data_id"],
                       formal=not args.smoke and not audit["provenance"].get("synthetic_fixture", False))
            write_json(args.output / "metrics.json", out)
    elif args.command == "compare-decode":
        from enfr.decode_comparison import compare_decode
        out = compare_decode(args.checkpoint, args.prepared, args.freeze, args.output,
                             args.device, args.precision, args.dry_run, args.smoke)
    elif args.command == "run-extensions":
        from enfr.extensions import run_extensions
        out = run_extensions(args.baseline_runs, args.baseline_freeze, args.prepared, args.output,
                             args.config, args.device, args.torch_python, args.dry_run, args.finalize, args.smoke)
    elif args.command == "_parity-export":
        from enfr.parity import export_tensorflow, export_torch
        c = ModelConfig(**load_config(args.config)["model"])
        save_npz(args.output, (export_torch if args.framework == "torch" else export_tensorflow)(c, args.seed))
        return
    elif args.command == "check-parity":
        from enfr.parity import check_parity
        report = check_parity(args.config, args.output, args.seed, args.torch_python, args.tf_python)
        out = {"passed": report["passed"], "checks": report["checks"], "report": str(args.output / "report.json")}
    elif args.command == "benchmark":
        from enfr.benchmark import benchmark
        out = benchmark(args.checkpoint, args.output, args.device, args.smoke, args.prepared)
    elif args.command == "diagnose":
        from enfr.diagnostics import attention_diagnostic, scaling_diagnostic
        if args.kind == "scaling":
            out = scaling_diagnostic(args.output)
        else:
            if not args.checkpoint:
                raise ValueError("Attention diagnostics require --checkpoint")
            results = attention_diagnostic(args.checkpoint, args.prepared, args.output, args.count)
            out = {"attention_groups": len(results), "output": str(args.output)}
    elif args.command == "freeze":
        from enfr.reporting import freeze
        out = freeze(args.runs, args.prepared, args.output)
    elif args.command == "summarize":
        from enfr.reporting import summarize
        out = summarize(args.runs, args.output)
    elif args.command == "bootstrap":
        from enfr.reporting import bootstrap
        out = bootstrap(args.left, args.right, args.output, args.repetitions)
    elif args.command == "error-review":
        from enfr.reporting import error_review
        out = {"samples": len(error_review({"A0": args.a0, "A1": args.a1, "A2": args.a2}, args.output))}
    elif args.command == "smoke-data":
        from enfr.data import build_prepared
        raw = {}
        for split, start, size in (("train", 0, 48), ("validation", 48, 8), ("test", 56, 8)):
            raw[split] = [{"id": f"fixture-{i}", "year": 2010, "doc_id": "fixture",
                "src_raw": f"The red bird sings number {i}.", "tgt_raw": f"L'oiseau rouge chante le nombre {i}."}
                for i in range(start, start + size)]
        out = build_prepared(raw, args.output, vocab_size=128, provenance={"synthetic_fixture": True, "not_benchmark_data": True})
    elif args.command == "overfit":
        from enfr.validation import overfit
        out = overfit(args.prepared, args.output, args.device, args.steps)
    elif args.command == "run-suite":
        from enfr.reporting import EXPERIMENTS, run_name
        from enfr.common import source_manifest
        from enfr.data import verify_prepared
        from enfr.evaluation import resolve_checkpoint
        audit = verify_prepared(args.prepared)
        if not args.dry_run:
            for framework, python in (("torch", args.torch_python), ("tensorflow", args.tf_python)):
                subprocess.run([python, str(ROOT / "scripts/run.py"), "doctor", "--framework", framework, "--require-gpu"], check=True)
            subprocess.run([args.torch_python, str(ROOT / "scripts/run.py"), "check-parity", "--config", str(args.config),
                "--torch-python", args.torch_python, "--tf-python", args.tf_python,
                "--output", str(args.runs.parent / "parity")], check=True)
        commands = []
        for variant, seed, framework in EXPERIMENTS:
            run = args.runs / run_name(variant, seed, framework)
            if (run / "status.json").exists() and read_json(run / "status.json")["status"] == "complete":
                previous = read_json(resolve_checkpoint(run) / "metadata.json")
                if previous["config"] != load_config(args.config, variant, seed) or previous["data_id"] != audit["data_id"] or previous.get("code_id") != source_manifest()["code_id"]:
                    raise ValueError(f"Existing completed run has different code, data or config: {run}")
                continue
            command = [args.torch_python if framework == "torch" else args.tf_python, str(ROOT / "scripts/run.py"), "train",
                "--config", str(args.config), "--prepared", str(args.prepared), "--run", str(run), "--framework", framework,
                "--variant", variant, "--seed", str(seed), "--device", args.device]
            if (run / "latest.json").exists():
                command += ["--resume", str(run / "latest.json")]
            commands.append(command)
            if not args.dry_run:
                subprocess.run(command, check=True)
        if args.finalize:
            base = [args.torch_python, str(ROOT / "scripts/run.py")]
            freeze_path = args.runs.parent / "freeze.json"
            final_commands = [base + ["freeze", "--runs", str(args.runs), "--prepared", str(args.prepared), "--output", str(freeze_path)]]
            for variant, seed, framework in EXPERIMENTS:
                run = args.runs / run_name(variant, seed, framework)
                final_commands.append([args.torch_python if framework == "torch" else args.tf_python, str(ROOT / "scripts/run.py"),
                    "evaluate", "--checkpoint", str(run), "--prepared", str(args.prepared), "--split", "test",
                    "--device", args.device, "--precision", "bf16", "--freeze", str(freeze_path), "--output", str(run / "test")])
            pred = lambda v: str(args.runs / run_name(v, 42, "torch") / "test/predictions.jsonl")
            report = args.runs.parent / "report"
            for variant in ("A1", "A2"):
                final_commands.append(base + ["bootstrap", "--left", pred("A0"), "--right", pred(variant),
                    "--output", str(report / f"bootstrap-A0-{variant}.json")])
            # Never overwrite manually reviewed labels when resuming finalization.
            if not (report / "error_review.jsonl").exists():
                final_commands.append(base + ["error-review", "--a0", pred("A0"), "--a1", pred("A1"), "--a2", pred("A2"),
                    "--output", str(report / "error_review.jsonl")])
            baseline = str(args.runs / run_name("A0", 42, "torch"))
            final_commands += [base + ["benchmark", "--checkpoint", baseline, "--prepared", str(args.prepared), "--device", args.device,
                "--output", str(args.runs.parent / "benchmark.json")],
                base + ["diagnose", "--kind", "scaling", "--output", str(args.runs.parent / "scaling")],
                base + ["diagnose", "--kind", "attention", "--checkpoint", baseline, "--prepared", str(args.prepared),
                        "--output", str(args.runs.parent / "attention")],
                base + ["diagnose", "--kind", "attention", "--checkpoint", str(args.runs / run_name("A2", 42, "torch")),
                        "--prepared", str(args.prepared), "--output", str(args.runs.parent / "attention-A2")],
                base + ["summarize", "--runs", str(args.runs), "--output", str(report)]]
            for command in final_commands:
                commands.append(command)
                if not args.dry_run:
                    subprocess.run(command, check=True)
        out = {"dry_run": args.dry_run, "commands": commands}
    elif args.command == "doctor":
        from enfr.runtime import doctor
        out = doctor(args.framework, args.require_gpu)
        if args.output:
            write_json(args.output, out)
    elif args.command == "profile":
        from enfr.runtime import profile_training
        variant = args.variant or ("A3" if args.framework == "tensorflow" else "A0")
        if (variant == "A3") != (args.framework == "tensorflow"):
            raise ValueError("A3 requires TensorFlow; A0/A4/A5 require PyTorch")
        out = profile_training(load_config(args.config, variant, args.seed, args.rdrop_alpha),
                               args.prepared, args.framework, args.device, args.output)
    else:
        raise ValueError(args.command)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
