from __future__ import annotations

import contextlib
import math
from pathlib import Path

import numpy as np

from .common import BOS, EOS, PAD, ModelConfig, length_bucket, load_npz, read_json, write_json, write_jsonl
from .data import collate, evaluation_batches, normalize


def token_text(sp, tokens):
    ids = []
    for token in tokens:
        if token == EOS:
            break
        if token not in (PAD, BOS):
            ids.append(int(token))
    return sp.decode(ids)


def scores(predictions):
    from sacrebleu.metrics import BLEU, CHRF
    if not predictions:
        return {"count": 0}
    hypotheses = [r["hypothesis"] for r in predictions]
    references = [[r["reference"] for r in predictions]]
    bleu = BLEU(tokenize="13a", lowercase=False, smooth_method="exp", effective_order=False)
    chrf = CHRF(char_order=6, word_order=0, beta=2)
    b, c = bleu.corpus_score(hypotheses, references), chrf.corpus_score(hypotheses, references)
    reference_words = sum(len(x.split()) for x in references[0])
    return {"count": len(predictions), "bleu": b.score, "chrf2": c.score,
            "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature()),
            "word_length_ratio": sum(len(x.split()) for x in hypotheses) / max(1, reference_words),
            "empty_fraction": sum(not x.strip() for x in hypotheses) / len(predictions),
            "length_cap_fraction": sum(r["truncated"] for r in predictions) / len(predictions)}


def torch_context(model, precision):
    import torch
    device = next(model.parameters()).device
    if precision == "bf16" and device.type != "cuda":
        raise RuntimeError("Requested BF16 CUDA evaluation on a non-CUDA model; use --precision fp32 for CPU.")
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" and device.type == "cuda" else contextlib.nullcontext()


def evaluate_model(model, framework, rows, sp, batch_size=16, max_new_tokens=256,
                   precision="fp32", use_cache=False, output=None):
    predictions, nll_sum, count = [None] * len(rows), 0.0, 0
    if framework == "torch":
        import torch
        from .torch_model import loss_sums
        model.eval()
        device = next(model.parameters()).device
    else:
        import tensorflow as tf
        from .tf_model import loss_sums
    for ids in evaluation_batches(rows, batch_size):
        subset = [rows[i] for i in ids]
        src, target, labels = collate(subset)
        if framework == "torch":
            with torch.no_grad(), torch_context(model, precision):
                src_t, tgt_t, lab_t = [torch.from_numpy(x).to(device) for x in (src, target, labels)]
                _, nll, valid = loss_sums(model(src_t, tgt_t), lab_t, 0.0)
                generated, capped = model.greedy(src_t, max_new_tokens, use_cache=use_cache)
            generated, capped = generated.cpu().numpy(), capped.cpu().numpy()
            nll_sum, count = nll_sum + float(nll), count + int(valid)
        else:
            src_t, tgt_t, lab_t = [tf.convert_to_tensor(x) for x in (src, target, labels)]
            _, nll, valid = loss_sums(model(src_t, tgt_t, training=False), lab_t, 0.0)
            generated, capped = model.greedy(src_t, max_new_tokens)
            generated, capped = generated.numpy(), capped.numpy()
            nll_sum, count = nll_sum + float(nll), count + int(valid)
        for j, idx in enumerate(ids):
            row = rows[idx]
            predictions[idx] = {"id": row["id"], "source": row["src_raw"], "reference": row["tgt_raw"],
                "hypothesis": token_text(sp, generated[j]), "truncated": bool(capped[j]),
                "year": row.get("year"), "doc_id": row.get("doc_id"),
                "src_tokens": len(row["src_ids"]) - 2, "length_bucket": length_bucket(len(row["src_ids"]) - 2)}
    metrics = scores(predictions)
    nll = nll_sum / max(1, count)
    metrics.update(nll=nll, ppl=math.exp(nll) if nll < 700 else None, target_tokens=count,
                   framework=framework, precision=precision, max_new_tokens=max_new_tokens,
                   decode="greedy", cache=bool(use_cache and framework == "torch"))
    metrics["by_year"] = {str(y): scores([r for r in predictions if r["year"] == y])
                           for y in sorted({r["year"] for r in predictions if r["year"] is not None})}
    metrics["by_length"] = {b: scores([r for r in predictions if r["length_bucket"] == b])
                            for b in ("01_<=16", "02_17-32", "03_33-64", "04_65-128", "05_>128")}
    if output:
        write_json(Path(output) / "metrics.json", metrics)
        write_jsonl(Path(output) / "predictions.jsonl", predictions)
        Path(output, "hypotheses.fr.txt").write_text("\n".join(r["hypothesis"] for r in predictions) + "\n", encoding="utf-8")
        Path(output, "references.fr.txt").write_text("\n".join(r["reference"] for r in predictions) + "\n", encoding="utf-8")
    return metrics, predictions


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_file() and path.suffix == ".json":
        return path.parent / read_json(path)["checkpoint"]
    if (path / "metadata.json").exists():
        return path
    if (path / "best.json").exists():
        return resolve_checkpoint(path / "best.json")
    raise FileNotFoundError(f"Not a checkpoint or completed run: {path}")


def load_model(checkpoint, framework=None, device="cpu", precision="fp32"):
    checkpoint = resolve_checkpoint(checkpoint)
    metadata = read_json(checkpoint / "metadata.json")
    framework = framework or metadata["framework"]
    c = ModelConfig(**metadata["config"]["model"])
    if framework == "torch":
        from .torch_model import Transformer
        model = Transformer(c).to(device).eval()
    else:
        import tensorflow as tf
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16" if precision == "bf16" else "float32")
        from .tf_model import Transformer
        model = Transformer(c)
    model.load_canonical(load_npz(checkpoint / "weights.npz"))
    return model, metadata, framework


def translate(model, framework, sp, text, max_new_tokens=256, precision="fp32", use_cache=True):
    text = normalize(text)
    if not text:
        raise ValueError("English input must not be empty")
    ids = np.array([[BOS] + sp.encode(text) + [EOS]], dtype=np.int64)
    if framework == "torch":
        import torch
        with torch.no_grad(), torch_context(model, precision):
            generated, capped = model.greedy(torch.from_numpy(ids).to(next(model.parameters()).device), max_new_tokens, use_cache)
        generated, capped = generated.cpu().numpy(), capped.cpu().numpy()
    else:
        generated, capped = model.greedy(ids, max_new_tokens)
        generated, capped = generated.numpy(), capped.numpy()
    return {"source": text, "translation": token_text(sp, generated[0]), "truncated": bool(capped[0])}
