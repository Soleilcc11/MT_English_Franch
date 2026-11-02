from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from .common import PAD, write_json


def scaling_diagnostic(output):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for dim in (16, 64, 256):
        torch.manual_seed(42)
        q0, k0 = torch.randn(32, 64, dim), torch.randn(32, 64, dim)
        for scaled in (False, True):
            q, k = q0.clone().requires_grad_(), k0.clone().requires_grad_()
            logits = q @ k.transpose(-2, -1) / (dim ** 0.5 if scaled else 1)
            logp = torch.log_softmax(logits, -1)
            p = logp.exp()
            loss = -logp[..., 0].mean()
            loss.backward()
            records.append({"d_k": dim, "scaled": scaled, "logits_std": float(logits.detach().std()),
                "entropy": float(-(p * logp).detach().sum(-1).mean()), "max_probability": float(p.detach().max(-1).values.mean()),
                "loss": float(loss), "q_gradient_rms": float(q.grad.square().mean().sqrt()),
                "k_gradient_rms": float(k.grad.square().mean().sqrt())})
    write_json(output / "scaling.json", {"assumption": "Independent standard normal Q/K; target key=0; not a trained-model claim", "results": records})
    for metric in ("logits_std", "entropy", "q_gradient_rms"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for scaled in (False, True):
            rows = [r for r in records if r["scaled"] == scaled]
            ax.plot([r["d_k"] for r in rows], [r[metric] for r in rows], marker="o", label="scaled" if scaled else "unscaled")
        ax.set(xlabel="Head dimension", ylabel=metric, xscale="log", xticks=[16, 64, 256])
        ax.legend(); fig.tight_layout(); fig.savefig(output / f"{metric}.png", dpi=160); plt.close(fig)
    return records


def permutation_diagnostic(model):
    import torch
    device = next(model.parameters()).device
    model.eval()
    src = torch.tensor([[2, 5, 6, 7, 8, 3, 0]], device=device)
    perm = torch.tensor([0, 4, 2, 1, 3, 5, 6], device=device)
    target = torch.tensor([[2, 9, 10]], device=device)
    with torch.no_grad():
        a, b = model.encode(src)[0], model.encode(src[:, perm])[0]
        y, z = model(src, target), model(src[:, perm], target)
    return {"encoder_positions": model.config.encoder_positions,
            "encoder_equivariance_max_error": float((a[:, perm] - b).abs().max()),
            "logits_invariance_max_error": float((y - z).abs().max()),
            "encoder_equivariant": bool(torch.allclose(a[:, perm], b, atol=1e-5, rtol=1e-4)),
            "logits_invariant": bool(torch.allclose(y, z, atol=1e-5, rtol=1e-4))}


def attention_diagnostic(checkpoint, prepared, output, count=20):
    import torch
    import sentencepiece as spm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .common import read_jsonl, save_npz
    from .data import collate
    from .evaluation import load_model
    from .torch_model import Attention
    model, metadata, _ = load_model(checkpoint, "torch", "cpu")
    output, prepared = Path(output), Path(prepared)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(prepared / "validation.jsonl")
    # Select deterministically across four diagnostic categories, without test data.
    import re
    candidates = [sorted(rows, key=lambda r: len(r["src_ids"])),
                  sorted(rows, key=lambda r: len(r["src_ids"]), reverse=True),
                  [r for r in rows if re.search(r"\b(not|never|no|cannot)\b|n't", r["src"], re.I)],
                  [r for r in rows if re.search(r"\d", r["src"])]]
    chosen, seen = [], set()
    for group in candidates:
        added = 0
        for row in group:
            if row["id"] not in seen:
                chosen.append(row); seen.add(row["id"]); added += 1
            if added >= max(1, count // 4):
                break
    for row in rows:
        if len(chosen) >= count:
            break
        if row["id"] not in seen:
            chosen.append(row); seen.add(row["id"])
    model.set_capture(True)
    summaries = []
    for index, row in enumerate(chosen[:count]):
        src, target, _ = [torch.from_numpy(x) for x in collate([row])]
        with torch.no_grad():
            model(src, target)
        matrices = {}
        for name, layer in model.named_modules():
            if not isinstance(layer, Attention):
                continue
            probs = layer.last["probs"][0].numpy()
            matrices[name] = probs
            entropy = -(probs * np.log(np.maximum(probs, 1e-30))).sum(-1).mean(-1)
            flat = probs.reshape(probs.shape[0], -1)
            unit = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
            summaries.append({"sample_id": row["id"], "layer": name,
                              "head_entropy": entropy.tolist(), "head_cosine_similarity": (unit @ unit.T).tolist()})
            fig, axes = plt.subplots(1, probs.shape[0], figsize=(4 * probs.shape[0], 4), squeeze=False)
            for h, ax in enumerate(axes[0]):
                ax.imshow(probs[h], aspect="auto", interpolation="nearest", vmin=0)
                ax.set(title=f"Head {h}", xlabel="Key token index", ylabel="Query token index")
            fig.suptitle(name); fig.tight_layout()
            fig.savefig(output / f"sample-{index:02d}-{name}.png", dpi=130); plt.close(fig)
        save_npz(output / f"sample-{index:02d}.npz", matrices)
    model.set_capture(False)
    sp = spm.SentencePieceProcessor(model_file=str(prepared / "tokenizer.model"))
    write_json(output / "samples.json", [{**r, "src_pieces": [sp.id_to_piece(i) for i in r["src_ids"]],
        "tgt_pieces": [sp.id_to_piece(i) for i in r["tgt_ids"]]} for r in chosen[:count]])
    write_json(output / "attention_summary.json", summaries)
    write_json(output / "permutation.json", permutation_diagnostic(model))
    return summaries
