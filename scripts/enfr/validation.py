from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .common import PAD, ModelConfig, read_jsonl, save_npz, write_json
from .data import collate, verify_prepared


def overfit(prepared, output, device="cpu", steps=2000):
    import torch
    from .torch_model import Transformer, loss_sums
    torch.set_num_threads(1)
    torch.manual_seed(42)
    audit = verify_prepared(prepared)
    # Small architecture validates the same implementation on 32 real corpus examples.
    c = ModelConfig(vocab_size=audit["vocab_size"], d_model=64, heads=4, d_ff=128,
                    encoder_layers=1, decoder_layers=1, dropout=0)
    rows = read_jsonl(Path(prepared) / "train.jsonl")[:32]
    src, target, labels = [torch.from_numpy(x).to(device) for x in collate(rows)]
    model = Transformer(c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(src, target)
        loss, _, count = loss_sums(logits, labels, 0)
        (loss / count).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            with torch.no_grad():
                pred = model(src, target).argmax(-1)
                accuracy = float(((pred == labels) & (labels != PAD)).sum() / count)
            row = {"step": step, "nll": float(loss / count), "token_accuracy": accuracy}
            history.append(row)
            print(row, flush=True)
            if accuracy > 0.99:
                break
    report = {"passed": history[-1]["token_accuracy"] > 0.99, "model": asdict(c),
        "data_id": audit["data_id"], "samples": [r["id"] for r in rows], "history": history,
        "scope": "32-sample memorization diagnostic on small architecture; not formal translation quality"}
    Path(output).mkdir(parents=True, exist_ok=True)
    save_npz(Path(output) / "weights.npz", model.canonical())
    write_json(Path(output) / "report.json", report)
    if not report["passed"]:
        raise AssertionError("32-example overfit did not exceed 99% token accuracy")
    return report
