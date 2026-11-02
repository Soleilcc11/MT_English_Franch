from __future__ import annotations

import json
import math
import pickle
import random
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from .common import (PAD, ModelConfig, environment, fingerprint, learning_rate, read_json,
                     read_jsonl, save_npz, set_seed, source_manifest, write_json)
from .data import batch_manifest, collate, verify_prepared
from .evaluation import evaluate_model, resolve_checkpoint


def append_log(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def identity_rows(rows):
    return rows


def publish_checkpoint(run, directory, best=False):
    # Readers only see a pointer after every checkpoint file has been written.
    write_json(run / "latest.json", {"checkpoint": str(directory.relative_to(run))})
    if best:
        write_json(run / "best.json", {"checkpoint": str(directory.relative_to(run))})
    retained = {read_json(run / name)["checkpoint"] for name in ("latest.json", "best.json") if (run / name).exists()}
    # Only remove generated, unreferenced snapshots inside this run; retain best+last.
    for old in (run / "checkpoints").glob("step-*"):
        if old.is_dir() and str(old.relative_to(run)) not in retained and (old / "metadata.json").exists():
            shutil.rmtree(old)


class TorchTrainer:
    def __init__(self, config, device):
        import torch
        from .torch_model import Transformer, make_optimizer
        self.torch = torch
        if config["training"]["precision"] == "bf16":
            if not device.startswith("cuda") or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("Formal BF16 training requires a CUDA GPU with BF16 support. Use smoke.json for CPU tests.")
        torch.manual_seed(config["seed"])
        self.device, self.config = device, config
        self.model = Transformer(ModelConfig(**config["model"]), config["seed"]).to(device)
        self.optimizer = make_optimizer(self.model, config["training"])
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

    def iter_batches(self, rows, batches):
        # The precomputed batch_sampler is the cross-framework ordering contract.
        return self.torch.utils.data.DataLoader(rows, batch_sampler=batches,
            collate_fn=identity_rows, num_workers=0, generator=self.torch.Generator().manual_seed(0))

    def update(self, rows, lr):
        from .torch_model import loss_sums
        from .evaluation import torch_context
        torch = self.torch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        denominator = sum(len(row["tgt_ids"]) - 1 for row in rows)
        loss_total = nll_total = 0.0
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        micro = self.config["training"]["micro_batch"]
        for start in range(0, len(rows), micro):
            src, target, labels = [torch.from_numpy(x).to(self.device) for x in collate(rows[start:start + micro])]
            with torch_context(self.model, self.config["training"]["precision"]):
                loss, nll, _ = loss_sums(self.model(src, target), labels, self.config["training"]["label_smoothing"])
                normalized = loss / denominator
            if not bool(torch.isfinite(normalized)):
                raise FloatingPointError("Non-finite loss; checkpoint and investigate before continuing")
            normalized.backward()
            loss_total += float(loss.detach())
            nll_total += float(nll.detach())
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["clip_norm"], error_if_nonfinite=True)
        self.optimizer.step()
        return loss_total, nll_total, denominator, float(norm)

    def save(self, folder):
        torch = self.torch
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, folder / "training.pt")

    def restore(self, folder):
        state = self.torch.load(folder / "training.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and self.torch.cuda.is_available():
            self.torch.cuda.set_rng_state_all(state["cuda_rng"])

    def memory(self):
        return int(self.torch.cuda.max_memory_allocated()) if self.device.startswith("cuda") else None


class TensorFlowTrainer:
    def __init__(self, config, device):
        import tensorflow as tf
        from .tf_model import Transformer, loss_sums
        self.tf, self.config = tf, config
        if config["training"]["precision"] == "bf16" and not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("Formal BF16 training requires the NVIDIA training host. Use smoke.json for CPU tests.")
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16" if config["training"]["precision"] == "bf16" else "float32")
        tf.random.set_seed(config["seed"])
        self.model = Transformer(ModelConfig(**config["model"]), config["seed"])
        c = config["training"]
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=c["lr"], weight_decay=c["weight_decay"],
            beta_1=c["betas"][0], beta_2=c["betas"][1], epsilon=c["eps"])
        self.optimizer.exclude_from_weight_decay(var_list=[v for v in self.model.trainable_variables if len(v.shape) == 1])
        self.optimizer.build(self.model.trainable_variables)
        self.accums = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.model.trainable_variables]
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, dropout_rng=self.model.rng)

        @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int64)] * 3 + [tf.TensorSpec([], tf.float32)])
        def accumulate(src, target, labels, denominator):
            with tf.GradientTape() as tape:
                loss, nll, count = loss_sums(self.model(src, target, training=True), labels, c["label_smoothing"])
                objective = loss / denominator
            tf.debugging.assert_all_finite(objective, "Non-finite loss")
            grads = tape.gradient(objective, self.model.trainable_variables)
            for accumulator, gradient in zip(self.accums, grads):
                gradient = tf.convert_to_tensor(gradient)
                tf.debugging.assert_all_finite(gradient, "Non-finite gradient")
                accumulator.assign_add(gradient)
            return loss, nll, count

        @tf.function
        def apply():
            grads, norm = tf.clip_by_global_norm(self.accums, c["clip_norm"])
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            for accumulator in self.accums:
                accumulator.assign(tf.zeros_like(accumulator))
            return norm

        self.accumulate, self.apply = accumulate, apply

    def iter_batches(self, rows, batches):
        tf = self.tf
        def generate():
            for indices in batches:
                yield collate([rows[i] for i in indices])
        dataset = tf.data.Dataset.from_generator(generate, output_signature=tuple(
            tf.TensorSpec([None, None], tf.int64) for _ in range(3)))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return dataset.with_options(options).prefetch(2)

    def update(self, rows, lr):
        tf = self.tf
        self.optimizer.learning_rate.assign(lr)
        arrays = tuple(tf.convert_to_tensor(x) for x in collate(rows)) if isinstance(rows, list) else rows
        denominator = int(tf.reduce_sum(tf.cast(arrays[2] != PAD, tf.int32)))
        micro = self.config["training"]["micro_batch"]
        loss_total = nll_total = 0.0
        for start in range(0, int(tf.shape(arrays[0])[0]), micro):
            src, target, labels = [x[start:start + micro] for x in arrays]
            loss, nll, _ = self.accumulate(src, target, labels, tf.constant(float(denominator)))
            loss_total += float(loss)
            nll_total += float(nll)
        norm = self.apply()
        return loss_total, nll_total, denominator, float(norm)

    def save(self, folder):
        self.checkpoint.write(str(folder / "training"))

    def restore(self, folder):
        self.checkpoint.read(str(folder / "training")).assert_consumed()

    def memory(self):
        try:
            return int(self.tf.config.experimental.get_memory_info("GPU:0")["peak"])
        except (ValueError, RuntimeError):
            return None


def train(config, prepared, run, framework="torch", device="cpu", resume=None, max_updates=None):
    import sentencepiece as spm
    prepared, run = Path(prepared), Path(run)
    audit = verify_prepared(prepared)
    source = source_manifest()
    if config["model"]["vocab_size"] != audit["vocab_size"]:
        raise ValueError("Model vocabulary must match prepared tokenizer")
    if config["data"]["max_train_tokens"] != audit["max_train_tokens"]:
        raise ValueError("Training length filter differs from audited dataset")
    if (run / "config.json").exists() and resume is None:
        raise FileExistsError(f"Run already exists. Use --resume {run / 'latest.json'} or a new --run path.")
    run.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])
    trainer = TorchTrainer(config, device) if framework == "torch" else TensorFlowTrainer(config, device)
    rows = read_jsonl(prepared / "train.jsonl")
    validation = read_jsonl(prepared / "validation.jsonl")
    sp = spm.SentencePieceProcessor(model_file=str(prepared / "tokenizer.model"))
    if not rows or not validation:
        raise ValueError("Training and validation sets must be nonempty")
    tc = config["training"]
    manifests = [batch_manifest(rows, prepared / "manifests", audit["data_id"], config["seed"], e,
                    tc["effective_batch"], config["data"]["bucket_window"]) for e in range(tc["epochs"])]
    total_steps = sum(map(len, manifests))
    state = {"epoch": 0, "next_batch": 0, "step": 0, "best_bleu": -1.0, "best_nll": 1e30,
             "epoch_loss": 0.0, "epoch_nll": 0.0, "epoch_tokens": 0, "training_seconds": 0.0}
    if resume:
        folder = resolve_checkpoint(resume)
        metadata = read_json(folder / "metadata.json")
        if metadata["data_id"] != audit["data_id"] or metadata["config"] != config or metadata["framework"] != framework:
            raise ValueError("Resume config, framework or dataset identity mismatch")
        if metadata.get("code_id") != source["code_id"]:
            raise ValueError("Source code changed since the checkpoint. Preserve this run and start a new controlled experiment.")
        state = metadata["state"]
        trainer.restore(folder)
        with (folder / "python_rng.pkl").open("rb") as f:
            py_rng, np_rng = pickle.load(f)
        random.setstate(py_rng)
        np.random.set_state(np_rng)
    write_json(run / "config.json", config)
    write_json(run / "source.json", source)
    env = environment()
    env.update(framework=framework, device=device)
    if framework == "torch" and device.startswith("cuda"):
        env.update(gpu=trainer.torch.cuda.get_device_name(), cuda=trainer.torch.version.cuda,
                   cudnn=trainer.torch.backends.cudnn.version())
    write_json(run / "environment.json", env)
    parameter_count = sum(int(np.prod(v.shape)) for v in trainer.model.canonical().values())
    write_json(run / "status.json", {"status": "running", "framework": framework, "data_id": audit["data_id"],
        "parameter_count": parameter_count, "total_updates": total_steps, "completed_updates": state["step"]})

    def save(best=False):
        folder = run / "checkpoints" / f"step-{state['step']:08d}-{uuid.uuid4().hex[:8]}"
        folder.mkdir(parents=True)
        trainer.save(folder)
        save_npz(folder / "weights.npz", trainer.model.canonical())
        with (folder / "python_rng.pkl").open("wb") as f:
            pickle.dump((random.getstate(), np.random.get_state()), f)
        write_json(folder / "metadata.json", {"framework": framework, "config": config, "data_id": audit["data_id"],
            "tokenizer_sha256": audit["tokenizer_sha256"], "state": state, "parameter_count": parameter_count,
            "code_id": source["code_id"]})
        publish_checkpoint(run, folder, best)

    try:
        for epoch in range(state["epoch"], tc["epochs"]):
            batches = manifests[epoch]
            begin = state["next_batch"] if epoch == state["epoch"] else 0
            for batch_index, batch in enumerate(trainer.iter_batches(rows, batches[begin:]), start=begin):
                if max_updates is not None and state["step"] >= max_updates:
                    save()
                    write_json(run / "status.json", {"status": "paused_at_update_limit", "completed_updates": state["step"],
                        "total_updates": total_steps, "formal_training_complete": False})
                    return run
                start = time.perf_counter()
                lr = learning_rate(state["step"] + 1, total_steps, tc)
                loss, nll, count, grad_norm = trainer.update(batch, lr)
                elapsed = time.perf_counter() - start
                state.update(epoch=epoch, next_batch=batch_index + 1, step=state["step"] + 1,
                    epoch_loss=state["epoch_loss"] + loss, epoch_nll=state["epoch_nll"] + nll,
                    epoch_tokens=state["epoch_tokens"] + count, training_seconds=state["training_seconds"] + elapsed)
                if state["step"] % tc["log_every"] == 0 or state["step"] == 1:
                    log = {"event": "update", "step": state["step"], "epoch": epoch + 1, "lr": lr,
                        "loss": loss / count, "nll": nll / count, "grad_norm": grad_norm,
                        "seconds": elapsed, "target_tokens_per_second": count / elapsed,
                        "peak_memory_bytes": trainer.memory()}
                    append_log(run / "log.jsonl", log)
                    print(json.dumps(log), flush=True)
                if state["step"] % tc["checkpoint_every"] == 0:
                    save()
            val_start = time.perf_counter()
            metrics, _ = evaluate_model(trainer.model, framework, validation, sp,
                **config["evaluation"], precision=tc["precision"], use_cache=False,
                output=run / "validation" / f"epoch-{epoch + 1:02d}")
            is_best = (metrics["bleu"] > state["best_bleu"] or
                       (metrics["bleu"] == state["best_bleu"] and metrics["nll"] < state["best_nll"]))
            if is_best:
                state.update(best_bleu=metrics["bleu"], best_nll=metrics["nll"])
            append_log(run / "log.jsonl", {"event": "epoch", "epoch": epoch + 1, "step": state["step"],
                "train_loss": state["epoch_loss"] / state["epoch_tokens"],
                "train_nll": state["epoch_nll"] / state["epoch_tokens"], "validation": metrics,
                "validation_seconds": time.perf_counter() - val_start, "best": is_best})
            state.update(epoch=epoch + 1, next_batch=0, epoch_loss=0.0, epoch_nll=0.0, epoch_tokens=0)
            save(is_best)
        write_json(run / "status.json", {"status": "complete", "training_complete": True,
            "formal_training_complete": tc["epochs"] == 30 and audit["provenance"].get("dataset") == "IWSLT/iwslt2017",
            "completed_updates": state["step"], "total_updates": total_steps, "epochs": tc["epochs"],
            "parameter_count": parameter_count, "best_validation_bleu": state["best_bleu"],
            "training_seconds": state["training_seconds"], "peak_memory_bytes": trainer.memory(),
            "data_id": audit["data_id"], "framework": framework})
    except BaseException as error:
        # Do not persist partly accumulated gradients as a successful update.
        write_json(run / "status.json", {"status": "interrupted", "error": repr(error),
            "last_completed_update_in_memory": state["step"], "resume": "latest.json", "formal_training_complete": False})
        raise
    return run
