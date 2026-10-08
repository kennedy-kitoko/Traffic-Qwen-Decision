"""Smoke or train Traffic-Qwen V1 Dual-Choice (phase + duration Clef heads)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/traffic_qwen_v1_dual_choice.json")
    parser.add_argument("--smoke", action="store_true", help="technical only: 10 examples and max_steps=10")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    smoke = args.smoke
    if smoke and (cfg.get("max_steps", 0) != 10):
        raise ValueError("Smoke must run exactly max_steps=10")
    if not str(cfg["train_data"]).endswith("/train.jsonl"):
        raise ValueError("Training is restricted to the training split")
    out = Path(cfg["artifact_dir"])
    model_out = Path(cfg["smoke_output_dir"] if smoke else cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)
    if any(model_out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty model output: {model_out}")
    os.environ.update(WANDB_MODE="offline", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import unsloth  # patch transformers before import
    import torch
    from transformers import TrainerCallback, TrainingArguments
    from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported

    seed = int(cfg["seed"])
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        np = None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    report = {
        "status": "started", "experiment": "Traffic-Qwen-V1-Dual-Choice",
        "scientific_status": "guarded Jinan action imitation baseline; not zero-shot or RL",
        "mode": "smoke" if smoke else "full", "config": cfg,
        "python": sys.version, "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "unsloth": unsloth.__version__, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": is_bfloat16_supported(), "started_unix": time.time(),
    }
    (out / ("smoke_config.json" if smoke else "config.json")).write_text(json.dumps(report, indent=2, default=str) + "\n")

    class FiniteLossCallback(TrainerCallback):
        def on_log(self, _args, _state, _control, logs=None, **kwargs):
            if logs and "loss" in logs and not math.isfinite(float(logs["loss"])):
                raise FloatingPointError(f"Non-finite loss: {logs['loss']}")

    try:
        train_rows = read_jsonl(cfg["train_data"])
        val_rows = read_jsonl(cfg["validation_data"])
        if smoke:
            train_rows = train_rows[:10]
            val_rows = val_rows[:4]
        if len(train_rows) < 10:
            raise ValueError("Need 10 train rows for the requested smoke test")
        started = time.perf_counter()
        model, tokenizer = FastDecisionModel.from_pretrained(
            cfg["model_name"], max_seq_length=cfg["max_seq_length"],
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            load_in_4bit=cfg["load_in_4bit"],
            use_gradient_checkpointing=cfg["gradient_checkpointing"],
            random_state=seed, local_files_only=cfg.get("local_files_only", True),
        )
        report["model_load_seconds"] = time.perf_counter() - started
        report["is_clef"] = bool(getattr(model, "is_clef", False))
        if not report["is_clef"]:
            raise RuntimeError("Base model does not have the Clef decision head")
        model = FastDecisionModel.get_peft_model(
            model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            use_gradient_checkpointing=cfg["gradient_checkpointing"], random_state=seed,
        )
        train_encoded, train_report = FastDecisionModel.build_dataset(train_rows, tokenizer, model)
        val_encoded, val_report = FastDecisionModel.build_dataset(val_rows, tokenizer, model)
        report["dataset_build"] = {"train_rows": len(train_rows), "train_items": len(train_encoded),
                                   "train_report": train_report, "validation_rows": len(val_rows),
                                   "validation_items": len(val_encoded), "validation_report": val_report}
        if len(train_encoded) != len(train_rows) or len(val_encoded) != len(val_rows):
            raise ValueError("FastDecisionModel rejected one or more dual-choice examples")
        batch = int(cfg["smoke_batch_size"] if smoke else cfg["per_device_train_batch_size"])
        grad_acc = int(cfg["smoke_gradient_accumulation_steps"] if smoke else cfg["gradient_accumulation_steps"])
        requested_optimizer_steps = 10 if smoke else math.ceil(
            len(train_encoded) * float(cfg["num_train_epochs"])
            / (batch * grad_acc)
        )
        warmup_steps = max(1, math.ceil(requested_optimizer_steps * float(cfg["warmup_ratio"])))
        report["warmup"] = {"requested_ratio": cfg["warmup_ratio"], "actual_steps": warmup_steps,
                             "note": "Transformers 5.17.0 TrainingArguments has warmup_steps, not warmup_ratio."}
        training_args = TrainingArguments(
            output_dir=str(out / ("trainer_smoke" if smoke else "trainer")),
            max_steps=10 if smoke else -1,
            num_train_epochs=1 if smoke else cfg["num_train_epochs"],
            per_device_train_batch_size=batch, per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_acc,
            learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
            lr_scheduler_type=cfg["lr_scheduler_type"], warmup_steps=warmup_steps,
            bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
            logging_steps=1 if smoke else cfg["logging_steps"],
            save_strategy="no" if smoke else cfg["save_strategy"],
            eval_strategy="no" if smoke else cfg["eval_strategy"],
            save_total_limit=cfg["save_total_limit"], report_to="none", seed=seed, data_seed=seed,
            dataloader_num_workers=0, optim="adamw_torch", remove_unused_columns=False,
        )
        trainer = DecisionTrainer(
            model=model, args=training_args, train_dataset=train_encoded,
            eval_dataset=None if smoke else val_encoded, tokenizer=tokenizer,
            head_learning_rate=cfg["head_learning_rate"], callbacks=[FiniteLossCallback()],
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        train_result = trainer.train()
        report["training_seconds"] = time.perf_counter() - t0
        report["train_metrics"] = {k: (float(v) if isinstance(v, (int, float)) else str(v))
                                   for k, v in train_result.metrics.items()}
        report["global_step"] = int(trainer.state.global_step)
        if smoke and report["global_step"] != 10:
            raise RuntimeError(f"Smoke expected 10 steps, got {report['global_step']}")
        if not smoke:
            report["eval_metrics"] = trainer.evaluate()
        trainer.save_model(str(model_out))
        tokenizer.save_pretrained(model_out)
        report["checkpoint_files"] = sorted(x.name for x in model_out.iterdir())
        report["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reloaded, reloaded_tokenizer = FastDecisionModel.from_pretrained(
            str(model_out), max_seq_length=cfg["max_seq_length"],
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            load_in_4bit=True, use_gradient_checkpointing=cfg["gradient_checkpointing"],
            random_state=seed, local_files_only=True,
        )
        if not getattr(reloaded, "is_clef", False):
            raise RuntimeError("Checkpoint reload did not restore Clef head")
        prediction = FastDecisionModel.predict(reloaded, reloaded_tokenizer,
                                                train_rows[0]["state"], train_rows[0]["questions"])
        report["reload_prediction"] = prediction
        report["probabilities_valid"] = {}
        for task, count in (("phase", 4), ("duration", 6)):
            answer = prediction[task]
            probs = answer.get("probabilities", {})
            values = [float(v) for v in probs.values()]
            valid = len(values) == count and all(math.isfinite(v) and v >= 0 for v in values) and abs(sum(values)-1) < .02
            report["probabilities_valid"][task] = {"valid": valid, "count": len(values), "sum": sum(values)}
            if not valid:
                raise ValueError(f"Invalid {task} probability output: {probs}")
            if answer.get("choice") not in train_rows[0]["questions"][task]["criteria"]:
                raise ValueError(f"Invalid {task} selected choice: {answer.get('choice')}")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if "torch" in locals() and torch.cuda.is_available():
            report["failure_gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
    report["finished_unix"] = time.time()
    report["elapsed_seconds"] = report["finished_unix"] - report["started_unix"]
    target = out / ("smoke_result.json" if smoke else "metrics.json")
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (out / ("smoke.log" if smoke else "train.log")).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
