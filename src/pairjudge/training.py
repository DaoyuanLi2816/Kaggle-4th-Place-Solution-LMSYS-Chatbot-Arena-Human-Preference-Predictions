"""Config-driven fine-tuning of a pairwise judge on any preference dataset.

Two label modes:

- ``hard`` — integer class labels (0 = A wins, 1 = B wins, 2 = tie), standard
  cross-entropy. Use for human-annotated data.
- ``soft`` — ``[p_a, p_b, p_tie]`` distributions, KL-divergence loss. Use for
  pseudo-label distillation: a judge trained on human data labels a large
  unlabeled pool (see :mod:`pairjudge.pseudo_label`), and a fresh judge is
  trained on the union. This two-phase loop was worth a significant log-loss
  improvement in the gold-medal solution.

The backbone is any ``AutoModelForSequenceClassification``-compatible model
(Gemma, Qwen, Llama, Mistral, ...), fine-tuned with LoRA by default.

Run from the command line::

    python -m pairjudge.training --cfg examples/configs/quickstart.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from .data import load_arena_csv
from .packing import PackerConfig, PairPacker


@dataclass
class JudgeTrainConfig:
    """Everything needed to train a judge. Loadable from YAML via :func:`load_config`."""

    # data
    train_path: str = ""  # csv (arena format) or parquet (canonical schema)
    label_mode: str = "hard"  # "hard" (CE) or "soft" (KL distillation)
    eval_holdout: float = 0.05  # fraction held out for evaluation
    seed: int = 42

    # model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_length: int = 2048
    disable_attn_logit_softcapping: bool = True  # Gemma-2 only; ignored elsewhere

    # lora
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # optimization
    output_dir: str = "./output/judge"
    n_epochs: float = 1.0
    lr: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 20
    optim: str = "adamw_torch"
    bf16: bool = True
    num_workers: int = 4
    merge_adapter: bool = True  # save a merged full model next to the adapter


def load_config(path: str) -> JudgeTrainConfig:
    """Load a YAML file into a :class:`JudgeTrainConfig`, rejecting unknown keys."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    known = {f_.name for f_ in JudgeTrainConfig.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return JudgeTrainConfig(**raw)


def soft_kl_loss(logits: "torch.Tensor", soft_labels: "torch.Tensor") -> "torch.Tensor":
    """Batch-mean KL(soft_labels || softmax(logits)) — the distillation loss."""
    import torch
    import torch.nn.functional as F

    log_probs = F.log_softmax(logits, dim=-1)
    return torch.nn.KLDivLoss(reduction="batchmean")(
        log_probs, soft_labels.to(torch.float32)
    )


def compute_metrics(eval_preds) -> Dict[str, float]:
    """Log-loss and accuracy; accepts hard int labels or soft distributions."""
    from sklearn.metrics import accuracy_score, log_loss

    preds, labels = eval_preds.predictions, eval_preds.label_ids
    if labels.ndim > 1 and labels.shape[-1] == 3:
        labels = labels.argmax(-1)
    exp = np.exp(preds - preds.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return {
        "log_loss": log_loss(y_true=labels, y_pred=probs, labels=[0, 1, 2]),
        "acc": accuracy_score(y_true=labels, y_pred=preds.argmax(-1)),
    }


def _load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return load_arena_csv(path)


def build_model_and_tokenizer(cfg: JudgeTrainConfig):
    """Load tokenizer + 3-class classification model, optionally LoRA-wrapped."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    from ._compat import model_dtype_kwargs

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=3,
        **model_dtype_kwargs(torch.bfloat16 if cfg.bf16 else torch.float16),
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    # Gemma-2 ships with attention logit soft-capping, which is incompatible
    # with fused attention kernels; the gold-medal run disabled it with no
    # measurable quality cost.
    if cfg.disable_attn_logit_softcapping and hasattr(
        model.config, "attn_logit_softcapping"
    ):
        model.config.attn_logit_softcapping = None

    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=cfg.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model, tokenizer


def train(cfg: JudgeTrainConfig) -> Dict[str, Any]:
    """Train a judge end to end; returns the final eval metrics."""
    import torch
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    df = _load_dataframe(cfg.train_path)
    model, tokenizer = build_model_and_tokenizer(cfg)

    packer = PairPacker(
        tokenizer,
        PackerConfig(max_length=cfg.max_length),
        label_mode=cfg.label_mode,
    )
    ds = Dataset.from_pandas(df)
    ds = ds.map(
        packer,
        batched=True,
        num_proc=cfg.num_workers or None,
        remove_columns=[
            c
            for c in ds.column_names
            if c
            not in ("input_ids", "attention_mask", "labels")
        ],
    )
    split = ds.train_test_split(test_size=cfg.eval_holdout, seed=cfg.seed)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        report_to="none",
        num_train_epochs=cfg.n_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.warmup_steps,
        optim=cfg.optim,
        bf16=cfg.bf16,
        fp16=not cfg.bf16,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="log_loss",
        greater_is_better=False,
        dataloader_num_workers=cfg.num_workers,
    )

    trainer_cls = Trainer if cfg.label_mode == "hard" else _build_soft_label_trainer()
    trainer = trainer_cls(
        args=training_args,
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()

    trainer.save_model(os.path.join(cfg.output_dir, "adapter"))
    if cfg.use_lora and cfg.merge_adapter:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(os.path.join(cfg.output_dir, "merged"))
        tokenizer.save_pretrained(os.path.join(cfg.output_dir, "merged"))

    return trainer.evaluate()


def _build_soft_label_trainer():
    from transformers import Trainer

    class SoftLabelTrainer(Trainer):
        """Trainer whose loss is KL against a soft label distribution."""

        def compute_loss(
            self, model, inputs, return_outputs=False, **kwargs
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = soft_kl_loss(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    return SoftLabelTrainer


def __getattr__(name: str):
    # Lazy so importing pairjudge.training never requires transformers.
    if name == "SoftLabelTrainer":
        return _build_soft_label_trainer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="Path to a YAML config")
    args = parser.parse_args(argv)
    metrics = train(load_config(args.cfg))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
