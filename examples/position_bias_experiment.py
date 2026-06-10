"""Measure position bias of a pairwise judge, and what swap debiasing buys.

Trains a small judge on a subset of the public Arena 55k preference dataset
(lmarena-ai/arena-human-preference-55k — the LMSYS competition training data),
then evaluates on a held-out split:

1. **flip rate** — fraction of pairs whose argmax verdict changes when the
   two responses are presented in the opposite order (0 = position-consistent)
2. **log-loss / accuracy** of a single (A, B) pass
3. **log-loss / accuracy** after swap debiasing (both orders, averaged)

Defaults fit a single consumer GPU (RTX 4080 16 GB, ~25 min end to end):

    python examples/position_bias_experiment.py

Scale up with --model / --train-size / --max-length for stronger judges.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from pairjudge import PackerConfig, PairwiseJudge, load_arena_csv, swap_average
from pairjudge.judge import SWAP_PERMUTATION
from pairjudge.packing import hard_label
from pairjudge.training import JudgeTrainConfig, train


def load_split(train_size: int, eval_size: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("lmarena-ai/arena-human-preference-55k", split="train")
    df = load_arena_csv(ds.to_pandas())
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    eval_df = df.iloc[:eval_size].reset_index(drop=True)
    train_df = df.iloc[eval_size : eval_size + train_size].reset_index(drop=True)
    return train_df, eval_df


def evaluate(judge: PairwiseJudge, eval_df: pd.DataFrame, batch_size: int):
    from sklearn.metrics import accuracy_score, log_loss

    labels = np.array(
        [
            hard_label(a, b)
            for a, b in zip(eval_df["winner_model_a"], eval_df["winner_model_b"])
        ]
    )

    # Pass 1: pairs as (A, B). Pass 2: same pairs as (B, A). Two forward
    # passes give every number we need.
    proba = judge.predict_proba(eval_df, batch_size=batch_size)
    swapped_df = eval_df.copy()
    swapped_df["response_a"], swapped_df["response_b"] = (
        eval_df["response_b"],
        eval_df["response_a"],
    )
    raw_swapped = judge.predict_proba(swapped_df, batch_size=batch_size)
    aligned = raw_swapped[:, list(SWAP_PERMUTATION)]  # back in the (A, B) frame
    debiased = swap_average(proba, raw_swapped)

    return {
        "n_eval": int(len(eval_df)),
        "flip_rate": float((proba.argmax(-1) != aligned.argmax(-1)).mean()),
        "single_pass": {
            "log_loss": float(log_loss(labels, proba, labels=[0, 1, 2])),
            "accuracy": float(accuracy_score(labels, proba.argmax(-1))),
        },
        "swap_debiased": {
            "log_loss": float(log_loss(labels, debiased, labels=[0, 1, 2])),
            "accuracy": float(accuracy_score(labels, debiased.argmax(-1))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-size", type=int, default=16000)
    parser.add_argument("--eval-size", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./output/posbias")
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse an already-trained judge in --output")
    args = parser.parse_args()

    import torch

    os.makedirs(args.output, exist_ok=True)
    train_df, eval_df = load_split(args.train_size, args.eval_size, args.seed)
    judge_dir = os.path.join(args.output, "judge")

    if not args.skip_train:
        train_path = os.path.join(args.output, "train.parquet")
        train_df.to_parquet(train_path)
        cfg = JudgeTrainConfig(
            train_path=train_path,
            label_mode="hard",
            seed=args.seed,
            model_name=args.model,
            max_length=args.max_length,
            lora_r=16,
            lora_alpha=16,
            output_dir=judge_dir,
            n_epochs=1.0,
            lr=2e-4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            bf16=True,
            num_workers=0,
        )
        print(f"Training {args.model} on {len(train_df)} pairs ...")
        print(train(cfg))

    judge = PairwiseJudge.from_pretrained(
        os.path.join(judge_dir, "merged"),
        packer_config=PackerConfig(max_length=args.max_length),
        torch_dtype=torch.bfloat16,
    )
    print(f"Evaluating on {len(eval_df)} held-out pairs (2 passes) ...")
    metrics = evaluate(judge, eval_df, args.batch_size)
    metrics["model"] = args.model
    metrics["train_size"] = int(len(train_df))
    metrics["max_length"] = args.max_length

    out_path = os.path.join(args.output, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    sp, sd = metrics["single_pass"], metrics["swap_debiased"]
    print("\n| metric | single pass (A,B) | swap debiased |")
    print("|---|---|---|")
    print(f"| log-loss | {sp['log_loss']:.4f} | {sd['log_loss']:.4f} |")
    print(f"| accuracy | {sp['accuracy']:.4f} | {sd['accuracy']:.4f} |")
    print(f"\nPosition flip rate: {metrics['flip_rate']:.1%}")
    print(f"Metrics written to {out_path}")


if __name__ == "__main__":
    main()
