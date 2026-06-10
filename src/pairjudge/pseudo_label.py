"""Pseudo-label an unlabeled preference pool with a trained judge.

The semi-supervised loop that drove the gold-medal result:

1. Train a judge on human-labeled data (``label_mode: hard``).
2. Run that judge over a large unlabeled pool of (prompt, A, B) pairs —
   this module — storing the *full probability distribution*, not the argmax.
3. Train a fresh judge on human labels + soft pseudo-labels
   (``label_mode: soft``, KL loss).

Keeping the distribution matters: an 0.9/0.1 verdict and a 0.4/0.35/0.25
verdict carry very different information, and KL distillation preserves that.

Run from the command line::

    python -m pairjudge.pseudo_label \
        --model ./output/judge/merged \
        --data pool.parquet --out pool_pl.parquet
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import pandas as pd

from .judge import PairwiseJudge
from .packing import PackerConfig


def pseudo_label(
    judge: PairwiseJudge,
    df: pd.DataFrame,
    batch_size: int = 4,
    swap_debias: bool = False,
) -> pd.DataFrame:
    """Attach soft winner columns predicted by ``judge`` to ``df``."""
    proba = judge.predict_proba(df, swap_debias=swap_debias, batch_size=batch_size)
    out = df.copy()
    out["winner_model_a"] = proba[:, 0]
    out["winner_model_b"] = proba[:, 1]
    out["winner_tie"] = proba[:, 2]
    return out


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained judge (path or hub id)")
    parser.add_argument("--data", required=True, help="Canonical-schema parquet to label")
    parser.add_argument("--out", required=True, help="Output parquet path")
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--swap-debias",
        action="store_true",
        help="Score each pair in both orders (2x compute, position-bias-free labels)",
    )
    args = parser.parse_args(argv)

    judge = PairwiseJudge.from_pretrained(
        args.model,
        packer_config=PackerConfig(max_length=args.max_length),
        load_in_8bit=args.load_in_8bit,
    )
    df = pd.read_parquet(args.data)
    out = pseudo_label(
        judge, df, batch_size=args.batch_size, swap_debias=args.swap_debias
    )
    out.to_parquet(args.out)
    print(f"Wrote {len(out)} pseudo-labeled rows to {args.out}")


if __name__ == "__main__":
    main()
