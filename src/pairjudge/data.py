"""Loaders that normalize preference data into one canonical schema.

Canonical columns (one row = one conversation):

- ``id``: str
- ``prompt``: list[str] — user prompt per round
- ``response_a`` / ``response_b``: list[str] — candidate responses per round
- ``winner_model_a`` / ``winner_model_b`` / ``winner_tie``: float — one-hot
  for human labels, arbitrary distribution for pseudo-labels

Everything downstream (packing, training, judging) consumes this schema, so
supporting a new dataset means writing one loader function.
"""

from __future__ import annotations

import json
import random
from typing import Iterable, Optional, Sequence

import pandas as pd

#: String encodings of an empty response as they appear in Chatbot Arena
#: exports. A response equal to any of these is "empty" for labeling and
#: guardrail purposes.
EMPTY_RESPONSE_PATTERNS = ('[null]', '[]', '[ ]', '[  ]', '[""]', '["",""]')

_LIST_COLUMNS = ("prompt", "response_a", "response_b")
_WINNER_COLUMNS = ("winner_model_a", "winner_model_b", "winner_tie")


def _is_empty(series: pd.Series) -> pd.Series:
    return series.isin(EMPTY_RESPONSE_PATTERNS)


def load_arena_csv(
    path_or_df,
    drop_identical: bool = True,
    relabel_empty: bool = True,
) -> pd.DataFrame:
    """Load a Chatbot-Arena-style CSV (LMSYS competition format).

    Expects JSON-encoded list columns ``prompt``/``response_a``/``response_b``
    and one-hot winner columns.

    Cleaning rules (from the gold-medal solution):

    - rows where *both* responses are empty are dropped (no signal);
    - rows where the two responses are byte-identical are dropped when
      ``drop_identical`` — the label is noise there, and inference handles
      that case with a guardrail instead (:func:`empty_and_identical_masks`);
    - rows where exactly one response is empty are relabeled so the non-empty
      side wins when ``relabel_empty`` — annotators almost always prefer *any*
      answer over a blank one, and the few contrary labels are noise.
    """
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df, encoding="utf-8")
    df = df.copy()

    a_empty, b_empty = _is_empty(df["response_a"]), _is_empty(df["response_b"])
    df = df[~(a_empty & b_empty)]
    if drop_identical:
        df = df[~(df["response_a"] == df["response_b"])]

    if relabel_empty:
        a_empty, b_empty = _is_empty(df["response_a"]), _is_empty(df["response_b"])
        df.loc[a_empty, list(_WINNER_COLUMNS)] = [0.0, 1.0, 0.0]
        df.loc[b_empty, list(_WINNER_COLUMNS)] = [1.0, 0.0, 0.0]

    for col in _LIST_COLUMNS:
        df[col] = df[col].apply(json.loads)

    df = df.reset_index(drop=True)
    df["id"] = df["id"].astype(str)
    return df


def load_ultrafeedback(
    path_or_df,
    seed: int = 42,
    dedup_by_prompt: bool = True,
) -> pd.DataFrame:
    """Convert an UltraFeedback-style chosen/rejected dataset to the canonical schema.

    Each record has ``prompt`` (str) and ``chosen``/``rejected`` conversations
    (list of ``{"role", "content"}`` messages, assistant reply at index 1).
    The chosen/rejected pair is assigned to A/B *uniformly at random* (seeded)
    so the resulting dataset is free of position bias by construction.
    """
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_parquet(path_or_df)
    rng = random.Random(seed)

    records = []
    for _, row in df.iterrows():
        chosen = [row["chosen"][1]["content"]]
        rejected = [row["rejected"][1]["content"]]
        if rng.random() > 0.5:
            response_a, response_b, winner = chosen, rejected, "a"
        else:
            response_a, response_b, winner = rejected, chosen, "b"
        records.append(
            {
                "prompt": [row["prompt"]],
                "response_a": response_a,
                "response_b": response_b,
                "winner_model_a": 1.0 if winner == "a" else 0.0,
                "winner_model_b": 1.0 if winner == "b" else 0.0,
                "winner_tie": 0.0,
            }
        )

    out = pd.DataFrame(records)
    if dedup_by_prompt:
        out["_key"] = out["prompt"].apply(lambda x: x[0])
        out = out.drop_duplicates(subset=["_key"], ignore_index=True)
        out = out.drop(columns=["_key"])
    out["id"] = out.index.astype(str)
    return out


def from_pairs(
    prompts: Sequence[str],
    responses_a: Sequence[str],
    responses_b: Sequence[str],
    winners: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Build a canonical dataframe from flat single-turn pairs.

    ``winners`` entries are ``"a"``, ``"b"`` or ``"tie"``; omit for unlabeled
    data (e.g. inference or pseudo-labeling inputs).
    """
    df = pd.DataFrame(
        {
            "prompt": [[p] for p in prompts],
            "response_a": [[r] for r in responses_a],
            "response_b": [[r] for r in responses_b],
        }
    )
    if winners is not None:
        winners = list(winners)
        bad = sorted({w for w in winners} - {"a", "b", "tie"})
        if bad:
            raise ValueError(f"winners must be 'a', 'b' or 'tie', got {bad}")
        df["winner_model_a"] = [1.0 if w == "a" else 0.0 for w in winners]
        df["winner_model_b"] = [1.0 if w == "b" else 0.0 for w in winners]
        df["winner_tie"] = [1.0 if w == "tie" else 0.0 for w in winners]
    df["id"] = df.index.astype(str)
    return df


def empty_and_identical_masks(df: pd.DataFrame):
    """Inference guardrail masks for degenerate pairs.

    Returns boolean Series ``(a_empty, b_empty, identical)`` over raw (still
    JSON-encoded) response columns. A judge should not be trusted on these
    rows: an empty response loses against a non-empty one, and identical
    responses are a tie. Overriding the model's prediction with fixed,
    *calibrated* probabilities (not 0/1 — labels are noisy and log-loss
    punishes overconfidence) is worth a measurable amount of log-loss; the
    gold-medal solution used ``[0.04, 0.88, 0.08]`` for empty-vs-non-empty
    and ``[0.06, 0.06, 0.88]`` for identical pairs.
    """
    return (
        _is_empty(df["response_a"]),
        _is_empty(df["response_b"]),
        df["response_a"] == df["response_b"],
    )
