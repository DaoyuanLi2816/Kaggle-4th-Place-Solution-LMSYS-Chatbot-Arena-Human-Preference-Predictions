"""Inference wrapper: load a trained pairwise judge and predict preferences.

Includes first-class *position-bias correction*: pairwise judges systematically
favor one presentation slot (well documented for LLM-as-judge setups), so
:meth:`PairwiseJudge.predict_proba` can score every pair twice — once as
(A, B) and once swapped as (B, A) — and average the aligned probabilities.
The swapped pass's (A-wins, B-wins) columns are permuted back before
averaging, so the result is invariant to presentation order by construction.

Heavy dependencies (torch, transformers) are imported lazily so the rest of
the package works without them.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .packing import PackerConfig, PairPacker

#: Column order of judge outputs.
CLASSES = ("a_wins", "b_wins", "tie")

#: Permutation that maps probabilities of a swapped (B, A) pass back to the
#: original (A, B) frame: A-wins and B-wins exchange places, tie is invariant.
SWAP_PERMUTATION = (1, 0, 2)


def swap_average(proba: np.ndarray, proba_swapped: np.ndarray) -> np.ndarray:
    """Average an (A, B) pass with a (B, A) pass in the (A, B) frame.

    ``proba_swapped[:, 0]`` is the probability that the *swapped* first slot —
    i.e. original B — wins, so it must be read as B-wins. This permutation is
    what makes the averaged prediction order-invariant.
    """
    return (proba + proba_swapped[:, SWAP_PERMUTATION]) / 2.0


class PairwiseJudge:
    """A 3-class (A wins / B wins / tie) preference judge.

    Wraps any Hugging Face ``*ForSequenceClassification`` model with
    ``num_labels=3``, the matching tokenizer, and a :class:`PairPacker`.

    Example::

        judge = PairwiseJudge.from_pretrained("path/to/judge")
        proba = judge.predict_proba(df)               # (n, 3) [A, B, tie]
        proba = judge.predict_proba(df, swap_debias=True)  # order-invariant
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        packer_config: Optional[PackerConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.packer = PairPacker(tokenizer, packer_config, label_mode="none")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        packer_config: Optional[PackerConfig] = None,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        torch_dtype: Optional[Any] = None,
        **model_kwargs: Any,
    ) -> "PairwiseJudge":
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if load_in_8bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=3,
            torch_dtype=torch_dtype or torch.float16,
            device_map=device_map,
            **model_kwargs,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.eval()
        return cls(model, tokenizer, packer_config)

    def _pack_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        packed = [
            self.packer.pack(p, ra, rb)
            for p, ra, rb in zip(df["prompt"], df["response_a"], df["response_b"])
        ]
        return pd.DataFrame(
            {
                "_order": np.arange(len(packed)),
                "input_ids": [x.input_ids for x in packed],
                "attention_mask": [x.attention_mask for x in packed],
                "length": [len(x.input_ids) for x in packed],
            }
        )

    def _forward(self, frame: pd.DataFrame, batch_size: int) -> np.ndarray:
        import torch

        try:
            from transformers.data.data_collator import (
                pad_without_fast_tokenizer_warning,
            )
        except ImportError:  # private helper; fall back to the public API
            def pad_without_fast_tokenizer_warning(tokenizer, inputs, **kwargs):
                return tokenizer.pad(inputs, **kwargs)

        # Sort by length so dynamic padding wastes as little compute as
        # possible, then restore the caller's order at the end.
        frame = frame.sort_values("length", ascending=False)
        probs: List[np.ndarray] = []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for start in range(0, len(frame), batch_size):
                batch = frame.iloc[start : start + batch_size]
                inputs = pad_without_fast_tokenizer_warning(
                    self.tokenizer,
                    {
                        "input_ids": batch["input_ids"].tolist(),
                        "attention_mask": batch["attention_mask"].tolist(),
                    },
                    padding="longest",
                    return_tensors="pt",
                )
                logits = self.model(**inputs.to(device)).logits
                probs.append(logits.float().softmax(-1).cpu().numpy())
        out = np.concatenate(probs, axis=0)
        return out[np.argsort(frame["_order"].to_numpy())]

    def predict_proba(
        self,
        df: pd.DataFrame,
        swap_debias: bool = False,
        batch_size: int = 4,
    ) -> np.ndarray:
        """Predict ``[p(A wins), p(B wins), p(tie)]`` for each conversation.

        Args:
            df: Canonical-schema dataframe (see :mod:`pairjudge.data`).
            swap_debias: Also score every pair with A and B swapped and
                average in the original frame. Doubles compute, removes
                position bias.
            batch_size: Inference batch size.
        """
        proba = self._forward(self._pack_frame(df), batch_size)
        if not swap_debias:
            return proba
        return swap_average(proba, self._swapped_proba(df, batch_size))

    def _swapped_proba(self, df: pd.DataFrame, batch_size: int) -> np.ndarray:
        swapped = df.copy()
        swapped["response_a"], swapped["response_b"] = (
            df["response_b"],
            df["response_a"],
        )
        return self._forward(self._pack_frame(swapped), batch_size)

    def position_flip_rate(
        self,
        df: pd.DataFrame,
        batch_size: int = 4,
    ) -> float:
        """Fraction of pairs whose argmax verdict changes when A/B are swapped.

        A perfectly position-consistent judge scores 0. Useful as a quick
        bias diagnostic before deciding whether ``swap_debias`` is worth the
        2x compute.
        """
        proba = self._forward(self._pack_frame(df), batch_size)
        aligned = self._swapped_proba(df, batch_size)[:, list(SWAP_PERMUTATION)]
        return float((proba.argmax(-1) != aligned.argmax(-1)).mean())
