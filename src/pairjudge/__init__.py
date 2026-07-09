"""pairjudge — train and serve pairwise LLM judges (A/B/tie).

Extracted and generalized from the 4th-place (gold medal, 4/1849) solution
to the Kaggle competition "LMSYS — Chatbot Arena Human Preference
Predictions". The packing defaults are byte-for-byte equivalent to the
competition tokenization (golden-tested).

Core pieces:

- :class:`pairjudge.PairPacker` — budget-aware packing of multi-turn
  (prompt, response A, response B) conversations. No heavy dependencies.
- :class:`pairjudge.PairwiseJudge` — inference with optional swap-based
  position-bias correction (requires ``pairjudge[judge]``).
- :mod:`pairjudge.training` — LoRA fine-tuning with hard (CE) or soft (KL
  distillation) labels (requires ``pairjudge[train]``).
- :mod:`pairjudge.data` — loaders normalizing Arena/UltraFeedback-style data
  to one canonical schema.
"""

from .data import (
    EMPTY_RESPONSE_PATTERNS,
    empty_and_identical_masks,
    from_pairs,
    load_arena_csv,
    load_ultrafeedback,
)
from .packing import PackedExample, PackerConfig, PairPacker, hard_label

__version__ = "0.1.2"

__all__ = [
    "PairPacker",
    "PackerConfig",
    "PackedExample",
    "hard_label",
    "load_arena_csv",
    "load_ultrafeedback",
    "from_pairs",
    "empty_and_identical_masks",
    "EMPTY_RESPONSE_PATTERNS",
    "PairwiseJudge",
    "swap_average",
    "__version__",
]


def __getattr__(name):
    # Lazy import: PairwiseJudge needs torch/transformers, which are optional.
    if name in ("PairwiseJudge", "swap_average"):
        from . import judge

        return getattr(judge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
