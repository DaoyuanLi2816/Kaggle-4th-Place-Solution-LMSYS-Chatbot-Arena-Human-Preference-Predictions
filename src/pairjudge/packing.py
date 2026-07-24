"""Budget-aware packing of multi-turn pairwise conversations.

The core problem: to train or run a pairwise judge, each example must fit a
multi-turn conversation *and two candidate responses per turn* into a fixed
token budget. Naive left/right truncation routinely destroys exactly the part
the judge needs — e.g. response B disappears entirely and the model learns
position artifacts instead of preferences.

This module packs rounds greedily and, when the budget runs out, truncates the
*last* round proportionally so that the prompt and both responses all remain
visible, each marked with an explicit ellipsis. The packing guarantees:

1. ``len(input_ids) <= max_length`` — always.
2. Every retained round shows the prompt and both responses (possibly
   shortened, never silently dropped).
3. If the remaining budget is too small to show a meaningful slice of a round
   (``min_tail_budget`` tokens), the round is dropped entirely rather than
   shown misleadingly.

With default settings the output is byte-for-byte identical to the
tokenization used by the 4th-place (gold medal) solution of the Kaggle
"LMSYS — Chatbot Arena Human Preference Predictions" competition; this is
enforced by a golden test against the original implementation
(``tests/test_packing.py``).

The packer is tokenizer-agnostic: anything that supports
``tokenizer(text, add_special_tokens=False)["input_ids"]`` plus
``bos_token_id`` / ``eos_token_id`` attributes works (every Hugging Face
tokenizer does).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class PackerConfig:
    """Configuration for :class:`PairPacker`.

    The defaults reproduce the competition-winning setup exactly.

    Attributes:
        max_length: Hard token budget per packed example (including BOS/EOS
            and the final instruction).
        ratios: Fraction of the remaining budget given to (prompt,
            response_a, response_b) when the final round must be truncated.
            Must sum to <= 1.0.
        min_tail_budget: If fewer than this many tokens remain for the final
            round's content, the round (and all later rounds) is dropped
            instead of truncated.
        ellipsis: Text appended to every truncated field so the judge can see
            that content was cut.
        final_instruction: Instruction appended after the conversation. The
            classification head reads the sequence representation, but
            instruction-tuned backbones benefit from an explicit question.
        round_header: Per-round header template; receives ``round=idx+1``.
        prompt_prefix / response_a_prefix / response_b_prefix: Field templates;
            receive the field text as ``{text}``.
    """

    max_length: int = 2048
    ratios: Sequence[float] = (0.2, 0.4, 0.4)
    min_tail_budget: int = 80
    ellipsis: str = "......"
    final_instruction: str = (
        "\n\n---\nWhich response is better? [A or B or tie]\nAnswer: "
    )
    round_header: str = "\n\n## Round {round}:"
    first_round_header: str = "## Round {round}:"
    prompt_prefix: str = "\n### Prompt:\n{text}"
    response_a_prefix: str = "\n\n### Response A:\n{text}"
    response_b_prefix: str = "\n\n### Response B:\n{text}"
    add_bos: bool = True
    add_eos: bool = True

    def __post_init__(self) -> None:
        if len(self.ratios) != 3:
            raise ValueError(f"ratios must have 3 entries, got {len(self.ratios)}")
        try:
            ratios = tuple(float(value) for value in self.ratios)
        except (TypeError, ValueError) as exc:
            raise ValueError("ratios must contain numeric values") from exc
        if any(not math.isfinite(value) or value < 0.0 for value in ratios):
            raise ValueError("ratios must contain finite, non-negative values")
        total = sum(ratios)
        if total <= 0.0 or total > 1.0 + 1e-9:
            raise ValueError(f"ratios must sum to <= 1.0 and be > 0.0, got {total}")
        self.ratios = ratios
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.min_tail_budget < 0:
            raise ValueError("min_tail_budget must be non-negative")


@dataclass
class PackedExample:
    """Result of packing one conversation."""

    input_ids: List[int]
    attention_mask: List[int]
    rounds_kept: int
    rounds_total: int
    truncated: bool

    @property
    def dropped_rounds(self) -> int:
        return self.rounds_total - self.rounds_kept


class PairPacker:
    """Packs (prompt, response_a, response_b) conversations into a token budget.

    Example::

        from transformers import AutoTokenizer
        from pairjudge import PairPacker, PackerConfig

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        packer = PairPacker(tok, PackerConfig(max_length=2048))
        packed = packer.pack(
            prompts=["What is the capital of France?"],
            responses_a=["Paris."],
            responses_b=["The capital of France is Paris, a city ..."],
        )
        packed.input_ids  # ready for a sequence-classification judge
    """

    def __init__(
        self,
        tokenizer: Any,
        config: Optional[PackerConfig] = None,
        label_mode: str = "hard",
    ):
        if label_mode not in ("hard", "soft", "none"):
            raise ValueError(
                f"label_mode must be 'hard', 'soft' or 'none', got {label_mode!r}"
            )
        self.tokenizer = tokenizer
        self.config = config or PackerConfig()
        self.label_mode = label_mode
        cfg = self.config
        self._ellipsis_ids = self._encode(cfg.ellipsis)
        self._final_ids = self._encode(cfg.final_instruction)
        # Some tokenizers define no BOS (e.g. Qwen2) or no EOS; skip them
        # rather than emitting None ids.
        self._bos = self.tokenizer.bos_token_id if cfg.add_bos else None
        self._eos = self.tokenizer.eos_token_id if cfg.add_eos else None
        n_special = int(self._bos is not None) + int(self._eos is not None)
        self._base_cost = n_special + len(self._final_ids)
        if cfg.max_length <= self._base_cost:
            raise ValueError(
                f"max_length={cfg.max_length} cannot even fit the fixed overhead "
                f"(special tokens + final instruction = {self._base_cost} tokens)"
            )

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def pack(
        self,
        prompts: Sequence[str],
        responses_a: Sequence[str],
        responses_b: Sequence[str],
    ) -> PackedExample:
        """Pack one multi-turn conversation.

        Args:
            prompts: One user prompt per round.
            responses_a: Model A's response per round.
            responses_b: Model B's response per round.

        Returns:
            A :class:`PackedExample` whose ``input_ids`` never exceed
            ``config.max_length``.
        """
        if not (len(prompts) == len(responses_a) == len(responses_b)):
            raise ValueError(
                "prompts, responses_a and responses_b must have the same number "
                f"of rounds, got {len(prompts)}/{len(responses_a)}/{len(responses_b)}"
            )
        cfg = self.config
        ellipsis = self._ellipsis_ids

        input_ids: List[int] = []
        if self._bos is not None:
            input_ids.append(self._bos)

        used = self._base_cost
        rounds_kept = 0
        truncated = False

        for idx, (p, ra, rb) in enumerate(zip(prompts, responses_a, responses_b)):
            header = cfg.first_round_header if idx == 0 else cfg.round_header
            r_tokens = self._encode(header.format(round=idx + 1))
            p_tokens = self._encode(cfg.prompt_prefix.format(text=p))
            ra_tokens = self._encode(cfg.response_a_prefix.format(text=ra))
            rb_tokens = self._encode(cfg.response_b_prefix.format(text=rb))

            total = (
                used + len(r_tokens) + len(p_tokens) + len(ra_tokens) + len(rb_tokens)
            )
            if total <= cfg.max_length:
                input_ids += r_tokens + p_tokens + ra_tokens + rb_tokens
                used = total
                rounds_kept += 1
                continue

            # Budget exceeded: give each field a proportional slice of what is
            # left (reserving room for up to three ellipsis markers), or drop
            # the round entirely if the slice would be too small to be honest.
            remain = used  # tokens already committed
            budget = cfg.max_length - remain - len(r_tokens) - 3 * len(ellipsis)
            if budget >= cfg.min_tail_budget:
                p_budget = int(budget * cfg.ratios[0])
                a_budget = int(budget * cfg.ratios[1])
                b_budget = int(budget * cfg.ratios[2])
                if len(p_tokens) > p_budget:
                    p_tokens = p_tokens[:p_budget] + ellipsis
                if len(ra_tokens) > a_budget:
                    ra_tokens = ra_tokens[:a_budget] + ellipsis
                if len(rb_tokens) > b_budget:
                    rb_tokens = rb_tokens[:b_budget] + ellipsis
                input_ids += r_tokens + p_tokens + ra_tokens + rb_tokens
                rounds_kept += 1
            truncated = True
            break

        input_ids += self._final_ids
        if self._eos is not None:
            input_ids.append(self._eos)

        return PackedExample(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            rounds_kept=rounds_kept,
            rounds_total=len(prompts),
            truncated=truncated or rounds_kept < len(prompts),
        )

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Batched mapper compatible with ``datasets.Dataset.map(batched=True)``.

        Expects columns ``prompt``, ``response_a``, ``response_b`` where each
        cell is a list of per-round strings. When winner columns are present
        and ``label_mode`` is not ``"none"``, adds ``labels``: class indices
        (``label_mode="hard"``) or ``[p_a, p_b, p_tie]`` float distributions
        (``label_mode="soft"``, used for pseudo-label distillation).
        """
        pair_columns = ("prompt", "response_a", "response_b")
        missing = [column for column in pair_columns if column not in example]
        if missing:
            raise ValueError(f"missing required batch columns: {missing}")
        pair_lengths = {column: len(example[column]) for column in pair_columns}
        if len(set(pair_lengths.values())) != 1:
            raise ValueError(
                "batched prompt/response columns must have the same length, got "
                + ", ".join(
                    f"{column}={length}" for column, length in pair_lengths.items()
                )
            )

        input_ids = []
        attention_mask = []
        for ps, ras, rbs in zip(
            example["prompt"], example["response_a"], example["response_b"]
        ):
            packed = self.pack(ps, ras, rbs)
            input_ids.append(packed.input_ids)
            attention_mask.append(packed.attention_mask)

        out: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        winner_columns = ("winner_model_a", "winner_model_b", "winner_tie")
        winner_presence = [column in example for column in winner_columns]
        if self.label_mode != "none" and any(winner_presence):
            if not all(winner_presence):
                missing = [
                    column
                    for column, present in zip(winner_columns, winner_presence)
                    if not present
                ]
                raise ValueError(f"missing winner batch columns: {missing}")
            winner_lengths = {column: len(example[column]) for column in winner_columns}
            expected = pair_lengths["prompt"]
            if any(length != expected for length in winner_lengths.values()):
                raise ValueError(
                    "winner columns must match the pair batch length, got "
                    + ", ".join(
                        f"{column}={length}"
                        for column, length in winner_lengths.items()
                    )
                    + f", pairs={expected}"
                )
            winners = list(
                zip(
                    example["winner_model_a"],
                    example["winner_model_b"],
                    example["winner_tie"],
                )
            )
            if self.label_mode == "hard":
                labels = []
                for index, (a, b, tie) in enumerate(winners):
                    try:
                        labels.append(hard_label(a, b, tie))
                    except ValueError as exc:
                        raise ValueError(
                            f"invalid hard winner labels at batch index {index}: {exc}"
                        ) from exc
                out["labels"] = labels
            else:
                labels = []
                for index, values in enumerate(winners):
                    try:
                        labels.append(_soft_label(*values))
                    except ValueError as exc:
                        raise ValueError(
                            f"invalid soft winner labels at batch index {index}: {exc}"
                        ) from exc
                out["labels"] = labels
        return out


def _probability_values(
    winner_a: float,
    winner_b: float,
    winner_tie: float,
) -> tuple[float, float, float]:
    try:
        values = (float(winner_a), float(winner_b), float(winner_tie))
    except (TypeError, ValueError) as exc:
        raise ValueError("winner values must be numeric") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError("winner values must be finite")
    return values


def _soft_label(
    winner_a: float,
    winner_b: float,
    winner_tie: float,
) -> List[float]:
    values = _probability_values(winner_a, winner_b, winner_tie)
    if any(value < 0.0 for value in values):
        raise ValueError("soft winner values must be non-negative")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"soft winner values must sum to 1.0, got {total}")
    return list(values)


def hard_label(
    winner_a: float,
    winner_b: float,
    winner_tie: Optional[float] = None,
) -> int:
    """Map one-hot winner indicators to 0 = A, 1 = B, or 2 = tie.

    ``winner_tie`` is optional for backward compatibility with the original
    two-indicator helper. When supplied, all three values are validated as a
    strict one-hot label before conversion.
    """
    if winner_tie is None:
        return 0 if winner_a else 1 if winner_b else 2
    values = _probability_values(winner_a, winner_b, winner_tie)
    if any(value not in (0.0, 1.0) for value in values) or sum(values) != 1.0:
        raise ValueError(f"hard winner values must be one-hot, got {values}")
    return values.index(1.0)
