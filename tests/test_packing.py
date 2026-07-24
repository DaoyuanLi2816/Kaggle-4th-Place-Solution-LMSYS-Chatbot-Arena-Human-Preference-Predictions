import random

import numpy as np
import pytest

from pairjudge import PackerConfig, PairPacker, hard_label
from tests.conftest import random_conversation
from tests.reference_impl import tokenize_cls_p3


def _one_hot(winner: str):
    return {
        "a": (1.0, 0.0, 0.0),
        "b": (0.0, 1.0, 0.0),
        "tie": (0.0, 0.0, 1.0),
    }[winner]


class TestCompetitionEquivalence:
    """The library's defaults must reproduce the gold-medal tokenization
    byte for byte. `tests/reference_impl.py` is a verbatim copy of the
    competition code; 500 fuzzed conversations are compared against it."""

    @pytest.mark.parametrize("max_length", [128, 512, 2048])
    def test_matches_competition_reference(self, tokenizer, max_length):
        rng = random.Random(max_length)
        packer = PairPacker(tokenizer, PackerConfig(max_length=max_length))

        prompts, ras, rbs, winners = [], [], [], []
        for _ in range(500):
            p, ra, rb = random_conversation(rng)
            prompts.append(p)
            ras.append(ra)
            rbs.append(rb)
            winners.append(rng.choice(["a", "b", "tie"]))

        example = {
            "prompt": prompts,
            "response_a": ras,
            "response_b": rbs,
            "winner_model_a": [_one_hot(w)[0] for w in winners],
            "winner_model_b": [_one_hot(w)[1] for w in winners],
            "winner_tie": [_one_hot(w)[2] for w in winners],
        }

        expected = tokenize_cls_p3(example, tokenizer, max_length)
        actual = packer(example)

        assert actual["input_ids"] == expected["input_ids"]
        assert actual["attention_mask"] == expected["attention_mask"]
        assert actual["labels"] == expected["labels"]


class TestBudgetInvariants:
    @pytest.mark.parametrize("max_length", [100, 128, 300, 1024])
    def test_never_exceeds_max_length(self, tokenizer, max_length):
        rng = random.Random(max_length)
        packer = PairPacker(tokenizer, PackerConfig(max_length=max_length))
        for _ in range(300):
            p, ra, rb = random_conversation(rng)
            packed = packer.pack(p, ra, rb)
            assert len(packed.input_ids) <= max_length
            assert packed.attention_mask == [1] * len(packed.input_ids)

    def test_all_fields_survive_truncation(self, tokenizer):
        """The point of proportional truncation: a giant response A must not
        push response B (or the prompt) out of the packed sequence."""
        cfg = PackerConfig(max_length=256)
        packer = PairPacker(tokenizer, cfg)
        packed = packer.pack(["short prompt"], ["A" * 5000], ["unique-b-content"])
        decoded = packed.input_ids
        b_marker = tokenizer("### Response B:", add_special_tokens=False)["input_ids"]
        as_str = ",".join(map(str, decoded))
        assert ",".join(map(str, b_marker)) in as_str
        assert packed.truncated
        assert packed.rounds_kept == 1

    def test_ellipsis_marks_truncated_fields(self, tokenizer):
        cfg = PackerConfig(max_length=256)
        packer = PairPacker(tokenizer, cfg)
        packed = packer.pack(["p" * 2000], ["a" * 2000], ["b" * 2000])
        ellipsis = tokenizer("......", add_special_tokens=False)["input_ids"]
        as_str = ",".join(map(str, packed.input_ids))
        assert ",".join(map(str, ellipsis)) in as_str

    def test_tiny_tail_budget_drops_round(self, tokenizer):
        """If less than min_tail_budget remains, the round is dropped rather
        than shown as a misleading sliver."""
        cfg = PackerConfig(max_length=2048)
        packer = PairPacker(tokenizer, cfg)
        # Round 1 fills almost the whole budget; round 2 has < 80 tokens left.
        packed = packer.pack(
            ["x" * 600, "second round prompt"],
            ["y" * 600, "second a"],
            ["z" * 700, "second b"],
        )
        assert packed.rounds_kept == 1
        assert packed.rounds_total == 2
        assert packed.dropped_rounds == 1
        assert packed.truncated

    def test_everything_fits_no_truncation(self, tokenizer):
        packer = PairPacker(tokenizer, PackerConfig(max_length=2048))
        packed = packer.pack(["hi", "again"], ["a1", "a2"], ["b1", "b2"])
        assert packed.rounds_kept == 2
        assert not packed.truncated
        assert packed.input_ids[0] == tokenizer.bos_token_id
        assert packed.input_ids[-1] == tokenizer.eos_token_id

    def test_deterministic(self, tokenizer):
        packer = PairPacker(tokenizer, PackerConfig(max_length=300))
        args = (["p" * 500], ["a" * 500], ["b" * 500])
        assert packer.pack(*args).input_ids == packer.pack(*args).input_ids

    def test_tokenizer_without_bos(self, tokenizer):
        """Qwen2-style tokenizers have bos_token_id=None; the packer must not
        emit None ids (regression test)."""

        class NoBosTokenizer(type(tokenizer)):
            bos_token_id = None

        packer = PairPacker(NoBosTokenizer(), PackerConfig(max_length=300))
        packed = packer.pack(["q" * 500], ["a" * 500], ["b" * 500])
        assert None not in packed.input_ids
        assert packed.input_ids[-1] == NoBosTokenizer.eos_token_id
        assert len(packed.input_ids) <= 300


class TestValidation:
    def test_mismatched_round_counts(self, tokenizer):
        packer = PairPacker(tokenizer)
        with pytest.raises(ValueError, match="same number"):
            packer.pack(["p1", "p2"], ["a1"], ["b1"])

    def test_max_length_smaller_than_overhead(self, tokenizer):
        with pytest.raises(ValueError, match="fixed overhead"):
            PairPacker(tokenizer, PackerConfig(max_length=10))

    def test_bad_ratios(self):
        with pytest.raises(ValueError, match="sum to <= 1.0"):
            PackerConfig(ratios=(0.5, 0.5, 0.5))
        with pytest.raises(ValueError, match="3 entries"):
            PackerConfig(ratios=(0.5, 0.5))
        with pytest.raises(ValueError, match="non-negative"):
            PackerConfig(ratios=(-1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="finite"):
            PackerConfig(ratios=(np.nan, 0.5, 0.5))
        with pytest.raises(ValueError, match="sum to"):
            PackerConfig(ratios=(0.0, 0.0, 0.0))

    def test_bad_min_tail_budget(self):
        with pytest.raises(ValueError, match="min_tail_budget"):
            PackerConfig(min_tail_budget=-1)

    def test_bad_label_mode(self, tokenizer):
        with pytest.raises(ValueError, match="label_mode"):
            PairPacker(tokenizer, label_mode="onehot")

    def test_mapper_rejects_mismatched_pair_batch_lengths(self, tokenizer):
        packer = PairPacker(tokenizer)
        with pytest.raises(ValueError, match="same length"):
            packer(
                {
                    "prompt": [["q1"], ["q2"]],
                    "response_a": [["a1"]],
                    "response_b": [["b1"], ["b2"]],
                }
            )

    def test_mapper_rejects_mismatched_winner_batch_lengths(self, tokenizer):
        packer = PairPacker(tokenizer)
        with pytest.raises(ValueError, match="winner columns"):
            packer(
                {
                    "prompt": [["q1"], ["q2"]],
                    "response_a": [["a1"], ["a2"]],
                    "response_b": [["b1"], ["b2"]],
                    "winner_model_a": [1.0],
                    "winner_model_b": [0.0],
                    "winner_tie": [0.0],
                }
            )


class TestLabels:
    def test_hard_label_mapping(self):
        assert hard_label(1, 0) == 0
        assert hard_label(0, 1) == 1
        assert hard_label(0, 0) == 2  # tie
        assert hard_label(0, 0, 1) == 2

    @pytest.mark.parametrize(
        "values",
        [
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (float("nan"), 0.0, 1.0),
        ],
    )
    def test_invalid_hard_labels_rejected(self, tokenizer, values):
        packer = PairPacker(tokenizer, label_mode="hard")
        with pytest.raises(ValueError, match="invalid hard winner labels"):
            packer(
                {
                    "prompt": [["q"]],
                    "response_a": [["a"]],
                    "response_b": [["b"]],
                    "winner_model_a": [values[0]],
                    "winner_model_b": [values[1]],
                    "winner_tie": [values[2]],
                }
            )

    def test_mapper_soft_labels(self, tokenizer):
        packer = PairPacker(tokenizer, label_mode="soft")
        out = packer(
            {
                "prompt": [["q"]],
                "response_a": [["a"]],
                "response_b": [["b"]],
                "winner_model_a": [0.7],
                "winner_model_b": [0.2],
                "winner_tie": [0.1],
            }
        )
        assert out["labels"] == [[0.7, 0.2, 0.1]]

    @pytest.mark.parametrize(
        "values",
        [
            (-0.1, 0.6, 0.5),
            (0.7, 0.2, 0.2),
            (float("inf"), 0.0, 0.0),
        ],
    )
    def test_invalid_soft_labels_rejected(self, tokenizer, values):
        packer = PairPacker(tokenizer, label_mode="soft")
        with pytest.raises(ValueError, match="invalid soft winner labels"):
            packer(
                {
                    "prompt": [["q"]],
                    "response_a": [["a"]],
                    "response_b": [["b"]],
                    "winner_model_a": [values[0]],
                    "winner_model_b": [values[1]],
                    "winner_tie": [values[2]],
                }
            )

    def test_mapper_label_mode_none(self, tokenizer):
        packer = PairPacker(tokenizer, label_mode="none")
        out = packer(
            {
                "prompt": [["q"]],
                "response_a": [["a"]],
                "response_b": [["b"]],
                "winner_model_a": [1.0],
                "winner_model_b": [0.0],
                "winner_tie": [0.0],
            }
        )
        assert "labels" not in out


class TestCustomTemplates:
    def test_custom_headers_change_output(self, tokenizer):
        default = PairPacker(tokenizer).pack(["q"], ["a"], ["b"])
        custom = PairPacker(
            tokenizer,
            PackerConfig(
                prompt_prefix="\nUSER: {text}",
                response_a_prefix="\nCANDIDATE 1: {text}",
                response_b_prefix="\nCANDIDATE 2: {text}",
                final_instruction="\nWhich candidate wins?",
            ),
        ).pack(["q"], ["a"], ["b"])
        assert default.input_ids != custom.input_ids
