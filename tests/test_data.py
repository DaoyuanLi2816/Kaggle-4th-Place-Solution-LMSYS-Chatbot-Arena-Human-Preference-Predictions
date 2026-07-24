import json

import pandas as pd
import pytest

from pairjudge import (
    empty_and_identical_masks,
    from_pairs,
    load_arena_csv,
    load_ultrafeedback,
)


def _arena_row(id_, prompt, ra, rb, winner):
    return {
        "id": id_,
        "model_a": "m1",
        "model_b": "m2",
        "prompt": json.dumps([prompt]),
        "response_a": ra if ra.startswith("[") else json.dumps([ra]),
        "response_b": rb if rb.startswith("[") else json.dumps([rb]),
        "winner_model_a": 1.0 if winner == "a" else 0.0,
        "winner_model_b": 1.0 if winner == "b" else 0.0,
        "winner_tie": 1.0 if winner == "tie" else 0.0,
    }


@pytest.fixture
def arena_df():
    return pd.DataFrame(
        [
            _arena_row(1, "q1", "good answer", "bad answer", "a"),
            _arena_row(2, "q2", "[null]", "real answer", "tie"),  # mislabeled
            _arena_row(3, "q3", "[null]", "[null]", "tie"),  # no signal
            _arena_row(4, "q4", "same", "same", "a"),  # identical
            _arena_row(5, "q5", "answer five", "[]", "b"),  # b empty, mislabeled
        ]
    )


class TestLoadArenaCsv:
    def test_cleaning_rules(self, arena_df):
        df = load_arena_csv(arena_df)

        ids = set(df["id"])
        assert "3" not in ids  # both empty -> dropped
        assert "4" not in ids  # identical -> dropped
        assert ids == {"1", "2", "5"}

        # one-sided empty rows are relabeled: non-empty side wins
        row2 = df[df["id"] == "2"].iloc[0]
        assert (row2["winner_model_a"], row2["winner_model_b"], row2["winner_tie"]) == (
            0.0,
            1.0,
            0.0,
        )
        row5 = df[df["id"] == "5"].iloc[0]
        assert (row5["winner_model_a"], row5["winner_model_b"], row5["winner_tie"]) == (
            1.0,
            0.0,
            0.0,
        )

    def test_json_columns_decoded(self, arena_df):
        df = load_arena_csv(arena_df)
        row = df[df["id"] == "1"].iloc[0]
        assert row["prompt"] == ["q1"]
        assert row["response_a"] == ["good answer"]

    def test_relabel_can_be_disabled(self, arena_df):
        df = load_arena_csv(arena_df, relabel_empty=False)
        row2 = df[df["id"] == "2"].iloc[0]
        assert row2["winner_tie"] == 1.0  # original (noisy) label kept

    def test_roundtrip_via_csv(self, arena_df, tmp_path):
        path = tmp_path / "train.csv"
        arena_df.to_csv(path, index=False)
        df = load_arena_csv(str(path))
        assert set(df["id"]) == {"1", "2", "5"}


class TestLoadUltrafeedback:
    @pytest.fixture
    def uf_df(self):
        def conv(text):
            return [
                {"role": "user", "content": "ignored"},
                {"role": "assistant", "content": text},
            ]

        return pd.DataFrame(
            {
                "prompt": ["p1", "p2", "p1", "p3"],
                "chosen": [conv("c1"), conv("c2"), conv("c1-dup"), conv("c3")],
                "rejected": [conv("r1"), conv("r2"), conv("r1-dup"), conv("r3")],
            }
        )

    def test_labels_match_assignment(self, uf_df):
        df = load_ultrafeedback(uf_df, seed=0)
        for _, row in df.iterrows():
            assert row["winner_tie"] == 0.0
            if row["winner_model_a"] == 1.0:
                assert row["response_a"][0].startswith("c")  # chosen on side A
            else:
                assert row["response_b"][0].startswith("c")  # chosen on side B

    def test_dedup_by_prompt(self, uf_df):
        df = load_ultrafeedback(uf_df, seed=0)
        assert len(df) == 3  # p1 duplicate removed

    def test_seed_controls_assignment(self, uf_df):
        a = load_ultrafeedback(uf_df, seed=0)
        b = load_ultrafeedback(uf_df, seed=0)
        pd.testing.assert_frame_equal(a, b)

    def test_uses_last_assistant_message(self):
        def conv(first, final):
            return [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "first prompt"},
                {"role": "assistant", "content": first},
                {"role": "user", "content": "follow-up"},
                {"role": "assistant", "content": final},
            ]

        source = pd.DataFrame(
            {
                "prompt": ["p"],
                "chosen": [conv("old chosen", "final chosen")],
                "rejected": [conv("old rejected", "final rejected")],
            }
        )
        row = load_ultrafeedback(source, seed=0).iloc[0]
        assert {row["response_a"][0], row["response_b"][0]} == {
            "final chosen",
            "final rejected",
        }

    def test_missing_assistant_message_rejected(self):
        messages = [[{"role": "user", "content": "question"}]]
        source = pd.DataFrame(
            {"prompt": ["p"], "chosen": messages, "rejected": messages}
        )
        with pytest.raises(ValueError, match="no assistant message"):
            load_ultrafeedback(source)


class TestFromPairs:
    def test_winner_mapping(self):
        df = from_pairs(["q"], ["a"], ["b"], winners=["tie"])
        assert df.iloc[0]["winner_tie"] == 1.0
        assert df.iloc[0]["prompt"] == ["q"]

    def test_unlabeled(self):
        df = from_pairs(["q"], ["a"], ["b"])
        assert "winner_model_a" not in df.columns

    def test_invalid_winner_rejected(self):
        with pytest.raises(ValueError, match="winners"):
            from_pairs(["q"], ["a"], ["b"], winners=["c"])

    def test_mismatched_pair_lengths_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            from_pairs(["q1", "q2"], ["a1"], ["b1", "b2"])

    def test_mismatched_winner_length_rejected(self):
        with pytest.raises(ValueError, match="pair count"):
            from_pairs(["q1", "q2"], ["a1", "a2"], ["b1", "b2"], winners=["a"])


class TestGuardrailMasks:
    def test_masks(self):
        df = pd.DataFrame(
            {
                "response_a": ['["x"]', "[null]", '["y"]'],
                "response_b": ['["x"]', '["z"]', "[]"],
            }
        )
        a_empty, b_empty, identical = empty_and_identical_masks(df)
        assert a_empty.tolist() == [False, True, False]
        assert b_empty.tolist() == [False, False, True]
        assert identical.tolist() == [True, False, False]
