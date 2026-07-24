import numpy as np
import pandas as pd

from pairjudge.pseudo_label import pseudo_label


class StubJudge:
    def predict_proba(self, df, swap_debias=False, batch_size=4):
        assert batch_size == 8
        assert swap_debias
        return np.tile([0.7, 0.2, 0.1], (len(df), 1))


def test_pseudo_label_adds_probabilities_without_mutating_input():
    source = pd.DataFrame(
        {
            "prompt": [["q1"], ["q2"]],
            "response_a": [["a1"], ["a2"]],
            "response_b": [["b1"], ["b2"]],
        }
    )
    out = pseudo_label(StubJudge(), source, batch_size=8, swap_debias=True)

    assert "winner_model_a" not in source
    assert out["winner_model_a"].tolist() == [0.7, 0.7]
    assert out["winner_model_b"].tolist() == [0.2, 0.2]
    assert out["winner_tie"].tolist() == [0.1, 0.1]
