import numpy as np
import pytest

from pairjudge import swap_average
from pairjudge.judge import SWAP_PERMUTATION


def _random_proba(rng, n):
    x = rng.random((n, 3))
    return x / x.sum(axis=1, keepdims=True)


class TestSwapAverage:
    def test_permutation_is_involution(self):
        assert [SWAP_PERMUTATION[i] for i in SWAP_PERMUTATION] == [0, 1, 2]

    def test_known_values(self):
        # Pass 1 sees (A, B): A wins with 0.6.
        proba = np.array([[0.6, 0.3, 0.1]])
        # Pass 2 sees (B, A): its "first slot wins" 0.2 means B wins with 0.2.
        proba_swapped = np.array([[0.2, 0.7, 0.1]])
        out = swap_average(proba, proba_swapped)
        # A: (0.6 + 0.7) / 2, B: (0.3 + 0.2) / 2, tie: (0.1 + 0.1) / 2
        np.testing.assert_allclose(out, [[0.65, 0.25, 0.1]])

    def test_order_invariance(self):
        """Presenting the pair as (A,B) or (B,A) must give the same verdict,
        with A/B columns exchanged."""
        rng = np.random.default_rng(0)
        p, q = _random_proba(rng, 50), _random_proba(rng, 50)
        forward = swap_average(p, q)
        backward = swap_average(q, p)
        np.testing.assert_allclose(backward, forward[:, list(SWAP_PERMUTATION)])

    def test_probabilities_stay_normalized(self):
        rng = np.random.default_rng(1)
        p, q = _random_proba(rng, 50), _random_proba(rng, 50)
        np.testing.assert_allclose(swap_average(p, q).sum(axis=1), np.ones(50))


class TestForwardOrderRestoration:
    def test_length_sorting_restores_original_order(self, tokenizer):
        """_forward sorts by length for padding efficiency; predictions must
        come back in the caller's row order."""
        torch = pytest.importorskip("torch")
        import pandas as pd

        from pairjudge.judge import PairwiseJudge

        class EchoModel:
            """Fake model whose 'logits' encode each sequence's length, so we
            can tell which row each prediction came from."""

            def parameters(self):
                yield torch.zeros(1)

            def __call__(self, input_ids=None, attention_mask=None):
                # Scaled down so softmax never saturates/overflows.
                lengths = attention_mask.sum(-1).float() / 1000.0

                class Out:
                    pass

                out = Out()
                # Make p(A wins) increase with sequence length.
                out.logits = torch.stack(
                    [lengths, -lengths, torch.zeros_like(lengths)], dim=-1
                )
                return out

        class PaddingTokenizer:
            bos_token_id = 1
            eos_token_id = 2
            pad_token_id = 0

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [ord(c) % 30000 + 10 for c in text]}

            def pad(self, inputs, padding=None, return_tensors=None, **kw):
                ids = inputs["input_ids"]
                longest = max(len(x) for x in ids)
                batch = {
                    "input_ids": torch.tensor(
                        [x + [0] * (longest - len(x)) for x in ids]
                    ),
                    "attention_mask": torch.tensor(
                        [
                            m + [0] * (longest - len(m))
                            for m in inputs["attention_mask"]
                        ]
                    ),
                }

                class B(dict):
                    def to(self, device):
                        return self

                return B(batch)

        tok = PaddingTokenizer()
        judge = PairwiseJudge(EchoModel(), tok)
        # Rows with strictly increasing content length, deliberately not sorted.
        df = pd.DataFrame(
            {
                "prompt": [["x" * n] for n in (40, 5, 90, 20, 60)],
                "response_a": [["a"]] * 5,
                "response_b": [["b"]] * 5,
            }
        )
        proba = judge.predict_proba(df, batch_size=2)
        # p(A) must be monotonic in the original row's length, proving each
        # prediction landed back on its own row.
        order_by_len = np.argsort([40, 5, 90, 20, 60])
        assert list(np.argsort(proba[:, 0])) == list(order_by_len)
