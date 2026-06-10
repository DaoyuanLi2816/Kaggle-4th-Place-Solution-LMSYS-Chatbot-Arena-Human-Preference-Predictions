import random

import pytest


class FakeTokenizer:
    """Deterministic char-level tokenizer with the minimal HF interface.

    One token per character, so token counts scale with text length — which is
    what the budget/truncation logic cares about. No network, no weights.
    """

    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) % 30000 + 10 for c in text]}


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def random_conversation(rng: random.Random, max_rounds: int = 5):
    """A random multi-turn conversation with adversarial length variety."""
    n_rounds = rng.randint(1, max_rounds)

    def text():
        kind = rng.random()
        if kind < 0.15:
            return ""  # empty field
        if kind < 0.5:
            return "".join(rng.choice("abcdefgh \n.,!?") for _ in range(rng.randint(1, 60)))
        # long fields force the truncation branch
        return "".join(rng.choice("abcdefgh 测试😀\n") for _ in range(rng.randint(200, 3000)))

    return (
        [text() for _ in range(n_rounds)],
        [text() for _ in range(n_rounds)],
        [text() for _ in range(n_rounds)],
    )
