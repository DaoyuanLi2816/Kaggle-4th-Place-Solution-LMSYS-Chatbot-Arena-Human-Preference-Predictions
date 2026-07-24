import math

import numpy as np
import pytest

from pairjudge.training import JudgeTrainConfig, compute_metrics, load_config


class TestConfig:
    def test_defaults_load(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("model_name: foo/bar\nlr: 0.001\n")
        cfg = load_config(str(cfg_file))
        assert cfg.model_name == "foo/bar"
        assert cfg.lr == 0.001
        assert cfg.label_mode == "hard"  # default preserved

    def test_unknown_keys_rejected(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("model_name: foo\nlearning_rate: 0.1\n")
        with pytest.raises(ValueError, match="learning_rate"):
            load_config(str(cfg_file))

    def test_empty_file_gives_defaults(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("")
        assert load_config(str(cfg_file)) == JudgeTrainConfig()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"label_mode": "none"}, "label_mode"),
            ({"eval_holdout": 0.0}, "eval_holdout"),
            ({"eval_holdout": 1.0}, "eval_holdout"),
            ({"max_length": 0}, "max_length"),
            ({"lr": 0.0}, "lr"),
            ({"num_workers": -1}, "num_workers"),
            ({"lora_dropout": 1.0}, "lora_dropout"),
            ({"lora_target_modules": []}, "lora_target_modules"),
        ],
    )
    def test_invalid_values_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            JudgeTrainConfig(**kwargs)


class TestSoftKLLoss:
    def test_matches_hand_computed_kl(self):
        torch = pytest.importorskip("torch")
        from pairjudge.training import soft_kl_loss

        logits = torch.tensor([[2.0, 1.0, 0.5], [0.0, 0.0, 0.0]])
        labels = torch.tensor([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]])

        loss = soft_kl_loss(logits, labels).item()

        # KL(p || q) = sum p * (log p - log q), batch-mean reduction.
        expected = 0.0
        for row_logits, row_p in zip(logits.tolist(), labels.tolist()):
            z = [math.exp(x) for x in row_logits]
            q = [x / sum(z) for x in z]
            expected += sum(p * (math.log(p) - math.log(qi)) for p, qi in zip(row_p, q))
        expected /= len(labels)

        assert loss == pytest.approx(expected, rel=1e-5)

    def test_zero_when_prediction_equals_target(self):
        torch = pytest.importorskip("torch")
        from pairjudge.training import soft_kl_loss

        labels = torch.tensor([[0.5, 0.25, 0.25]])
        logits = torch.log(labels)
        assert soft_kl_loss(logits, labels).item() == pytest.approx(0.0, abs=1e-6)


class TestComputeMetrics:
    class _Preds:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions
            self.label_ids = label_ids

    def test_hard_labels(self):
        pytest.importorskip("sklearn")
        preds = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        labels = np.array([0, 1])
        metrics = compute_metrics(self._Preds(preds, labels))
        assert metrics["acc"] == 1.0
        assert metrics["log_loss"] < 0.1

    def test_soft_labels_reduced_to_argmax(self):
        pytest.importorskip("sklearn")
        preds = np.array([[3.0, 0.0, 0.0]])
        labels = np.array([[0.9, 0.05, 0.05]])
        metrics = compute_metrics(self._Preds(preds, labels))
        assert metrics["acc"] == 1.0
        exp = np.exp(preds - preds.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        expected = -float(np.sum(labels * np.log(probs)))
        assert metrics["log_loss"] == pytest.approx(expected)
