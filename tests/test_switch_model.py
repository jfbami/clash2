from __future__ import annotations

import unittest

import torch

from crdata.models.history_encoder import LinearSwitchHead, NextSwitchHead
from crdata.models.switch_model import SwitchPredictionModel


class SwitchPredictionModelTests(unittest.TestCase):
    def _inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            torch.randint(0, 20, (3, 10, 8)),
            torch.randint(10, 16, (3, 10, 8)).float(),
            torch.randn(3, 10, 7),
            torch.randn(3, 8),
        )

    def test_runs_end_to_end_with_nonlinear_head(self) -> None:
        model = SwitchPredictionModel(20, 13.0, 1.5, head_hidden_dim=128)

        logits = model(*self._inputs())

        self.assertEqual(logits.shape, (3,))
        self.assertIsInstance(model.switch_head, NextSwitchHead)

    def test_runs_end_to_end_with_linear_head(self) -> None:
        model = SwitchPredictionModel(20, 13.0, 1.5, head_hidden_dim=None)

        logits = model(*self._inputs())

        self.assertEqual(logits.shape, (3,))
        self.assertIsInstance(model.switch_head, LinearSwitchHead)


if __name__ == "__main__":
    unittest.main()
