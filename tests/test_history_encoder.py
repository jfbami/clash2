from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from crdata.models.history_encoder import (
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_INPUT_DIM,
    GRUHistoryEncoder,
    NextSwitchHead,
    NextSwitchModel,
)
from crdata.sequences import SUMMARY_FEATURE_NAMES


class GRUHistoryEncoderTests(unittest.TestCase):
    def test_baseline_is_one_layer_and_forward_only(self) -> None:
        encoder = GRUHistoryEncoder()

        self.assertEqual(encoder.gru.num_layers, 1)
        self.assertFalse(encoder.gru.bidirectional)
        self.assertTrue(encoder.gru.batch_first)
        self.assertEqual(encoder.gru.dropout, 0.0)

    def test_returns_the_final_hidden_state(self) -> None:
        encoder = GRUHistoryEncoder()
        tokens = torch.randn(3, 10, DEFAULT_INPUT_DIM)

        encoded = encoder(tokens)
        _, final_hidden = encoder.gru(tokens)

        self.assertEqual(encoded.shape, (3, DEFAULT_HIDDEN_DIM))
        torch.testing.assert_close(encoded, final_hidden[0])

    def test_recurrent_gate_weights_are_orthogonal(self) -> None:
        encoder = GRUHistoryEncoder(hidden_dim=16)
        identity = torch.eye(16)

        for gate_weights in encoder.gru.weight_hh_l0.chunk(3, dim=0):
            torch.testing.assert_close(
                gate_weights @ gate_weights.T,
                identity,
                atol=1e-5,
                rtol=1e-5,
            )

    def test_input_gate_weights_use_xavier_bounds(self) -> None:
        encoder = GRUHistoryEncoder(input_dim=12, hidden_dim=16)
        bound = math.sqrt(6.0 / (12 + 16))

        for gate_weights in encoder.gru.weight_ih_l0.chunk(3, dim=0):
            self.assertLessEqual(float(gate_weights.detach().abs().max()), bound)

    def test_all_gru_biases_start_at_zero(self) -> None:
        encoder = GRUHistoryEncoder()

        torch.testing.assert_close(
            encoder.gru.bias_ih_l0,
            torch.zeros_like(encoder.gru.bias_ih_l0),
        )
        torch.testing.assert_close(
            encoder.gru.bias_hh_l0,
            torch.zeros_like(encoder.gru.bias_hh_l0),
        )


class NextSwitchHeadTests(unittest.TestCase):
    def test_fuses_history_and_summary_into_one_logit(self) -> None:
        head = NextSwitchHead()
        history = torch.randn(4, DEFAULT_HIDDEN_DIM)
        summary = torch.randn(4, len(SUMMARY_FEATURE_NAMES))

        logits = head(history, summary)

        self.assertEqual(logits.shape, (4,))
        self.assertEqual(head.network[0].in_features, 72)
        self.assertIsInstance(head.network[2], nn.Dropout)
        self.assertEqual(head.network[2].p, DEFAULT_DROPOUT)

    def test_rejects_mismatched_batch_sizes(self) -> None:
        head = NextSwitchHead()

        with self.assertRaisesRegex(ValueError, "same batch size"):
            head(
                torch.randn(4, DEFAULT_HIDDEN_DIM),
                torch.randn(3, len(SUMMARY_FEATURE_NAMES)),
            )


class NextSwitchModelTests(unittest.TestCase):
    def test_backpropagation_reaches_sequence_and_summary_inputs(self) -> None:
        model = NextSwitchModel()
        tokens = torch.randn(3, 10, DEFAULT_INPUT_DIM, requires_grad=True)
        summary = torch.randn(
            3,
            len(SUMMARY_FEATURE_NAMES),
            requires_grad=True,
        )

        model(tokens, summary).sum().backward()

        self.assertIsNotNone(tokens.grad)
        self.assertIsNotNone(summary.grad)
        self.assertGreater(float(tokens.grad.abs().sum()), 0.0)
        self.assertGreater(float(summary.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
