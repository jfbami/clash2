from __future__ import annotations

import unittest

import torch

from crdata.models.card_encoder import CardEmbedding, CardFeatures, CardMLP
from crdata.vocabulary import CardVocabulary


class CardEmbeddingTests(unittest.TestCase):
    def test_vocabulary_lookup_preserves_battle_and_card_axes(self) -> None:
        vocabulary = CardVocabulary(range(10, 18))
        indices = torch.tensor(vocabulary.encode([[10, 11, 999], [11, 10, 999]]))
        encoder = CardEmbedding(vocabulary.embedding_rows, embedding_dim=24)

        vectors = encoder(indices)

        self.assertEqual(vectors.shape, (2, 3, 24))
        torch.testing.assert_close(vectors[0, 0], vectors[1, 1])
        torch.testing.assert_close(vectors[0, 2], vectors[1, 2])

    def test_backpropagation_updates_selected_rows_including_unknown(self) -> None:
        encoder = CardEmbedding(embedding_rows=4, embedding_dim=8)
        before = encoder.embedding.weight.detach().clone()
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)

        encoder(torch.tensor([0, 2])).sum().backward()
        optimizer.step()

        after = encoder.embedding.weight.detach()
        torch.testing.assert_close(after[[1, 3]], before[[1, 3]])
        torch.testing.assert_close(after[[0, 2]], before[[0, 2]] - 0.1)


class CardFeaturesTests(unittest.TestCase):
    def test_standardized_level_is_appended_to_each_embedding(self) -> None:
        encoder = CardFeatures(
            embedding_rows=4,
            embedding_dim=3,
            level_mean=12.0,
            level_standard_deviation=2.0,
        )
        indices = torch.tensor([[1, 2]])
        levels = torch.tensor([[10.0, 14.0]])

        features = encoder(indices, levels)

        self.assertEqual(features.shape, (1, 2, 4))
        torch.testing.assert_close(features[..., -1], torch.tensor([[-1.0, 1.0]]))


class CardMLPTests(unittest.TestCase):
    def test_transforms_each_card_without_changing_sequence_axes(self) -> None:
        card_mlp = CardMLP(input_dim=25, hidden_dim=48)
        card_features = torch.randn(2, 10, 8, 25, requires_grad=True)

        transformed = card_mlp(card_features)
        transformed.square().mean().backward()

        self.assertEqual(transformed.shape, (2, 10, 8, 48))
        self.assertTrue(torch.isfinite(transformed).all())
        self.assertIsNotNone(card_features.grad)
        self.assertTrue(torch.isfinite(card_features.grad).all())


if __name__ == "__main__":
    unittest.main()
