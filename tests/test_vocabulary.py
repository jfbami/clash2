from __future__ import annotations

import unittest

import numpy as np

from crdata.vocabulary import UNKNOWN_INDEX, CardVocabulary


class CardVocabularyTests(unittest.TestCase):
    def test_mapping_is_sorted_and_independent_of_input_order(self) -> None:
        first = CardVocabulary([30, 10, 20, 10])
        second = CardVocabulary([20, 30, 10])

        for card_id in (10, 20, 30):
            self.assertEqual(first.index_for(card_id), second.index_for(card_id))
        self.assertEqual(first.index_for(10), 1)
        self.assertEqual(first.index_for(20), 2)
        self.assertEqual(first.index_for(30), 3)

    def test_unknown_card_uses_reserved_zero_index(self) -> None:
        vocabulary = CardVocabulary([10, 20, 30])

        self.assertEqual(vocabulary.index_for(999), UNKNOWN_INDEX)
        self.assertIsNone(vocabulary.card_id_for(UNKNOWN_INDEX))
        self.assertEqual(vocabulary.embedding_rows, 4)

    def test_encode_preserves_deck_array_shape(self) -> None:
        vocabulary = CardVocabulary(range(10, 26))
        decks = np.asarray([range(10, 18), range(18, 26)], dtype=np.int64)

        encoded = vocabulary.encode(decks)

        self.assertEqual(encoded.shape, (2, 8))
        np.testing.assert_array_equal(encoded[0], np.arange(1, 9))
        np.testing.assert_array_equal(encoded[1], np.arange(9, 17))

    def test_known_indices_reverse_to_original_ids(self) -> None:
        vocabulary = CardVocabulary([30, 10, 20])

        decoded = [vocabulary.card_id_for(index) for index in range(1, 4)]

        self.assertEqual(decoded, [10, 20, 30])

    def test_empty_vocabulary_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            CardVocabulary([])


if __name__ == "__main__":
    unittest.main()
