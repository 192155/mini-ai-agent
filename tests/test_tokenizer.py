"""
test_tokenizer.py - Unit Tests for Tokenizer & Vocabulary
"""

import unittest
from tokenizer import Tokenizer, PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN
import config


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.build_vocab_from_file(config.TRAINING_DATA_PATH)

    def test_special_tokens_initialized(self):
        """Verify that special tokens are mapped to starting indices."""
        self.assertEqual(self.tokenizer.word2idx[PAD_TOKEN], 0)
        self.assertEqual(self.tokenizer.word2idx[UNK_TOKEN], 1)
        self.assertEqual(self.tokenizer.word2idx[SOS_TOKEN], 2)
        self.assertEqual(self.tokenizer.word2idx[EOS_TOKEN], 3)

    def test_clean_text(self):
        """Verify text cleaning normalizes casing and punctuation."""
        raw = "What is Python?!?"
        cleaned = self.tokenizer.clean_text(raw)
        self.assertIn("what", cleaned)
        self.assertIn("python", cleaned)

    def test_encode_and_decode(self):
        """Verify round-trip token encoding and decoding."""
        text = "python is a programming language"
        encoded = self.tokenizer.encode(text)
        self.assertIsInstance(encoded, list)
        self.assertTrue(len(encoded) > 0)

        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_unknown_word_handling(self):
        """Verify that out-of-vocabulary words are mapped to UNK_TOKEN."""
        text = "xyzzy12345nonexistentword"
        encoded = self.tokenizer.encode(text)
        self.assertEqual(encoded, [self.tokenizer.unk_id])


if __name__ == "__main__":
    unittest.main()
