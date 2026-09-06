"""
tokenizer.py - Custom Word Tokenizer & Vocabulary Builder

This module handles text normalization, tokenization, vocabulary creation,
and bidirectional mapping between words and numerical token IDs.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Union
import config

# Special Tokens
PAD_TOKEN = "<PAD>"  # Padding for uniform sequence lengths
UNK_TOKEN = "<UNK>"  # Unknown words encountered during inference
SOS_TOKEN = "<SOS>"  # Start of sequence marker
EOS_TOKEN = "<EOS>"  # End of sequence marker

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


class Tokenizer:
    """
    Word-level Tokenizer with vocabulary mapping and serialization capabilities.
    """

    def __init__(self):
        # Word-to-ID mapping dictionary
        self.word2idx: Dict[str, int] = {}
        # ID-to-Word mapping dictionary
        self.idx2word: Dict[int, str] = {}
        
        # Initialize special token mappings
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            
        self.pad_id = self.word2idx[PAD_TOKEN]
        self.unk_id = self.word2idx[UNK_TOKEN]
        self.sos_id = self.word2idx[SOS_TOKEN]
        self.eos_id = self.word2idx[EOS_TOKEN]

    def clean_text(self, text: str) -> str:
        """
        Normalize text: lowercases, separates punctuation, and removes excess whitespace.
        
        Example:
            "What is Python?" -> "what is python ?"
        """
        text = text.lower()
        # Separate punctuation with spaces so they become distinct tokens if desired,
        # or remove non-alphanumeric characters while keeping basic punctuation.
        text = re.sub(r"([?.!,;:\-\(\)])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Split normalized text into a list of word tokens.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        return cleaned.split(" ")

    def build_vocab_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Read text lines from a file, tokenize them, and construct the vocabulary.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Training data file not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Extract words from non-empty, non-comment lines
        all_tokens = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = self.tokenize(line)
            all_tokens.extend(tokens)

        # Build unique vocabulary
        for token in all_tokens:
            if token not in self.word2idx:
                new_id = len(self.word2idx)
                self.word2idx[token] = new_id
                self.idx2word[new_id] = token

        print(f"[Tokenizer] Vocabulary built successfully! Total unique tokens: {len(self.word2idx)}")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Convert a text string into a list of token IDs.
        
        Args:
            text: Raw input string.
            add_special_tokens: If True, prepends <SOS> and appends <EOS>.
            
        Returns:
            List of integer token IDs.
        """
        tokens = self.tokenize(text)
        token_ids = [self.word2idx.get(token, self.unk_id) for token in tokens]
        
        if add_special_tokens:
            token_ids = [self.sos_id] + token_ids + [self.eos_id]
            
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert a list of token IDs back into a readable string.
        """
        words = []
        for tid in token_ids:
            word = self.idx2word.get(tid, UNK_TOKEN)
            if skip_special_tokens and word in SPECIAL_TOKENS:
                continue
            words.append(word)
            
        # Recombine words and clean up spacing around punctuation
        text = " ".join(words)
        text = re.sub(r"\s+([?.!,;:\)])", r"\1", text)
        text = re.sub(r"\(\s+", r"(", text)
        return text

    def save_vocab(self, save_path: Union[str, Path]) -> None:
        """Save vocabulary mappings to a JSON file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, indent=2, ensure_ascii=False)
        print(f"[Tokenizer] Saved vocabulary to {save_path}")

    def load_vocab(self, load_path: Union[str, Path]) -> None:
        """Load vocabulary mappings from an existing JSON file."""
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found at: {load_path}")

        with open(load_path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
            
        # Reconstruct reverse mapping (int ID to word)
        self.idx2word = {int(idx): word for word, idx in self.word2idx.items()}
        self.pad_id = self.word2idx.get(PAD_TOKEN, 0)
        self.unk_id = self.word2idx.get(UNK_TOKEN, 1)
        self.sos_id = self.word2idx.get(SOS_TOKEN, 2)
        self.eos_id = self.word2idx.get(EOS_TOKEN, 3)
        print(f"[Tokenizer] Loaded vocabulary ({len(self.word2idx)} tokens) from {load_path}")

    def __len__(self) -> int:
        return len(self.word2idx)


if __name__ == "__main__":
    # Self-Test for Tokenizer
    print("=" * 60)
    print("TESTING TOKENIZER MODULE")
    print("=" * 60)
    
    tokenizer = Tokenizer()
    tokenizer.build_vocab_from_file(config.TRAINING_DATA_PATH)
    
    sample_sentence = "What is Python?"
    print(f"\nOriginal sentence: '{sample_sentence}'")
    
    tokens = tokenizer.tokenize(sample_sentence)
    print(f"Tokenized: {tokens}")
    
    encoded = tokenizer.encode(sample_sentence)
    print(f"Encoded Token IDs: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded String: '{decoded}'")
    
    # Save vocabulary for subsequent modules
    tokenizer.save_vocab(config.VOCAB_SAVE_PATH)
    print("\nTokenizer self-test passed successfully!")