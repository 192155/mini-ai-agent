"""
prepare_data.py - Next-Token Training Dataset & DataLoader

This module prepares input sequences (X) and target next-tokens (y)
for training the miniature language model using PyTorch Dataset and DataLoader.

How Next-Token Prediction Works:
--------------------------------
Given a sequence of words:
    "Python is a high-level interpreted programming language"
With sequence_length = 4:
    Input X:   ["Python", "is", "a", "high-level"] -> Target y: "interpreted"
    Input X:   ["is", "a", "high-level", "interpreted"] -> Target y: "programming"
    Input X:   ["a", "high-level", "interpreted", "programming"] -> Target y: "language"
"""

from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader

import config
from tokenizer import Tokenizer


class NextTokenDataset(Dataset):
    """
    PyTorch Dataset that extracts sliding window sequences for next-token prediction.
    """

    def __init__(self, token_ids: List[int], sequence_length: int = config.SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.inputs = []
        self.targets = []

        # Generate sliding windows of (sequence_length) inputs -> 1 target
        for i in range(len(token_ids) - sequence_length):
            input_seq = token_ids[i : i + sequence_length]
            target_token = token_ids[i + sequence_length]
            self.inputs.append(input_seq)
            self.targets.append(target_token)

        # Convert to PyTorch Tensors
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def prepare_training_data(
    file_path: Path = config.TRAINING_DATA_PATH,
    sequence_length: int = config.SEQUENCE_LENGTH,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
) -> Tuple[DataLoader, Tokenizer]:
    """
    Loads text, trains tokenizer vocabulary, generates sequence tensors,
    and returns a PyTorch DataLoader and the Tokenizer instance.
    """
    tokenizer = Tokenizer()
    tokenizer.build_vocab_from_file(file_path)

    # Read training file line-by-line and convert to a continuous sequence with <EOS>
    all_token_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Encode each line and append <EOS> to mark sentence boundary
            line_ids = tokenizer.encode(line, add_special_tokens=False)
            if line_ids:
                all_token_ids.extend(line_ids + [tokenizer.eos_id])

    dataset = NextTokenDataset(all_token_ids, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, tokenizer


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TRAINING DATA PREPARATION")
    print("=" * 60)

    loader, tok = prepare_training_data(config.TRAINING_DATA_PATH, sequence_length=4, batch_size=4)
    print(f"Total training samples: {len(loader.dataset)}")
    print(f"Vocabulary size: {len(tok)}")

    # Inspect a single batch
    for batch_x, batch_y in loader:
        print("\nSample Batch Inputs Shape:", batch_x.shape)
        print("Sample Batch Targets Shape:", batch_y.shape)
        
        print("\nFirst sample in batch:")
        input_words = [tok.idx2word[idx.item()] for idx in batch_x[0]]
        target_word = tok.idx2word[batch_y[0].item()]
        print(f"  Input Tokens:  {input_words} (IDs: {batch_x[0].tolist()})")
        print(f"  Target Token:  '{target_word}' (ID: {batch_y[0].item()})")
        break

    print("\nData preparation test passed successfully!")