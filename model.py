"""
model.py - Miniature Educational Neural Language Model

This module implements a small, educational PyTorch Language Model (MiniLM).

Architecture:
=============
  Input Token IDs: (Batch Size, Sequence Length)
         │
         ▼
  ┌────────────────────────────────────────────────────────┐
  │ 1. Embedding Layer: nn.Embedding(vocab_size, embed_dim)│
  │    Converts discrete word indices into continuous,     │
  │    learnable dense vector representations.             │
  └────────────────────────┬───────────────────────────────┘
                           │  Shape: (Batch, Seq_Len, Embed_Dim)
                           ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. Recurrent Hidden Layer: nn.GRU(embed_dim, hidden_dim│
  │    Processes sequential context and captures temporal   │
  │    word dependencies across the sliding window.        │
  └────────────────────────┬───────────────────────────────┘
                           │  Last Time Step Hidden State
                           ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3. Dropout & Dense Layer: nn.Linear(hidden_dim, hidden)│
  │    Applies non-linear activation (ReLU) & regularizes. │
  └────────────────────────┬───────────────────────────────┘
                           │
                           ▼
  ┌────────────────────────────────────────────────────────┐
  │ 4. Output Projection: nn.Linear(hidden_dim, vocab_size)│
  │    Produces unnormalized log-probabilities (Logits)   │
  │    for every word in the vocabulary.                   │
  └────────────────────────┬───────────────────────────────┘
                           │
                           ▼
              Vocabulary Logits -> Softmax -> Next-Token
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class MiniLM(nn.Module):
    """
    Miniature Educational Language Model built with PyTorch.
    Demonstrates word embeddings, recurrent feature extraction, and next-token logits.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = config.EMBED_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        dropout: float = 0.2,
    ):
        super(MiniLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # 2. Sequential Hidden Layer (GRU)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # 3. Dense Hidden & Regularization
        self.dropout = nn.Dropout(p=dropout)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

        # 4. Output Projection Layer (Logits over Vocabulary)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for next-token prediction.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length) containing token IDs.
            
        Returns:
            logits: Output tensor of shape (batch_size, vocab_size).
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embeds = self.embedding(x)

        # (batch_size, seq_len, embed_dim) -> gru_out: (batch_size, seq_len, hidden_dim)
        gru_out, _ = self.gru(embeds)

        # Extract features from the final sequence position
        last_hidden = gru_out[:, -1, :]  # (batch_size, hidden_dim)

        # Dense transform with non-linearity and dropout
        hidden = self.relu(self.fc_hidden(last_hidden))
        hidden = self.dropout(hidden)

        # Project to vocabulary size to obtain unnormalized logits
        logits = self.fc_out(hidden)  # (batch_size, vocab_size)
        return logits

    @torch.no_grad()
    def generate_next_token(
        self,
        token_ids: List[int],
        temperature: float = 0.7,
        top_k: int = 5,
    ) -> int:
        """
        Predict the next token ID given a prompt context using temperature sampling.
        """
        self.eval()
        
        # Ensure context is at least sequence_length tokens (pad if shorter)
        if len(token_ids) < config.SEQUENCE_LENGTH:
            token_ids = [0] * (config.SEQUENCE_LENGTH - len(token_ids)) + token_ids
        else:
            token_ids = token_ids[-config.SEQUENCE_LENGTH:]

        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        logits = self.forward(input_tensor)[0]  # (vocab_size,)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            # Filter top-k candidates to prevent sampling low-probability artifacts
            top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(0)))
            probs = F.softmax(top_k_values, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1).item()
            return top_k_indices[sampled_idx].item()
        else:
            # Greedy argmax
            return torch.argmax(logits).item()


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MINI LANGUAGE MODEL ARCHITECTURE")
    print("=" * 60)

    sample_vocab_size = 350
    dummy_model = MiniLM(vocab_size=sample_vocab_size)
    print(dummy_model)

    dummy_input = torch.randint(0, sample_vocab_size, (2, config.SEQUENCE_LENGTH))
    dummy_logits = dummy_model(dummy_input)

    print("\nInput Tensor Shape: ", dummy_input.shape)
    print("Output Logits Shape:", dummy_logits.shape)
    assert dummy_logits.shape == (2, sample_vocab_size), "Output shape mismatch!"
    print("\nMini Language Model architecture verified successfully!")