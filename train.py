"""
train.py - Training Pipeline for Miniature Language Model

This script trains the MiniLM model using CrossEntropyLoss and the Adam optimizer,
and saves the trained weights and vocabulary for use in CLI chat, RAG, and Streamlit.
"""

import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

import config
from prepare_data import prepare_training_data
from model import MiniLM


def train_model(
    epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    embed_dim: int = config.EMBED_DIM,
    hidden_dim: int = config.HIDDEN_DIM,
    sequence_length: int = config.SEQUENCE_LENGTH,
    batch_size: int = config.BATCH_SIZE,
    model_save_path: Path = config.MODEL_SAVE_PATH,
    vocab_save_path: Path = config.VOCAB_SAVE_PATH,
):
    """
    Executes the training loop for the MiniLM model and saves model checkpoint.
    """
    print("=" * 65)
    print("      MINI AI AGENT - LANGUAGE MODEL TRAINING")
    print("=" * 65)
    print(f"[*] Loading training corpus from: {config.TRAINING_DATA_PATH}")

    # 1. Prepare data and vocabulary
    dataloader, tokenizer = prepare_training_data(
        file_path=config.TRAINING_DATA_PATH,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )

    vocab_size = len(tokenizer)
    print(f"[*] Vocabulary size: {vocab_size} unique tokens")
    print(f"[*] Training samples: {len(dataloader.dataset)} sequences")
    print(f"[*] Hyperparameters -> Epochs: {epochs}, LR: {learning_rate}, Batch: {batch_size}, Hidden: {hidden_dim}")

    # 2. Instantiate Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    model = MiniLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    # 3. Loss Function and Optimizer
    # CrossEntropyLoss computes softmax + negative log likelihood
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    start_time = time.time()
    print("\n--- Starting Training Loop ---")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Zero out gradients from previous step
            optimizer.zero_grad()

            # Forward pass: compute predictions
            logits = model(batch_x)

            # Calculate Cross-Entropy Loss
            loss = criterion(logits, batch_y)

            # Backward pass: compute gradients
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # Update weights
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(1, batch_count)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Loss: {avg_loss:.4f} | Elapsed: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\n[+] Training completed in {total_time:.2f} seconds!")

    # 5. Save Model and Vocabulary Checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "sequence_length": sequence_length,
        "epoch": epochs,
        "loss": avg_loss,
    }

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_save_path)
    tokenizer.save_vocab(vocab_save_path)

    print(f"[+] Model checkpoint successfully saved to: {model_save_path}")
    print(f"[+] Vocabulary mappings saved to: {vocab_save_path}")
    print("=" * 65)


if __name__ == "__main__":
    train_model()