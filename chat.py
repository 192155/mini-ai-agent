"""
chat.py - CLI Chat Interface for Mini Language Model

This module loads the trained MiniLM checkpoint and vocabulary, accepts user prompts,
generates next-token predictions, and decodes them into text responses.
"""

from pathlib import Path
from typing import Tuple, List
import torch
import torch.nn.functional as F

import config
from tokenizer import Tokenizer
from model import MiniLM


def load_trained_model(
    model_path: Path = config.MODEL_SAVE_PATH,
    vocab_path: Path = config.VOCAB_SAVE_PATH,
) -> Tuple[MiniLM, Tokenizer]:
    """
    Loads the trained model weights and vocabulary mappings from disk.
    """
    model_path = Path(model_path)
    vocab_path = Path(vocab_path)

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found at: {vocab_path}. "
            "Please train the model first by running: python train.py"
        )

    tokenizer = Tokenizer()
    tokenizer.load_vocab(vocab_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights file not found at: {model_path}. "
            "Please train the model first by running: python train.py"
        )

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    vocab_size = checkpoint.get("vocab_size", len(tokenizer))
    embed_dim = checkpoint.get("embed_dim", config.EMBED_DIM)
    hidden_dim = checkpoint.get("hidden_dim", config.HIDDEN_DIM)

    model = MiniLM(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer


def generate_text(
    model: MiniLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 25,
    temperature: float = config.TEMPERATURE,
) -> str:
    """
    Autoregressively generates text continuation starting from the input prompt.
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        return "Please provide a valid prompt."

    generated_ids = list(input_ids)

    with torch.no_grad():
        for _ in range(max_tokens):
            # Take last SEQUENCE_LENGTH tokens as context
            context = generated_ids[-config.SEQUENCE_LENGTH:]
            if len(context) < config.SEQUENCE_LENGTH:
                context = [tokenizer.pad_id] * (config.SEQUENCE_LENGTH - len(context)) + context

            context_tensor = torch.tensor([context], dtype=torch.long)
            logits = model(context_tensor)[0]

            # Apply temperature scaling
            if temperature > 0:
                scaled_logits = logits / temperature
                # Suppress special non-generation tokens
                scaled_logits[tokenizer.pad_id] = -1e9
                scaled_logits[tokenizer.unk_id] = -1e9
                scaled_logits[tokenizer.sos_id] = -1e9

                # Top-k sampling
                top_k = min(5, scaled_logits.size(0))
                top_k_values, top_k_indices = torch.topk(scaled_logits, top_k)
                probs = F.softmax(top_k_values, dim=-1)
                choice = torch.multinomial(probs, num_samples=1).item()
                next_token_id = top_k_indices[choice].item()
            else:
                next_token_id = torch.argmax(logits).item()

            # Stop generating if End-Of-Sequence token is produced
            if next_token_id == tokenizer.eos_id:
                break

            generated_ids.append(next_token_id)

    # Decode entire generated sequence
    decoded_output = tokenizer.decode(generated_ids)
    return decoded_output


def run_cli_chat():
    """Starts the interactive CLI session."""
    print("=" * 65)
    print("       MINI AI AGENT - INTERACTIVE MODEL CHAT (CLI)")
    print("=" * 65)
    print("Note: This is an educational miniature language model trained on a")
    print("small domain dataset. Type 'exit' or 'quit' to end the conversation.\n")

    try:
        model, tokenizer = load_trained_model()
        print("[+] Model and vocabulary loaded successfully!\n")
    except Exception as e:
        print(f"[-] Error loading model: {e}")
        return

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Mini AI: Goodbye! Have a great day.")
                break

            response = generate_text(model, tokenizer, user_input, max_tokens=20)
            print(f"Mini AI: {response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nMini AI: Chat session closed.")
            break


if __name__ == "__main__":
    # Test programmatic generation
    print("Testing programmatic generation:")
    m, t = load_trained_model()
    for test_prompt in ["Python is", "Java is", "Machine learning", "What is"]:
        out = generate_text(m, t, test_prompt, max_tokens=15, temperature=0.1)
        print(f"  Prompt: '{test_prompt}' -> Generated: '{out}'")
    print("\nStarting interactive CLI session...")