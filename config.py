"""
config.py - Central Configuration for Mini AI Agent

This module defines all paths, hyperparameters, and default settings
for the Mini AI Agent project in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base Project Directory
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCUMENTS_DIR = BASE_DIR / "documents"
TESTS_DIR = BASE_DIR / "tests"

# Ensure essential directories exist
for directory in [DATA_DIR, MODELS_DIR, DOCUMENTS_DIR, TESTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Training Data & Checkpoint Paths
TRAINING_DATA_PATH = BASE_DIR / "training_data.txt"
MODEL_SAVE_PATH = MODELS_DIR / "mini_model.pth"
VOCAB_SAVE_PATH = MODELS_DIR / "vocab.json"
VECTOR_STORE_PATH = DATA_DIR / "vector_store.json"

# Mini Language Model Hyperparameters
# (Optimized for fast CPU training and clear educational demonstration)
EMBED_DIM = 64          # Size of word embedding vectors
HIDDEN_DIM = 128        # Number of hidden units in neural network
SEQUENCE_LENGTH = 4     # Number of previous words used to predict the next word
BATCH_SIZE = 16         # Number of training samples per optimization step
LEARNING_RATE = 0.005   # Step size for gradient descent (Adam)
EPOCHS = 60             # Number of complete passes through training dataset
TEMPERATURE = 0.7       # Sampling temperature for text generation (lower = more deterministic)

# Text Chunking & RAG Parameters
CHUNK_SIZE = 250        # Approximate number of characters/words per document chunk
CHUNK_OVERLAP = 50      # Overlapping characters between consecutive chunks
TOP_K_RESULTS = 3       # Number of relevant chunks retrieved for RAG context
SIMILARITY_THRESHOLD = 0.12  # Minimum similarity score required to consider a match valid

# Conversation Memory
MAX_MEMORY_TURNS = 10   # Maximum number of conversation turns retained in buffer

# External API Configuration (Optional - for advanced LLM expansion)
AI_API_KEY = os.getenv("AI_API_KEY", "")
