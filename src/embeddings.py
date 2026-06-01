"""Embedding utilities using sentence-transformers.

This module wraps a SentenceTransformer to create embeddings for texts.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load model once; this may download weights on first run.
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return float32 numpy array of embeddings for the input texts."""
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return np.array(embs, dtype="float32")
