"""Retrieval pipeline: embed query, search FAISS, and return top chunks with metadata."""
from pathlib import Path
from .embeddings import Embedder
from .faiss_utils import load_index, load_metadata, search
import numpy as np
from typing import List, Dict


def query_pipeline(query: str, index_path: Path, metadata_path: Path, model_name: str = "all-MiniLM-L6-v2", top_k: int = 5):
    """Return list of retrieved chunk dicts with scores."""
    embedder = Embedder(model_name=model_name)
    q_emb = embedder.embed([query])[0]

    index = load_index(index_path)
    metadata = load_metadata(metadata_path)

    scores, ids = search(index, q_emb, top_k=top_k)
    results: List[Dict] = []
    for score, idx in zip(scores, ids):
        # FAISS ids are integer positions; use that to index metadata list
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx].copy()
        m["score"] = float(score)
        results.append(m)
    return results
