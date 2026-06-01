"""FAISS utilities: create, save, load index and manage metadata.

Metadata is stored as JSON mapping from internal index id -> chunk metadata.
"""
from pathlib import Path
import faiss
import numpy as np
import json
from typing import List, Dict, Any


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create an IndexFlatIP FAISS index and add normalized embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # normalize vectors for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.IndexFlatIP:
    return faiss.read_index(str(path))


def save_metadata(metadata: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_metadata(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def search(index: faiss.IndexFlatIP, query_emb: np.ndarray, top_k: int = 5):
    """Search the index with a single query embedding (1xdim). Returns (scores, ids)."""
    # ensure normalized
    qe = np.array(query_emb, dtype="float32")
    faiss.normalize_L2(qe)
    D, I = index.search(qe.reshape(1, -1), top_k)
    return D[0].tolist(), I[0].tolist()
