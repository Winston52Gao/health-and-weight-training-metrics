"""Main CLI for the local RAG fitness assistant.

Provides two commands:
- ingest : read PDFs and build/save FAISS index + metadata
- query  : run a retrieval-augmented query against the local Ollama model

Run `python main.py ingest` then `python main.py query "your question"`
"""
import argparse
from pathlib import Path
from src.ingest import ingest_pdfs
from src.embeddings import Embedder
from src.faiss_utils import build_faiss_index, save_index, save_metadata
from src.ollama_client import OllamaClient
from src.prompts import build_prompt
from src.query import query_pipeline
import numpy as np


# Default locations
PDF_DIR = Path("data/pdfs")
VSTORE_DIR = Path("vectorstore")
INDEX_PATH = VSTORE_DIR / "faiss.index"
META_PATH = VSTORE_DIR / "metadata.json"


def ingest_command(args):
    print("Ingesting PDFs from:", args.pdf_dir)
    chunks = ingest_pdfs(Path(args.pdf_dir), chunk_size=args.chunk_size, overlap=args.overlap)
    texts = [c["text"] for c in chunks]
    print(f"Created {len(texts)} chunks. Computing embeddings...")
    embedder = Embedder(model_name=args.embed_model)
    embs = embedder.embed(texts)
    print("Building FAISS index...")
    index = build_faiss_index(embs)
    save_index(index, INDEX_PATH)
    # metadata must be in same order as vectors (0..n-1)
    save_metadata(chunks, META_PATH)
    print("Ingestion complete. Index and metadata saved to vectorstore/.")


def query_command(args):
    print("Querying for:", args.question)
    results = query_pipeline(args.question, INDEX_PATH, META_PATH, model_name=args.embed_model, top_k=args.top_k)
    # build prompt and call Ollama
    prompt = build_prompt(args.question, results, )
    client = OllamaClient(host=args.ollama_host, model=args.ollama_model)
    resp = client.generate(prompt, max_tokens=args.max_tokens, temperature=args.temperature)
    print("\n=== Answer ===\n")
    print(resp)
    print("\n=== Retrieved Sources ===\n")
    for r in results:
        print(f"- {r.get('source')} (score={r.get('score'):.3f})")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    ing = sub.add_parser("ingest")
    ing.add_argument("--pdf_dir", default=str(PDF_DIR))
    ing.add_argument("--chunk_size", type=int, default=1000)
    ing.add_argument("--overlap", type=int, default=200)
    ing.add_argument("--embed_model", default="all-MiniLM-L6-v2")

    qry = sub.add_parser("query")
    qry.add_argument("question")
    qry.add_argument("--top_k", type=int, default=5)
    qry.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    qry.add_argument("--ollama_host", default="http://localhost:11434")
    qry.add_argument("--ollama_model", default="llama3.1:8b")
    qry.add_argument("--max_tokens", type=int, default=512)
    qry.add_argument("--temperature", type=float, default=0.0)

    args = p.parse_args()
    if args.cmd == "ingest":
        ingest_command(args)
    elif args.cmd == "query":
        query_command(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
