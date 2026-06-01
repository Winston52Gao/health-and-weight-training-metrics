"""Prompt construction for RAG queries.

This module creates a system prompt that instructs the LLM to answer based on
retrieved evidence and to cite sources. It also provides a helper to build
the full prompt sent to the model.
"""
from typing import List, Dict


SYSTEM_INSTRUCTIONS = (
    "You are an evidence-based fitness assistant. Answer questions about strength training,"
    " hypertrophy, recovery, and nutrition using ONLY the provided context from training books."
    " If the answer is not contained in the context, say you don't know instead of guessing."
)


def build_prompt(question: str, retrieved: List[Dict], max_context_chars: int = 4000) -> str:
    """Build a single-text prompt combining system instructions, retrieved chunks, and question.

    retrieved: list of dicts with keys 'text' and 'source'
    """
    parts = ["SYSTEM: " + SYSTEM_INSTRUCTIONS, "\n\nCONTEXT:\n"]
    total = 0
    for r in retrieved:
        snippet = r.get("text", "")
        src = r.get("source", "unknown")
        entry = f"Source: {src}\n{snippet}\n---\n"
        if total + len(entry) > max_context_chars:
            break
        parts.append(entry)
        total += len(entry)

    parts.append("\nQUESTION:\n" + question)
    parts.append("\nINSTRUCTIONS: Use the context above to answer; cite sources by filename. Do not hallucinate.")
    return "\n".join(parts)
