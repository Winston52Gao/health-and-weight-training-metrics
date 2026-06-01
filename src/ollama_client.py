"""Simple Ollama HTTP API wrapper.

This sends prompts to a local Ollama instance. Configure host and model.
"""
import requests
from typing import Optional


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self.host = host.rstrip("/")
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Call local Ollama /api/generate and return the generated text.

        Note: Ollama may stream results or return JSON depending on version; this simple wrapper
        attempts a JSON POST and falls back to raw text.
        """
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            # try parse JSON
            try:
                data = resp.json()
                # common shape: {'id':..., 'choices': [{'text': '...'}]} or similar
                if isinstance(data, dict):
                    # try common fields
                    if "text" in data:
                        return data["text"]
                    if "choices" in data and len(data["choices"]) > 0:
                        c = data["choices"][0]
                        return c.get("text") or c.get("message") or str(c)
                    # Ollama may return 'result' or other structure
                    if "result" in data:
                        return str(data["result"])
                return resp.text
            except Exception:
                return resp.text
        except Exception as e:
            return f"Ollama request failed: {e}"
