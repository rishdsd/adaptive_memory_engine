from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Minimal Ollama client using Python's standard library."""

    def __init__(self, model: str = "qwen3.5:2b", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, system_prompt: str, user_prompt: str, timeout: int = 300) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.2},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(
                "Could not reach Ollama. Check that Ollama is running and the model is installed. "
                f"Details: {exc}"
            ) from exc
        except (json.JSONDecodeError, KeyError) as exc:
            raise OllamaError(f"Ollama returned an invalid response: {exc}") from exc

        message = result.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError(f"Ollama response did not contain message.content: {result}")
        return content.strip()
