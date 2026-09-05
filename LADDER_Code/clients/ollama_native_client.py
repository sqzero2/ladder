"""Minimal synchronous Ollama client compatible with the project's LLM adapter.

Uses Ollama's native API so thinking can be disabled. The OpenAI-compatible
endpoint currently ignores `think=false` for qwen3.5 and may return an empty
content field when the prediction budget is consumed by hidden reasoning.
"""

from __future__ import annotations

import json
import urllib.request


class OllamaNativeClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        max_completion_tokens: int = 1800,
        timeout: int = 600,
        keep_alive: str = "30m",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.last_usage = None
        print("Ollama native client initialized")
        print(f"   Base URL: {self.base_url}")
        print(f"   Model: {self.model}")
        print(f"   Temperature: {self.temperature}")

    def chat_messages(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_completion_tokens,
            },
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        self.last_usage = {
            "prompt_tokens": result.get("prompt_eval_count"),
            "completion_tokens": result.get("eval_count"),
            "prompt_duration_ns": result.get("prompt_eval_duration"),
            "completion_duration_ns": result.get("eval_duration"),
        }
        return str((result.get("message") or {}).get("content") or "").strip()
