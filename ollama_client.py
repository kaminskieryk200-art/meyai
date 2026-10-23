"""Small local-only Ollama client used by the Mey CLI."""

import json
from typing import Dict, Iterable, List

import requests


OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_MAP = {
    "mey:1.0-nano": "mey:1.0-nano",
    "mey:1.0": "mey:1.0",
    "mey:pro": "mey-1.0-pro:latest",
}


def available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def installed_models() -> List[str]:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        response.raise_for_status()
        return [item["name"] for item in response.json().get("models", [])]
    except (requests.RequestException, ValueError, KeyError):
        return []


def chat_stream(model: str, messages: List[Dict[str, str]]) -> Iterable[str]:
    payload = {"model": model, "messages": messages, "stream": True}
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=(5, 600),
    )
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        item = json.loads(line)
        content = item.get("message", {}).get("content", "")
        if content:
            yield content
