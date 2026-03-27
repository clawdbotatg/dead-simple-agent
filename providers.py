"""
providers.py - Pluggable LLM provider backends.

Each provider is a function:

    chat(model, messages, tool_specs) -> dict | None

Returns {"content": "...", "tool_calls": [...]} on success, None on error.
tool_calls use the shape: [{"id": "...", "function": {"name": "...", "arguments": ...}}]

To add a new provider:
  1. Write a chat function with the signature above
  2. Add it to PROVIDERS at the bottom
  3. (Optional) Add prefix rules to detect_provider()
"""

import json
import os
import sys

import requests


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def ollama_chat(model, messages, tool_specs):
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
    try:
        resp = requests.post(url, json={
            "model": model,
            "messages": messages,
            "tools": tool_specs,
            "stream": False,
        }, timeout=300)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to Ollama at {url}", file=sys.stderr)
        print("Is Ollama running? Check OLLAMA_URL in .env", file=sys.stderr)
        return None
    except requests.Timeout:
        print("ERROR: Ollama request timed out (300s)", file=sys.stderr)
        return None

    if resp.status_code == 404:
        print(f"ERROR: Model '{model}' not found at {url}", file=sys.stderr)
        print("Try: ollama pull <model>", file=sys.stderr)
        return None
    if not resp.ok:
        print(f"ERROR: Ollama returned {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        return None

    msg = resp.json().get("message", {})
    return {
        "content": msg.get("content", ""),
        "tool_calls": msg.get("tool_calls", []),
    }


# ---------------------------------------------------------------------------
# Venice AI  (OpenAI-compatible)
# ---------------------------------------------------------------------------

def venice_chat(model, messages, tool_specs):
    api_key = os.environ.get("VENICE_API_KEY", "")
    base_url = os.environ.get("VENICE_BASE_URL", "https://api.venice.ai/api/v1")

    if not api_key:
        print("ERROR: VENICE_API_KEY not set in .env", file=sys.stderr)
        return None

    return _openai_compatible_chat(model, messages, tool_specs, api_key, base_url)


# ---------------------------------------------------------------------------
# Shared: OpenAI-compatible request/response handling
#
# Any provider that speaks the OpenAI /v1/chat/completions format can reuse
# this.  Just call it with the right api_key and base_url.
# ---------------------------------------------------------------------------

def _openai_compatible_chat(model, messages, tool_specs, api_key, base_url):
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
    }
    if tool_specs:
        body["tools"] = tool_specs

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=300)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {url}", file=sys.stderr)
        return None
    except requests.Timeout:
        print(f"ERROR: Request to {url} timed out (300s)", file=sys.stderr)
        return None

    # If the API rejects the tools param, retry without tools so the model
    # can still answer (just without tool use).
    if resp.status_code == 400 and "tools" in resp.text.lower() and "tools" in body:
        print(f"WARN: {model} does not support tools, retrying without", file=sys.stderr)
        del body["tools"]
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=300)
        except (requests.ConnectionError, requests.Timeout):
            return None

    if not resp.ok:
        print(f"ERROR: {url} returned {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
        return None

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        print("ERROR: No choices in response", file=sys.stderr)
        return None

    msg = choices[0].get("message", {})
    tool_calls = msg.get("tool_calls", [])

    return {
        "content": msg.get("content", "") or "",
        "tool_calls": tool_calls,
    }


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDERS = {
    "ollama": ollama_chat,
    "venice": venice_chat,
}


def detect_provider(model):
    """Guess the provider from the model name. Falls back to ollama."""
    model_lower = model.lower()

    # Venice models
    if "venice" in model_lower:
        return "venice"

    # OpenAI models (for when you add an openai provider)
    # if model_lower.startswith(("gpt-", "o1", "o3", "o4")):
    #     return "openai"

    # Anthropic models
    # if model_lower.startswith("claude"):
    #     return "anthropic"

    return "ollama"


def get_chat_fn(provider_name):
    """Look up a provider by name. Raises KeyError with a helpful message."""
    if provider_name not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS.keys()))
        raise KeyError(f"Unknown provider '{provider_name}'. Available: {available}")
    return PROVIDERS[provider_name]
