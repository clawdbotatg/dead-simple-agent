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
import time as _time
from datetime import datetime as _datetime

import requests


# ---------------------------------------------------------------------------
# Cost tracking and per-call logging
# ---------------------------------------------------------------------------

PRICING = {
    "claude-opus-4.6": (15.0, 75.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-sonnet-4.6": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-3.5": (0.80, 4.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "minimax-m2.7": (0.50, 1.50),
}

cumulative_cost = 0.0
cumulative_input_tokens = 0
cumulative_output_tokens = 0


def estimate_cost(model, input_tokens, output_tokens):
    prices = PRICING.get(model, (5.0, 15.0))
    return (input_tokens / 1_000_000 * prices[0]) + (output_tokens / 1_000_000 * prices[1])


def context_chars(messages):
    total = 0
    for m in messages:
        total += len(m.get("content", "") or "")
        for tc in m.get("tool_calls", []):
            args = tc.get("function", {}).get("arguments", "")
            total += len(args) if isinstance(args, str) else len(json.dumps(args))
    return total


def _log_api(model, messages, usage, elapsed_s):
    """Update cumulative cost counters from API response usage data."""
    global cumulative_cost, cumulative_input_tokens, cumulative_output_tokens
    tok_in = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    tok_out = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    cost = estimate_cost(model, tok_in, tok_out)
    cumulative_cost += cost
    cumulative_input_tokens += tok_in
    cumulative_output_tokens += tok_out


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
# Bankr LLM Gateway  (OpenAI-compatible, multi-provider)
# https://docs.bankr.bot/llm-gateway/overview
# ---------------------------------------------------------------------------

def bankr_chat(model, messages, tool_specs):
    api_key = os.environ.get("BANKR_API_KEY", "")
    base_url = os.environ.get("BANKR_BASE_URL", "https://llm.bankr.bot/v1")

    if not api_key:
        print("ERROR: BANKR_API_KEY not set in .env", file=sys.stderr)
        return None

    return _openai_compatible_chat(model, messages, tool_specs, api_key, base_url)


# ---------------------------------------------------------------------------
# OpenRouter  (OpenAI-compatible, multi-provider)
# https://openrouter.ai/docs
# ---------------------------------------------------------------------------

def openrouter_chat(model, messages, tool_specs):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in .env", file=sys.stderr)
        return None

    return _openai_compatible_chat(model, messages, tool_specs, api_key, base_url)


# ---------------------------------------------------------------------------
# Anthropic  (native Messages API)
# ---------------------------------------------------------------------------

def _convert_tools_to_anthropic(tool_specs):
    """OpenAI tool specs -> Anthropic tool specs."""
    tools = []
    for spec in tool_specs:
        fn = spec.get("function", {})
        tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return tools


def _convert_messages_to_anthropic(messages):
    """OpenAI-style messages -> Anthropic (system, messages) pair.

    Handles: system extraction, assistant tool_calls -> tool_use blocks,
    tool role -> user tool_result blocks, and merging consecutive same-role
    messages (which Anthropic forbids).
    """
    system = ""
    anthropic_msgs = []

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            system = msg.get("content", "")
            continue

        if role == "user":
            anthropic_msgs.append({"role": "user", "content": msg["content"]})

        elif role == "assistant":
            content_blocks = []
            text = (msg.get("content") or "").strip()
            if text:
                content_blocks.append({"type": "text", "text": text})

            for tc in msg.get("tool_calls", []):
                fn = tc["function"]
                args = fn["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", "call_0"),
                    "name": fn["name"],
                    "input": args,
                })

            anthropic_msgs.append({
                "role": "assistant",
                "content": content_blocks if content_blocks else [{"type": "text", "text": ""}],
            })

        elif role == "tool":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", "call_0"),
                "content": msg.get("content", ""),
            }
            if anthropic_msgs and anthropic_msgs[-1]["role"] == "user" and isinstance(anthropic_msgs[-1]["content"], list):
                anthropic_msgs[-1]["content"].append(tool_result)
            else:
                anthropic_msgs.append({"role": "user", "content": [tool_result]})

    return system, anthropic_msgs


def anthropic_chat(model, messages, tool_specs):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
        return None

    system, anthropic_msgs = _convert_messages_to_anthropic(messages)

    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = {"anthropic-version": "2023-06-01", "Content-Type": "application/json", "x-api-key": api_key}
    body = {
        "model": model,
        "max_tokens": 4096,
        "messages": anthropic_msgs,
    }
    if system:
        body["system"] = system

    anthropic_tools = _convert_tools_to_anthropic(tool_specs) if tool_specs else []
    if anthropic_tools:
        body["tools"] = anthropic_tools

    t0 = _time.time()
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=300)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {url}", file=sys.stderr)
        return None
    except requests.Timeout:
        print(f"ERROR: Request to {url} timed out (300s)", file=sys.stderr)
        return None
    elapsed = _time.time() - t0

    if not resp.ok:
        print(f"ERROR: {url} returned {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
        return None

    data = resp.json()
    usage = data.get("usage", {})
    _log_api(model, messages, usage, elapsed)

    content_blocks = data.get("content", [])

    text_parts = []
    tool_calls = []
    for block in content_blocks:
        if block["type"] == "text":
            text_parts.append(block["text"])
        elif block["type"] == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "function": {
                    "name": block["name"],
                    "arguments": block["input"],
                },
            })

    return {
        "content": "\n".join(text_parts),
        "tool_calls": tool_calls,
    }


# ---------------------------------------------------------------------------
# Shared: OpenAI-compatible request/response handling
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

    t0 = _time.time()
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=300)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {url}", file=sys.stderr)
        return None
    except requests.Timeout:
        print(f"ERROR: Request to {url} timed out (300s)", file=sys.stderr)
        return None

    if resp.status_code == 400 and "tools" in resp.text.lower() and "tools" in body:
        print(f"WARN: {model} does not support tools, retrying without", file=sys.stderr)
        del body["tools"]
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=300)
        except (requests.ConnectionError, requests.Timeout):
            return None

    elapsed = _time.time() - t0

    if not resp.ok:
        print(f"ERROR: {url} returned {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
        return None

    data = resp.json()
    usage = data.get("usage", {})
    _log_api(model, messages, usage, elapsed)

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
    "bankr": bankr_chat,
    "openrouter": openrouter_chat,
    "anthropic": anthropic_chat,
}


_BANKR_PREFIXES = ("claude-", "gpt-", "gemini-", "kimi-", "qwen3-")

def detect_provider(model):
    """Guess the provider from the model name. Falls back to ollama."""
    model_lower = model.lower()

    if "venice" in model_lower:
        return "venice"

    if model_lower.startswith("openrouter/") and os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"

    if model_lower.startswith("claude") and os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"

    if model_lower.startswith(_BANKR_PREFIXES) and os.environ.get("BANKR_API_KEY"):
        return "bankr"

    return "ollama"


def get_chat_fn(provider_name):
    """Look up a provider by name. Raises KeyError with a helpful message."""
    if provider_name not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS.keys()))
        raise KeyError(f"Unknown provider '{provider_name}'. Available: {available}")
    return PROVIDERS[provider_name]
