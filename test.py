#!/usr/bin/env python3
"""
test.py - Manual test script for tools, memory, and providers.

Usage:
  python test.py              Run all tests
  python test.py tools        Run only tool tests
  python test.py memory       Run only memory integration tests
  python test.py providers    Run only provider tests
"""

import os
import sys

# ---------------------------------------------------------------------------
# Load .env (same logic as run.py)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))

_env_path = os.path.join(_script_dir, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0
skipped = 0


def ok(name, detail=""):
    global passed
    passed += 1
    suffix = f"  {DIM}{detail}{RESET}" if detail else ""
    print(f"  {GREEN}[PASS]{RESET} {name}{suffix}")


def fail(name, reason=""):
    global failed
    failed += 1
    suffix = f"  {RED}{reason}{RESET}" if reason else ""
    print(f"  {RED}[FAIL]{RESET} {name}{suffix}")


def skip(name, reason=""):
    global skipped
    skipped += 1
    suffix = f"  {DIM}{reason}{RESET}" if reason else ""
    print(f"  {YELLOW}[SKIP]{RESET} {name}{suffix}")


def section(title):
    print(f"\n{BOLD}--- {title} ---{RESET}")


# ---------------------------------------------------------------------------
# 1. Tool tests
# ---------------------------------------------------------------------------

def test_tools():
    from tools import run_tool

    section("Tools")

    # shell
    out = run_tool("shell", {"cmd": "echo hello"})
    if "hello" in out:
        ok("shell")
    else:
        fail("shell", f"expected 'hello', got: {out[:100]}")

    # write_file + read_file
    tmp = os.path.join(_script_dir, "_test_tmp.txt")
    payload = "dead-simple-agent test content 12345"
    out = run_tool("write_file", {"path": tmp, "content": payload})
    if "ERROR" in out:
        fail("write_file", out)
    else:
        ok("write_file")
        out = run_tool("read_file", {"path": tmp})
        if out.strip() == payload:
            ok("read_file")
        else:
            fail("read_file", f"content mismatch: {out[:80]}")
        os.unlink(tmp)

    # fetch_url
    out = run_tool("fetch_url", {"url": "https://httpbin.org/get", "as_text": False})
    if "headers" in out.lower() or "origin" in out.lower():
        ok("fetch_url")
    elif "ERROR" in out:
        fail("fetch_url", out[:120])
    else:
        fail("fetch_url", f"unexpected response: {out[:120]}")

    # memory_write
    test_content = "# Test Memory\n\nThis is a test memory about ethereum and solidity."
    out = run_tool("memory_write", {"filename": "_test_memory.md", "content": test_content})
    if "saved" in out.lower():
        ok("memory_write")
    else:
        fail("memory_write", out[:100])

    # memory_read
    out = run_tool("memory_read", {"filename": "_test_memory.md"})
    if out.strip() == test_content.strip():
        ok("memory_read")
    else:
        fail("memory_read", f"content mismatch: {out[:80]}")

    # memory_list
    out = run_tool("memory_list", {})
    if "_test_memory.md" in out and "# Test Memory" in out:
        ok("memory_list")
    else:
        fail("memory_list", f"test file not found in listing: {out[:120]}")

    # memory_list with limit
    out = run_tool("memory_list", {"limit": 1})
    lines = [l for l in out.strip().split("\n") if l.strip()]
    if len(lines) <= 1:
        ok("memory_list (limit)")
    else:
        fail("memory_list (limit)", f"expected <=1 line, got {len(lines)}")

    # memory_search
    out = run_tool("memory_search", {"query": "ethereum"})
    if "_test_memory.md" in out and "ethereum" in out.lower():
        ok("memory_search", "found 'ethereum' in test file")
    else:
        fail("memory_search", f"search miss: {out[:120]}")

    # memory_search (no match)
    out = run_tool("memory_search", {"query": "zzz_nonexistent_zzz"})
    if "no memories match" in out.lower():
        ok("memory_search (no match)")
    else:
        fail("memory_search (no match)", f"expected no-match message: {out[:80]}")

    # cleanup
    test_file = os.path.join(_script_dir, "memory", "_test_memory.md")
    if os.path.exists(test_file):
        os.unlink(test_file)


# ---------------------------------------------------------------------------
# 2. Memory system integration
# ---------------------------------------------------------------------------

def test_memory_integration():
    from run import SYSTEM_PROMPT

    section("Memory Integration")

    if "{{MEMORY}}" in SYSTEM_PROMPT:
        fail("{{MEMORY}} replaced", "raw placeholder still in prompt")
    else:
        ok("{{MEMORY}} replaced")

    if "{{TOOLS}}" in SYSTEM_PROMPT:
        fail("{{TOOLS}} replaced", "raw placeholder still in prompt")
    else:
        ok("{{TOOLS}} replaced")

    critical_path = os.path.join(_script_dir, "memory", "critical.md")
    if os.path.exists(critical_path):
        with open(critical_path) as f:
            content = f.read().strip()
        if content and content in SYSTEM_PROMPT:
            ok("critical.md in prompt", f"{len(content)} chars loaded")
        else:
            fail("critical.md in prompt", "file exists but content not found in prompt")
    else:
        if "No critical memories yet" in SYSTEM_PROMPT:
            ok("critical.md fallback", "placeholder text present")
        else:
            fail("critical.md fallback", "no fallback text found")

    if "memory_list" in SYSTEM_PROMPT and "memory_search" in SYSTEM_PROMPT:
        ok("memory tools in prompt")
    else:
        fail("memory tools in prompt", "memory tool names missing from system prompt")


# ---------------------------------------------------------------------------
# 3. Provider connectivity
# ---------------------------------------------------------------------------

def _detect_ollama_model():
    """Pick the smallest available ollama model, or None."""
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.strip().split("\n")[1:]:
            name = line.split()[0]
            if name:
                return name
    except Exception:
        pass
    return None


PROVIDER_TESTS = {
    "ollama": {
        "model": _detect_ollama_model(),
        "skip_if": lambda: _detect_ollama_model() is None,
        "skip_reason": "no ollama models available (try: ollama pull qwen2.5:7b)",
    },
    "venice": {
        "model": "llama-3.3-70b",
        "skip_if": lambda: not os.environ.get("VENICE_API_KEY"),
        "skip_reason": "no VENICE_API_KEY",
    },
    "anthropic": {
        "model": "claude-sonnet-4-20250514",
        "skip_if": lambda: not os.environ.get("ANTHROPIC_API_KEY"),
        "skip_reason": "no ANTHROPIC_API_KEY",
    },
    "bankr": {
        "model": "gpt-5.4-nano",
        "skip_if": lambda: not os.environ.get("BANKR_API_KEY"),
        "skip_reason": "no BANKR_API_KEY",
    },
}


def test_providers():
    from providers import get_chat_fn

    section("Providers")

    for name, cfg in PROVIDER_TESTS.items():
        label = f"{name} ({cfg['model']})"

        if cfg["skip_if"]():
            skip(label, cfg["skip_reason"])
            continue

        chat_fn = get_chat_fn(name)
        messages = [
            {"role": "system", "content": "You are a test bot."},
            {"role": "user", "content": "Say hello in exactly one word."},
        ]

        try:
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")
            result = chat_fn(cfg["model"], messages, [])
            sys.stderr.close()
            sys.stderr = old_stderr
        except Exception as e:
            sys.stderr = old_stderr
            fail(label, str(e)[:120])
            continue

        if result is None:
            fail(label, "auth or connection error")
        elif result.get("content", "").strip():
            ok(label, result["content"].strip()[:60])
        else:
            fail(label, "empty content in response")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sections = sys.argv[1:] if len(sys.argv) > 1 else ["tools", "memory", "providers"]

    if "tools" in sections:
        test_tools()
    if "memory" in sections:
        test_memory_integration()
    if "providers" in sections:
        test_providers()

    print(f"\n{BOLD}{passed} passed, {failed} failed, {skipped} skipped{RESET}")
    sys.exit(1 if failed else 0)
