#!/usr/bin/env python3
"""
run.py - Run any LLM with tool support from the command line.

Usage:
  run.py <model>                     Continue last session (or start new)
  run.py <model> <prompt>            One-shot: run prompt and exit
  run.py <model> --new               Force a fresh session
  run.py <model> --resume <id>       Resume a specific session by id
  run.py <model> --provider <name>   Use a specific provider (default: auto-detect)

Providers:
  ollama (default), venice — see providers.py to add more.

Config:
  .env              - API keys and URLs
  system_prompt.md  - system prompt (editable markdown)
  tools.py          - tool definitions + implementations
"""

import argparse
import json
import os
import readline  # noqa: F401 — imported for input() line-editing support
import sys
import textwrap

from providers import detect_provider, get_chat_fn
from sessions import (
    append_messages,
    create_session,
    export_markdown,
    latest_session,
    list_sessions,
    load_session,
)
from tools import get_tool_specs, run_tool

# ---------------------------------------------------------------------------
# Load .env from same directory as this script
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
# Load system prompt from markdown file
# ---------------------------------------------------------------------------
_prompt_path = os.path.join(_script_dir, "system_prompt.md")
with open(_prompt_path, "r") as _f:
    SYSTEM_PROMPT = _f.read().strip()


# ---------------------------------------------------------------------------
# Agent loop — runs one user turn (may involve multiple tool-call rounds)
# Returns the list of new messages added during this turn.
# ---------------------------------------------------------------------------

def agent_turn(chat_fn, model, messages):
    new_messages = []

    for _ in range(10):
        result = chat_fn(model, messages, get_tool_specs())
        if result is None:
            return new_messages

        content = result.get("content", "").strip()
        tool_calls = result.get("tool_calls", [])

        if not tool_calls:
            assistant_msg = {"role": "assistant", "content": content}
            messages.append(assistant_msg)
            new_messages.append(assistant_msg)
            print(content)
            return new_messages

        assistant_msg = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }
        messages.append(assistant_msg)
        new_messages.append(assistant_msg)

        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
            call_id = call.get("id", "call_0")

            print(f"🔧 {name}({json.dumps(args)})", file=sys.stderr)
            output = run_tool(name, args)
            print(f"   → {output[:300]}{'...' if len(output) > 300 else ''}", file=sys.stderr)

            tool_msg = {"role": "tool", "tool_call_id": call_id, "content": output}
            messages.append(tool_msg)
            new_messages.append(tool_msg)

    print("ERROR: max iterations reached", file=sys.stderr)
    return new_messages


# ---------------------------------------------------------------------------
# One-shot mode (no session persistence)
# ---------------------------------------------------------------------------

def run_once(chat_fn, model, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    print(f"🤖 {model}", file=sys.stderr)
    agent_turn(chat_fn, model, messages)


# ---------------------------------------------------------------------------
# Session recap display
# ---------------------------------------------------------------------------

_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

def _display_recap(messages, max_turns=4):
    """Show last N user/assistant exchanges so you remember where you left off."""
    pairs = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if role == "user":
            pairs.append({"user": content, "assistant": None})
        elif role == "assistant" and content and pairs:
            pairs[-1]["assistant"] = content

    recent = [p for p in pairs if p["user"]][-max_turns:]
    if not recent:
        return

    width = min(os.get_terminal_size().columns, 90)
    print(f"\n{_DIM}{'─' * width}")
    print(f"  Last {len(recent)} exchange{'s' if len(recent) != 1 else ''}:{_RESET}")

    for p in recent:
        user_text = textwrap.shorten(p["user"], width=width - 8, placeholder="…")
        print(f"  {_BOLD}You:{_RESET} {user_text}")
        if p["assistant"]:
            assistant_text = textwrap.shorten(p["assistant"], width=width - 8, placeholder="…")
            print(f"  {_CYAN}AI:{_RESET}  {assistant_text}")
        print()

    print(f"{_DIM}{'─' * width}{_RESET}\n")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def _handle_slash(cmd, session_id, model, messages):
    """Handle a slash command. Returns 'exit' to quit, 'new_session' to clear, or None."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    if command in ("/exit", "/quit"):
        return "exit"

    elif command == "/clear":
        return "new_session"

    elif command == "/history":
        sessions = list_sessions(model=model)
        if not sessions:
            print("No saved sessions.")
        else:
            print(f"{'ID':<32} {'Model':<20} {'Updated'}")
            print("-" * 76)
            for s in sessions[:20]:
                marker = " ← current" if s["id"] == session_id else ""
                print(f"{s['id']:<32} {s['model']:<20} {s['updated']}{marker}")

    elif command == "/save":
        path = arg or f"session-{session_id}.md"
        md = export_markdown(session_id)
        with open(path, "w") as f:
            f.write(md)
        print(f"Saved to {path}")

    elif command == "/help":
        print("Commands:")
        print("  /help             Show this help")
        print("  /exit  /quit      Exit the session")
        print("  /clear            Start a fresh conversation")
        print("  /history          List past sessions")
        print("  /save [path]      Export session as markdown")

    else:
        print(f"Unknown command: {command}")
        print("Type /help for available commands.")

    return None


def run_interactive(chat_fn, model, resume_id=None):
    if resume_id:
        session_id = resume_id
        meta, messages = load_session(session_id)
        messages[0] = {"role": "system", "content": SYSTEM_PROMPT}
        turn_count = sum(1 for m in messages if m.get("role") == "user")
        print(f"🤖 {model} (resumed · {turn_count} turns · {session_id[:15]}…)", file=sys.stderr)
        _display_recap(messages)
    else:
        session_id = create_session(model)
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        messages = [system_msg]
        append_messages(session_id, [system_msg])
        print(f"🤖 {model} (new session)", file=sys.stderr)

    print("Type /help for commands.\n", file=sys.stderr)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            result = _handle_slash(user_input, session_id, model, messages)
            if result == "exit":
                break
            elif result == "new_session":
                session_id = create_session(model)
                system_msg = {"role": "system", "content": SYSTEM_PROMPT}
                messages = [system_msg]
                append_messages(session_id, [system_msg])
                print(f"🗑  Cleared. New session {session_id}", file=sys.stderr)
            continue

        user_msg = {"role": "user", "content": user_input}
        messages.append(user_msg)
        append_messages(session_id, [user_msg])

        new_messages = agent_turn(chat_fn, model, messages)
        if new_messages:
            append_messages(session_id, new_messages)

        print()

    print(f"Session saved: {session_id}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run an LLM with tool support.",
        usage="%(prog)s <model> [prompt...] [--provider NAME] [--new | --resume SESSION_ID]",
    )
    parser.add_argument("model", help="Model name (e.g. qwen3:32b, venice-uncensored)")
    parser.add_argument("prompt", nargs="*", help="Prompt (omit for interactive mode)")
    parser.add_argument(
        "--provider", default=None,
        help="Provider backend: ollama, venice, ... (default: auto-detect from model name)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--new", action="store_true",
        help="Start a fresh session instead of continuing the last one",
    )
    group.add_argument(
        "--resume", default=None, metavar="SESSION_ID",
        help="Resume a specific session by id",
    )

    args = parser.parse_args()
    prompt = " ".join(args.prompt) if args.prompt else None

    if prompt and args.resume:
        parser.error("Cannot combine a prompt with --resume")

    provider_name = args.provider or detect_provider(args.model)
    try:
        chat_fn = get_chat_fn(provider_name)
    except KeyError as e:
        parser.error(str(e))

    print(f"Provider: {provider_name}", file=sys.stderr)

    if prompt:
        run_once(chat_fn, args.model, prompt)
    else:
        resume_id = None
        if args.resume:
            resume_id = args.resume
        elif not args.new:
            resume_id = latest_session(model=args.model)
        run_interactive(chat_fn, args.model, resume_id=resume_id)


if __name__ == "__main__":
    main()
