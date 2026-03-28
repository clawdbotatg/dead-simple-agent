"""
core.py - Agent class that wires tools, providers, sessions, and memory together.

Usage:
    from agent import Agent
    Agent().cli()                          # default agent (uses CWD files)
    Agent(extra_tools=MY_TOOLS).cli()      # custom agent with extra tools
"""

import argparse
import json
import os
import readline  # noqa: F401 — imported for input() line-editing support
import sys
import textwrap

from .providers import detect_provider, get_chat_fn
from .sessions import (
    append_messages,
    create_session,
    export_markdown,
    latest_session,
    list_sessions,
    load_session,
)
from .tools import BASE_TOOLS, get_tool_specs, get_tool_summary, make_memory_tools, run_tool

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PROMPT = os.path.join(_PACKAGE_DIR, "default_prompt.md")


def _load_env(env_file):
    """Load a .env file into os.environ (setdefault, won't override)."""
    if not os.path.exists(env_file):
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class Agent:
    """A configurable agent that can be extended with custom tools and prompts.

    All paths default to CWD-relative, so each agent is just a directory.
    """

    def __init__(
        self,
        system_prompt=None,
        extra_tools=None,
        memory_dir=None,
        sessions_dir=None,
        env_file=None,
        max_iterations=10,
    ):
        cwd = os.getcwd()

        # Load .env
        env_file = env_file or os.path.join(cwd, ".env")
        _load_env(env_file)

        # Resolve directories
        self.memory_dir = memory_dir or os.path.join(cwd, "memory")
        self.sessions_dir = sessions_dir or os.path.join(cwd, "sessions")

        # Build tool registry: base + memory (bound to dir) + extras
        self.tools = list(BASE_TOOLS) + make_memory_tools(self.memory_dir)
        if extra_tools:
            self.tools.extend(extra_tools)

        # Config
        self.max_iterations = max_iterations

        # Build system prompt
        self.system_prompt = self._build_prompt(system_prompt)

    def _build_prompt(self, system_prompt):
        """Load and render the system prompt with {{TOOLS}} and {{MEMORY}}."""
        if system_prompt and not os.path.isfile(system_prompt):
            raw = system_prompt
        else:
            prompt_path = system_prompt or os.path.join(os.getcwd(), "system_prompt.md")
            if not os.path.exists(prompt_path):
                prompt_path = _DEFAULT_PROMPT
            with open(prompt_path, "r") as f:
                raw = f.read().strip()

        tool_summary = get_tool_summary(self.tools)

        critical_path = os.path.join(self.memory_dir, "critical.md")
        critical_memory = ""
        if os.path.exists(critical_path):
            with open(critical_path, "r") as f:
                critical_memory = f.read().strip()
        critical_memory = critical_memory or "(No critical memories yet. Use memory_write with filename 'critical.md' to create one.)"

        return raw.replace("{{TOOLS}}", tool_summary).replace("{{MEMORY}}", critical_memory)

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    def agent_turn(self, chat_fn, model, messages):
        """Run one user turn (may involve multiple tool-call rounds)."""
        new_messages = []

        for _ in range(self.max_iterations):
            result = chat_fn(model, messages, get_tool_specs(self.tools))
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

                print(f"\U0001f527 {name}({json.dumps(args)})", file=sys.stderr)
                output = run_tool(self.tools, name, args)
                print(f"   \u2192 {output[:300]}{'...' if len(output) > 300 else ''}", file=sys.stderr)

                tool_msg = {"role": "tool", "tool_call_id": call_id, "content": output}
                messages.append(tool_msg)
                new_messages.append(tool_msg)

        print("ERROR: max iterations reached", file=sys.stderr)
        return new_messages

    # ------------------------------------------------------------------
    # One-shot mode
    # ------------------------------------------------------------------

    def run_once(self, chat_fn, model, prompt):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        print(f"\U0001f916 {model}", file=sys.stderr)
        self.agent_turn(chat_fn, model, messages)

    # ------------------------------------------------------------------
    # Interactive mode
    # ------------------------------------------------------------------

    _DIM = "\033[2m"
    _BOLD = "\033[1m"
    _CYAN = "\033[36m"
    _RESET = "\033[0m"

    def _display_recap(self, messages, max_turns=4):
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
        print(f"\n{self._DIM}{'─' * width}")
        print(f"  Last {len(recent)} exchange{'s' if len(recent) != 1 else ''}:{self._RESET}")

        for p in recent:
            user_text = textwrap.shorten(p["user"], width=width - 8, placeholder="…")
            print(f"  {self._BOLD}You:{self._RESET} {user_text}")
            if p["assistant"]:
                assistant_text = textwrap.shorten(p["assistant"], width=width - 8, placeholder="…")
                print(f"  {self._CYAN}AI:{self._RESET}  {assistant_text}")
            print()

        print(f"{self._DIM}{'─' * width}{self._RESET}\n")

    def _handle_slash(self, cmd, session_id, model, messages):
        parts = cmd.strip().split(None, 1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        if command in ("/exit", "/quit"):
            return "exit"

        elif command == "/clear":
            return "new_session"

        elif command == "/history":
            sessions = list_sessions(self.sessions_dir, model=model)
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
            md = export_markdown(self.sessions_dir, session_id)
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

    def run_interactive(self, chat_fn, model, resume_id=None):
        sd = self.sessions_dir

        if resume_id:
            session_id = resume_id
            meta, messages = load_session(sd, session_id)
            messages[0] = {"role": "system", "content": self.system_prompt}
            turn_count = sum(1 for m in messages if m.get("role") == "user")
            print(f"\U0001f916 {model} (resumed · {turn_count} turns · {session_id[:15]}…)", file=sys.stderr)
            self._display_recap(messages)
        else:
            session_id = create_session(sd, model)
            system_msg = {"role": "system", "content": self.system_prompt}
            messages = [system_msg]
            append_messages(sd, session_id, [system_msg])
            print(f"\U0001f916 {model} (new session)", file=sys.stderr)

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
                result = self._handle_slash(user_input, session_id, model, messages)
                if result == "exit":
                    break
                elif result == "new_session":
                    session_id = create_session(sd, model)
                    system_msg = {"role": "system", "content": self.system_prompt}
                    messages = [system_msg]
                    append_messages(sd, session_id, [system_msg])
                    print(f"\U0001f5d1  Cleared. New session {session_id}", file=sys.stderr)
                continue

            user_msg = {"role": "user", "content": user_input}
            messages.append(user_msg)
            append_messages(sd, session_id, [user_msg])

            new_messages = self.agent_turn(chat_fn, model, messages)
            if new_messages:
                append_messages(sd, session_id, new_messages)

            print()

        print(f"Session saved: {session_id}", file=sys.stderr)

    # ------------------------------------------------------------------
    # CLI entry point
    # ------------------------------------------------------------------

    def cli(self):
        """Full CLI with argparse — same flags as the original run.py."""
        parser = argparse.ArgumentParser(
            description="Run an LLM with tool support.",
            usage="%(prog)s <model> [prompt...] [--provider NAME] [--new | --resume SESSION_ID]",
        )
        parser.add_argument("model", help="Model name (e.g. qwen3:32b, openrouter/auto)")
        parser.add_argument("prompt", nargs="*", help="Prompt (omit for interactive mode)")
        parser.add_argument(
            "--provider", default=None,
            help="Provider backend: ollama, venice, openrouter, ... (default: auto-detect)",
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
            self.run_once(chat_fn, args.model, prompt)
        else:
            resume_id = None
            if args.resume:
                resume_id = args.resume
            elif not args.new:
                resume_id = latest_session(self.sessions_dir, model=args.model)
            self.run_interactive(chat_fn, args.model, resume_id=resume_id)


def _cli_entry():
    """Entry point for the `dead-simple-agent` console script."""
    Agent().cli()
