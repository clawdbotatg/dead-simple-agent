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
import time

from .providers import detect_provider, get_chat_fn, context_chars
from .sessions import (
    append_messages,
    create_session,
    export_markdown,
    latest_session,
    list_sessions,
    load_session,
)
from .tools import BASE_TOOLS, get_tool_specs, get_tool_summary, make_memory_tools, run_tool
from .subagent import SkillCache, make_subagent_tools, _fetch_skill

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PROMPT = os.path.join(_PACKAGE_DIR, "default_prompt.md")


def _log_agent(msg):
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)


def _truncate_args(args):
    """Summarize tool args for logging — truncate long values like file content."""
    if not args:
        return "{}"
    short = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 120:
            short[k] = v[:60] + f"...({len(v)} chars)"
        else:
            short[k] = v
    return json.dumps(short)


def _compact_context(messages, keep_recent=10):
    """Truncate old tool outputs to reduce context size.

    Preserves: system msg, user msg, and the last `keep_recent` messages.
    Truncates tool message content to 200 chars for everything older.
    Aggressively strips write_file content args (already on disk).
    """
    if len(messages) <= keep_recent + 2:
        return 0
    cutoff = len(messages) - keep_recent
    saved = 0
    for i in range(cutoff):
        msg = messages[i]
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if len(content) > 250:
                original_len = len(content)
                msg["content"] = content[:200] + f"\n... [truncated from {original_len} chars]"
                saved += original_len - len(msg["content"])
        elif msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                if isinstance(args, dict):
                    # write_file content is on disk — replace with summary
                    if name == "write_file" and "content" in args:
                        content_len = len(args.get("content", ""))
                        if content_len > 100:
                            saved += content_len - 60
                            args["content"] = f"[written to disk, {content_len} chars]"
                    else:
                        args_str = json.dumps(args)
                        if len(args_str) > 250:
                            fn["arguments"] = {"_truncated": args_str[:200] + f"... [{len(args_str)} chars]"}
                            saved += len(args_str) - 250
                elif isinstance(args, str) and len(args) > 250:
                    original_len = len(args)
                    fn["arguments"] = args[:200] + f"... [truncated from {original_len} chars]"
                    saved += original_len - len(fn["arguments"])
    return saved


_STAGE_SIGNALS = {
    "deep_fetch": "loading skill docs",
    "leftclaw_get_job": "reading job description",
    "leftclaw_get_messages": "checking client messages",
    "leftclaw_accept_job": "accepting job",
    "leftclaw_log_work": "logging stage progress on-chain",
    "leftclaw_complete_job": "completing job",
    "github_list_repos": "reconnaissance — scanning repos",
    "github_read_file": "reconnaissance — reading repo files",
    "github_create_issue": "filing audit issues",
    "github_create_pr": "creating pull request",
    "github_write_file": "pushing code to GitHub",
    "read_file": "reading local files",
    "source_grep": "searching codebase",
    "delegate": "sub-agent delegation",
    "patch_file": "patching file",
}

def _detect_phase(messages):
    """Detect the current phase from recent tool calls and assistant thinking."""
    import re

    phase = None
    intent = None

    for msg in reversed(messages[-20:]):
        if msg.get("role") == "assistant":
            text = (msg.get("content") or "").strip()
            if text and not intent:
                think_match = re.search(r"<think>\s*(.+?)(?:\n|</think>)", text, re.DOTALL)
                if think_match:
                    raw = think_match.group(1).strip()
                    first_line = raw.split("\n")[0].strip()
                    intent = first_line[:120]
                elif not text.startswith("<"):
                    intent = text.split("\n")[0].strip()[:120]

            for tc in msg.get("tool_calls", []):
                name = tc.get("function", {}).get("name", "")
                if not phase and name in _STAGE_SIGNALS:
                    phase = _STAGE_SIGNALS[name]
                if name == "delegate":
                    args = tc.get("function", {}).get("arguments", {})
                    if isinstance(args, dict):
                        skills = args.get("skills", [])
                        files = args.get("files", [])
                        sub_model = args.get("model", "")
                        parts = ["sub-agent"]
                        if skills:
                            parts.append(f"skills={skills}")
                        if files:
                            parts.append(f"{len(files)} file(s)")
                        if sub_model:
                            parts.append(f"model={sub_model}")
                        phase = " — ".join(parts)
                if name == "patch_file":
                    args = tc.get("function", {}).get("arguments", {})
                    path = args.get("path", "") if isinstance(args, dict) else ""
                    phase = f"patching {os.path.basename(path)}" if path else "patching file"
                if name == "write_file":
                    args = tc.get("function", {}).get("arguments", {})
                    path = args.get("path", "") if isinstance(args, dict) else ""
                    if ".sol" in path:
                        phase = "writing Solidity contracts"
                    elif ".tsx" in path or ".jsx" in path:
                        phase = "writing frontend code"
                    elif ".t.sol" in path:
                        phase = "writing tests"
                    else:
                        phase = "writing files"
                if name == "shell":
                    args = tc.get("function", {}).get("arguments", {})
                    cmd = args.get("cmd", "") if isinstance(args, dict) else ""
                    if "forge build" in cmd:
                        phase = "compiling contracts"
                    elif "forge test" in cmd:
                        phase = "running tests"
                    elif "yarn deploy" in cmd or "forge script" in cmd or "forge create" in cmd:
                        phase = "deploying contracts"
                    elif "bgipfs" in cmd:
                        phase = "deploying to IPFS"
                    elif "git push" in cmd or "git commit" in cmd:
                        phase = "pushing to GitHub"
                    elif "ls " in cmd or "cat " in cmd or "find " in cmd:
                        if not phase:
                            phase = "exploring codebase"
                    elif "npx" in cmd or "create-eth" in cmd:
                        phase = "scaffolding project"

        if phase and intent:
            break

    return phase or "starting", intent


_debug_autopilot = False

def _debug_dashboard(iteration, max_iter, model, messages, total_tool_calls,
                     skill_cache=None):
    """Print the step-debugger dashboard and wait for user input.

    Cost tracking is handled by the LLM proxy (see ~/clawd/llm-proxy).
    """
    global _debug_autopilot
    from .providers import context_chars
    ctx = context_chars(messages)
    ctx_k = f"{ctx // 1000}K" if ctx >= 1000 else str(ctx)
    est_tokens = ctx // 4

    phase, intent = _detect_phase(messages)

    last_actions = []
    for msg in reversed(messages[-10:]):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            preview = content[:80].replace("\n", " ")
            last_actions.append(f"    -> {preview}{'...' if len(content) > 80 else ''}")
        elif msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                if name == "delegate" and isinstance(args, dict):
                    skills = args.get("skills", [])
                    files = args.get("files", [])
                    sub_model = args.get("model", "")
                    task_preview = (args.get("task", "")[:60] + "...") if len(args.get("task", "")) > 60 else args.get("task", "")
                    detail = f"    delegate(\"{task_preview}\""
                    if skills:
                        detail += f", skills={skills}"
                    if files:
                        fnames = [os.path.basename(f) for f in files]
                        detail += f", files={fnames}"
                    if sub_model:
                        detail += f", model={sub_model}"
                    detail += ")"
                    last_actions.append(detail)
                elif name == "patch_file" and isinstance(args, dict):
                    path = os.path.basename(args.get("path", "?"))
                    old = args.get("old_string", "")[:30].replace("\n", "\\n")
                    new = args.get("new_string", "")[:30].replace("\n", "\\n")
                    last_actions.append(f"    patch_file({path}: '{old}' -> '{new}')")
                else:
                    last_actions.append(f"    {name}")
        if len(last_actions) >= 8:
            break
    last_actions.reverse()

    bar = "=" * 60
    print(f"\n{bar}", file=sys.stderr)
    print(f"  ITERATION {iteration + 1} / {max_iter}  |  Tool calls: {total_tool_calls}", file=sys.stderr)
    print(f"  Phase: {phase}", file=sys.stderr)
    if intent:
        print(f"  Intent: {intent}", file=sys.stderr)
    print(f"  Model: {model}", file=sys.stderr)
    print(f"  Context: {len(messages)} msgs, ~{ctx_k} chars (~{est_tokens:,} tokens)", file=sys.stderr)
    if skill_cache:
        cached = skill_cache.cached_tags()
        if cached:
            print(f"  Skill cache: {cached}", file=sys.stderr)
    if last_actions:
        print(f"  Last actions:", file=sys.stderr)
        for a in last_actions:
            print(f"  {a}", file=sys.stderr)
    print(bar, file=sys.stderr)
    print("  [Enter] continue | [a] autopilot | [c] compact context | [q] quit", file=sys.stderr)

    if _debug_autopilot:
        print("  > (autopilot)", file=sys.stderr)
        return "continue"

    try:
        cmd = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        cmd = "q"

    if cmd == "a":
        _debug_autopilot = True
        print("  Autopilot ON — will no longer pause", file=sys.stderr)
        return "continue"
    elif cmd == "q":
        print("DEBUG: user quit", file=sys.stderr)
        return "quit"
    elif cmd == "c":
        saved = _compact_context(messages)
        new_ctx = context_chars(messages)
        new_k = f"{new_ctx // 1000}K" if new_ctx >= 1000 else str(new_ctx)
        print(f"  Compacted: saved {saved:,} chars. Context now: {new_k}", file=sys.stderr)
        return "continue"
    return "continue"


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
        max_iterations=200,
        max_cost=None,
        debug=False,
        preload_skills=None,
        default_skills=None,
        exclude_tools=None,
        warn_tools=None,
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

        # Tools the orchestrator is NOT allowed to call directly.
        # Sub-agents are unaffected — they build their own tool set.
        self._exclude_tools = set(exclude_tools or [])

        # Tools that work but emit a soft warning nudging toward delegate().
        self._warn_tools = set(warn_tools or [])

        # Config
        self.max_iterations = max_iterations
        self.max_cost = max_cost
        self.debug = debug

        # Skill cache for sub-agent delegation (populated as skills are fetched)
        self.skill_cache = SkillCache()
        self._subagent_tools_wired = False
        self._default_skills = default_skills or []

        # Pre-seed skill cache so sub-agents don't need to fetch at runtime
        if preload_skills:
            for url in preload_skills:
                content = _fetch_skill(url)
                if content:
                    self.skill_cache.put(url, content)
                    _log_agent(f"preloaded skill: {url[:60]}")

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

        if self._exclude_tools:
            visible = [t for t in self.tools
                       if t["spec"]["function"]["name"] not in self._exclude_tools]
        else:
            visible = self.tools
        tool_summary = get_tool_summary(visible)

        critical_path = os.path.join(self.memory_dir, "critical.md")
        critical_memory = ""
        if os.path.exists(critical_path):
            with open(critical_path, "r") as f:
                critical_memory = f.read().strip()
        critical_memory = critical_memory or "(No critical memories yet. Use memory_write with filename 'critical.md' to create one.)"

        raw = raw.replace("{{TOOLS}}", tool_summary).replace("{{MEMORY}}", critical_memory)

        if "{{WORKER_ADDRESS}}" in raw:
            try:
                from .leftclaw import worker_address
                raw = raw.replace("{{WORKER_ADDRESS}}", worker_address())
            except Exception:
                raw = raw.replace("{{WORKER_ADDRESS}}", "(could not derive — ETH_PRIVATE_KEY missing?)")

        return raw

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    def agent_turn(self, chat_fn, model, messages):
        """Run one user turn (may involve multiple tool-call rounds)."""
        from . import providers as _prov

        new_messages = []
        turn_start = time.time()
        total_tool_calls = 0
        error_streak = 0
        last_error_key = None
        _MAX_SAME_ERROR = 3
        _MAX_API_RETRIES = 2
        api_retries = 0

        _AUTO_COMPACT_THRESHOLD = 80_000

        for iteration in range(self.max_iterations):
            phase, intent = _detect_phase(messages)
            os.environ["LLM_PROXY_ITERATION"] = str(iteration)
            os.environ["LLM_PROXY_PHASE"] = phase or ""
            os.environ["LLM_PROXY_INTENT"] = intent or ""
            ctx = context_chars(messages)

            # Auto-compact when context grows too large
            if ctx > _AUTO_COMPACT_THRESHOLD:
                saved = _compact_context(messages)
                if saved > 0:
                    ctx = context_chars(messages)
                    _log_agent(f"auto-compacted: saved {saved:,} chars, now {ctx:,}")

            est_tokens = ctx // 4
            intent_short = (intent[:50] + "...") if intent and len(intent) > 50 else (intent or "")

            if self.debug:
                action = _debug_dashboard(
                    iteration, self.max_iterations, model, messages,
                    total_tool_calls,
                    skill_cache=self.skill_cache,
                )
                if action == "quit":
                    _log_agent(f"debug quit after {iteration} iterations, {total_tool_calls} calls")
                    return new_messages
            else:
                _log_agent(
                    f"{iteration+1}/{self.max_iterations} [{total_tool_calls}] "
                    f"({phase}) \"{intent_short}\" -- {model} "
                    f"~{est_tokens:,} tok"
                )

            if self._exclude_tools:
                visible_tools = [t for t in self.tools
                                 if t["spec"]["function"]["name"] not in self._exclude_tools]
            else:
                visible_tools = self.tools
            result = chat_fn(model, messages, get_tool_specs(visible_tools))

            if result is None:
                if api_retries < _MAX_API_RETRIES:
                    api_retries += 1
                    if api_retries == _MAX_API_RETRIES:
                        saved = _compact_context(messages, keep_recent=6)
                        if saved > 0:
                            _log_agent(f"  compacted {saved:,} chars before final retry")
                    _log_agent(f"iter={iteration+1} api returned None, retry {api_retries}/{_MAX_API_RETRIES}")
                    time.sleep(2 ** api_retries)
                    continue
                _log_agent(f"iter={iteration+1} api returned None after {api_retries} retries")
                return new_messages

            api_retries = 0  # reset on successful API call

            # Cost ceiling check
            if self.max_cost and _prov.cumulative_cost >= self.max_cost:
                _log_agent(
                    f"COST LIMIT: ${_prov.cumulative_cost:.2f} >= ${self.max_cost:.2f} "
                    f"after {iteration+1} iters, {total_tool_calls} calls"
                )
                content = result.get("content", "").strip()
                if content:
                    messages.append({"role": "assistant", "content": content})
                    new_messages.append(messages[-1])
                    print(content)
                print(f"COST LIMIT REACHED: ${_prov.cumulative_cost:.2f} (limit: ${self.max_cost:.2f})",
                      file=sys.stderr)
                return new_messages

            content = result.get("content", "").strip()
            tool_calls = result.get("tool_calls", [])

            if not tool_calls:
                assistant_msg = {"role": "assistant", "content": content}
                messages.append(assistant_msg)
                new_messages.append(assistant_msg)
                elapsed = int(time.time() - turn_start)
                _log_agent(f"done: {iteration+1} iters, {total_tool_calls} calls, {elapsed}s")
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
                try:
                    args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
                except (json.JSONDecodeError, TypeError) as e:
                    args = {}
                    _log_agent(f"WARNING: could not parse arguments for {name}: {e}")
                call_id = call.get("id", "call_0")

                total_tool_calls += 1
                if name in self._exclude_tools:
                    output = (
                        f"BLOCKED: '{name}' is not available to you directly. "
                        f"Use delegate() to run this in a sub-agent."
                    )
                    _log_agent(f"  ⛔ {name} — blocked (excluded tool)")
                elif name in self._warn_tools:
                    _log_agent(f"  ⚠️  {name}({_truncate_args(args)}) — prefer delegate()")
                    output = run_tool(self.tools, name, args)
                    output += (
                        f"\n\n⚠️ You called '{name}' directly. "
                        f"For complex multi-step work, prefer delegate() so a sub-agent "
                        f"can handle the full task with its own context."
                    )
                else:
                    _log_agent(f"  {name}({_truncate_args(args)})")
                    output = run_tool(self.tools, name, args)
                out_preview = output[:120].replace("\n", " ")
                _log_agent(f"    → {out_preview}{'...' if len(output) > 120 else ''}")

                # Auto-cache skill docs when fetched
                if name in ("fetch_url", "deep_fetch") and not output.startswith("ERROR:"):
                    url = args.get("url", "")
                    if self.skill_cache.try_cache_url(url, output):
                        _log_agent(f"    📚 cached skill: {url[:60]}")

                is_error = output.startswith("ERROR:")
                error_key = f"{name}:{output[:80]}" if is_error else None

                if is_error and error_key == last_error_key:
                    error_streak += 1
                elif is_error:
                    error_streak = 1
                    last_error_key = error_key
                else:
                    error_streak = 0
                    last_error_key = None

                if error_streak >= _MAX_SAME_ERROR:
                    output += (
                        f"\n\n⚠️ STOP: You have made the same failing call {error_streak} times in a row. "
                        f"Retrying will NOT work. You MUST try a different approach. "
                        f"For write_file errors: break the content into smaller chunks or use "
                        f"shell with heredoc (cat >> file << 'EOF'). "
                        f"For shell errors: make sure you pass the 'cmd' parameter."
                    )
                    _log_agent(f"⚠️  repeated error x{error_streak}: {name} — injected guidance")

                tool_msg = {"role": "tool", "tool_call_id": call_id, "content": output}
                messages.append(tool_msg)
                new_messages.append(tool_msg)

        elapsed = int(time.time() - turn_start)
        _log_agent(f"ERROR: max iterations ({self.max_iterations}) reached after "
                   f"{total_tool_calls} calls, {elapsed}s")
        print("ERROR: max iterations reached", file=sys.stderr)
        return new_messages

    def _ensure_subagent_tools(self, chat_fn, model):
        """Wire sub-agent tools once, when chat_fn and model are known."""
        if not self._subagent_tools_wired:
            self.tools.extend(make_subagent_tools(
                chat_fn, model, self.skill_cache,
                all_tools=self.tools,
                default_skills=self._default_skills,
            ))
            self._subagent_tools_wired = True
            # Rebuild system prompt so {{TOOLS}} includes delegate + patch_file
            self.system_prompt = self._build_prompt(None)

    # ------------------------------------------------------------------
    # One-shot mode
    # ------------------------------------------------------------------

    def run_once(self, chat_fn, model, prompt):
        self._ensure_subagent_tools(chat_fn, model)
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
        self._ensure_subagent_tools(chat_fn, model)
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
        parser.add_argument(
            "--debug", action="store_true",
            help="Step-debugger: pause before each LLM call showing cost and context stats",
        )
        parser.add_argument(
            "--max-cost", type=float, default=None, metavar="DOLLARS",
            help="Stop the agent when cumulative cost reaches this amount (e.g. --max-cost 2.00)",
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

        if args.debug:
            self.debug = True
        if args.max_cost is not None:
            self.max_cost = args.max_cost

        provider_name = args.provider or detect_provider(args.model)
        try:
            chat_fn = get_chat_fn(provider_name)
        except KeyError as e:
            parser.error(str(e))

        limits = f"Provider: {provider_name}"
        limit_parts = []
        if self.max_cost:
            limit_parts.append(f"cost<=${self.max_cost:.2f}")
        limit_parts.append(f"iter<={self.max_iterations}")
        limits += f" | Limits: {', '.join(limit_parts)}"
        print(limits, file=sys.stderr)

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
