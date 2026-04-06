"""
subagent.py - Sub-agent delegation with skill-aware context preparation.

Provides:
  - SkillCache: auto-caches skill docs fetched via deep_fetch/fetch_url
  - make_subagent_tools(): factory returning a `delegate` tool that spawns
    focused sub-agents with minimal context (selected skills + files only)
"""

import json
import os
import re
import sys
import urllib.request

from .tools import get_tool_specs, run_tool


def _log_sub(msg):
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [sub-agent] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Skill cache
# ---------------------------------------------------------------------------

class SkillCache:
    """Cache for skill docs, keyed by URL. Supports short tag aliases."""

    TAG_MAP = {
        "leftclaw":      "https://leftclaw.services/admin/skill.md",
        "ethskills":     "https://ethskills.com/SKILL.md",
        "scaffoldeth":   "https://docs.scaffoldeth.io/SKILL.md",
        "bgipfs":        "https://www.bgipfs.com/SKILL.md",
        "orchestration": "https://ethskills.com/orchestration/SKILL.md",
        "frontend":      "https://ethskills.com/frontend-playbook/SKILL.md",
        "ux":            "https://ethskills.com/frontend-ux/SKILL.md",
        "security":      "https://ethskills.com/security/SKILL.md",
        "testing":       "https://ethskills.com/testing/SKILL.md",
        "audit":         "https://ethskills.com/audit/SKILL.md",
        "qa":            "https://ethskills.com/qa/SKILL.md",
    }

    SKILL_DOMAINS = ("leftclaw.services", "ethskills.com", "scaffoldeth.io", "bgipfs.com")

    def __init__(self):
        self._cache = {}  # url -> content

    def put(self, url, content):
        self._cache[url] = content

    def get(self, tag_or_url):
        """Look up by tag name or full URL."""
        url = self.TAG_MAP.get(tag_or_url, tag_or_url)
        if url in self._cache:
            return self._cache[url]
        # Try without trailing slash
        stripped = url.rstrip("/")
        for k, v in self._cache.items():
            if k.rstrip("/") == stripped:
                return v
        return None

    def try_cache_url(self, url, content):
        """Auto-cache if url belongs to a known skill domain. Returns True if cached."""
        if not url or not content:
            return False
        for domain in self.SKILL_DOMAINS:
            if domain in url:
                self._cache[url] = content
                return True
        return False

    def cached_tags(self):
        """Return list of tag names that are currently cached."""
        tags = []
        for tag, url in self.TAG_MAP.items():
            if self.get(tag) is not None:
                tags.append(tag)
        return tags


# ---------------------------------------------------------------------------
# Skill fetcher (for on-demand fetch inside delegate)
# ---------------------------------------------------------------------------

def _fetch_skill(url, timeout=15, max_chars=16000):
    """Fetch a skill doc URL. Returns content string or None on error."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 dead-simple-agent/sub"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        raw = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw[:max_chars]
    except Exception as e:
        _log_sub(f"fetch skill {url} failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Limited tool set for sub-agents (no GitHub, no memory, no fetch)
# ---------------------------------------------------------------------------

def _make_limited_tools():
    """Build the restricted tool set available inside a delegate call."""
    from .tools import (
        _run_shell, _run_read_file, _run_write_file, _run_patch_file,
    )

    return [
        {
            "spec": {"type": "function", "function": {
                "name": "shell",
                "description": "Run a shell command.",
                "parameters": {"type": "object", "properties": {
                    "cmd": {"type": "string", "description": "The shell command to run"},
                }, "required": ["cmd"]},
            }},
            "run": _run_shell,
        },
        {
            "spec": {"type": "function", "function": {
                "name": "read_file",
                "description": "Read the contents of a file.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                }, "required": ["path"]},
            }},
            "run": _run_read_file,
        },
        {
            "spec": {"type": "function", "function": {
                "name": "write_file",
                "description": "Write content to a file.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string", "description": "Path to write to"},
                    "content": {"type": "string", "description": "Content to write"},
                }, "required": ["path", "content"]},
            }},
            "run": _run_write_file,
        },
        {
            "spec": {"type": "function", "function": {
                "name": "patch_file",
                "description": "Replace a specific string in a file (first occurrence).",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "old_string": {"type": "string", "description": "Text to find"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                }, "required": ["path", "old_string", "new_string"]},
            }},
            "run": _run_patch_file,
        },
    ]


# ---------------------------------------------------------------------------
# Sub-agent tool factory
# ---------------------------------------------------------------------------

_BGIPFS_KEYWORDS = ("deploy", "upload", "ipfs", "bgipfs", "build the frontend")
_DEFAULT_MAX_ITERS = 15
_MAX_SUB_COST = 0.50
_MAX_SUB_COST_EXPENSIVE = 1.50
_SUB_COMPACT_THRESHOLD = 30_000  # chars — keep sub-agents lean
_OPUS_BUDGET_LIMIT = 2.00  # total opus spend across all sub-agents per job
_opus_cumulative_cost = 0.0

def make_subagent_tools(chat_fn, default_model, skill_cache, all_tools=None,
                        default_skills=None, debug_log=None):
    """Return tools that can spawn focused sub-agents.

    Parameters:
        chat_fn:        The LLM chat function (same signature as providers)
        default_model:  Fallback model name if not overridden
        skill_cache:    SkillCache instance (shared with main agent)
        all_tools:      The main agent's full tool registry, for selective forwarding
        default_skills: Skill tags always injected into every sub-agent (e.g. ["ethskills", "scaffoldeth"])
        debug_log:      Optional DebugLog instance for audit logging
    """
    base_limited = _make_limited_tools()
    base_names = {t["spec"]["function"]["name"] for t in base_limited}

    def _build_tool_set(requested_names):
        """Build a tool set: always includes base tools, plus requested from all_tools."""
        if not requested_names or not all_tools:
            return list(base_limited)
        tools = list(base_limited)
        seen = set(base_names)
        for t in all_tools:
            name = t["spec"]["function"]["name"]
            if name in requested_names and name not in seen:
                tools.append(t)
                seen.add(name)
        return tools

    def _run_delegate(args):
        global _opus_cumulative_cost
        task = args.get("task")
        if not task:
            return "ERROR: 'task' parameter is required."

        files = args.get("files", [])
        requested_skills = list(args.get("skills", []))
        skills = list(requested_skills)
        tool_names = args.get("tools", [])
        model = args.get("model") or default_model
        max_iters = args.get("max_iterations", _DEFAULT_MAX_ITERS)
        _EXPENSIVE_PREFIXES = ("claude-",)
        is_expensive = any(model.startswith(p) for p in _EXPENSIVE_PREFIXES)

        if "opus" in model and _opus_cumulative_cost >= _OPUS_BUDGET_LIMIT:
            return (
                f"ERROR: Opus budget exhausted (${_opus_cumulative_cost:.2f} >= ${_OPUS_BUDGET_LIMIT:.2f}). "
                f"Use a cheaper model (claude-sonnet-4.6 or minimax-m2.7) for this task."
            )

        default_cost = _MAX_SUB_COST_EXPENSIVE if is_expensive else _MAX_SUB_COST
        cost_limit = args.get("max_cost", default_cost)
        if is_expensive and cost_limit > _MAX_SUB_COST_EXPENSIVE:
            _log_sub(f"  capping cost from ${cost_limit:.2f} to ${_MAX_SUB_COST_EXPENSIVE:.2f} for {model}")
            cost_limit = _MAX_SUB_COST_EXPENSIVE

        # Auto-skip default skills for expensive models to save tokens
        skip_defaults = args.get("skip_default_skills", False)
        if not skip_defaults and is_expensive:
            skip_defaults = True
            _log_sub(f"  auto-skipping default skills for expensive model {model}")

        if default_skills and not skip_defaults:
            for s in default_skills:
                if s not in skills:
                    skills.append(s)
        if not is_expensive:
            task_lower = task.lower()
            if "bgipfs" not in skills and any(kw in task_lower for kw in _BGIPFS_KEYWORDS):
                skills.append("bgipfs")

        active_tools = _build_tool_set(tool_names)
        active_names = [t["spec"]["function"]["name"] for t in active_tools]

        _log_sub(f"delegate: model={model} skills={skills} files={len(files)} "
                 f"tools={active_names}")

        # 1. Build skill context from cache (fetch on demand if missing)
        skill_text = ""
        for tag in skills:
            doc = skill_cache.get(tag)
            if not doc:
                url = SkillCache.TAG_MAP.get(tag, tag)
                if url.startswith("http"):
                    _log_sub(f"  fetching uncached skill: {tag} -> {url}")
                    doc = _fetch_skill(url)
                    if doc:
                        skill_cache.put(url, doc)
            if doc:
                skill_text += f"\n---\n## Rules: {tag}\n{doc}\n"

        # 2. Handle AGENTS.md for SE2 projects.
        #    For expensive models: extract only the frontend-relevant sections
        #    (hook API, components, styling) instead of the full 8K+ file.
        #    For cheap models: leave the full file in the preload list.
        _agents_md_extra = ""
        _AGENTS_SECTIONS = [
            "### Frontend Contract Interaction",
            "### UI Components",
            "### Notifications",
            "### Styling",
            "### Import Paths",
            "### Creating Pages",
        ]

        def _extract_agents_sections(path):
            with open(path) as _af:
                _full = _af.read()
            _sections = []
            for header in _AGENTS_SECTIONS:
                idx = _full.find(header)
                if idx >= 0:
                    end = _full.find("\n### ", idx + len(header))
                    if end < 0:
                        end = _full.find("\n## ", idx + len(header))
                    if end < 0:
                        end = len(_full)
                    _sections.append(_full[idx:end].strip())
            return "\n\n".join(_sections) if _sections else ""

        # If AGENTS.md is explicitly preloaded, remove it from files for
        # expensive models (we'll inject extracted sections instead)
        _explicit_agents = [p for p in files if p.endswith("AGENTS.md")]
        if _explicit_agents and is_expensive:
            files = [p for p in files if not p.endswith("AGENTS.md")]
            try:
                _agents_md_extra = _extract_agents_sections(_explicit_agents[0])
                if _agents_md_extra:
                    _log_sub(f"  extracted AGENTS.md sections ({len(_agents_md_extra)} chars, "
                             f"saved ~{8000 - len(_agents_md_extra)} chars)")
            except Exception:
                files = list(files) + _explicit_agents  # fallback: keep full file

        # If no AGENTS.md was preloaded, auto-detect it from sibling dirs
        if not _explicit_agents and not _agents_md_extra and files:
            _agents_path = None
            for p in files:
                candidate = os.path.join(os.path.dirname(p), "AGENTS.md")
                while candidate and len(candidate) > 10:
                    if os.path.isfile(candidate):
                        _agents_path = candidate
                        break
                    parent = os.path.dirname(os.path.dirname(candidate))
                    candidate = os.path.join(parent, "AGENTS.md")
                    if parent == os.path.dirname(parent):
                        break
                if _agents_path:
                    break
            if _agents_path:
                try:
                    if is_expensive:
                        _agents_md_extra = _extract_agents_sections(_agents_path)
                        if _agents_md_extra:
                            _log_sub(f"  auto-injected AGENTS.md sections ({len(_agents_md_extra)} chars)")
                    else:
                        files = list(files) + [_agents_path]
                        _log_sub(f"  auto-preloaded AGENTS.md: {_agents_path}")
                except Exception as e:
                    _log_sub(f"  failed to read AGENTS.md: {e}")

        # Read file contents (cap more aggressively for expensive models)
        _max_file = 6000 if is_expensive else 12000
        file_text = ""
        for path in files:
            try:
                with open(os.path.expanduser(path), "r") as f:
                    content = f.read()
                if len(content) > _max_file:
                    content = content[:_max_file] + f"\n... [truncated to {_max_file} chars]"
                file_text += f"\n## File: {path}\n```\n{content}\n```\n"
            except Exception as e:
                file_text += f"\n## File: {path}\nERROR reading: {e}\n"

        # 3. Build fresh messages (NO main conversation history)
        system = (
            "You are a focused coding agent. Complete the task below efficiently.\n"
            f"You have access to: {', '.join(active_names)}.\n"
            "RULES:\n"
            "- You have LIMITED iterations. Use preloaded files below instead of exploring with ls/find.\n"
            "- Prefer patch_file for surgical edits, write_file for new files.\n"
            "- Complete the task and stop. Do not explore beyond what is needed.\n"
        )
        if skill_text:
            system += f"\n# Domain Rules\n{skill_text}"

        user = f"## Task\n\n{task}"
        if _agents_md_extra:
            user += f"\n\n# Project Conventions (from AGENTS.md)\n{_agents_md_extra}"
        if file_text:
            user += f"\n\n# Files\n{file_text}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        ctx_chars = sum(len(m.get("content", "")) for m in messages)
        _log_sub(f"  sub-context: {ctx_chars:,} chars ({ctx_chars // 4:,} est tokens)")

        _debug_sid = None
        if debug_log:
            _debug_sid = debug_log.subagent_start(
                model=model, task=task, files=files, skills=skills,
                max_iters=max_iters,
                system_prompt_len=len(system),
                user_prompt_len=len(user),
            )

        # 4. Select chat function — use native Anthropic for Claude models
        #    to enable prompt caching (90% cheaper on repeat prefixes).
        _sub_chat_fn = chat_fn
        if model.startswith("claude-"):
            try:
                from .providers import anthropic_chat
                if os.environ.get("ANTHROPIC_API_KEY"):
                    _sub_chat_fn = anthropic_chat
                    _log_sub(f"  using native Anthropic provider (prompt caching enabled)")
            except Exception:
                pass

        # Add cache_control markers to system and initial user message
        # so Anthropic caches the prefix across sub-agent iterations.
        if _sub_chat_fn is not chat_fn:
            if isinstance(messages[0].get("content"), str):
                messages[0]["content"] = [
                    {"type": "text", "text": messages[0]["content"],
                     "cache_control": {"type": "ephemeral"}}
                ]
            if len(messages) > 1 and isinstance(messages[1].get("content"), str):
                messages[1]["content"] = [
                    {"type": "text", "text": messages[1]["content"],
                     "cache_control": {"type": "ephemeral"}}
                ]

        # 5. Run mini agent loop
        from .providers import subagent_tracking
        from .core import _compact_context
        from . import providers as _prov
        cost_before = _prov.subagent_cost
        total_calls = 0
        sub_api_retries = 0
        _compact_at = 20_000 if is_expensive else _SUB_COMPACT_THRESHOLD
        for iteration in range(max_iters):
            sub_ctx = sum(len(m.get("content", "") or "") for m in messages)

            if sub_ctx > _compact_at:
                saved = _compact_context(messages, keep_recent=6)
                if saved > 0:
                    sub_ctx = sum(len(m.get("content", "") or "") for m in messages)
                    _log_sub(f"  compacted: saved {saved:,} chars, now {sub_ctx:,}")

            sub_ctx_k = f"{sub_ctx // 1000}K" if sub_ctx >= 1000 else str(sub_ctx)
            est_tokens = sub_ctx // 4

            this_sub_cost = _prov.subagent_cost - cost_before
            if this_sub_cost >= cost_limit:
                _log_sub(f"  SUB-AGENT COST LIMIT after {iteration} iters, {total_calls} calls")
                if _debug_sid and debug_log:
                    debug_log.subagent_end(_debug_sid, "cost_limit", iteration + 1, total_calls, this_sub_cost)
                last_out = ""
                for m in reversed(messages):
                    if m.get("role") == "tool":
                        last_out = m.get("content", "")[:200]
                        break
                return f"Sub-agent stopped (cost limit). {total_calls} tool calls. Last output: {last_out}"

            _log_sub(f"  iter {iteration + 1} ctx={sub_ctx_k} (~{est_tokens:,} tok)")

            with subagent_tracking():
                result = _sub_chat_fn(model, messages, get_tool_specs(active_tools))
            if result is None:
                if sub_api_retries < 3:
                    sub_api_retries += 1
                    if sub_api_retries == 3:
                        saved = _compact_context(messages, keep_recent=4)
                        if saved > 0:
                            _log_sub(f"  compacted {saved:,} chars before final retry")
                    _log_sub(f"  api returned None, retry {sub_api_retries}/3")
                    import time as _time
                    _time.sleep(2 ** sub_api_retries)
                    continue
                return "ERROR: sub-agent LLM call failed after retries"
            sub_api_retries = 0

            content = result.get("content", "").strip()
            tool_calls = result.get("tool_calls", [])

            if not tool_calls:
                _log_sub(f"  done: {iteration + 1} iters, {total_calls} tool calls")
                if "opus" in model:
                    _opus_cumulative_cost += _prov.subagent_cost - cost_before
                _sub_cost = _prov.subagent_cost - cost_before
                if _debug_sid and debug_log:
                    debug_log.subagent_end(_debug_sid, "completed", iteration + 1, total_calls, _sub_cost)
                return f"Sub-agent done: {content[:500]}" if content else "Sub-agent completed (no output)"

            if _debug_sid and debug_log and content:
                debug_log.subagent_thinking(_debug_sid, content)

            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })

            _iter_tc_data = []
            for call in tool_calls:
                fn = call["function"]
                name = fn["name"]
                try:
                    tc_args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
                except (json.JSONDecodeError, TypeError):
                    tc_args = {}

                total_calls += 1
                output = run_tool(active_tools, name, tc_args)

                if name == "read_file":
                    label = f"read {os.path.basename(tc_args.get('path', '?'))}"
                elif name == "write_file":
                    label = f"write {os.path.basename(tc_args.get('path', '?'))} ({len(tc_args.get('content', ''))} chars)"
                elif name == "patch_file":
                    label = f"patch {os.path.basename(tc_args.get('path', '?'))}"
                elif name == "shell":
                    cmd = tc_args.get("cmd", "")
                    label = f"$ {cmd[:80]}{'...' if len(cmd) > 80 else ''}"
                else:
                    arg_preview = json.dumps(tc_args)[:60]
                    label = f"{name}({arg_preview}{'...' if len(json.dumps(tc_args)) > 60 else ''})"

                is_err = output.startswith("ERROR:")
                err_tag = " ✗" if is_err else ""
                _log_sub(f"  {label}{err_tag}")

                _iter_tc_data.append({
                    "name": name, "label": label,
                    "output": output, "is_error": is_err,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", "sub_0"),
                    "content": output,
                })

            if _debug_sid and debug_log:
                debug_log.subagent_iter(_debug_sid, iteration + 1, est_tokens, _iter_tc_data)

        _log_sub(f"  max iterations ({max_iters}) reached, {total_calls} tool calls")
        _sub_cost = _prov.subagent_cost - cost_before
        if "opus" in model:
            _opus_cumulative_cost += _sub_cost
        if _debug_sid and debug_log:
            debug_log.subagent_end(_debug_sid, "max_iterations", max_iters, total_calls, _sub_cost)
        last_content = ""
        for m in reversed(messages):
            if m.get("role") == "tool":
                last_content = m.get("content", "")[:200]
                break
        return f"Sub-agent completed (max iterations). Last output: {last_content}"

    return [
        {
            "spec": {"type": "function", "function": {
                "name": "delegate",
                "description": (
                    "Delegate a task to a sub-agent with minimal, focused context. "
                    "The sub-agent gets ONLY the task description, specified skill docs, "
                    "and files — NOT the main conversation history. Its context starts "
                    "fresh and stays small. Use delegate as your PRIMARY execution "
                    "mechanism: repo setup, writing code, fixing errors, running builds, "
                    "deploying — anything that involves multiple tool calls. "
                    "You orchestrate (plan, delegate, log progress). Sub-agents execute. "
                    "Write a clear, self-contained task description with all info the "
                    "sub-agent needs. Use 'tools' to grant access to specific tools "
                    "beyond the defaults (shell, read_file, write_file, patch_file). "
                    "Skill tags: ethskills, scaffoldeth, bgipfs, orchestration, "
                    "frontend, ux, security, testing, audit, qa."
                ),
                "parameters": {"type": "object", "properties": {
                    "task": {
                        "type": "string",
                        "description": "Self-contained task description. Include all context the sub-agent needs: file paths, error messages, repo URLs, expected outcomes.",
                    },
                    "files": {
                        "type": "array", "items": {"type": "string"},
                        "description": "File paths to read and include in sub-agent context",
                    },
                    "skills": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Skill tags or URLs to include (e.g. 'scaffoldeth', 'security')",
                    },
                    "tools": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Additional tool names to make available beyond the base set. "
                                       "E.g. ['github_write_file', 'github_list_repos', 'fetch_url', 'deep_fetch', 'bgipfs_upload']. "
                                       "shell, read_file, write_file, patch_file are always included.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model override for the sub-agent (default: same as main agent)",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Max tool-call rounds (default: 30). Keep tasks small — if a sub-agent needs more than 30 rounds, the task should be broken up.",
                    },
                    "skip_default_skills": {
                        "type": "boolean",
                        "description": "If true, only explicitly listed skills are injected (no auto-injected defaults). Use for focused tasks that don't need all skill docs.",
                    },
                    "max_cost": {
                        "type": "number",
                        "description": "Cost ceiling in USD for this sub-agent (default: 0.50). Increase for larger tasks like full frontend builds.",
                    },
                }, "required": ["task"]},
            }},
            "run": _run_delegate,
        },
    ]
