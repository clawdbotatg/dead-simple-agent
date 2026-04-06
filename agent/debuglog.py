"""
debuglog.py - Structured audit logging for agent runs.

When --debug is passed, creates a timestamped folder with markdown files
documenting every decision, tool call, sub-agent delegation, and outcome.

Usage:
    log = DebugLog("./debug-runs", job_name="job-38")
    log.iteration(1, phase="scaffold", intent="...", model="minimax", ctx_tokens=5000)
    log.tool_call("delegate", args, output)
    log.subagent_start(1, model="opus", task="...", files=[...], skills=[...])
    log.subagent_iter(1, 1, tool_calls=[...])
    log.subagent_end(1, result="done", iters=3, cost=0.30)
    log.finalize(total_cost=1.20, total_iters=25)
"""

import json
import os
import sys
from datetime import datetime


class DebugLog:
    """Write structured audit logs to a per-run directory."""

    def __init__(self, base_dir, job_name=None):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"run-{job_name}-{ts}" if job_name else f"run-{ts}"
        self.run_dir = os.path.join(base_dir, folder_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self._run_file = os.path.join(self.run_dir, "RUN.md")
        self._subagent_counter = 0
        self._start_time = datetime.now()

        with open(self._run_file, "w") as f:
            f.write(f"# Agent Run: {folder_name}\n\n")
            f.write(f"**Started:** {self._start_time.isoformat()}\n\n")
            if job_name:
                f.write(f"**Job:** {job_name}\n\n")
            f.write("---\n\n")

        print(f"[debug] audit log: {self.run_dir}/", file=sys.stderr)

    def _ts(self):
        return datetime.now().strftime("%H:%M:%S")

    def _append_run(self, text):
        with open(self._run_file, "a") as f:
            f.write(text)

    # ------------------------------------------------------------------
    # Orchestrator events
    # ------------------------------------------------------------------

    def iteration(self, iteration, max_iter, phase, intent, model,
                  ctx_tokens, ctx_chars, total_tool_calls):
        self._append_run(
            f"## Iteration {iteration}/{max_iter}\n\n"
            f"- **Time:** {self._ts()}\n"
            f"- **Phase:** {phase}\n"
            f"- **Intent:** {intent or '(none)'}\n"
            f"- **Model:** {model}\n"
            f"- **Context:** ~{ctx_tokens:,} tokens ({ctx_chars:,} chars)\n"
            f"- **Tool calls so far:** {total_tool_calls}\n\n"
        )

    def thinking(self, content):
        if not content or not content.strip():
            return
        text = content.strip()
        if len(text) > 2000:
            text = text[:2000] + "\n\n... [truncated]"
        self._append_run(
            f"### Thinking\n\n"
            f"```\n{text}\n```\n\n"
        )

    def tool_call(self, name, args, output, is_error=False):
        args_display = dict(args) if isinstance(args, dict) else {}
        if name == "delegate":
            task_preview = (args_display.get("task", ""))[:200]
            args_summary = {
                "task": task_preview + ("..." if len(args_display.get("task", "")) > 200 else ""),
                "model": args_display.get("model", ""),
                "skills": args_display.get("skills", []),
                "files": [os.path.basename(f) for f in args_display.get("files", [])],
                "max_iterations": args_display.get("max_iterations", ""),
            }
            args_str = json.dumps(args_summary, indent=2)
        elif name == "write_file":
            args_summary = {"path": args_display.get("path", ""), "content_length": len(args_display.get("content", ""))}
            args_str = json.dumps(args_summary, indent=2)
        else:
            safe = {}
            for k, v in args_display.items():
                if isinstance(v, str) and len(v) > 200:
                    safe[k] = v[:200] + "..."
                else:
                    safe[k] = v
            args_str = json.dumps(safe, indent=2)

        status = "ERROR" if is_error else "OK"
        out_preview = output[:500] if output else "(empty)"

        self._append_run(
            f"### Tool: `{name}` [{status}]\n\n"
            f"**Args:**\n```json\n{args_str}\n```\n\n"
            f"**Output:**\n```\n{out_preview}\n```\n\n"
        )

    def api_error(self, iteration, retry_num, max_retries):
        self._append_run(
            f"### API Error (iter {iteration}, retry {retry_num}/{max_retries})\n\n"
        )

    def cost_limit(self, cost, limit, iteration, tool_calls):
        self._append_run(
            f"## COST LIMIT REACHED\n\n"
            f"- **Cost:** ${cost:.2f} / ${limit:.2f}\n"
            f"- **Iteration:** {iteration}\n"
            f"- **Tool calls:** {tool_calls}\n\n"
        )

    def circuit_breaker(self, goal_key, fail_count):
        self._append_run(
            f"## CIRCUIT BREAKER: `{goal_key}` ({fail_count} failures)\n\n"
        )

    def compaction(self, saved_chars, new_ctx_chars):
        self._append_run(
            f"> Compacted context: saved {saved_chars:,} chars, now {new_ctx_chars:,}\n\n"
        )

    # ------------------------------------------------------------------
    # Sub-agent events — each gets its own MD file
    # ------------------------------------------------------------------

    def subagent_start(self, model, task, files, skills, max_iters, system_prompt_len, user_prompt_len):
        self._subagent_counter += 1
        sid = self._subagent_counter
        filename = f"subagent-{sid:02d}.md"
        filepath = os.path.join(self.run_dir, filename)

        with open(filepath, "w") as f:
            f.write(f"# Sub-agent #{sid}\n\n")
            f.write(f"- **Model:** {model}\n")
            f.write(f"- **Started:** {self._ts()}\n")
            f.write(f"- **Max iterations:** {max_iters}\n")
            f.write(f"- **Skills:** {skills or '(none)'}\n")
            f.write(f"- **Files:** {[os.path.basename(p) for p in files] if files else '(none)'}\n")
            f.write(f"- **System prompt:** {system_prompt_len:,} chars\n")
            f.write(f"- **User prompt:** {user_prompt_len:,} chars\n\n")
            f.write(f"## Task\n\n{task}\n\n---\n\n")

        self._append_run(
            f"### Sub-agent #{sid} started\n\n"
            f"- **Model:** {model}\n"
            f"- **Files:** {[os.path.basename(p) for p in files] if files else '(none)'}\n"
            f"- **Task:** {task[:150]}{'...' if len(task) > 150 else ''}\n\n"
            f"*Details: [{filename}]({filename})*\n\n"
        )

        return sid

    def subagent_iter(self, sid, iteration, ctx_tokens, tool_calls_data=None):
        filepath = os.path.join(self.run_dir, f"subagent-{sid:02d}.md")
        with open(filepath, "a") as f:
            f.write(f"## Iteration {iteration} (~{ctx_tokens:,} tokens)\n\n")
            if tool_calls_data:
                for tc in tool_calls_data:
                    name = tc.get("name", "?")
                    label = tc.get("label", "")
                    output = tc.get("output", "")[:300]
                    is_err = tc.get("is_error", False)
                    status = "ERROR" if is_err else "OK"
                    f.write(f"### `{name}` [{status}]\n\n")
                    if label:
                        f.write(f"```\n{label}\n```\n\n")
                    if output:
                        f.write(f"**Output:**\n```\n{output}\n```\n\n")

    def subagent_thinking(self, sid, content):
        if not content or not content.strip():
            return
        filepath = os.path.join(self.run_dir, f"subagent-{sid:02d}.md")
        text = content.strip()
        if len(text) > 2000:
            text = text[:2000] + "\n\n... [truncated]"
        with open(filepath, "a") as f:
            f.write(f"### Thinking\n\n```\n{text}\n```\n\n")

    def subagent_end(self, sid, result, iterations, tool_calls, cost=None):
        filepath = os.path.join(self.run_dir, f"subagent-{sid:02d}.md")
        with open(filepath, "a") as f:
            f.write(f"\n---\n\n## Result\n\n")
            f.write(f"- **Iterations:** {iterations}\n")
            f.write(f"- **Tool calls:** {tool_calls}\n")
            if cost is not None:
                f.write(f"- **Cost:** ${cost:.4f}\n")
            f.write(f"- **Outcome:** {result[:300]}\n\n")

        self._append_run(
            f"### Sub-agent #{sid} finished\n\n"
            f"- **Iterations:** {iterations}, **Tool calls:** {tool_calls}"
            f"{f', **Cost:** ${cost:.4f}' if cost is not None else ''}\n"
            f"- **Result:** {result[:150]}{'...' if len(result) > 150 else ''}\n\n"
        )

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self, total_cost=None, total_iters=None, total_tool_calls=None, outcome="completed"):
        elapsed = (datetime.now() - self._start_time).total_seconds()
        self._append_run(
            f"\n---\n\n## Run Complete\n\n"
            f"- **Outcome:** {outcome}\n"
            f"- **Elapsed:** {elapsed:.0f}s\n"
            f"- **Iterations:** {total_iters}\n"
            f"- **Tool calls:** {total_tool_calls}\n"
        )
        if total_cost is not None:
            self._append_run(f"- **Total cost:** ${total_cost:.4f}\n")
        self._append_run("\n")

        print(f"[debug] audit complete: {self.run_dir}/RUN.md", file=sys.stderr)
