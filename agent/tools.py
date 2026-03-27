"""
tools.py - Tool definitions and implementations for the agent.

Each tool is a dict with:
  - "spec": the tool schema (passed to the model)
  - "run":  a callable(args: dict) -> str that executes the tool

BASE_TOOLS are always available.
make_memory_tools(dir) returns memory tools bound to a specific directory.
"""

import os
import json
import subprocess
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _run_shell(args):
    try:
        env = os.environ.copy()
        env["PATH"] = (
            "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            ":/opt/homebrew/bin:/opt/homebrew/opt/node@22/bin:"
            + env.get("PATH", "")
        )
        result = subprocess.run(
            args["cmd"], shell=True, capture_output=True, text=True,
            timeout=30, env=env,
        )
        output = result.stdout or ""
        if result.stderr:
            output += "\nSTDERR: " + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out"
    except Exception as e:
        return f"ERROR: {e}"


def _run_fetch_url(args):
    try:
        import urllib.request
        import re
        url = args["url"]
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 clawd-cli/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        if args.get("as_text", True):
            raw = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
            raw = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.DOTALL)
            raw = re.sub(r"<[^>]+>", " ", raw)
            raw = re.sub(r"\s+", " ", raw).strip()
        return raw[:4000] + ("..." if len(raw) > 4000 else "")
    except Exception as e:
        return f"ERROR: {e}"


def _run_read_file(args):
    try:
        with open(os.path.expanduser(args["path"]), "r") as f:
            return f.read()
    except Exception as e:
        return f"ERROR: {e}"


def _run_write_file(args):
    try:
        path = os.path.expanduser(args["path"])
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(args["content"])
        return f"Written to {path}"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# GitHub tool implementations (requires `gh` CLI authenticated)
# ---------------------------------------------------------------------------

def _gh(cmd):
    try:
        result = subprocess.run(
            f"gh {cmd}", shell=True, capture_output=True, text=True, timeout=30,
        )
        out = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"ERROR (exit {result.returncode}): {err or out}"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: gh command timed out"
    except Exception as e:
        return f"ERROR: {e}"


def _run_github_list_repos(args):
    owner = args.get("owner", "")
    limit = args.get("limit", 20)
    if owner:
        return _gh(f"repo list {owner} --limit {limit} --json name,description,visibility,updatedAt")
    return _gh(f"repo list --limit {limit} --json name,description,visibility,updatedAt")


def _run_github_read_file(args):
    repo = args["repo"]
    path = args["path"]
    ref = args.get("ref", "")
    cmd = f"api repos/{repo}/contents/{path}"
    if ref:
        cmd += f" -f ref={ref}"
    cmd += " -H 'Accept: application/vnd.github.raw'"
    return _gh(cmd)


def _run_github_write_file(args):
    repo = args["repo"]
    path = args["path"]
    content = args["content"]
    message = args.get("message", f"Update {path}")
    branch = args.get("branch", "")

    import base64 as _b64
    import tempfile
    b64 = _b64.b64encode(content.encode()).decode()

    existing_sha = ""
    check = subprocess.run(
        f'gh api repos/{repo}/contents/{path}' + (f' --ref {branch}' if branch else '') + ' --jq .sha',
        shell=True, capture_output=True, text=True, timeout=15,
    )
    if check.returncode == 0 and check.stdout.strip():
        existing_sha = check.stdout.strip()

    body = {"message": message, "content": b64}
    if existing_sha:
        body["sha"] = existing_sha
    if branch:
        body["branch"] = branch

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(body, f)
        tmp = f.name

    try:
        result = _gh(f"api repos/{repo}/contents/{path} --method PUT --input {tmp} --jq '.content.html_url'")
    finally:
        os.unlink(tmp)
    return result


def _run_github_list_issues(args):
    repo = args["repo"]
    state = args.get("state", "open")
    limit = args.get("limit", 20)
    return _gh(f"issue list --repo {repo} --state {state} --limit {limit} --json number,title,state,author,labels,updatedAt")


def _run_github_create_issue(args):
    repo = args["repo"]
    title = args["title"]
    body = args.get("body", "")
    labels = args.get("labels", "")
    cmd = f'issue create --repo {repo} --title "{title}"'
    if body:
        cmd += f' --body "{body}"'
    if labels:
        cmd += f' --label "{labels}"'
    return _gh(cmd)


def _run_github_search_code(args):
    query = args["query"]
    owner = args.get("owner", "")
    limit = args.get("limit", 20)
    q = f"{query} user:{owner}" if owner else query
    return _gh(f'search code "{q}" --limit {limit} --json repository,path,textMatches')


def _run_github_create_pr(args):
    repo = args["repo"]
    title = args["title"]
    body = args.get("body", "")
    head = args["head"]
    base = args.get("base", "main")
    cmd = f'pr create --repo {repo} --title "{title}" --head {head} --base {base}'
    if body:
        cmd += f' --body "{body}"'
    return _gh(cmd)


# ---------------------------------------------------------------------------
# Base tool registry (everything except memory — those are per-agent)
# ---------------------------------------------------------------------------

BASE_TOOLS = [
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command and get the output. Use for system info, file ops, blockchain queries with cast, curl requests, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string", "description": "The shell command to run"},
                    },
                    "required": ["cmd"],
                },
            },
        },
        "run": _run_shell,
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch and read the content of a URL. Use for web pages, APIs, GitHub repos, dashboards, eth.limo ENS sites, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to fetch"},
                        "as_text": {"type": "boolean", "description": "Return as plain text (default: true). Set false for raw JSON APIs."},
                    },
                    "required": ["url"],
                },
            },
        },
        "run": _run_fetch_url,
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                    },
                    "required": ["path"],
                },
            },
        },
        "run": _run_read_file,
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to write to"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        "run": _run_write_file,
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "github_list_repos",
                "description": "List GitHub repositories. Lists your own repos by default, or repos for a given owner/org.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "GitHub user or org (omit for your own repos)"},
                        "limit": {"type": "integer", "description": "Max repos to return (default 20)"},
                    },
                    "required": [],
                },
            },
        },
        "run": _run_github_list_repos,
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "github_read_file",
                "description": "Read a file from a GitHub repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "owner/repo (e.g. octocat/Hello-World)"},
                        "path": {"type": "string", "description": "File path in the repo (e.g. README.md)"},
                        "ref": {"type": "string", "description": "Branch, tag, or commit SHA (default: repo default branch)"},
                    },
                    "required": ["repo", "path"],
                },
            },
        },
        "run": _run_github_read_file,
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "github_write_file",
                "description": "Create or update a file in a GitHub repository. Creates the file if it doesn't exist, updates it if it does.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "owner/repo"},
                        "path": {"type": "string", "description": "File path in the repo"},
                        "content": {"type": "string", "description": "The file content to write"},
                        "message": {"type": "string", "description": "Commit message (default: 'Update <path>')"},
                        "branch": {"type": "string", "description": "Target branch (default: repo default branch)"},
                    },
                    "required": ["repo", "path", "content"],
                },
            },
        },
        "run": _run_github_write_file,
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "github_list_issues",
                "description": "List issues on a GitHub repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "owner/repo"},
                        "state": {"type": "string", "description": "Filter by state: open, closed, all (default: open)"},
                        "limit": {"type": "integer", "description": "Max issues to return (default 20)"},
                    },
                    "required": ["repo"],
                },
            },
        },
        "run": _run_github_list_issues,
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "github_create_issue",
                "description": "Create a new issue on a GitHub repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "owner/repo"},
                        "title": {"type": "string", "description": "Issue title"},
                        "body": {"type": "string", "description": "Issue body/description"},
                        "labels": {"type": "string", "description": "Comma-separated labels (e.g. 'bug,help wanted')"},
                    },
                    "required": ["repo", "title"],
                },
            },
        },
        "run": _run_github_create_issue,
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "github_search_code",
                "description": "Search for code across GitHub repositories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (e.g. 'useState language:typescript')"},
                        "owner": {"type": "string", "description": "Limit search to a specific user/org"},
                        "limit": {"type": "integer", "description": "Max results (default 20)"},
                    },
                    "required": ["query"],
                },
            },
        },
        "run": _run_github_search_code,
    },
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "github_create_pr",
                "description": "Create a pull request on a GitHub repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "owner/repo"},
                        "title": {"type": "string", "description": "PR title"},
                        "body": {"type": "string", "description": "PR description"},
                        "head": {"type": "string", "description": "Source branch name"},
                        "base": {"type": "string", "description": "Target branch (default: main)"},
                    },
                    "required": ["repo", "title", "head"],
                },
            },
        },
        "run": _run_github_create_pr,
    },
]


# ---------------------------------------------------------------------------
# Memory tools factory — returns tools bound to a specific directory
# ---------------------------------------------------------------------------

def make_memory_tools(memory_dir):
    """Create memory tool dicts bound to a specific directory."""

    def _run_memory_list(args):
        try:
            os.makedirs(memory_dir, exist_ok=True)
            files = [f for f in os.listdir(memory_dir) if os.path.isfile(os.path.join(memory_dir, f))]
            if not files:
                return "(no memory files yet)"

            entries = []
            for fname in files:
                fpath = os.path.join(memory_dir, fname)
                mtime = os.path.getmtime(fpath)
                modified = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                with open(fpath, "r") as f:
                    first_line = f.readline().strip() or "(empty)"
                entries.append((mtime, fname, modified, first_line))

            entries.sort(key=lambda e: e[0], reverse=True)
            limit = args.get("limit", 20)
            if limit:
                entries = entries[:limit]
            lines = [f"{fname}  (modified: {mod})  -- {first_line}" for _, fname, mod, first_line in entries]
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    def _run_memory_read(args):
        try:
            fname = args["filename"]
            fpath = os.path.join(memory_dir, fname)
            with open(fpath, "r") as f:
                return f.read()
        except FileNotFoundError:
            return f"ERROR: memory file '{args['filename']}' not found"
        except Exception as e:
            return f"ERROR: {e}"

    def _run_memory_write(args):
        try:
            os.makedirs(memory_dir, exist_ok=True)
            fname = args["filename"]
            fpath = os.path.join(memory_dir, fname)
            with open(fpath, "w") as f:
                f.write(args["content"])
            return f"Memory saved to {fname}"
        except Exception as e:
            return f"ERROR: {e}"

    def _run_memory_search(args):
        try:
            os.makedirs(memory_dir, exist_ok=True)
            query = args["query"].lower()
            files = [f for f in os.listdir(memory_dir) if os.path.isfile(os.path.join(memory_dir, f))]
            if not files:
                return "(no memory files yet)"

            results = []
            for fname in files:
                fpath = os.path.join(memory_dir, fname)
                with open(fpath, "r") as f:
                    lines = f.readlines()
                matches = [(i + 1, line.strip()) for i, line in enumerate(lines) if query in line.lower()]
                if matches:
                    snippets = [f"  L{num}: {text[:120]}" for num, text in matches[:5]]
                    results.append(f"{fname} ({len(matches)} match{'es' if len(matches) != 1 else ''}):\n" + "\n".join(snippets))

            if not results:
                return f"No memories match '{args['query']}'"
            return "\n\n".join(results)
        except Exception as e:
            return f"ERROR: {e}"

    return [
        {
            "spec": {
                "type": "function",
                "function": {
                    "name": "memory_list",
                    "description": "List memory files sorted by most recently modified. Returns filename, modified date, and first-line preview.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "Max files to return (default 20, 0 for all)"},
                        },
                        "required": [],
                    },
                },
            },
            "run": _run_memory_list,
        },
        {
            "spec": {
                "type": "function",
                "function": {
                    "name": "memory_read",
                    "description": "Read the contents of a memory file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Name of the memory file (e.g. 'critical.md', 'projects.md')"},
                        },
                        "required": ["filename"],
                    },
                },
            },
            "run": _run_memory_read,
        },
        {
            "spec": {
                "type": "function",
                "function": {
                    "name": "memory_write",
                    "description": "Create or update a memory file. Use 'critical.md' for facts that should always be in context. Use other filenames for topic-specific notes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Name of the memory file (e.g. 'critical.md', 'projects.md')"},
                            "content": {"type": "string", "description": "Content to write to the memory file"},
                        },
                        "required": ["filename", "content"],
                    },
                },
            },
            "run": _run_memory_write,
        },
        {
            "spec": {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "Search across all memory files for a keyword or phrase. Returns matching filenames with line snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Text to search for (case-insensitive)"},
                        },
                        "required": ["query"],
                    },
                },
            },
            "run": _run_memory_search,
        },
    ]


# ---------------------------------------------------------------------------
# Helpers for working with tool registries
# ---------------------------------------------------------------------------

def get_tool_specs(registry):
    """Return the list of tool schemas to pass to the model."""
    return [t["spec"] for t in registry]


def get_tool_summary(registry):
    """Auto-generate a readable tool list for injection into the system prompt."""
    lines = []
    for t in registry:
        fn = t["spec"]["function"]
        name = fn["name"]
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])

        param_parts = []
        for pname, pinfo in params.items():
            req = " (required)" if pname in required else ""
            param_parts.append(f"  - {pname}: {pinfo.get('description', pinfo.get('type', ''))}{req}")

        lines.append(f"- **{name}**: {desc}")
        if param_parts:
            lines.extend(param_parts)
    return "\n".join(lines)


def run_tool(registry, name, args):
    """Dispatch a tool call by name. Returns the tool's string output."""
    for t in registry:
        if t["spec"]["function"]["name"] == name:
            return t["run"](args)
    return f"ERROR: unknown tool '{name}'"
