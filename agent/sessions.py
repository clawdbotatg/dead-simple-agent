"""
sessions.py - JSONL-based session persistence for the agent.

Sessions are stored in a sessions/ directory as .jsonl files.
  Line 1:  metadata  {"id": "...", "model": "...", "created": "...", "updated": "..."}
  Line 2+: messages  {"role": "...", "content": "...", ...}

Append-only: new messages are appended without rewriting the file.
All functions take sessions_dir as first argument.
"""

import json
import os
import uuid
from datetime import datetime, timezone


def _session_path(sessions_dir, session_id):
    return os.path.join(sessions_dir, f"{session_id}.jsonl")


def create_session(sessions_dir, model):
    """Create a new session file and return its id."""
    os.makedirs(sessions_dir, exist_ok=True)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    meta = {
        "id": session_id,
        "model": model,
        "created": datetime.now(timezone.utc).isoformat(),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    with open(_session_path(sessions_dir, session_id), "w") as f:
        f.write(json.dumps(meta) + "\n")
    return session_id


def append_messages(sessions_dir, session_id, messages):
    """Append one or more messages to a session file and touch the updated timestamp."""
    path = _session_path(sessions_dir, session_id)
    with open(path, "a") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    _touch_updated(sessions_dir, session_id)


def _touch_updated(sessions_dir, session_id):
    """Rewrite the metadata line's 'updated' field."""
    path = _session_path(sessions_dir, session_id)
    with open(path, "r") as f:
        lines = f.readlines()
    meta = json.loads(lines[0])
    meta["updated"] = datetime.now(timezone.utc).isoformat()
    lines[0] = json.dumps(meta) + "\n"
    with open(path, "w") as f:
        f.writelines(lines)


def load_session(sessions_dir, session_id):
    """Load a session. Returns (metadata_dict, list_of_messages)."""
    path = _session_path(sessions_dir, session_id)
    with open(path, "r") as f:
        lines = f.readlines()
    meta = json.loads(lines[0])
    messages = [json.loads(line) for line in lines[1:] if line.strip()]
    return meta, messages


def latest_session(sessions_dir, model=None):
    """Find the most recent session id, optionally filtered by model. Returns None if none exist."""
    sessions = list_sessions(sessions_dir, model=model)
    if not sessions:
        return None
    return sessions[0]["id"]


def list_sessions(sessions_dir, model=None):
    """List sessions sorted by updated (newest first). Each entry is the metadata dict."""
    os.makedirs(sessions_dir, exist_ok=True)
    result = []
    for fname in os.listdir(sessions_dir):
        if not fname.endswith(".jsonl"):
            continue
        try:
            with open(os.path.join(sessions_dir, fname), "r") as f:
                meta = json.loads(f.readline())
            if model and meta.get("model") != model:
                continue
            result.append(meta)
        except (json.JSONDecodeError, OSError):
            continue
    result.sort(key=lambda m: m.get("updated", ""), reverse=True)
    return result


def export_markdown(sessions_dir, session_id):
    """Export a session as a readable markdown string."""
    meta, messages = load_session(sessions_dir, session_id)
    lines = [
        f"# Session {meta['id']}",
        f"Model: {meta['model']}  ",
        f"Created: {meta['created']}  ",
        f"Updated: {meta['updated']}",
        "",
        "---",
        "",
    ]
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "system":
            lines.append(f"**System:**\n{content}\n")
        elif role == "user":
            lines.append(f"**You:**\n{content}\n")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if content:
                lines.append(f"**Assistant:**\n{content}\n")
            for tc in tool_calls:
                fn = tc.get("function", {})
                lines.append(f"*Tool call: {fn.get('name')}({json.dumps(fn.get('arguments', {}))})*\n")
        elif role == "tool":
            lines.append(f"*Tool result:*\n```\n{content}\n```\n")
    return "\n".join(lines)
