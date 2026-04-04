"""
leftclaw.py - LeftClaw Services tools for agents.

Usage:
    from agent.leftclaw import make_leftclaw_tools

    TOOLS = make_leftclaw_tools(service_type_id=7)
    Agent(extra_tools=TOOLS).cli()
"""

import json
import os
import subprocess
import urllib.request

_LEFTCLAW_BASE = "https://leftclaw.services"
_CAST = os.path.expanduser("~/.foundry/bin/cast")

# ---------------------------------------------------------------------------
# Shared helpers (module-level, contract cached across all tool instances)
# ---------------------------------------------------------------------------

_CONTRACT = None
_WORKER_ADDRESS = None


def _fetch_contract():
    global _CONTRACT
    if _CONTRACT:
        return _CONTRACT
    try:
        req = urllib.request.Request(
            f"{_LEFTCLAW_BASE}/api/services",
            headers={"User-Agent": "leftclaw-agent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        addr = data.get("contract", "")
        if addr and addr.startswith("0x"):
            _CONTRACT = addr
            return _CONTRACT
    except Exception:
        pass
    raise RuntimeError("Could not fetch contract address from leftclaw.services/api/services")


def _contract():
    return _CONTRACT or _fetch_contract()


def _rpc():
    return os.environ.get("BASE_RPC_URL", "https://mainnet.base.org")


def _privkey():
    return os.environ.get("ETH_PRIVATE_KEY", "")


def worker_address():
    """Derive the worker address from ETH_PRIVATE_KEY using cast."""
    global _WORKER_ADDRESS
    if _WORKER_ADDRESS:
        return _WORKER_ADDRESS
    pk = _privkey()
    if not pk:
        raise RuntimeError("ETH_PRIVATE_KEY not set in .env")
    try:
        result = subprocess.run(
            [_CAST, "wallet", "address", "--private-key", pk],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip().startswith("0x"):
            _WORKER_ADDRESS = result.stdout.strip()
            return _WORKER_ADDRESS
    except Exception:
        pass
    raise RuntimeError("Could not derive worker address from ETH_PRIVATE_KEY")


def _cast_send(func_sig, *call_args):
    pk = _privkey()
    if not pk:
        return "ERROR: ETH_PRIVATE_KEY not set in .env"
    cmd = [
        _CAST, "send", _contract(), func_sig, *[str(a) for a in call_args],
        "--rpc-url", _rpc(),
        "--private-key", pk,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout.strip()
        if result.returncode != 0:
            return f"ERROR: {result.stderr.strip() or output}"
        return output or "(transaction sent)"
    except subprocess.TimeoutExpired:
        return "ERROR: transaction timed out (60s)"
    except Exception as e:
        return f"ERROR: {e}"


def _cast_call(func_sig, *call_args):
    cmd = [
        _CAST, "call", _contract(), func_sig, *[str(a) for a in call_args],
        "--rpc-url", _rpc(),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if result.returncode != 0:
            return f"ERROR: {result.stderr.strip() or output}"
        return output
    except Exception as e:
        return f"ERROR: {e}"


def _get_next_job_id():
    try:
        cmd = [_CAST, "call", _contract(), "nextJobId()", "--rpc-url", _rpc()]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            raw = result.stdout.strip()
            return int(raw, 16) if raw.startswith("0x") else int(raw)
    except Exception:
        pass
    return 20


def _parse_job_words(h):
    """Parse hex-encoded getJob() return data into a dict."""
    words = [h[i:i + 64] for i in range(0, len(h), 64)]
    if len(words) <= 14:
        return None

    status = int(words[7], 16)
    worker = "0x" + words[12][24:]
    stype = int(words[3], 16)
    client = "0x" + words[2][24:]

    desc = ""
    try:
        offset = int(words[6], 16) // 32
        length = int(words[offset + 1], 16) if offset + 1 < len(words) else 0
        data_start = (offset + 2) * 64
        raw_hex = h[data_start:data_start + length * 2]
        desc = bytes.fromhex(raw_hex).decode("utf-8", errors="replace")
    except Exception:
        pass

    return {
        "status": status,
        "worker": worker,
        "serviceTypeId": stype,
        "client": client,
        "description": desc,
    }


_STATUS_MAP = {
    0: "OPEN",
    1: "IN_PROGRESS",
    2: "COMPLETED",
    3: "DECLINED",
    4: "CANCELLED",
    5: "REASSIGNED",
}


def is_job_complete(job_id):
    """Check on-chain if the job status is COMPLETED (status == 2)."""
    try:
        cmd = [_CAST, "call", _contract(), "getJob(uint256)", str(job_id), "--rpc-url", _rpc()]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False
        raw = result.stdout.strip()
        h = raw[2:] if raw.startswith("0x") else raw
        info = _parse_job_words(h)
        return info is not None and info["status"] == 2
    except Exception:
        return False


def get_active_jobs(worker_addr, service_type_id):
    """Return on-chain IN_PROGRESS jobs assigned to worker_addr for the given service type.

    If worker_addr is None, derives it from ETH_PRIVATE_KEY.
    """
    if worker_addr is None:
        worker_addr = worker_address()
    next_id = _get_next_job_id()
    active = []
    for job_id in range(1, min(next_id, 100)):
        try:
            cmd = [_CAST, "call", _contract(), "getJob(uint256)", str(job_id), "--rpc-url", _rpc()]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                continue
            raw = r.stdout.strip()
            h = raw[2:] if raw.startswith("0x") else raw
            info = _parse_job_words(h)
            if not info:
                continue
            if (info["status"] == 1
                    and info["worker"].lower() == worker_addr.lower()
                    and info["serviceTypeId"] == service_type_id):
                active.append({"id": job_id, "status": "IN_PROGRESS", **info})
        except Exception:
            continue
    return active


def get_ready_jobs(service_type_id):
    """Fetch open jobs from the LeftClaw API for a given service type."""
    try:
        url = f"{_LEFTCLAW_BASE}/api/job/ready"
        req = urllib.request.Request(url, headers={"User-Agent": "leftclaw-agent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        jobs = data.get("jobs", data if isinstance(data, list) else [])
        return [j for j in jobs if str(j.get("serviceTypeId")) == str(service_type_id)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_leftclaw_tools(service_type_id, worker_addr=None):
    """Return a list of LeftClaw tool dicts parameterized for this agent.

    Args:
        service_type_id: The LeftClaw service type this agent handles (e.g. 7=research, 5=QA).
        worker_addr: Optional. The on-chain address of this agent/worker.
            If None, derived from ETH_PRIVATE_KEY at call time.
    """
    _addr = worker_addr

    def _resolve_addr():
        return _addr or worker_address()

    def _run_check_jobs(args):
        try:
            url = f"{_LEFTCLAW_BASE}/api/job/ready"
            req = urllib.request.Request(url, headers={"User-Agent": "leftclaw-agent/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            if isinstance(data, dict):
                jobs = data.get("jobs", data.get("data", []))
            elif isinstance(data, list):
                jobs = data
            else:
                return json.dumps(data, indent=2)[:8000]

            filtered = [
                j for j in jobs
                if str(j.get("serviceTypeId")) == str(service_type_id)
            ]

            if not filtered:
                return f"No open jobs for service type {service_type_id} right now."

            lines = []
            for j in filtered:
                lines.append(f"Job #{j.get('id')} — {j.get('description', '(no description)')[:200]}")
                lines.append(f"  client: {j.get('client', '?')}  status: {j.get('status', '?')}  stage: {j.get('currentStage', '?')}")
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    def _run_check_my_jobs(args):
        try:
            active = get_active_jobs(_resolve_addr(), service_type_id)
            if active:
                lines = [
                    f"Job #{j['id']} — IN_PROGRESS — {j['description'][:200]}"
                    for j in active
                ]
                return "IN-PROGRESS jobs assigned to you:\n" + "\n".join(lines)
            return "No in-progress jobs assigned to you."
        except Exception as e:
            return f"ERROR checking active jobs: {e}"

    def _run_get_job(args):
        try:
            job_id = str(args["job_id"])
            cmd = [_CAST, "call", _contract(), "getJob(uint256)", job_id, "--rpc-url", _rpc()]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return f"ERROR: {result.stderr.strip()}"

            raw = result.stdout.strip()
            h = raw[2:] if raw.startswith("0x") else raw
            words = [h[i:i + 64] for i in range(0, len(h), 64)]

            def addr(w):
                return "0x" + w[24:]

            def uint(w):
                return int(w, 16)

            def text_at(wds, offset_word_idx):
                offset = uint(wds[offset_word_idx]) // 32
                length = uint(wds[offset + 1]) if offset + 1 < len(wds) else 0
                data_start = (offset + 2) * 64
                raw_hex = h[data_start:data_start + length * 2]
                try:
                    return bytes.fromhex(raw_hex).decode("utf-8", errors="replace")
                except Exception:
                    return ""

            client = addr(words[2])
            stype = uint(words[3])
            price_usd = uint(words[5])
            status_int = uint(words[7])
            worker = addr(words[12])
            created = uint(words[8])

            desc = text_at(words, 6)
            stage = ""
            for w in words:
                try:
                    t = bytes.fromhex(w).decode("utf-8", errors="ignore").strip("\x00")
                    if t in ("accepted", "research", "create_repo", "prototype", "ready", "qa_audit"):
                        stage = t
                except Exception:
                    pass

            return (
                f"Job #{job_id}\n"
                f"description: {desc}\n"
                f"client: {client}\n"
                f"worker: {worker}\n"
                f"serviceTypeId: {stype}\n"
                f"status: {_STATUS_MAP.get(status_int, status_int)}\n"
                f"currentStage: {stage}\n"
                f"priceUsd: ${price_usd / 10000:.2f}\n"
                f"createdAt: {created}"
            )
        except Exception as e:
            return f"ERROR: {e}"

    def _run_get_messages(args):
        try:
            job_id = args["job_id"]
            url = f"{_LEFTCLAW_BASE}/api/job/{job_id}/messages"
            req = urllib.request.Request(url, headers={"User-Agent": "leftclaw-agent/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            return json.dumps(data, indent=2)[:8000]
        except Exception as e:
            return f"ERROR: {e}"

    def _run_post_message(args):
        try:
            job_id = args["job_id"]
            msg_type = args.get("type", "bot_message")
            content = args["content"]
            metadata = args.get("metadata", {})

            url = f"{_LEFTCLAW_BASE}/api/job/{job_id}/messages"
            body = json.dumps({"type": msg_type, "from": "bot", "content": content, "metadata": metadata}).encode()
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/json",
                "User-Agent": "leftclaw-agent/1.0",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode()[:4000]
        except Exception as e:
            return f"ERROR: {e}"

    def _run_accept_job(args):
        return _cast_send("acceptJob(uint256)", args["job_id"])

    def _run_log_work(args):
        return _cast_send("logWork(uint256,string,string)", args["job_id"], args["note"], args["stage"])

    def _run_complete_job(args):
        return _cast_send("completeJob(uint256,string)", args["job_id"], args["result_url"])

    return [
        {
            "spec": {"type": "function", "function": {
                "name": "leftclaw_check_jobs",
                "description": (
                    f"Check LeftClaw Services for NEW open jobs (Service Type {service_type_id} only). "
                    "Call leftclaw_check_my_jobs FIRST to resume unfinished work, then call this only if you have no in-progress jobs."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            }},
            "run": _run_check_jobs,
        },
        {
            "spec": {"type": "function", "function": {
                "name": "leftclaw_check_my_jobs",
                "description": (
                    f"Check on-chain for jobs (Service Type {service_type_id}) already assigned to you that are "
                    "IN_PROGRESS but not yet completed. Call this FIRST before leftclaw_check_jobs — resume unfinished work before taking new jobs."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            }},
            "run": _run_check_my_jobs,
        },
        {
            "spec": {"type": "function", "function": {
                "name": "leftclaw_get_job",
                "description": "Get full details for a specific LeftClaw job by ID.",
                "parameters": {"type": "object", "properties": {
                    "job_id": {"type": "integer", "description": "The job ID"},
                }, "required": ["job_id"]},
            }},
            "run": _run_get_job,
        },
        {
            "spec": {"type": "function", "function": {
                "name": "leftclaw_get_messages",
                "description": "Get all messages for a LeftClaw job. Check this before starting work — client messages may contain scope changes.",
                "parameters": {"type": "object", "properties": {
                    "job_id": {"type": "integer", "description": "The job ID"},
                }, "required": ["job_id"]},
            }},
            "run": _run_get_messages,
        },
        {
            "spec": {"type": "function", "function": {
                "name": "leftclaw_post_message",
                "description": "Post a message to a LeftClaw job (escalation or bot response).",
                "parameters": {"type": "object", "properties": {
                    "job_id": {"type": "integer", "description": "The job ID"},
                    "content": {"type": "string", "description": "Message content"},
                    "type": {"type": "string", "description": "Message type: bot_message or escalation (default: bot_message)"},
                }, "required": ["job_id", "content"]},
            }},
            "run": _run_post_message,
        },
        {
            "spec": {"type": "function", "function": {
                "name": "leftclaw_accept_job",
                "description": "Accept a LeftClaw job on-chain. This claims the job — you must complete it.",
                "parameters": {"type": "object", "properties": {
                    "job_id": {"type": "integer", "description": "The job ID to accept"},
                }, "required": ["job_id"]},
            }},
            "run": _run_accept_job,
        },
        {
            "spec": {"type": "function", "function": {
                "name": "leftclaw_log_work",
                "description": "Log work progress on-chain for a LeftClaw job. Sets the job's current stage.",
                "parameters": {"type": "object", "properties": {
                    "job_id": {"type": "integer", "description": "The job ID"},
                    "note": {"type": "string", "description": "Work note (max 500 chars)"},
                    "stage": {"type": "string", "description": "Stage name (e.g. 'research', 'qa_audit')"},
                }, "required": ["job_id", "note", "stage"]},
            }},
            "run": _run_log_work,
        },
        {
            "spec": {"type": "function", "function": {
                "name": "leftclaw_complete_job",
                "description": "Complete a LeftClaw job on-chain. Pass the FULL BGIPFS gateway URL as result_url.",
                "parameters": {"type": "object", "properties": {
                    "job_id": {"type": "integer", "description": "The job ID"},
                    "result_url": {"type": "string", "description": "Full IPFS gateway URL: https://{CID}.ipfs.community.bgipfs.com/"},
                }, "required": ["job_id", "result_url"]},
            }},
            "run": _run_complete_job,
        },
    ]
