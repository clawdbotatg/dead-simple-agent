"""
bgipfs.py - BGIPFS upload tool for agents.

Usage:
    from agent.bgipfs import BGIPFS_TOOLS

    Agent(extra_tools=BGIPFS_TOOLS).cli()
"""

import json
import os
import subprocess


def _run_bgipfs_upload(args):
    key = os.environ.get("BGIPFS_API_KEY", "")
    if not key:
        return "ERROR: BGIPFS_API_KEY not set in .env"

    filepath = os.path.expanduser(args["filepath"])
    if not os.path.exists(filepath):
        return f"ERROR: file not found: {filepath}"

    try:
        cmd = [
            "curl", "-s", "-X", "POST",
            "https://upload.bgipfs.com/api/v0/add",
            "-H", f"X-API-Key: {key}",
            "-F", f"file=@{filepath}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return f"ERROR: {result.stderr.strip()}"

        data = json.loads(result.stdout)
        cid_v0 = data.get("Hash", "")
        if not cid_v0:
            return f"ERROR: no Hash in response: {result.stdout}"

        conv = subprocess.run(
            ["npx", "cid-tool", "base32", cid_v0],
            capture_output=True, text=True, timeout=30,
        )
        if conv.returncode == 0 and conv.stdout.strip():
            cid = conv.stdout.strip()
        else:
            cid = cid_v0

        gateway_url = f"https://{cid}.ipfs.community.bgipfs.com/"
        return f"Uploaded successfully.\nCID: {cid}\nGateway URL: {gateway_url}"
    except Exception as e:
        return f"ERROR: {e}"


BGIPFS_TOOLS = [
    {
        "spec": {"type": "function", "function": {
            "name": "bgipfs_upload",
            "description": "Upload a file to BGIPFS and get back the gateway URL. Use this to upload finished reports and deliverables.",
            "parameters": {"type": "object", "properties": {
                "filepath": {"type": "string", "description": "Path to the file to upload"},
            }, "required": ["filepath"]},
        }},
        "run": _run_bgipfs_upload,
    },
]
