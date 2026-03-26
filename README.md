# dead-simple-agent

A local AI agent in ~400 lines of Python. Runs Qwen (or any Ollama model) with tool use, session persistence, and auto-resume.

No frameworks. No abstractions. Just a loop that calls a model, executes tools, and saves the conversation.

## Setup

```bash
pip install requests

cp .env.example .env
# Edit .env if your Ollama is on a different host

# Pull a model
ollama pull qwen3:32b
```

## Usage

```bash
# Interactive — auto-resumes your last session
python qwen-run.py qwen3:32b

# One-shot
python qwen-run.py qwen3:32b "how much RAM does this machine have?"

# Force a fresh session
python qwen-run.py qwen3:32b --new

# Resume a specific session by id
python qwen-run.py qwen3:32b --resume 20260326-180721-da21e0bf
```

When you resume, it shows a recap of your last few exchanges so you remember where you left off.

### Slash commands (inside interactive mode)

| Command | What it does |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Start a fresh conversation |
| `/history` | List past sessions |
| `/save [path]` | Export session as markdown |
| `/exit` | Quit |

## Tools

The agent can use these tools (defined in `tools.py`):

- **shell** — run any shell command
- **fetch_url** — fetch web pages and APIs
- **read_file** / **write_file** — local file access
- **github_*** — list repos, read/write files, issues, PRs, search code (requires [`gh` CLI](https://cli.github.com))

Adding a tool is ~20 lines: write a function, add a spec dict to `TOOL_REGISTRY`.

## Sessions

Conversations are saved as `.jsonl` files in `sessions/`. Each file is append-only — metadata on line 1, messages after that. Running the agent again picks up right where you left off.

## Files

```
qwen-run.py      — main script: CLI, agent loop, Ollama API
tools.py         — tool definitions and implementations
sessions.py      — session persistence (create, load, append, list)
system_prompt.md — system prompt (edit this to change personality)
.env             — config (OLLAMA_URL)
```

## Requirements

- [Ollama](https://ollama.ai) running locally (or on your network)
- Python 3.8+
- `requests` (`pip install requests`)
