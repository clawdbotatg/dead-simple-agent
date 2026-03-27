# dead-simple-agent

A local AI agent in ~1000 lines of Python. Runs any LLM with tool use, session persistence, and auto-resume.

No frameworks. No abstractions. Just a loop that calls a model, executes tools, and saves the conversation.

## Extending — Create Your Own Agent

The base agent is pip-installable. You can create specialized agents (a "builder", a "researcher", etc.) that add their own tools, prompts, and memory while pulling updates from the base.

### Install the base

```bash
pip3 install git+https://github.com/austingriffith/dead-simple-agent.git
```

### Create a new agent

Each agent is just a directory:

```bash
mkdir ~/agents/builder && cd ~/agents/builder
```

**`system_prompt.md`** — your agent's personality:

```markdown
You are a builder agent specialized in scaffolding and deploying projects.

## Memory

{{MEMORY}}

## Available Tools

{{TOOLS}}
```

**`tools.py`** — your extra tools:

```python
def _run_scaffold(args):
    import subprocess
    name = args["name"]
    template = args.get("template", "react")
    result = subprocess.run(
        f"npx create-{template}-app {name}",
        shell=True, capture_output=True, text=True, timeout=60,
    )
    return result.stdout or result.stderr

TOOLS = [
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "scaffold_project",
                "description": "Scaffold a new project from a template",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Project name"},
                        "template": {"type": "string", "description": "Template: react, next, vite (default: react)"},
                    },
                    "required": ["name"],
                },
            },
        },
        "run": _run_scaffold,
    },
]
```

**`run.py`** — the 3-line entry point:

```python
#!/usr/bin/env python3
from agent import Agent
from tools import TOOLS

Agent(extra_tools=TOOLS).cli()
```

**`.env`** — copy your API keys or symlink from another agent.

### Run it

```bash
cd ~/agents/builder
python3 run.py openrouter/auto
```

It picks up `system_prompt.md`, `tools.py`, `.env`, `memory/`, and `sessions/` from the current directory. Each agent has its own memory and sessions.

### Example: researcher agent

```bash
mkdir ~/agents/researcher && cd ~/agents/researcher
```

`system_prompt.md`:
```markdown
You are a research agent. You investigate topics thoroughly using web searches, read papers and documentation, and compile your findings into memory files.

## Memory

{{MEMORY}}

## Available Tools

{{TOOLS}}
```

`tools.py`:
```python
import subprocess

def _run_arxiv_search(args):
    import urllib.request, json
    query = args["query"].replace(" ", "+")
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results=5"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode()[:4000]

TOOLS = [
    {
        "spec": {
            "type": "function",
            "function": {
                "name": "arxiv_search",
                "description": "Search arXiv for academic papers on a topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        "run": _run_arxiv_search,
    },
]
```

`run.py`:
```python
#!/usr/bin/env python3
from agent import Agent
from tools import TOOLS

Agent(extra_tools=TOOLS).cli()
```

### Update the base

```bash
pip3 install --upgrade git+https://github.com/austingriffith/dead-simple-agent.git
```

Your agent directories are untouched — only the `agent` package gets updated.

## Setup

```bash
pip3 install requests

cp .env.example .env
# Edit .env — add API keys for the providers you want to use
```

## Usage

```bash
# Interactive — auto-resumes your last session
python3 run.py qwen3:32b

# One-shot
python3 run.py qwen3:32b "how much RAM does this machine have?"

# Force a fresh session
python3 run.py qwen3:32b --new

# Resume a specific session by id
python3 run.py qwen3:32b --resume 20260326-180721-da21e0bf

# Use a specific provider
python3 run.py openrouter/auto --provider openrouter
python3 run.py qwen3:32b --provider ollama
```

The provider is auto-detected from the model name, or you can set it explicitly with `--provider`.

When you resume, it shows a recap of your last few exchanges so you remember where you left off.

### Slash commands (inside interactive mode)

| Command | What it does |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Start a fresh conversation |
| `/history` | List past sessions |
| `/save [path]` | Export session as markdown |
| `/exit` | Quit |

## Providers

Providers live in `agent/providers.py`. Each one is a function that talks to an LLM API and returns a normalized response. Built-in:

- **ollama** — local models via [Ollama](https://ollama.ai) (default)
- **venice** — [Venice AI](https://venice.ai) inference API (OpenAI-compatible)
- **openrouter** — [OpenRouter](https://openrouter.ai) — 300+ models through a single API
- **anthropic** — [Anthropic](https://anthropic.com) native Messages API (Claude models)
- **bankr** — [Bankr LLM Gateway](https://docs.bankr.bot/llm-gateway/overview) — Claude, GPT, Gemini, and more through a single API

### Adding a provider

~30 lines. Write a chat function, register it:

```python
# in agent/providers.py

def my_provider_chat(model, messages, tool_specs):
    api_key = os.environ.get("MY_PROVIDER_KEY", "")
    return _openai_compatible_chat(
        model, messages, tool_specs,
        api_key=api_key,
        base_url="https://api.myprovider.com/v1",
    )

PROVIDERS["myprovider"] = my_provider_chat
```

If the API is OpenAI-compatible (most are), you can reuse `_openai_compatible_chat()`. Otherwise, write your own request/response handling.

Add auto-detection rules in `detect_provider()` if you want model names to route automatically.

## Tools

The agent can use these tools (defined in `agent/tools.py`):

- **shell** — run any shell command
- **fetch_url** — fetch web pages and APIs
- **read_file** / **write_file** — local file access
- **memory_list** / **memory_read** / **memory_write** / **memory_search** — persistent memory (see below)
- **github_*** — list repos, read/write files, issues, PRs, search code (requires [`gh` CLI](https://cli.github.com))

Adding a tool is ~20 lines: write a function, add a spec dict to `BASE_TOOLS`.

## Memory

The agent has a persistent memory system backed by files in `memory/`.

- **`memory/critical.md`** — automatically loaded into the system prompt on every conversation. Use this for facts the agent should always know (who you are, preferences, key context).
- All other memory files are accessible on-demand via tools.

The agent manages its own memory through four tools:

| Tool | What it does |
|------|-------------|
| `memory_list` | Browse memory files sorted by most recently modified (default: 20, pass `limit: 0` for all) |
| `memory_read` | Read a specific memory file |
| `memory_write` | Create or update a memory file |
| `memory_search` | Search across all memory files by keyword |

Memory files are gitignored — they stay local to your machine.

## Sessions

Conversations are saved as `.jsonl` files in `sessions/`. Each file is append-only — metadata on line 1, messages after that. Running the agent again picks up right where you left off.

## Testing

```bash
python3 test.py              # run all tests
python3 test.py tools        # just tool tests (no LLM calls)
python3 test.py memory       # just memory integration tests
python3 test.py providers    # just provider connectivity tests
```

Tests tools, memory system wiring, and provider connectivity. Provider tests are skipped if the API key isn't set. Tool and memory tests run in under a second with no LLM calls.

## Files

```
agent/               — the installable package
  __init__.py        — exports Agent
  core.py            — Agent class: CLI, agent loop, prompt builder
  providers.py       — LLM provider backends (ollama, venice, openrouter, anthropic, bankr)
  tools.py           — base tools + memory tool factory
  sessions.py        — session persistence (create, load, append, list)
  default_prompt.md  — default system prompt (used when no local one exists)
run.py               — thin wrapper: from agent import Agent; Agent().cli()
test.py              — manual test script for tools, memory, and providers
system_prompt.md     — system prompt (edit to change personality)
pyproject.toml       — packaging (pip install)
.env                 — config (API keys, URLs)
memory/              — agent memory files (gitignored, created at runtime)
sessions/            — saved conversations (gitignored, created at runtime)
```

## Requirements

- Python 3.8+
- `requests` (`pip3 install requests`)
- [Ollama](https://ollama.ai) if using the ollama provider
