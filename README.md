# dead-simple-agent

A local AI agent in ~1000 lines of Python. Runs any LLM with tool use, session persistence, and auto-resume.

No frameworks. No abstractions. Just a loop that calls a model, executes tools, and saves the conversation.

## Setup

```bash
pip install requests

cp .env.example .env
# Edit .env — add API keys for the providers you want to use
```

## Usage

```bash
# Interactive — auto-resumes your last session
python run.py qwen3:32b

# One-shot
python run.py qwen3:32b "how much RAM does this machine have?"

# Force a fresh session
python run.py qwen3:32b --new

# Resume a specific session by id
python run.py qwen3:32b --resume 20260326-180721-da21e0bf

# Use a specific provider
python run.py venice-uncensored --provider venice
python run.py qwen3:32b --provider ollama
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

Providers live in `providers.py`. Each one is a function that talks to an LLM API and returns a normalized response. Built-in:

- **ollama** — local models via [Ollama](https://ollama.ai) (default)
- **venice** — [Venice AI](https://venice.ai) inference API

### Adding a provider

~30 lines. Write a chat function, register it:

```python
# in providers.py

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
run.py           — main script: CLI, agent loop, provider dispatch
providers.py     — LLM provider backends (ollama, venice, add your own)
tools.py         — tool definitions and implementations
sessions.py      — session persistence (create, load, append, list)
system_prompt.md — system prompt (edit this to change personality)
.env             — config (API keys, URLs)
```

## Requirements

- Python 3.8+
- `requests` (`pip install requests`)
- [Ollama](https://ollama.ai) if using the ollama provider
