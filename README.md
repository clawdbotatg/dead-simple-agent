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
python3 run.py qwen3:32b

# One-shot
python3 run.py qwen3:32b "how much RAM does this machine have?"

# Force a fresh session
python3 run.py qwen3:32b --new

# Resume a specific session by id
python3 run.py qwen3:32b --resume 20260326-180721-da21e0bf

# Use a specific provider
python3 run.py venice-uncensored --provider venice
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

Providers live in `providers.py`. Each one is a function that talks to an LLM API and returns a normalized response. Built-in:

- **ollama** — local models via [Ollama](https://ollama.ai) (default)
- **venice** — [Venice AI](https://venice.ai) inference API (OpenAI-compatible)
- **anthropic** — [Anthropic](https://anthropic.com) native Messages API (Claude models)
- **bankr** — [Bankr LLM Gateway](https://docs.bankr.bot/llm-gateway/overview) — Claude, GPT, Gemini, and more through a single API

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
- **memory_list** / **memory_read** / **memory_write** / **memory_search** — persistent memory (see below)
- **github_*** — list repos, read/write files, issues, PRs, search code (requires [`gh` CLI](https://cli.github.com))

Adding a tool is ~20 lines: write a function, add a spec dict to `TOOL_REGISTRY`.

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
run.py           — main script: CLI, agent loop, provider dispatch
providers.py     — LLM provider backends (ollama, venice, anthropic, bankr)
tools.py         — tool definitions and implementations
sessions.py      — session persistence (create, load, append, list)
system_prompt.md — system prompt (edit this to change personality)
test.py          — manual test script for tools, memory, and providers
.env             — config (API keys, URLs)
memory/          — agent memory files (gitignored, created at runtime)
sessions/        — saved conversations (gitignored, created at runtime)
```

## Requirements

- Python 3.8+
- `requests` (`pip3 install requests`)
- [Ollama](https://ollama.ai) if using the ollama provider
