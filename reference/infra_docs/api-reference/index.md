# API Reference

Complete API documentation for all services in the Shared Docker Infrastructure.

## API Categories

### [LiteLLM API](litellm.md) - Inference Gateway

Base URL: `http://litellm.local/v1`  
Auth: `Authorization: Bearer sk-litellm-master`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (LLM) |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/audio/transcriptions` | POST | Speech-to-text (ASR) |
| `/v1/audio/speech` | POST | Text-to-speech (TTS) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

**Documentation:** [LiteLLM API](litellm.md)

### [Orchestrator API](orchestrator.md) - Resource Management

Base URL: `http://orchestrator.local`  
Auth: None required

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Full system status |
| `/vram` | GET | VRAM usage |
| `/load/{slot}/{model}` | POST | Load model |
| `/unload/{slot}` | POST | Unload model |
| `/llm/models` | GET | LLM model status |
| `/llm/health` | GET | LLM health check |
| `/llm/props` | GET | LLM properties |
| `/health` | GET | Health check |

**Documentation:** [Orchestrator API](orchestrator.md)

### [Langfuse API](langfuse.md) - Observability

Base URL: `http://langfuse.local/api`  
Auth: Basic auth (API keys)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/traces` | GET/POST | Trace management |
| `/sessions` | GET | Session tracking |
| `/prompts` | GET/POST | Prompt management |
| `/scores` | POST | Add scores |
| `/generations` | GET/POST | Generation tracking |

**Documentation:** [Langfuse API](langfuse.md)

### [MCP Servers](mcp.md) - Web Tools

Base URL: `http://webtools.local`  
Access: Via LiteLLM tools parameter

| Tool | Description |
|------|-------------|
| `search_web` | Web search with reranking |
| `scrape_url` | Scrape single pages |
| `map_domain` | Discover URLs from sitemaps |
| `crawl_site` | Deep site crawling |
| `scrape_structured` | Schema-based JSON extraction |

**Documentation:** [MCP Servers](mcp.md)

## Quick Reference

### Chat Completion (LiteLLM)

```bash
curl http://litellm.local/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-35-27",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Load Model (Orchestrator)

```bash
curl -X POST http://orchestrator.local/load/tts/kokoro
```

### Check Status (Orchestrator)

```bash
curl http://orchestrator.local/status
```

### Create Trace (Langfuse)

```bash
curl -X POST http://langfuse.local/api/traces \
  -u "pk-lf-...:sk-lf-..." \
  -H "Content-Type: application/json" \
  -d '{
    "name": "chat-completion",
    "userId": "user-123"
  }'
```

## SDKs

### Python (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://litellm.local/v1",
    api_key="sk-litellm-master"
)

response = client.chat.completions.create(
    model="qwen-35-27",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Python (Langfuse)

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="http://langfuse.local"
)

trace = langfuse.trace(name="my-trace")
```

## Next Steps

- **[LiteLLM API](litellm.md)** - Complete inference API
- **[Orchestrator API](orchestrator.md)** - Resource management
- **[Langfuse API](langfuse.md)** - Observability
- **[MCP Servers](mcp.md)** - Web tools
- **[Integration](../integration/index.md)** - Integration guides
