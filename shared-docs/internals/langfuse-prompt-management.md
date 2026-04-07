# LangFuse Prompt Management

**LangFuse is the single source of truth for all prompts.**

No prompts in code. No fallbacks. No duplication.

## Quick Start

### 1. Start LangFuse

```bash
cd ../../shared-docker-infra
make up-langfuse
```

### 2. Get API Keys

1. Open http://langfuse.local
2. Settings → API Keys
3. Copy Public Key and Secret Key

### 3. Configure Environment

```bash
cp .env.example .env
vim .env  # Add your keys
```

### 4. Upload Prompts

```bash
uv run python scripts/upload_prompts_to_langfuse.py
```

### 5. Verify

Open http://langfuse.local → Prompts

## Using Prompts

```python
from deep_research_engine.prompts import get_prompt, PromptRegistry

# Get prompt from LangFuse
prompt = await get_prompt(PromptRegistry.RESEARCH_PLANNER)

# Compile with variables
compiled = prompt.compile(
    strategy="complexity_based",
    min_workers=1,
    max_workers=3,
    query="AI trends in 2025"
)

# Get model config from LangFuse
model_config = prompt.get_model_config()
# {"model": "qwen-35-27", "temperature": 0.7}
```

## Prompt Catalog

| Prompt | Variables | Used By |
|--------|-----------|---------|
| `research-planner` | strategy, min_workers, max_workers, query | Boss Node |
| `research-evaluator` | criteria_text, results_text | Critic Node |
| `research-report` | findings, worker_count, citations_enabled, citations_format | Synthesis |
| `markdown-format` | findings, worker_count, citations_enabled, citations_format | Synthesis |
| `structured-json` | findings, worker_count, citations_enabled, citations_format | Synthesis |

## Editing Prompts

### Via LangFuse UI (Recommended)

1. Open http://langfuse.local
2. Prompts → Select prompt → Edit
3. Save → Label as "production"

**Changes are instant. No code deployment needed.**

### Via Upload Script

```bash
# Edit prompts in scripts/upload_prompts_to_langfuse.py
vim scripts/upload_prompts_to_langfuse.py

# Re-upload
uv run python scripts/upload_prompts_to_langfuse.py
```

## Architecture

```
┌─────────────────────────────────────┐
│         Your Application            │
│                                     │
│  prompt = await get_prompt(name)    │
│  compiled = prompt.compile(**vars)  │
└─────────────────────────────────────┘
                 ↓
        ┌────────────────┐
        │   LangFuse     │
        │  (prompts DB)  │
        └────────────────┘
```

**YAML config stays in your codebase.** Only prompt templates live in LangFuse.

```yaml
# configs/rag.yaml - Application config
llm:
  models:
    main: qwen-35-27  # ← Stays in YAML

# LangFuse - Prompt templates only
# research-planner prompt:
# "You are... {{ strategy }}..."  ← Lives in LangFuse
```

## Troubleshooting

### Missing Variables Error

```python
# Error: Missing required variables for prompt 'research-planner': 
#        ['min_workers']. Provided: ['strategy']. Required: ['strategy', 'min_workers', ...]

# Fix: Provide all required variables
prompt.compile(
    strategy="...",
    min_workers=1,  # ← Was missing
    max_workers=3,
    query="..."
)
```

### LangFuse Unavailable

```bash
# Check LangFuse is running
curl http://langfuse.local/api/health

# Check credentials
echo $LANGFUSE_PUBLIC_KEY
echo $LANGFUSE_SECRET_KEY
```

### Connection Issues

```bash
# Test connection
uv run python -c "from langfuse import Langfuse; print(Langfuse().client._client.health.get())"

# View logs
docker compose logs langfuse
```

## Next Steps

- **Upload prompts**: `uv run python scripts/upload_prompts_to_langfuse.py`
- **LangFuse docs**: https://langfuse.com/docs/prompt-management
- **Edit prompts**: http://langfuse.local → Prompts
