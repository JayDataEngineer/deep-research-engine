# LiteLLM

LiteLLM is the inference gateway providing an OpenAI-compatible API for all AI models.

## Overview

- **Purpose:** Unified inference API (OpenAI-compatible)
- **Domain:** `litellm.local`
- **Internal URL:** `http://litellm:8080`
- **API:** `http://litellm.local/v1`
- **Auth:** `Authorization: Bearer sk-litellm-master`

## Key Features

- ✅ **OpenAI-compatible API** - Drop-in replacement
- ✅ **Model routing** - Route requests to different backends
- ✅ **Automatic Langfuse tracing** - Built-in observability
- ✅ **Multi-modal support** - Chat, embeddings, ASR, TTS
- ✅ **Load balancing** - Multiple model instances
- ✅ **Rate limiting** - Protect backends
- ✅ **Fallback models** - Automatic retry on failure

## Quick Start

```bash
# Start LiteLLM
make up-litellm

# Or using Docker Compose
docker compose -f docker-compose.litellm.yml up -d

# Test connection
curl http://litellm.local/v1/models \
  -H "Authorization: Bearer sk-litellm-master"

# Chat completion
curl http://litellm.local/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-35-27", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Configuration

### Environment Variables

```bash
# API Key
LITELLM_MASTER_KEY=sk-litellm-master

# Langfuse Integration
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://langfuse.local

# Logging
LITELLM_LOG_LEVEL=INFO
```

### Model Configuration

```yaml
# litellm/config.yaml
model_list:
  # LLM Models
  - model_name: qwen-35-27
    litellm_params:
      model: ollama/qwen-35-27
      api_base: http://llama-cpp:8080
    
  - model_name: qwen-35-27-fast
    litellm_params:
      model: ollama/qwen-35-27-fast
      api_base: http://llama-cpp:8080
  
  # Embedding Models
  - model_name: jina-v5-embed
    litellm_params:
      model: ollama/jina-v5-embed
      api_base: http://llama-cpp:8080
  
  # ASR Models
  - model_name: distil-whisper
    litellm_params:
      model: whisper/faster-whisper
      api_base: http://faster-whisper:8080
  
  - model_name: qwen3-asr
    litellm_params:
      model: whisper/qwen-asr
      api_base: http://qwen-asr:8080
  
  # TTS Models
  - model_name: kokoro
    litellm_params:
      model: tts/kokoro
      api_base: http://kokoro-tts:8080
  
  - model_name: indextts2
    litellm_params:
      model: tts/index-tts
      api_base: http://index-tts:8080

# Langfuse Integration
litellm_settings:
  success_callback: ["langfuse"]
  failure_callback: ["langfuse"]

langfuse_public_key: "pk-lf-..."
langfuse_secret_key: "sk-lf-..."
langfuse_host: "http://langfuse.local"
```

## API Reference

### Chat Completions

```bash
# Basic chat
curl http://litellm.local/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-35-27",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Hello!"}
    ]
  }'

# Streaming
curl http://litellm.local/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-35-27",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'

# With parameters
curl http://litellm.local/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-35-27",
    "messages": [{"role": "user", "content": "Be creative"}],
    "temperature": 0.9,
    "max_tokens": 2048,
    "top_p": 0.95
  }'
```

### Embeddings

```bash
# Single text
curl http://litellm.local/v1/embeddings \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-v5-embed",
    "input": "Hello world"
  }'

# Multiple texts
curl http://litellm.local/v1/embeddings \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-v5-embed",
    "input": ["Hello", "World", "Test"]
  }'
```

### Audio Transcription (ASR)

```bash
# Basic transcription
curl http://litellm.local/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-litellm-master" \
  -F model="distil-whisper" \
  -F file="@audio.mp3"

# With options
curl http://litellm.local/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-litellm-master" \
  -F model="qwen3-asr" \
  -F file="@audio.mp3" \
  -F language="en" \
  -F response_format="verbose_json"
```

### Text-to-Speech (TTS)

```bash
# Basic TTS
curl http://litellm.local/v1/audio/speech \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello, this is a test."
  }' \
  --output speech.wav

# With voice parameters
curl http://litellm.local/v1/audio/speech \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello world",
    "voice": "default",
    "speed": 1.0
  }' \
  --output speech.wav
```

### Vision

```bash
# From file
curl http://litellm.local/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-35-27-vision",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }]
  }'

# From URL
curl http://litellm.local/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-35-27-vision",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }]
  }'
```

### List Models

```bash
curl http://litellm.local/v1/models \
  -H "Authorization: Bearer sk-litellm-master"
```

## Python SDK

### OpenAI-Compatible Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://litellm.local/v1",
    api_key="sk-litellm-master"
)

# Chat completion
response = client.chat.completions.create(
    model="qwen-35-27",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen-35-27",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Embeddings
response = client.embeddings.create(
    model="jina-v5-embed",
    input="Hello world"
)
print(response.data[0].embedding)

# TTS
response = client.audio.speech.create(
    model="kokoro",
    input="Hello, this is a test."
)
with open("output.wav", "wb") as f:
    f.write(response.content)

# ASR
with open("audio.mp3", "rb") as f:
    response = client.audio.transcriptions.create(
        model="distil-whisper",
        file=f
    )
print(response.text)
```

### LangChain Integration

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://litellm.local/v1",
    api_key="sk-litellm-master",
    model="qwen-35-27"
)

# Simple invocation
response = llm.invoke("What is the capital of France?")
print(response.content)

# With chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm

response = chain.invoke({"topic": "programming"})
print(response.content)
```

## Available Models

### LLM Models

| Model | Type | Description |
|-------|------|-------------|
| `qwen-35-27` | Chat | Qwen3.5-27B (thinking mode) |
| `qwen-35-27-fast` | Chat | Qwen3.5-27B (non-thinking) |
| `qwen-35-27-vision` | Vision | Qwen3.5-27B with vision |
| `zai-glm-4-5` | Chat | Zhipu AI GLM-4.5 (cloud) |
| `zai-glm-4-6` | Chat | Zhipu AI GLM-4.6 (cloud) |
| `zai-glm-5` | Chat | Zhipu AI GLM-5 (cloud) |

### Embedding Models

| Model | Dimensions | Description |
|-------|-----------|-------------|
| `jina-v5-embed` | 1024 | Jina v5 embeddings |

### ASR Models

| Model | Language | Description |
|-------|----------|-------------|
| `distil-whisper` | Multi | Fast transcription |
| `qwen3-asr` | 52 langs | Accurate transcription |

### TTS Models

| Model | Voice | Description |
|-------|-------|-------------|
| `kokoro` | Multiple | Fast, lightweight |
| `indextts2` | Multiple | High quality |
| `vibevoice` | Multiple | TTS + ASR unified |

## Langfuse Integration

### Automatic Tracing

LiteLLM automatically sends traces to Langfuse when configured:

```yaml
# litellm/config.yaml
litellm_settings:
  success_callback: ["langfuse"]
  failure_callback: ["langfuse"]

langfuse_public_key: "pk-lf-..."
langfuse_secret_key: "sk-lf-..."
langfuse_host: "http://langfuse.local"
```

All API calls are automatically traced with:
- Request/response
- Token usage
- Latency
- Model information
- Session tracking (via metadata)

### Session Tracking

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://litellm.local/v1",
    api_key="sk-litellm-master"
)

response = client.chat.completions.create(
    model="qwen-35-27",
    messages=[{"role": "user", "content": "Hello!"}],
    metadata={
        "session_id": "session-123",
        "user_id": "user-456",
        "tags": ["production", "chatbot"]
    }
)
```

## Monitoring

### Health Check

```bash
curl http://litellm.local/health
```

### Logs

```bash
# View logs
make logs-litellm

# Or using Docker Compose
docker compose logs litellm

# Follow logs
docker compose logs -f litellm
```

### Metrics

```bash
# Token usage (via Langfuse)
open http://langfuse.local

# API metrics (via LiteLLM)
curl http://litellm.local/metrics
```

## Troubleshooting

### Can't Connect

```bash
# Check LiteLLM is running
docker compose ps litellm

# Test health
curl http://litellm.local/health

# Check logs
docker compose logs litellm
```

### Model Not Found

```bash
# List available models
curl http://litellm.local/v1/models \
  -H "Authorization: Bearer sk-litellm-master"

# Check config
cat litellm/config.yaml
```

### Backend Unavailable

```bash
# Check backend health
curl http://llama-cpp:8080/health

# Check Orchestrator status
curl http://orchestrator.local/status

# Load model if needed
curl -X POST http://orchestrator.local/load/asr/whisper
```

## Best Practices

1. **Use environment variables** - Don't hardcode API keys
2. **Enable Langfuse tracing** - For observability
3. **Use streaming** - For better UX with long responses
4. **Implement retry logic** - For transient failures
5. **Monitor token usage** - Via Langfuse dashboard
6. **Use appropriate models** - Match model to task

## Next Steps

- **[API Reference](../api-reference/litellm.md)** - Complete API docs
- **[Integration](../integration/index.md)** - Integrate your apps
- **[Orchestrator](orchestrator.md)** - Resource management
- **[Langfuse](langfuse.md)** - Observability
