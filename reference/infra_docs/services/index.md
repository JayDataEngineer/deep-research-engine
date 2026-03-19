# Services Documentation

Complete documentation for all services in the Shared Docker Infrastructure.

## Service Categories

### Core Services

| Service | Purpose | Domain | Documentation |
|---------|---------|--------|---------------|
| [Traefik](traefik.md) | Reverse proxy | `traefik.local` | [Guide](traefik.md) |
| [LiteLLM](litellm.md) | Inference gateway | `litellm.local` | [Guide](litellm.md) |
| [Orchestrator](orchestrator.md) | Resource management | `orchestrator.local` | [Guide](orchestrator.md) |
| [Langfuse](langfuse.md) | Observability | `langfuse.local` | [Guide](langfuse.md) |

### Model Services (Internal Only)

#### LLM Services

| Service | Purpose | Access | Documentation |
|---------|---------|--------|---------------|
| [llama.cpp](llm/index.md) | LLM inference | Via LiteLLM | [Guide](llm/index.md) |

#### TTS Services

| Service | Purpose | VRAM | Access | Documentation |
|---------|---------|------|--------|---------------|
| [Kokoro](tts/kokoro.md) | Lightweight TTS (82M) | ~1GB | LiteLLM, Orchestrator | [Guide](tts/kokoro.md) |
| [IndexTTS2](tts/index-tts.md) | High quality TTS | ~2GB | LiteLLM, Orchestrator | [Guide](tts/index-tts.md) |
| [VibeVoice](tts/vibevoice.md) | TTS + ASR unified | ~2GB | LiteLLM, Orchestrator | [Guide](tts/vibevoice.md) |
| [GPT-SoVITS](tts/gpt-sovits.md) | Voice cloning | ~2GB | LiteLLM, Orchestrator | [Guide](tts/gpt-sovits.md) |
| [Maya1](tts/maya1.md) | Natural language TTS | ~1GB | LiteLLM, Orchestrator | [Guide](tts/maya1.md) |
| [Qwen3-TTS](tts/qwen3-tts.md) | Multi-speaker TTS | ~2GB | LiteLLM, Orchestrator | [Guide](tts/qwen3-tts.md) |
| [eSpeak-NG](tts/espeak-ng.md) | Lightweight CPU TTS | CPU | LiteLLM | [Guide](tts/espeak-ng.md) |

#### ASR Services

| Service | Purpose | VRAM | Access | Documentation |
|---------|---------|------|--------|---------------|
| [Faster-Whisper](asr/faster-whisper.md) | Fast ASR | ~1.5GB | LiteLLM, Orchestrator | [Guide](asr/faster-whisper.md) |
| [Qwen-ASR](asr/qwen-asr.md) | Accurate ASR (52 langs) | ~4GB | LiteLLM, Orchestrator | [Guide](asr/qwen-asr.md) |

### Data Services

| Service | Purpose | Port | Access | Documentation |
|---------|---------|------|--------|---------------|
| [PostgreSQL](postgresql.md) | Relational database | 5433 | Traefik TCP | [Guide](postgresql.md) |
| [MongoDB](mongodb.md) | Document database | 27018 | Traefik TCP | [Guide](mongodb.md) |
| [Neo4j](neo4j.md) | Knowledge graph | 7688 | Traefik TCP | [Guide](neo4j.md) |
| [MinIO](minio.md) | Object storage (S3) | 9001 | Traefik HTTP | [Guide](minio.md) |

### Application Services

| Service | Purpose | Domain | Documentation |
|---------|---------|--------|---------------|
| [ComfyUI](comfyui.md) | Image generation | `comfyui.local` | [Guide](comfyui.md) |
| [Mongo Express](mongo-express.md) | MongoDB UI | `mongo-express.local` | [Guide](mongo-express.md) |

## Service Management

### Start Services

```bash
# Core services
infra core up

# Databases
infra db up

# Individual services
infra up traefik
infra up litellm
infra up orchestrator
```

### Check Status

```bash
# All services
infra status

# Specific category
infra status core
infra status db

# JSON output
infra status --json
```

### View Logs

```bash
# All logs
infra logs

# Specific service
infra logs litellm
infra logs llama-cpp

# Follow logs
infra logs -f litellm
```

### Stop Services

```bash
# All services
infra down

# Specific category
infra core down
infra db down
```

## Service Access Patterns

### Inference (LiteLLM)

```bash
# Chat completion
curl http://litellm.local/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-35-27", "messages": [{"role": "user", "content": "Hello"}]}'

# Embeddings
curl http://litellm.local/v1/embeddings \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{"model": "jina-v5-embed", "input": "Hello world"}'

# TTS
curl http://litellm.local/v1/audio/speech \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello world"}' \
  --output speech.wav

# ASR
curl http://litellm.local/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-litellm-master" \
  -F model="distil-whisper" \
  -F file="@audio.mp3"
```

### Management (Orchestrator)

```bash
# Status
curl http://orchestrator.local/status

# VRAM
curl http://orchestrator.local/vram

# Load model
curl -X POST http://orchestrator.local/load/tts/kokoro

# Unload model
curl -X POST http://orchestrator.local/unload/tts
```

### Databases

```bash
# PostgreSQL
psql -h localhost -p 5433 -U postgres

# MongoDB
mongosh mongodb://localhost:27018

# Neo4j
# Browser: http://neo4j.local
# Bolt: neo4j://localhost:7688
```

## Service Dependencies

```
Traefik (required first)
    ├── LiteLLM
    │   └── llama-cpp
    │   └── TTS services
    │   └── ASR services
    ├── Orchestrator
    │   └── TTS services
    │   └── ASR services
    ├── Langfuse (optional)
    ├── Databases (optional)
    └── Applications (optional)
```

## Health Checks

```bash
# Traefik
curl http://traefik.local:8081/ping

# LiteLLM
curl http://litellm.local/health

# Orchestrator
curl http://orchestrator.local/health

# llama.cpp
curl http://orchestrator.local/llm/health

# Langfuse
curl http://langfuse.local/api/health
```

## Resource Allocation

### GPU Services

| Service | VRAM | GPU Required |
|---------|------|--------------|
| llama.cpp | 8-16GB | Yes (for LLM) |
| Kokoro | ~1GB | Yes (recommended) |
| IndexTTS2 | ~2GB | Yes |
| Faster-Whisper | ~1.5GB | Yes (recommended) |
| Qwen-ASR | ~4GB | Yes |

### CPU Services

| Service | CPU | Memory |
|---------|-----|--------|
| Traefik | 0.5 core | 256MB |
| LiteLLM | 1 core | 512MB |
| Orchestrator | 0.5 core | 256MB |
| eSpeak-NG | 0.5 core | 256MB |
| Whisper-CPU | 2 cores | 2GB |

## Next Steps

- **[Traefik](traefik.md)** - Reverse proxy configuration
- **[LiteLLM](litellm.md)** - Inference gateway
- **[Orchestrator](orchestrator.md)** - Resource management
- **[LLM Services](llm/index.md)** - LLM backends
- **[TTS Services](tts/index.md)** - Text-to-Speech
- **[ASR Services](asr/index.md)** - Speech-to-Text
- **[Databases](postgresql.md)** - Data services
