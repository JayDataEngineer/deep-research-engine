# Quick Start Guide

Get your Shared Docker Infrastructure up and running in 5 minutes!

## Prerequisites

Ensure you've completed the [Installation](installation.md) steps:
- [x] Docker and Docker Compose installed
- [x] Repository cloned
- [x] Environment files created (`.env` and `traefik/.env`)
- [x] Secrets generated
- [x] Hostnames configured in `/etc/hosts`
- [x] Docker network created (`shared-infra`)

## Quick Start (5 Minutes)

### Option 1: Using Makefile (Simplest)

```bash
cd ~/shared-docker-infra

# Start core services (Traefik, LiteLLM, Orchestrator, Llama)
make up-core

# Or start everything
make up

# Check status
make status
```

### Option 2: Using Python CLI (Recommended)

```bash
cd ~/shared-docker-infra

# Start all core services
infra core up

# Check status
infra status

# View logs
infra logs
```

### Option 3: Using Docker Compose Directly

```bash
cd ~/shared-docker-infra

# Start Traefik (required first)
docker compose -f docker-compose.traefik.yml up -d

# Start LiteLLM
docker compose -f docker-compose.litellm.yml up -d

# Start Orchestrator
docker compose -f docker-compose.orchestrator.yml up -d

# Start Llama (LLM backend)
docker compose -f llama-cpp/docker-compose.yml up -d
```

## Verify Installation

### 1. Check Services Are Running

```bash
# Using Docker Compose
docker compose ps

# Or using CLI
infra status
```

Expected output:
```
NAME                    STATUS          PORTS
traefik                 Up (healthy)    0.0.0.0:8880->80/tcp
litellm                 Up (healthy)    
orchestrator            Up (healthy)    
llama-cpp               Up (healthy)    
```

### 2. Test LiteLLM (Inference Gateway)

```bash
# Check LiteLLM health
curl http://litellm.local/v1/models \
  -H "Authorization: Bearer sk-litellm-master"

# Test chat completion
curl http://litellm.local/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-35-27",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 3. Test Orchestrator (Control Plane)

```bash
# Check orchestrator status
curl http://orchestrator.local/status

# Check VRAM usage
curl http://orchestrator.local/vram
```

### 4. Access Web UIs

Open in your browser:

| Service | URL |
|---------|-----|
| Traefik Dashboard | http://traefik.local:8081 |
| Langfuse | http://langfuse.local |
| Neo4j Browser | http://neo4j.local |
| MinIO Console | http://minio-console.local |
| Mongo Express | http://mongo-express.local |
| ComfyUI | http://comfyui.local |

## Your First AI Requests

### Chat with LLM

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://litellm.local/v1",
    api_key="sk-litellm-master"
)

response = client.chat.completions.create(
    model="qwen-35-27",
    messages=[{"role": "user", "content": "What is machine learning?"}]
)

print(response.choices[0].message.content)
```

### Generate Embeddings

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://litellm.local/v1",
    api_key="sk-litellm-master"
)

response = client.embeddings.create(
    model="jina-v5-embed",
    input="Hello, world!"
)

print(response.data[0].embedding)
```

### Text-to-Speech

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://litellm.local/v1",
    api_key="sk-litellm-master"
)

response = client.audio.speech.create(
    model="kokoro",
    input="Hello, this is a test of text to speech."
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Speech-to-Text

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://litellm.local/v1",
    api_key="sk-litellm-master"
)

with open("audio.mp3", "rb") as f:
    response = client.audio.transcriptions.create(
        model="distil-whisper",
        file=f
    )

print(response.text)
```

## Common Tasks

### Start Specific Services

```bash
# Using Makefile
make up-traefik
make up-litellm
make up-orchestrator
make up-llama
make up-langfuse
make up-postgres
make up-mongodb
make up-neo4j

# Using CLI
infra db up          # Start all databases
infra core up        # Start core services
```

### View Logs

```bash
# All services
make logs

# Specific service
make logs-litellm
make logs-orchestrator

# Using CLI
infra logs
infra logs litellm
infra logs -f litellm  # Follow logs
```

### Stop Services

```bash
# Stop all
make down

# Stop specific service
make down-traefik

# Using CLI
infra down
infra core down
```

### Check Resource Usage

```bash
# VRAM usage
curl http://orchestrator.local/vram

# Full status
curl http://orchestrator.local/status

# Docker stats
docker stats
```

## Load Models via Orchestrator

```bash
# Load TTS model
curl -X POST http://orchestrator.local/load/tts/kokoro

# Load ASR model
curl -X POST http://orchestrator.local/load/asr/whisper

# Check loaded models
curl http://orchestrator.local/status
```

## Next Steps

1. **[Configuration](configuration.md)** - Customize ports and settings
2. **[Architecture](../architecture/index.md)** - Understand the system
3. **[API Reference](../api-reference/index.md)** - Learn the APIs
4. **[Integration](../integration/index.md)** - Integrate your apps

## Troubleshooting Quick Fixes

### Can't Access Services?

```bash
# Verify hosts file
grep "local" /etc/hosts

# Restart Traefik
make down-traefik && make up-traefik

# Check Traefik logs
make logs-traefik
```

### Services Not Starting?

```bash
# Check Docker logs
docker compose ps
docker compose logs <service-name>

# Check resource usage
docker stats
```

### Port Conflicts?

```bash
# Find what's using a port
sudo lsof -i :8880

# Change port in .env
TRAEFIK_HTTP_PORT=8881

# Restart
make down-traefik && make up-traefik
```

## Getting Help

- **[FAQ](../troubleshooting/faq.md)** - Common questions
- **[Troubleshooting](../troubleshooting/index.md)** - Detailed guides
- **[Architecture](../architecture/index.md)** - System design

---

**Tip**: Bookmark the [Quick Reference](../index.md#quick-reference) for common commands and endpoints!
