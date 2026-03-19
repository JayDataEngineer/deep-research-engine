# Model Orchestrator

The Model Orchestrator manages VRAM allocation and model lifecycle for TTS/ASR services.

## Overview

- **Purpose:** VRAM management and model lifecycle
- **Domain:** `orchestrator.local`
- **Internal URL:** `http://orchestrator:8080`
- **API:** `http://orchestrator.local`
- **Auth:** None required (internal service)

## Key Features

- ✅ **VRAM tracking** - Real-time GPU memory monitoring
- ✅ **Model lifecycle** - Load/unload models on demand
- ✅ **CPU fallback** - Automatic fallback to CPU models
- ✅ **Status proxy** - Health checks for internal services
- ✅ **LLM proxy** - Status for llama.cpp models

## Quick Start

```bash
# Start Orchestrator
make up-orchestrator

# Or using Docker Compose
docker compose -f docker-compose.orchestrator.yml up -d

# Check status
curl http://orchestrator.local/status

# Check VRAM
curl http://orchestrator.local/vram

# Load TTS model
curl -X POST http://orchestrator.local/load/tts/kokoro

# Load ASR model
curl -X POST http://orchestrator.local/load/asr/whisper
```

## API Reference

### Status & Monitoring

#### Full Status

```bash
curl http://orchestrator.local/status
```

**Response:**
```json
{
  "vram": {
    "free_mb": 4000,
    "used_mb": 20000,
    "total_mb": 24000
  },
  "slots": {
    "asr": {
      "current_model": {
        "key": "whisper",
        "vram_mb": 1500,
        "loaded": true
      },
      "loading": false
    },
    "tts": {
      "current_model": {
        "key": "kokoro",
        "vram_mb": 1000,
        "loaded": true
      },
      "loading": false
    }
  },
  "available_models": {
    "asr": {
      "whisper": {"vram_mb": 1500},
      "whisper-cpu": {"vram_mb": 0},
      "qwen": {"vram_mb": 4000}
    },
    "tts": {
      "kokoro": {"vram_mb": 1000},
      "kokoro-cpu": {"vram_mb": 0},
      "index": {"vram_mb": 2000}
    }
  }
}
```

#### VRAM Usage

```bash
curl http://orchestrator.local/vram
```

**Response:**
```json
{
  "free_mb": 4000,
  "used_mb": 20000,
  "total_mb": 24000,
  "percent_used": 83.3
}
```

### Model Lifecycle

#### Load Model

```bash
# Load TTS model
curl -X POST http://orchestrator.local/load/tts/kokoro

# Load ASR model
curl -X POST http://orchestrator.local/load/asr/whisper

# Load specific model
curl -X POST http://orchestrator.local/load/asr/qwen
```

**Success Response:**
```json
{
  "status": "success",
  "message": "Loaded kokoro TTS model"
}
```

**Error Response (Insufficient VRAM):**
```json
{
  "detail": "Insufficient VRAM. Required: 2000MB, Available: 1000MB"
}
```

#### Unload Model

```bash
# Unload TTS
curl -X POST http://orchestrator.local/unload/tts

# Unload ASR
curl -X POST http://orchestrator.local/unload/asr

# Unload all
curl -X POST http://orchestrator.local/unload/all
```

**Response:**
```json
{
  "status": "success",
  "message": "Unloaded TTS models"
}
```

### LLM Status Proxy

#### LLM Models

```bash
curl http://orchestrator.local/llm/models
```

**Response:**
```json
{
  "data": [
    {
      "id": "qwen-35-27",
      "status": {"value": "loaded"}
    },
    {
      "id": "qwen-35-27-fast",
      "status": {"value": "available"}
    }
  ]
}
```

#### LLM Health

```bash
curl http://orchestrator.local/llm/health
```

**Response:**
```json
{
  "status": "ok",
  "message": "llama.cpp is healthy"
}
```

#### LLM Properties

```bash
curl http://orchestrator.local/llm/props
```

**Response:**
```json
{
  "version": "1.0.0",
  "models_loaded": 2,
  "context_length": 32768
}
```

## Available Models

### ASR Slot

| Model | VRAM | Description |
|-------|------|-------------|
| `whisper` | 1.5 GB | GPU, fast |
| `whisper-cpu` | 0 GB | CPU, always available |
| `qwen` | 4 GB | GPU, most accurate (52 languages) |

### TTS Slot

| Model | VRAM | Description |
|-------|------|-------------|
| `kokoro` | 1 GB | GPU, fast, lightweight |
| `kokoro-cpu` | 0 GB | CPU, always available |
| `index` | 2 GB | GPU, highest quality |
| `vibevoice` | 2 GB | GPU, TTS + ASR unified |
| `gpt-sovits` | 2 GB | GPU, voice cloning |
| `maya1` | 1 GB | GPU, natural language |
| `qwen3-tts` | 2 GB | GPU, multi-speaker |

## Usage Patterns

### VRAM-Aware Loading

```python
import httpx

ORCHESTRATOR_URL = "http://orchestrator.local"

def load_model_safe(slot: str, model: str) -> bool:
    """Load model only if enough VRAM available."""
    # Check current status
    status = httpx.get(f"{ORCHESTRATOR_URL}/status").json()
    
    # Check if already loaded
    current = status["slots"].get(slot, {}).get("current_model")
    if current and current.get("key") == model:
        print(f"{model} already loaded")
        return True
    
    # Check VRAM
    vram = status["vram"]
    model_vram = status["available_models"][slot][model]["vram_mb"]
    
    if vram["free_mb"] < model_vram:
        print(f"Insufficient VRAM: need {model_vram}MB, have {vram['free_mb']}MB")
        # Try CPU fallback
        cpu_model = model + "-cpu" if "-cpu" not in model else model
        if cpu_model in status["available_models"][slot]:
            print(f"Falling back to CPU model: {cpu_model}")
            resp = httpx.post(f"{ORCHESTRATOR_URL}/load/{slot}/{cpu_model}")
            return resp.status_code == 200
        return False
    
    # Load the model
    resp = httpx.post(f"{ORCHESTRATOR_URL}/load/{slot}/{model}")
    return resp.status_code == 200

# Usage
load_model_safe("tts", "kokoro")
load_model_safe("asr", "whisper")
```

### Model Switching

```python
def switch_tts_model(new_model: str):
    """Switch to a different TTS model."""
    # Unload current
    httpx.post(f"{ORCHESTRATOR_URL}/unload/tts")
    
    # Load new
    resp = httpx.post(f"{ORCHESTRATOR_URL}/load/tts/{new_model}")
    
    if resp.status_code == 200:
        print(f"Switched to {new_model}")
    else:
        print(f"Failed to load {new_model}: {resp.text}")

# Usage
switch_tts_model("index")  # High quality
switch_tts_model("kokoro")  # Fast
```

### Pipeline with Model Management

```python
import httpx
from openai import OpenAI

ORCHESTRATOR_URL = "http://orchestrator.local"
LITELLM_URL = "http://litellm.local/v1"
API_KEY = "sk-litellm-master"

client = OpenAI(base_url=LITELLM_URL, api_key=API_KEY)

def transcribe_with_model(audio_file: str, model: str = "whisper") -> str:
    """Transcribe audio with automatic model loading."""
    # Load model
    resp = httpx.post(f"{ORCHESTRATOR_URL}/load/asr/{model}")
    if resp.status_code != 200:
        raise Exception(f"Failed to load model: {resp.text}")
    
    try:
        # Transcribe
        with open(audio_file, "rb") as f:
            result = client.audio.transcriptions.create(
                model=model,
                file=f
            )
        return result.text
    finally:
        # Unload model (optional, keep loaded for repeated use)
        # httpx.post(f"{ORCHESTRATOR_URL}/unload/asr")
        pass

def tts_with_model(text: str, model: str = "kokoro") -> bytes:
    """Generate speech with automatic model loading."""
    # Load model
    resp = httpx.post(f"{ORCHESTRATOR_URL}/load/tts/{model}")
    if resp.status_code != 200:
        raise Exception(f"Failed to load model: {resp.text}")
    
    try:
        # Generate speech
        response = client.audio.speech.create(
            model=model,
            input=text
        )
        return response.content
    finally:
        # Unload model (optional)
        # httpx.post(f"{ORCHESTRATOR_URL}/unload/tts")
        pass

# Usage
transcript = transcribe_with_model("meeting.mp3")
audio = tts_with_model("Hello, this is a test.")
```

## Monitoring

### Health Check

```bash
curl http://orchestrator.local/health
```

### Logs

```bash
# View logs
make logs-orchestrator

# Or using Docker Compose
docker compose logs orchestrator

# Follow logs
docker compose logs -f orchestrator
```

### Real-time VRAM Monitoring

```bash
# Watch VRAM usage
watch -n 1 'curl -s http://orchestrator.local/vram | jq'

# Or using jq for formatted output
curl -s http://orchestrator.local/vram | jq '.percent_used'
```

## Configuration

### Environment Variables

```bash
# VRAM threshold (MB)
ORCHESTRATOR_VRAM_THRESHOLD=2000

# Auto-unload inactive models
ORCHESTRATOR_AUTO_UNLOAD=true
ORCHESTRATOR_INACTIVITY_TIMEOUT=300  # seconds

# Default models
ASR_DEFAULT_MODEL=whisper
TTS_DEFAULT_MODEL=kokoro

# CPU fallback
ASR_CPU_FALLBACK=whisper-cpu
TTS_CPU_FALLBACK=kokoro-cpu
```

### Docker Compose

```yaml
services:
  orchestrator:
    build: ./model-orchestrator
    expose:
      - "8080"
    networks:
      - shared-infra
    environment:
      - ORCHESTRATOR_VRAM_THRESHOLD=2000
      - ORCHESTRATOR_AUTO_UNLOAD=true
    labels:
      - "traefik.http.routers.orchestrator.rule=Host(`orchestrator.local`)"
      - "traefik.http.services.orchestrator.loadbalancer.server.port=8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
```

## Troubleshooting

### Can't Load Model

```bash
# Check VRAM
curl http://orchestrator.local/vram

# Check available models
curl http://orchestrator.local/status | jq '.available_models'

# Check logs
docker compose logs orchestrator
```

### VRAM Issues

```bash
# Check what's loaded
curl http://orchestrator.local/status | jq '.slots'

# Unload all models
curl -X POST http://orchestrator.local/unload/all

# Check GPU usage
nvidia-smi
```

### Model Not Responding

```bash
# Check model health via proxy
curl http://orchestrator.local/llm/health

# Restart model container
docker compose restart kokoro-tts

# Reload model
curl -X POST http://orchestrator.local/unload/tts
curl -X POST http://orchestrator.local/load/tts/kokoro
```

## Best Practices

1. **Check VRAM before loading** - Use `/vram` endpoint
2. **Use CPU fallback** - For low-priority tasks
3. **Keep models loaded** - For repeated use (avoid reload overhead)
4. **Unload when done** - Free VRAM for other tasks
5. **Monitor VRAM** - Use `/status` for full picture
6. **Handle errors gracefully** - Implement retry logic

## Next Steps

- **[API Reference](../api-reference/orchestrator.md)** - Complete API docs
- **[LiteLLM](litellm.md)** - Inference gateway
- **[TTS Services](tts/index.md)** - Text-to-Speech models
- **[ASR Services](asr/index.md)** - Speech-to-Text models
