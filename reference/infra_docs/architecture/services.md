# Service Architecture

This document describes how services in the Shared Docker Infrastructure interact with each other.

## Service Categories

Services are organized into five categories:

1. **Edge Services** - Traefik (external access)
2. **Gateway Services** - LiteLLM, Orchestrator
3. **Model Services** - LLM, TTS, ASR (internal only)
4. **Data Services** - Databases, Storage
5. **Application Services** - UIs, Monitoring

## Service Directory

### Edge Services

#### Traefik
- **Purpose:** Reverse proxy and load balancer
- **Port:** All external ports
- **Access:** Only service exposing ports to host
- **Documentation:** [../services/traefik.md](../services/traefik.md)

### Gateway Services

#### LiteLLM
- **Purpose:** Inference gateway (OpenAI-compatible API)
- **Domain:** `litellm.local`
- **Internal URL:** `http://litellm:8080`
- **Access:** Via Traefik HTTP routing
- **Routes requests to:** LLM, TTS, ASR, Embedding models
- **Documentation:** [../services/litellm.md](../services/litellm.md)

#### Model Orchestrator
- **Purpose:** VRAM management and model lifecycle
- **Domain:** `orchestrator.local`
- **Internal URL:** `http://orchestrator:8080`
- **Access:** Via Traefik HTTP routing
- **Manages:** TTS, ASR model loading/unloading
- **Documentation:** [../services/orchestrator.md](../services/orchestrator.md)

### Model Services (Internal Only)

#### LLM Services

**llama.cpp**
- **Purpose:** LLM inference
- **Internal URL:** `http://llama-cpp:8080`
- **Access:** Via LiteLLM only
- **Models:** Qwen3.5-27B, etc.
- **Documentation:** [../services/llm/index.md](../services/llm/index.md)

#### TTS Services

| Service | Internal URL | Access | VRAM |
|---------|-------------|--------|------|
| Kokoro | `http://kokoro-tts:8080` | LiteLLM, Orchestrator | ~1GB |
| IndexTTS2 | `http://index-tts:8080` | LiteLLM, Orchestrator | ~2GB |
| VibeVoice | `http://vibevoice:8080` | LiteLLM, Orchestrator | ~2GB |
| GPT-SoVITS | `http://gpt-sovits:8080` | LiteLLM, Orchestrator | ~2GB |
| Maya1 | `http://maya1-tts:8080` | LiteLLM, Orchestrator | ~1GB |
| Qwen3-TTS | `http://qwen3-tts:8080` | LiteLLM, Orchestrator | ~2GB |
| eSpeak-NG | `http://espeak-ng:8080` | LiteLLM | CPU |

**Documentation:** [../services/tts/index.md](../services/tts/index.md)

#### ASR Services

| Service | Internal URL | Access | VRAM |
|---------|-------------|--------|------|
| Faster-Whisper | `http://faster-whisper:8080` | LiteLLM, Orchestrator | ~1.5GB |
| Qwen-ASR | `http://qwen-asr:8080` | LiteLLM, Orchestrator | ~4GB |
| Whisper-CPU | `http://whisper-cpu:8080` | LiteLLM | CPU |

**Documentation:** [../services/asr/index.md](../services/asr/index.md)

### Data Services

#### PostgreSQL
- **Purpose:** Relational database
- **Internal Port:** 5432
- **External Port:** `${TRAEFIK_POSTGRES_PORT}` (5433)
- **Access:** Via Traefik TCP routing
- **Documentation:** [../services/postgresql.md](../services/postgresql.md)

#### MongoDB
- **Purpose:** Document database
- **Internal Port:** 27017
- **External Port:** `${TRAEFIK_MONGODB_PORT}` (27018)
- **Access:** Via Traefik TCP routing
- **Documentation:** [../services/mongodb.md](../services/mongodb.md)

#### Neo4j
- **Purpose:** Knowledge graph database
- **Internal Ports:** 7474 (HTTP), 7687 (Bolt)
- **External Ports:** `${TRAEFIK_HTTP_PORT}` (HTTP), `${TRAEFIK_NEO4J_PORT}` (Bolt)
- **Access:** HTTP via Traefik, Bolt via TCP routing
- **Documentation:** [../services/neo4j.md](../services/neo4j.md)

#### MinIO
- **Purpose:** S3-compatible object storage
- **Internal Ports:** 9000 (API), 9001 (Console)
- **External:** Console via Traefik HTTP
- **Access:** API via SDK, Console via browser
- **Documentation:** [../services/minio.md](../services/minio.md)

### Application Services

#### Langfuse
- **Purpose:** LLM observability and tracing
- **Domain:** `langfuse.local`
- **Internal URL:** `http://langfuse:3000`
- **Access:** Via Traefik HTTP routing
- **Documentation:** [../services/langfuse.md](../services/langfuse.md)

#### ComfyUI
- **Purpose:** Image/video generation
- **Domain:** `comfyui.local`
- **Internal URL:** `http://comfyui:8188`
- **Access:** Via Traefik HTTP routing
- **Documentation:** [../services/comfyui.md](../services/comfyui.md)

#### Mongo Express
- **Purpose:** MongoDB web UI
- **Domain:** `mongo-express.local`
- **Internal URL:** `http://mongo-express:8081`
- **Access:** Via Traefik HTTP routing
- **Documentation:** [../services/mongo-express.md](../services/mongo-express.md)

## Service Interactions

### Inference Request Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │────▶│ Traefik  │────▶│ LiteLLM  │────▶│llama.cpp │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                                         │
                                         ▼
                                   ┌──────────┐
                                   │ Langfuse │
                                   │ (trace)  │
                                   └──────────┘
```

**Steps:**
1. Client sends request to `litellm.local/v1/chat/completions`
2. Traefik routes to LiteLLM container
3. LiteLLM routes to llama.cpp based on model name
4. llama.cpp generates response
5. LiteLLM sends trace to Langfuse
6. Response returned to client

### Model Loading Flow

```
┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐
│  Client  │────▶│ Traefik  │────▶│ Orchestrator │────▶│ kokoro   │
└──────────┘     └──────────┘     └──────────────┘     └──────────┘
                                         │
                                         ▼
                                   ┌──────────┐
                                   │ VRAM     │
                                   │ Check    │
                                   └──────────┘
```

**Steps:**
1. Client sends POST to `orchestrator.local/load/tts/kokoro`
2. Traefik routes to Orchestrator
3. Orchestrator checks VRAM availability
4. If VRAM sufficient, loads model into kokoro container
5. Status returned to client

### Database Connection Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │────▶│ Traefik  │────▶│ Postgres │
└──────────┘     └──────────┘     └──────────┘
   TCP              TCP               TCP
localhost:5433    Routing          :5432
```

**Steps:**
1. Client connects to `localhost:5433`
2. Traefik TCP router forwards to PostgreSQL
3. PostgreSQL responds through Traefik
4. Client receives response

## Service Dependencies

### Core Dependencies

```
Traefik (required first)
    ├── LiteLLM
    │   └── llama-cpp (for LLM)
    │   └── TTS services (for audio)
    │   └── ASR services (for transcription)
    ├── Orchestrator
    │   └── TTS services (for management)
    │   └── ASR services (for management)
    ├── Langfuse (optional)
    ├── Databases (optional)
    └── Applications (optional)
```

### Startup Order

```bash
# Required order
1. Traefik (always first)
2. LiteLLM + Orchestrator (can start together)
3. Model services (llama-cpp, TTS, ASR)
4. Databases (if needed)
5. Applications (if needed)
```

## Service Communication Patterns

### HTTP Communication

Services communicate via HTTP within the `shared-infra` network:

```python
# LiteLLM to llama.cpp
response = requests.post(
    "http://llama-cpp:8080/completion",
    json={"prompt": "Hello"}
)

# Orchestrator to TTS
response = requests.post(
    "http://kokoro-tts:8080/load",
    json={"model": "kokoro-v1"}
)
```

### Database Communication

Applications connect to databases via Traefik TCP routing:

```python
# PostgreSQL connection
conn = psycopg2.connect(
    host="localhost",
    port=5433,  # Traefik port
    database="mydb",
    user="postgres",
    password="secret"
)

# MongoDB connection
client = MongoClient(
    "mongodb://localhost:27018/"  # Traefik port
)
```

## Service Health Checks

### Traefik Health Checks

Traefik performs health checks on all labeled services:

```yaml
labels:
  - "traefik.http.services.myservice.loadbalancer.server.scheme=http"
  - "traefik.http.services.myservice.loadbalancer.passHostHeader=true"
```

### Application Health Endpoints

| Service | Health Endpoint |
|---------|----------------|
| LiteLLM | `/health` |
| Orchestrator | `/health` |
| llama.cpp | `/health` |
| Langfuse | `/api/health` |
| Traefik | `/ping` |

### Check Service Health

```bash
# Via Traefik dashboard
curl http://traefik.local:8081/api/http/routers

# Direct health checks
curl http://litellm.local/health
curl http://orchestrator.local/health
```

## Service Resource Allocation

### GPU Resources

```yaml
# llama.cpp (LLM - high VRAM)
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']
          capabilities: [gpu]

# Kokoro TTS (medium VRAM)
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']
          capabilities: [gpu]
```

### CPU Resources

```yaml
# Traefik (low CPU)
deploy:
  resources:
    limits:
      cpus: '0.5'
      memory: 256M

# llama.cpp (high CPU for preprocessing)
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 16G
```

## Service Discovery

### DNS Resolution

Services discover each other via Docker DNS:

```bash
# From LiteLLM container
curl http://llama-cpp:8080/health

# From Orchestrator container
curl http://kokoro-tts:8080/status
```

### Service Registration

Services register with Traefik via labels:

```yaml
labels:
  - "traefik.http.routers.myservice.rule=Host(`myservice.local`)"
  - "traefik.http.services.myservice.loadbalancer.server.port=8080"
```

## Next Steps

- **[Traefik](../services/traefik.md)** - Reverse proxy details
- **[LiteLLM](../services/litellm.md)** - Inference gateway
- **[Orchestrator](../services/orchestrator.md)** - Resource management
- **[LLM Services](../services/llm/index.md)** - LLM backends
- **[TTS Services](../services/tts/index.md)** - Text-to-Speech
- **[ASR Services](../services/asr/index.md)** - Speech-to-Text
