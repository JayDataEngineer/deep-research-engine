# Architecture

Deep dive into the Shared Docker Infrastructure system design and architecture.

## Documentation in This Section

- **[Security Model](security.md)** - Traefik-only external access
- **[Network Topology](network.md)** - Docker networking and isolation
- **[Service Architecture](services.md)** - How services interact
- **[Port Configuration](ports.md)** - Understanding port management

## Overview

The infrastructure follows a **security-first** design with centralized access control through Traefik reverse proxy.

### Key Principles

1. **Traefik-Only Exposure**: Only Traefik exposes ports to the host
2. **Internal Networking**: All other services use internal Docker networking
3. **Centralized Management**: LiteLLM for inference, Orchestrator for resources
4. **Configurable**: All ports managed via environment variables

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      EXTERNAL ACCESS                         │
│              (Only through Traefik - configurable ports)     │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────▼───────────────┐
              │           TRAEFIK             │
              │  (Only service exposing ports)│
              │  HTTP:  :8880 (configurable)  │
              │  HTTPS: :8443 (configurable)  │
              │  TCP:   DB ports (configurable)│
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
      ┌───────────────┐               ┌───────────────┐
      │    LiteLLM    │               │  Orchestrator │
      │  (Inference)  │               │   (Control)   │
      │               │               │               │
      │ All model     │               │ VRAM tracking │
      │ requests      │               │ Lifecycle mgmt│
      └───────┬───────┘               │ Status proxy  │
              │                       └───────┬───────┘
              │                               │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │       INTERNAL NETWORK        │
              │   (No External Access)        │
              └───────────────┬───────────────┘
                              │
      ┌───────────┬───────────┼───────────┬───────────┐
      ▼           ▼           ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│llama.cpp │ │ whisper  │ │ kokoro   │ │ index    │ │  other   │
│  (LLM)   │ │  (ASR)   │ │  (TTS)   │ │  (TTS)   │ │ services │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

## Service Layers

### Layer 1: Edge (Traefik)

**Traefik** is the only service exposing ports to the host:
- HTTP/HTTPS entry points (configurable ports)
- TCP routing for databases
- Dashboard for monitoring
- SSL termination (optional)

### Layer 2: Gateway Services

**LiteLLM** - Inference Gateway:
- OpenAI-compatible API
- Routes requests to model containers
- Automatic Langfuse tracing
- Model routing and fallback

**Orchestrator** - Control Plane:
- VRAM management
- Model lifecycle (load/unload)
- Health monitoring
- Status proxy for internal services

### Layer 3: Model Containers (Internal Only)

**LLM Services**:
- llama.cpp for LLM inference
- Internal only, accessed via LiteLLM

**TTS Services**:
- Kokoro, IndexTTS2, VibeVoice, etc.
- Internal only, accessed via LiteLLM or Orchestrator

**ASR Services**:
- Faster-Whisper, Qwen-ASR
- Internal only, accessed via LiteLLM or Orchestrator

### Layer 4: Databases (TCP via Traefik)

- **PostgreSQL** - Relational data
- **MongoDB** - Document storage
- **Neo4j** - Knowledge graphs
- **MinIO** - Object storage (S3-compatible)

### Layer 5: Applications

- **Langfuse** - Observability
- **ComfyUI** - Image generation
- **Mongo Express** - Database UI
- **Neo4j Browser** - Graph UI

## Access Patterns

### Inference Requests

```
Client → Traefik → LiteLLM → Model Container
                    ↓
                Langfuse (auto-tracing)
```

### Management Requests

```
Client → Traefik → Orchestrator → Model Container
                          ↓
                    VRAM check, lifecycle
```

### Database Connections

```
Client → Traefik (TCP) → Database
         (localhost:port)
```

## Security Model

### Network Isolation

```yaml
# All services use internal network
networks:
  shared-infra:
    external: true
    name: shared-infra

# Only Traefik exposes ports
services:
  traefik:
    ports:
      - "${TRAEFIK_HTTP_PORT}:80"
  
  litellm:
    expose:
      - "8080"
    # No ports: section!
```

### Access Control

| Service | External Access | Internal Access |
|---------|----------------|-----------------|
| Traefik | ✅ Yes (all ports) | ✅ Yes |
| LiteLLM | ❌ No | ✅ Via Traefik |
| Orchestrator | ❌ No | ✅ Via Traefik |
| Model Containers | ❌ No | ✅ Via LiteLLM/Orchestrator |
| Databases | ❌ No | ✅ Via Traefik TCP |

### Benefits

- ✅ **Centralized security** - Single entry point
- ✅ **Network isolation** - Internal services protected
- ✅ **Simplified firewall** - Only Traefik ports needed
- ✅ **Audit trail** - All access through Traefik logs

## Data Flow

### Chat Completion Flow

1. Client sends request to `litellm.local/v1/chat/completions`
2. Traefik routes to LiteLLM container
3. LiteLLM routes to appropriate model (e.g., llama.cpp)
4. Model generates response
5. LiteLLM sends trace to Langfuse
6. Response returned to client

### Model Loading Flow

1. Client sends POST to `orchestrator.local/load/asr/whisper`
2. Traefik routes to Orchestrator
3. Orchestrator checks VRAM availability
4. If VRAM sufficient, loads model
5. If VRAM insufficient, returns error or uses CPU fallback
6. Status returned to client

## Scalability

### Horizontal Scaling

- Multiple LiteLLM instances behind Traefik load balancer
- Multiple model containers for same model type
- Database replication (PostgreSQL, MongoDB)

### Vertical Scaling

- GPU resource allocation via Docker
- VRAM management via Orchestrator
- CPU limits via Traefik configuration

## Monitoring

### Health Checks

- Traefik health checks for all services
- LiteLLM health endpoint
- Orchestrator status endpoint
- Database health checks

### Observability

- **Traefik logs** - Access logs, error logs
- **Langfuse** - LLM traces, metrics
- **Docker stats** - Resource usage
- **Orchestrator** - VRAM monitoring

## Next Steps

- **[Security Model](security.md)** - Detailed security guide
- **[Network Topology](network.md)** - Network configuration
- **[Service Architecture](services.md)** - Service details
- **[Port Configuration](ports.md)** - Port management
