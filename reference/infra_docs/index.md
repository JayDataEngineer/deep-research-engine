# Shared Docker Infrastructure Documentation

Welcome to the comprehensive documentation for the shared AI infrastructure. This documentation covers all aspects of the unified AI platform with internal-only model containers, exposed through LiteLLM (inference) and Orchestrator (control plane).

---

## 🚀 Quick Start

New to the infrastructure? Start here:

1. **[Installation & Setup](getting-started/installation.md)** - Get up and running
2. **[Quick Start Guide](getting-started/quickstart.md)** - Start your first services
3. **[Configuration](getting-started/configuration.md)** - Configure ports and secrets

---

## 📚 Documentation Sections

### [Getting Started](getting-started/index.md)
Everything you need to begin using the infrastructure:
- Installation and prerequisites
- Quick start guide
- Configuration and environment setup
- First steps with the CLI

### [Architecture](architecture/index.md)
Deep dive into the system design:
- [Security Model](architecture/security.md) - Traefik-only external access
- [Network Topology](architecture/network.md) - Docker networking and isolation
- [Service Architecture](architecture/services.md) - How services interact
- [Port Configuration](architecture/ports.md) - Understanding port management

### [Services](services/index.md)
Documentation for all services in the infrastructure:

#### Core Services
- [Traefik](services/traefik.md) - Reverse proxy and load balancer
- [LiteLLM](services/litellm.md) - Inference gateway (OpenAI-compatible API)
- [Model Orchestrator](services/orchestrator.md) - VRAM management and lifecycle
- [Langfuse](services/langfuse.md) - LLM observability and tracing

#### Model Services (Internal Only)
- [LLM Services](services/llm/index.md) - llama.cpp and LLM backends
- [TTS Services](services/tts/index.md) - Text-to-Speech services
  - [Kokoro](services/tts/kokoro.md) - Lightweight TTS (82M params)
  - [IndexTTS2](services/tts/index-tts.md) - High quality TTS
  - [VibeVoice](services/tts/vibevoice.md) - TTS + ASR unified
  - [GPT-SoVITS](services/tts/gpt-sovits.md) - Voice cloning
  - [Maya1](services/tts/maya1.md) - Natural language TTS
  - [Qwen3-TTS](services/tts/qwen3-tts.md) - Multi-speaker TTS
  - [eSpeak-NG](services/tts/espeak-ng.md) - Lightweight CPU TTS
- [ASR Services](services/asr/index.md) - Speech-to-Text services
  - [Faster-Whisper](services/asr/faster-whisper.md) - Fast ASR
  - [Qwen-ASR](services/asr/qwen-asr.md) - Accurate ASR (52 languages)

#### Databases
- [PostgreSQL](services/postgresql.md) - Relational database
- [MongoDB](services/mongodb.md) - Document database
- [Neo4j](services/neo4j.md) - Knowledge graph database
- [MinIO](services/minio.md) - S3-compatible object storage

#### Applications
- [ComfyUI](services/comfyui.md) - Image/video generation
- [Mongo Express](services/mongo-express.md) - MongoDB web UI

### [API Reference](api-reference/index.md)
Complete API documentation:
- [LiteLLM API](api-reference/litellm.md) - Inference endpoints
- [Orchestrator API](api-reference/orchestrator.md) - Management endpoints
- [MCP Servers](api-reference/mcp.md) - Web tools (search, scraping)
- [Langfuse API](api-reference/langfuse.md) - Observability SDK

### [Integration Guides](integration/index.md)
How to integrate your applications:
- [LangChain Integration](integration/langchain.md) - Using LangChain with LiteLLM
- [LiteLLM SDK](integration/litellm-sdk.md) - Direct SDK usage
- [Deep Agents](integration/deep-agents.md) - LangGraph and agent frameworks
- [OpenAI SDK](integration/openai-sdk.md) - Drop-in OpenAI replacement
- [MCP Tools](integration/mcp-tools.md) - Using web tools

### [Operations](operations/index.md)
Running and maintaining the infrastructure:
- [CLI Reference](operations/cli.md) - Python CLI commands
- [Makefile Commands](operations/makefile.md) - Legacy make commands
- [Monitoring](operations/monitoring.md) - Health checks and metrics
- [Backup & Recovery](operations/backup.md) - Data backup strategies
- [Updates](operations/updates.md) - Updating services and models
- [Security Best Practices](operations/security.md) - Hardening guide

### [Troubleshooting](troubleshooting/index.md)
Common issues and solutions:
- [Connection Issues](troubleshooting/connection.md) - Network and port problems
- [VRAM Management](troubleshooting/vram.md) - Out of memory errors
- [Service Failures](troubleshooting/services.md) - Debugging failed services
- [Traefik Issues](troubleshooting/traefik.md) - Proxy and routing problems
- [FAQ](troubleshooting/faq.md) - Frequently asked questions

---

## 🏗️ Architecture Overview

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

**Key Principle:** Model containers are internal-only. Access through:
- **LiteLLM** (`litellm.local`) - All inference: LLM, embeddings, ASR, TTS
- **Orchestrator** (`orchestrator.local`) - Lifecycle management, status, VRAM

---

## 📋 Quick Reference

### Common Commands

```bash
# Setup
make setup-secrets      # Generate passwords
make setup-hosts        # Add .local domains to /etc/hosts

# Start services
make up-traefik         # Start Traefik (required first)
make up-orchestrator    # Start control plane
make up-litellm         # Start inference gateway
make up-llama           # Start LLM backend

# Using the Python CLI (recommended)
infra core up           # Start all core services
infra status            # Check service status
infra logs litellm      # View LiteLLM logs

# Verify
curl http://litellm.local/v1/models
curl http://orchestrator.local/status
```

### Service Endpoints

| Service | Domain | Purpose |
|---------|--------|---------|
| Traefik | `traefik.local` | Reverse proxy dashboard |
| LiteLLM | `litellm.local` | **All inference** (LLM, ASR, TTS, embeddings) |
| Orchestrator | `orchestrator.local` | **Control plane** (status, VRAM, lifecycle) |
| Langfuse | `langfuse.local` | LLM observability |
| Neo4j | `neo4j.local` | Knowledge graph (UI) |
| MinIO | `s3.local` | S3-compatible storage |
| MinIO Console | `minio-console.local` | S3 management UI |
| ComfyUI | `comfyui.local` | Image/video generation |
| Mongo Express | `mongo-express.local` | MongoDB UI |

### Database Connections

| Database | Host | Port | Protocol |
|----------|------|------|----------|
| PostgreSQL | `localhost` | `$(TRAEFIK_POSTGRES_PORT)` | TCP |
| MongoDB | `localhost` | `$(TRAEFIK_MONGODB_PORT)` | TCP |
| Neo4j | `localhost` | `$(TRAEFIK_NEO4J_PORT)` | Bolt (TCP) |

---

## 🔗 Additional Resources

- [GitHub Repository](https://github.com/your-org/shared-docker-infra)
- [API Usage Guide](../INTEGRATION_GUIDE.md)
- [Setup API Guide](../SETUP_API_GUIDE.md)
- [Port Configuration](../PORT_CONFIG_NOTE.md)
- [Session Recovery](../SESSION_RECOVERY.txt)

---

## 📞 Support

For issues and questions:
1. Check the [Troubleshooting](troubleshooting/index.md) section
2. Review the [FAQ](troubleshooting/faq.md)
3. Open an issue on GitHub

---

**Last Updated:** March 17, 2026  
**Documentation Version:** 1.0.0
