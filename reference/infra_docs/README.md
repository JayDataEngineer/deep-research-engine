# Shared Docker Infrastructure - Documentation

Welcome to the comprehensive documentation for the Shared Docker Infrastructure.

## 🚀 Quick Start

**New users:** Start with the [Getting Started Guide](getting-started/installation.md)

**Already set up?** See the [Quick Start Guide](getting-started/quickstart.md)

## 📚 Documentation Sections

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started/) | Installation, setup, and first steps |
| [Architecture](architecture/) | System design, security, and networking |
| [Services](services/) | Documentation for all services |
| [API Reference](api-reference/) | Complete API documentation |
| [Integration](integration/) | Integration guides for applications |
| [Operations](operations/) | Maintenance, monitoring, and troubleshooting |
| [Troubleshooting](troubleshooting/) | Common issues and solutions |

## 🔧 Quick Reference

### Common Commands

```bash
# Setup
make setup-secrets      # Generate passwords
make setup-hosts        # Add .local domains to /etc/hosts

# Start services
make up-traefik         # Start Traefik (required first)
make up-litellm         # Start inference gateway
make up-orchestrator    # Start control plane
make up-llama           # Start LLM backend

# Using Python CLI (recommended)
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
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
      ┌───────────────┐               ┌───────────────┐
      │    LiteLLM    │               │  Orchestrator │
      │  (Inference)  │               │   (Control)   │
      └───────┬───────┘               └───────┬───────┘
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

## 📖 Navigation by Task

### I want to...

| Task | Documentation |
|------|---------------|
| Install the infrastructure | [Installation Guide](getting-started/installation.md) |
| Start services quickly | [Quick Start](getting-started/quickstart.md) |
| Configure ports | [Configuration Guide](getting-started/configuration.md) |
| Set up the CLI | [CLI Setup](getting-started/cli-setup.md) |
| Understand the architecture | [Architecture Overview](architecture/) |
| Learn about security | [Security Model](architecture/security.md) |
| Configure networking | [Network Topology](architecture/network.md) |
| Manage ports | [Port Configuration](architecture/ports.md) |
| Use LiteLLM API | [LiteLLM API](api-reference/litellm.md) |
| Use Orchestrator API | [Orchestrator API](api-reference/orchestrator.md) |
| Integrate LangChain | [LangChain Integration](integration/langchain.md) |
| Use LiteLLM SDK | [LiteLLM SDK](integration/litellm-sdk.md) |
| Set up monitoring | [Monitoring Guide](operations/monitoring.md) |
| Backup data | [Backup & Recovery](operations/backup.md) |
| Troubleshoot issues | [Troubleshooting](troubleshooting/) |

## 🛠️ Service Documentation

### Core Services

- [Traefik](services/traefik.md) - Reverse proxy
- [LiteLLM](services/litellm.md) - Inference gateway
- [Orchestrator](services/orchestrator.md) - Resource management
- [Langfuse](services/langfuse.md) - Observability

### Model Services (Internal Only)

#### LLM
- [llama.cpp](services/llm/index.md) - LLM inference

#### TTS
- [Kokoro](services/tts/kokoro.md) - Lightweight TTS
- [IndexTTS2](services/tts/index-tts.md) - High quality TTS
- [VibeVoice](services/tts/vibevoice.md) - TTS + ASR
- [GPT-SoVITS](services/tts/gpt-sovits.md) - Voice cloning
- [Maya1](services/tts/maya1.md) - Natural language TTS
- [Qwen3-TTS](services/tts/qwen3-tts.md) - Multi-speaker TTS
- [eSpeak-NG](services/tts/espeak-ng.md) - CPU TTS

#### ASR
- [Faster-Whisper](services/asr/faster-whisper.md) - Fast ASR
- [Qwen-ASR](services/asr/qwen-asr.md) - Accurate ASR

### Databases

- [PostgreSQL](services/postgresql.md) - Relational database
- [MongoDB](services/mongodb.md) - Document database
- [Neo4j](services/neo4j.md) - Knowledge graph
- [MinIO](services/minio.md) - Object storage

### Applications

- [ComfyUI](services/comfyui.md) - Image generation
- [Mongo Express](services/mongo-express.md) - MongoDB UI

## 📚 Legacy Documentation

The following legacy documents are still available:

- [Integration Guide](../INTEGRATION_GUIDE.md) - Detailed integration patterns
- [Setup API Guide](../SETUP_API_GUIDE.md) - API usage examples
- [Port Configuration](../PORT_CONFIG_NOTE.md) - Port management
- [Session Recovery](../SESSION_RECOVERY.txt) - Recovery procedures

## 🆘 Getting Help

1. Check the [Troubleshooting](troubleshooting/) section
2. Review the [FAQ](troubleshooting/faq.md)
3. Search existing [GitHub Issues](https://github.com/your-org/shared-docker-infra/issues)
4. Open a new issue if needed

---

**Last Updated:** March 17, 2026  
**Documentation Version:** 1.0.0
