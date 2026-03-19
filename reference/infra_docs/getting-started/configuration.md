# Configuration Guide

This guide covers all configuration options for the Shared Docker Infrastructure.

## Environment Files

The infrastructure uses two main environment files:

1. **`.env`** - Main configuration file
2. **`traefik/.env`** - Traefik-specific configuration

### Setup

```bash
# Copy example files
cp .env.example .env
cp traefik/.env.example traefik/.env
```

## Port Configuration

All external ports are configurable via environment variables.

### HTTP/HTTPS Entry Points

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAEFIK_HTTP_PORT` | `8880` | HTTP traffic port |
| `TRAEFIK_HTTPS_PORT` | `8443` | HTTPS traffic port |

### Traefik Dashboard

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAEFIK_DASHBOARD_PORT` | `8081` | Web UI port |
| `TRAEFIK_DASHBOARD_ENABLED` | `true` | Enable/disable dashboard |
| `TRAEFIK_DASHBOARD_INSECURE` | `true` | Allow HTTP access |

### Database Ports (TCP Routing)

| Variable | Default | Service |
|----------|---------|---------|
| `TRAEFIK_POSTGRES_PORT` | `5433` | PostgreSQL |
| `TRAEFIK_MONGODB_PORT` | `27018` | MongoDB |
| `TRAEFIK_NEO4J_PORT` | `7688` | Neo4j Bolt |

### Example Configuration

```bash
# .env
TRAEFIK_HTTP_PORT=8880
TRAEFIK_HTTPS_PORT=8443
TRAEFIK_DASHBOARD_PORT=8081
TRAEFIK_POSTGRES_PORT=5433
TRAEFIK_MONGODB_PORT=27018
TRAEFIK_NEO4J_PORT=7688

# traefik/.env
TRAEFIK_HTTP_PORT=8880
TRAEFIK_HTTPS_PORT=8443
TRAEFIK_DASHBOARD_PORT=8081
```

## Secrets Configuration

### Database Credentials

```bash
# PostgreSQL
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<generated-secret>

# MongoDB
MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=<generated-secret>

# Neo4j
NEO4J_AUTH=neo4j/<generated-secret>
```

### Service API Keys

```bash
# LiteLLM
LITELLM_MASTER_KEY=sk-litellm-master

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-<generated>
LANGFUSE_SECRET_KEY=sk-lf-<generated>

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=<generated-secret>
```

### Generate New Secrets

```bash
# Using Makefile
make setup-secrets

# Or manually
bash scripts/generate-secrets.sh

# Or individual secrets
openssl rand -hex 32  # 64-char hex string
```

## LiteLLM Configuration

### Model Configuration

Edit `litellm/config.yaml`:

```yaml
model_list:
  - model_name: qwen-35-27
    litellm_params:
      model: ollama/qwen-35-27
      api_base: http://llama-cpp:8080
    
  - model_name: jina-v5-embed
    litellm_params:
      model: ollama/jina-v5-embed
      api_base: http://llama-cpp:8080

  - model_name: distil-whisper
    litellm_params:
      model: whisper/faster-whisper
      api_base: http://faster-whisper:8080

  - model_name: kokoro
    litellm_params:
      model: tts/kokoro
      api_base: http://kokoro-tts:8080
```

### Langfuse Integration

```yaml
litellm_settings:
  success_callback: ["langfuse"]
  failure_callback: ["langfuse"]

langfuse_public_key: "pk-lf-..."
langfuse_secret_key: "sk-lf-..."
langfuse_host: "http://langfuse.local"
```

## Orchestrator Configuration

### VRAM Management

```bash
# Set VRAM threshold (MB)
ORCHESTRATOR_VRAM_THRESHOLD=2000

# Enable auto-unload
ORCHESTRATOR_AUTO_UNLOAD=true
```

### Model Slots

```bash
# ASR slot
ASR_DEFAULT_MODEL=whisper
ASR_CPU_FALLBACK=whisper-cpu

# TTS slot
TTS_DEFAULT_MODEL=kokoro
TTS_CPU_FALLBACK=kokoro-cpu
```

## Traefik Configuration

### Resource Limits

```bash
# traefik/.env
TRAEFIK_CPU_LIMIT=0.5
TRAEFIK_CPU_RESERVATION=0.1
TRAEFIK_MEMORY_LIMIT=256M
TRAEFIK_MEMORY_RESERVATION=64M
```

### Logging

```bash
TRAEFIK_LOG_LEVEL=INFO
TRAEFIK_ACCESSLOG_ENABLED=true
```

### Dashboard Security

```bash
# Enable basic auth (recommended for production)
TRAEFIK_DASHBOARD_AUTH_ENABLED=true
TRAEFIK_DASHBOARD_USERNAME=admin
TRAEFIK_DASHBOARD_PASSWORD=<hashed-password>
```

## Docker Compose Overrides

Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.8'

services:
  traefik:
    environment:
      - TRAEFIK_HTTP_PORT=9999
  
  litellm:
    environment:
      - LITELLM_LOG_LEVEL=DEBUG
```

## Network Configuration

### Internal Network

All services use the `shared-infra` network:

```bash
# Create network
docker network create shared-infra

# Inspect network
docker network inspect shared-infra
```

### External Access

Only Traefik exposes ports to the host. All other services use internal networking.

## Hostname Configuration

### /etc/hosts

Add entries for local development:

```bash
127.0.0.1 traefik.local langfuse.local litellm.local
127.0.0.1 orchestrator.local neo4j.local s3.local
127.0.0.1 minio-console.local mongo-express.local comfyui.local
```

### Using setup-hosts.sh

```bash
# Automatic setup
sudo bash scripts/setup-hosts.sh

# Or using Makefile
make setup-hosts
```

## Environment Variable Reference

### Core Services

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAEFIK_HTTP_PORT` | `8880` | HTTP port |
| `TRAEFIK_HTTPS_PORT` | `8443` | HTTPS port |
| `TRAEFIK_DASHBOARD_PORT` | `8081` | Dashboard port |
| `LITELLM_MASTER_KEY` | `sk-litellm-master` | LiteLLM API key |
| `ORCHESTRATOR_VRAM_THRESHOLD` | `2000` | VRAM threshold (MB) |

### Databases

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `postgres` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `<generated>` | PostgreSQL password |
| `MONGO_INITDB_ROOT_USERNAME` | `root` | MongoDB username |
| `MONGO_INITDB_ROOT_PASSWORD` | `<generated>` | MongoDB password |
| `NEO4J_AUTH` | `neo4j/<generated>` | Neo4j credentials |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-...` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | `sk-lf-...` | Langfuse secret key |
| `LANGFUSE_HOST` | `http://langfuse.local` | Langfuse URL |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_ROOT_USER` | `minioadmin` | MinIO username |
| `MINIO_ROOT_PASSWORD` | `<generated>` | MinIO password |

## Changing Ports at Runtime

### Step 1: Update Environment Files

```bash
# Edit .env
sed -i 's/TRAEFIK_HTTP_PORT=.*/TRAEFIK_HTTP_PORT=9999/' .env

# Edit traefik/.env
sed -i 's/TRAEFIK_HTTP_PORT=.*/TRAEFIK_HTTP_PORT=9999/' traefik/.env
```

### Step 2: Restart Services

```bash
# Restart Traefik
make down-traefik && make up-traefik

# Or all services
make down && make up
```

### Step 3: Update Connection Strings

```bash
# PostgreSQL
postgresql://user:pass@localhost:5433/dbname

# MongoDB
mongodb://localhost:27018

# Neo4j
neo4j://localhost:7688
```

## Validation

### Check Configuration

```bash
# Verify environment files
cat .env | grep TRAEFIK
cat traefik/.env | grep TRAEFIK

# Check Docker Compose
docker compose config

# Test connectivity
curl http://litellm.local/health
```

### Port Switching Test

```bash
# Run E2E port test
bash scripts/e2e-port-switching-test.sh

# Or using CLI
infra test ports
```

## Best Practices

1. **Never commit `.env` files** - They're in `.gitignore`
2. **Use strong secrets** - Use `make setup-secrets`
3. **Document custom ports** - Update team documentation
4. **Test port changes** - Use `infra test ports`
5. **Backup configurations** - Version control `.env.example`

## Next Steps

- **[Quick Start](quickstart.md)** - Start using your configuration
- **[Architecture](../architecture/index.md)** - Understand the system
- **[Operations](../operations/index.md)** - Learn maintenance tasks
