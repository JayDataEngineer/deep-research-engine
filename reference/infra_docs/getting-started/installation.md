# Installation Guide

This guide covers the complete installation process for the Shared Docker Infrastructure.

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or WSL2
- **CPU**: 4+ cores (8+ recommended for LLM workloads)
- **RAM**: 16GB minimum, 32GB+ recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM for accelerated inference (optional)
- **Storage**: 50GB+ free space for models and data
- **Docker**: Version 24.0+ with Docker Compose v2.20+

### Required Software

1. **Docker Engine**
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   
   # Verify installation
   docker --version
   docker compose version
   ```

2. **Git**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git
   
   # Verify
   git --version
   ```

3. **Make** (optional, for Makefile commands)
   ```bash
   sudo apt-get install make
   ```

4. **UV** (recommended, for Python CLI)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Installation Steps

### Step 1: Clone the Repository

```bash
cd ~
git clone https://github.com/your-org/shared-docker-infra.git
cd shared-docker-infra
```

### Step 2: Create Environment Files

```bash
# Copy example environment files
cp .env.example .env
cp traefik/.env.example traefik/.env
```

### Step 3: Generate Secrets

```bash
# Using Makefile
make setup-secrets

# Or manually
bash scripts/generate-secrets.sh
```

This generates secure random passwords for:
- PostgreSQL
- MongoDB
- Neo4j
- MinIO
- Langfuse
- LiteLLM API keys

### Step 4: Configure Hostnames

Add `.local` domains to your `/etc/hosts`:

```bash
# Using Makefile (recommended)
make setup-hosts

# Or manually
sudo bash scripts/setup-hosts.sh
```

This adds entries like:
```
127.0.0.1 traefik.local langfuse.local litellm.local
127.0.0.1 orchestrator.local neo4j.local s3.local
127.0.0.1 minio-console.local mongo-express.local comfyui.local
```

### Step 5: Create Docker Network

```bash
# Create the shared internal network
docker network create shared-infra
```

### Step 6: Verify Installation

```bash
# Check Docker is running
docker info

# Check Docker Compose
docker compose version

# Verify network exists
docker network ls | grep shared-infra

# Check environment files
ls -la .env traefik/.env
```

## Post-Installation

### Configure Ports (Optional)

Edit `.env` and `traefik/.env` to customize ports:

```bash
# .env
TRAEFIK_HTTP_PORT=8880
TRAEFIK_HTTPS_PORT=8443
TRAEFIK_DASHBOARD_PORT=8081
TRAEFIK_POSTGRES_PORT=5433
TRAEFIK_MONGODB_PORT=27018
TRAEFIK_NEO4J_PORT=7688
```

See [Configuration](configuration.md) for all available options.

### Set Up Python CLI (Recommended)

```bash
cd scripts/cli
uv sync

# Test the CLI
infra --help
```

See [CLI Setup](cli-setup.md) for detailed instructions.

## Next Steps

1. **[Quick Start](quickstart.md)** - Start your first services
2. **[Configuration](configuration.md)** - Learn about all configuration options
3. **[Architecture](../architecture/index.md)** - Understand the system design

## Troubleshooting

### Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Port Already in Use

```bash
# Check what's using a port
sudo lsof -i :8880

# Change the port in .env
TRAEFIK_HTTP_PORT=8881
```

### Network Creation Failed

```bash
# Remove existing network and recreate
docker network rm shared-infra
docker network create shared-infra
```

### DNS Resolution Issues

Verify `/etc/hosts` entries:
```bash
grep -E "traefik.local|litellm.local|orchestrator.local" /etc/hosts
```

## Getting Help

- Check the [FAQ](../troubleshooting/faq.md)
- Review [Troubleshooting](../troubleshooting/index.md)
- Open an issue on GitHub
