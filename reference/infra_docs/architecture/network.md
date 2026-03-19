# Network Topology

This document describes the Docker networking configuration for the Shared Docker Infrastructure.

## Overview

The infrastructure uses a **single internal Docker network** (`shared-infra`) for all service communication, with **Traefik as the only bridge** to the external world.

```
┌─────────────────────────────────────────────────────────┐
│                         HOST                            │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              TRAEFIK CONTAINER                   │   │
│  │  Ports: 8880, 8443, 8081, 5433, 27018, 7688    │   │
│  │  Networks: shared-infra, host                   │   │
│  └──────────────────┬──────────────────────────────┘   │
│                     │                                    │
│  ┌──────────────────┴──────────────────────────────┐   │
│  │           SHARED-INFRA NETWORK                   │   │
│  │  (Internal Docker Network)                       │   │
│  │                                                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │   │
│  │  │ LiteLLM  │  │Orchestrator│ │ Langfuse  │      │   │
│  │  └──────────┘  └──────────┘  └──────────┘      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │   │
│  │  │llama.cpp │  │ whisper  │  │ kokoro   │      │   │
│  │  └──────────┘  └──────────┘  └──────────┘      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │   │
│  │  │ Postgres │  │ MongoDB  │  │  Neo4j   │      │   │
│  │  └──────────┘  └──────────┘  └──────────┘      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Network Types

### 1. Shared-Infra Network (Internal)

**Purpose:** Internal service communication

```bash
# Create network
docker network create shared-infra

# Network properties
docker network inspect shared-infra
```

**Properties:**
- **Driver:** bridge
- **Scope:** local
- **Internal:** false (but no external ports exposed)
- **IPAM:** Docker default (172.x.x.x range)

**Connected Services:**
- All application containers
- Traefik (for routing to internal services)
- Model containers (llama.cpp, TTS, ASR)
- Databases
- Monitoring services

### 2. Host Network (External)

**Purpose:** External access to Traefik

**Properties:**
- Only Traefik connects to host
- Ports exposed via `ports:` in docker-compose.yml
- Configurable via environment variables

## Service Communication

### External → Internal Flow

```
Client (Browser/Curl)
    ↓
Host Network (localhost:8880)
    ↓
Traefik (ports: section)
    ↓
Shared-Infra Network
    ↓
Target Service (LiteLLM, Orchestrator, etc.)
```

### Internal → Internal Flow

```
LiteLLM Container
    ↓
Shared-Infra Network (DNS resolution)
    ↓
llama.cpp Container
```

**DNS Resolution:**
- Services can reach each other by service name
- Example: `http://llama-cpp:8080` from LiteLLM
- Docker's embedded DNS handles resolution

## Docker Compose Configuration

### Network Definition

```yaml
# docker-compose.network.yml
version: '3.8'

networks:
  shared-infra:
    external: true
    name: shared-infra
```

### Service Configuration

**Traefik (Bridge between host and internal):**
```yaml
services:
  traefik:
    image: traefik:v3.0
    ports:
      - "${TRAEFIK_HTTP_PORT}:80"
      - "${TRAEFIK_HTTPS_PORT}:443"
      - "${TRAEFIK_DASHBOARD_PORT}:8080"
    networks:
      - shared-infra
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
```

**LiteLLM (Internal only):**
```yaml
services:
  litellm:
    image: ghcr.io/berriai/litellm:main
    expose:
      - "8080"
    networks:
      - shared-infra
    labels:
      - "traefik.http.routers.litellm.rule=Host(`litellm.local`)"
      - "traefik.http.services.litellm.loadbalancer.server.port=8080"
```

**Model Container (Internal only):**
```yaml
services:
  llama-cpp:
    image: ghcr.io/ggerganov/llama.cpp:server
    expose:
      - "8080"
    networks:
      - shared-infra
    # No Traefik labels - accessed via LiteLLM only
```

**Database (TCP routing via Traefik):**
```yaml
services:
  postgres:
    image: postgres:16
    expose:
      - "5432"
    networks:
      - shared-infra
    labels:
      - "traefik.tcp.routers.postgres.rule=HostSNI(`*`)"
      - "traefik.tcp.routers.postgres.entrypoints=postgres"
      - "traefik.tcp.services.postgres.loadbalancer.server.port=5432"
```

## Network Inspection

### View Network

```bash
# List networks
docker network ls

# Inspect shared-infra
docker network inspect shared-infra
```

**Output includes:**
- Connected containers
- IP addresses
- Network configuration

### View Container Connectivity

```bash
# Check container network
docker inspect <container-name> | grep -A 20 Networks

# Or use docker inspect format
docker inspect --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}: {{.IPAddress}}{{end}}' <container-name>
```

### Test Connectivity

```bash
# From Traefik container
docker compose exec traefik ping litellm

# From LiteLLM container
docker compose exec litellm ping llama-cpp

# From host (via Traefik TCP routing)
psql -h localhost -p ${TRAEFIK_POSTGRES_PORT} -U postgres
```

## Port Mapping

### HTTP Services

| Service | Internal Port | External (via Traefik) |
|---------|--------------|------------------------|
| LiteLLM | 8080 | `${TRAEFIK_HTTP_PORT}` (8880) |
| Orchestrator | 8080 | `${TRAEFIK_HTTP_PORT}` (8880) |
| Langfuse | 3000 | `${TRAEFIK_HTTP_PORT}` (8880) |
| Neo4j Browser | 7474 | `${TRAEFIK_HTTP_PORT}` (8880) |
| MinIO Console | 9001 | `${TRAEFIK_HTTP_PORT}` (8880) |

### TCP Services

| Service | Internal Port | External (via Traefik) |
|---------|--------------|------------------------|
| PostgreSQL | 5432 | `${TRAEFIK_POSTGRES_PORT}` (5433) |
| MongoDB | 27017 | `${TRAEFIK_MONGODB_PORT}` (27018) |
| Neo4j Bolt | 7687 | `${TRAEFIK_NEO4J_PORT}` (7688) |

### Traefik Dashboard

| Service | Internal Port | External |
|---------|--------------|----------|
| Dashboard API | 8080 | `${TRAEFIK_DASHBOARD_PORT}` (8081) |

## DNS Resolution

### Service Discovery

Docker's embedded DNS resolves service names:

```bash
# From any container in shared-infra
curl http://litellm:8080/health
curl http://orchestrator:8080/status
curl http://llama-cpp:8080/health
```

### From Host

Add to `/etc/hosts`:

```bash
127.0.0.1 traefik.local langfuse.local litellm.local
127.0.0.1 orchestrator.local neo4j.local s3.local
127.0.0.1 minio-console.local mongo-express.local comfyui.local
```

Or use setup script:
```bash
make setup-hosts
```

## Network Security

### Isolation

- Services only accessible within `shared-infra`
- No direct external access (except Traefik)
- Firewall rules only needed for Traefik ports

### Access Control

```yaml
# Traefik middleware for authentication
labels:
  - "traefik.http.middlewares.auth.basicauth.users=admin:$$apr1$$..."
  - "traefik.http.routers.dashboard.middlewares=auth"
```

### Rate Limiting

```yaml
# Traefik rate limiting
labels:
  - "traefik.http.middlewares.rate-limit.ratelimit.average=100"
  - "traefik.http.middlewares.rate-limit.ratelimit.burst=50"
```

## Troubleshooting

### Can't Connect Between Services

```bash
# Check both services on same network
docker network inspect shared-infra

# Test DNS resolution
docker compose exec litellm nslookup llama-cpp

# Test connectivity
docker compose exec litellm curl http://llama-cpp:8080/health
```

### Port Conflicts

```bash
# Check what's using a port
sudo lsof -i :8880
sudo netstat -tlnp | grep 8880

# Change port in .env
TRAEFIK_HTTP_PORT=8881
```

### Network Issues

```bash
# Recreate network
docker network rm shared-infra
docker network create shared-infra

# Restart all services
docker compose down
docker compose up -d
```

## Performance Considerations

### Network Performance

- Docker bridge network has minimal overhead
- Services on same host communicate via veth pairs
- No NAT overhead for internal communication

### DNS Caching

Docker caches DNS entries:
- TTL: 600 seconds (default)
- Can cause delays when services restart
- Clear cache: `docker network prune`

## Best Practices

1. **Use service names for internal communication**
   ```python
   # ✅ Good
   llama_url = "http://llama-cpp:8080"
   
   # ❌ Bad
   llama_url = "http://172.18.0.5:8080"
   ```

2. **Never expose internal services directly**
   ```yaml
   # ✅ Good
   expose:
     - "8080"
   
   # ❌ Bad
   ports:
     - "8080:8080"
   ```

3. **Use environment variables for ports**
   ```yaml
   ports:
     - "${TRAEFIK_HTTP_PORT}:80"
   ```

4. **Keep network simple**
   - Single `shared-infra` network
   - Avoid complex multi-network setups
   - Use Traefik for all external access

## Next Steps

- **[Security Model](security.md)** - Security architecture
- **[Port Configuration](ports.md)** - Port management
- **[Service Architecture](services.md)** - Service details
