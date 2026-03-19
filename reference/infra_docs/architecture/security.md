# Security Model

The Shared Docker Infrastructure follows a **security-first** design with all external access controlled through Traefik reverse proxy.

## Core Security Principle

**Only Traefik exposes ports to the host.** All other services use internal Docker networking exclusively.

```
┌─────────────────────────────────────────────────────────┐
│                    EXTERNAL ACCESS                       │
│              (Only through Traefik ports)                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │     TRAEFIK     │
                  │  (Only service  │
                  │  exposing ports)│
                  └────────┬────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ LiteLLM  │     │Orchestrator│    │ Database │
   │(internal)│     │(internal) │    │(internal)│
   └──────────┘     └──────────┘     └──────────┘
```

## Benefits

### 1. Centralized Access Control

Single entry point for all external traffic:
- ✅ All requests logged in one place
- ✅ Single point for rate limiting
- ✅ Centralized SSL termination
- ✅ Unified authentication layer

### 2. Network Isolation

Internal services cannot be accessed directly:
- ✅ Model containers protected from external access
- ✅ Databases only accessible via Traefik TCP routing
- ✅ No direct container-to-container access from outside
- ✅ Reduced attack surface

### 3. Simplified Firewall Rules

Only Traefik ports need to be allowed:
```bash
# Only these ports need firewall rules
TRAEFIK_HTTP_PORT=8880
TRAEFIK_HTTPS_PORT=8443
TRAEFIK_DASHBOARD_PORT=8081
TRAEFIK_POSTGRES_PORT=5433
TRAEFIK_MONGODB_PORT=27018
TRAEFIK_NEO4J_PORT=7688
```

### 4. Audit Trail

All access logged by Traefik:
```yaml
# Enable access logging
TRAEFIK_ACCESSLOG_ENABLED=true
TRAEFIK_LOG_LEVEL=INFO
```

## Implementation

### Docker Compose Configuration

**Correct (Traefik-only exposure):**
```yaml
version: '3.8'

services:
  traefik:
    image: traefik:v3.0
    ports:
      - "${TRAEFIK_HTTP_PORT}:80"
      - "${TRAEFIK_HTTPS_PORT}:443"
      - "${TRAEFIK_DASHBOARD_PORT}:8080"
    networks:
      - shared-infra
  
  litellm:
    image: ghcr.io/berriai/litellm:main
    expose:
      - "8080"
    # ✅ No ports: section - internal only
    networks:
      - shared-infra
    labels:
      - "traefik.http.routers.litellm.rule=Host(`litellm.local`)"
  
  llama-cpp:
    image: ghcr.io/ggerganov/llama.cpp:server
    expose:
      - "8080"
    # ✅ Internal only - accessed via LiteLLM
    networks:
      - shared-infra
```

**Incorrect (direct port exposure):**
```yaml
# ❌ WRONG - Don't do this!
services:
  llama-cpp:
    ports:
      - "8080:8080"  # ❌ Exposes directly to host!
  
  postgres:
    ports:
      - "5432:5432"  # ❌ Direct database access!
```

### Network Configuration

All services use the internal `shared-infra` network:

```yaml
networks:
  shared-infra:
    external: true
    name: shared-infra
```

This ensures:
- Services can communicate internally
- No external access without Traefik
- Isolated from other Docker networks

## Access Patterns

### Web Services (HTTP/HTTPS)

All web services accessed through Traefik HTTP routing:

| Service | Domain | Traefik Router |
|---------|--------|----------------|
| LiteLLM | `litellm.local` | `traefik.http.routers.litellm` |
| Orchestrator | `orchestrator.local` | `traefik.http.routers.orchestrator` |
| Langfuse | `langfuse.local` | `traefik.http.routers.langfuse` |
| Neo4j Browser | `neo4j.local` | `traefik.http.routers.neo4j` |
| MinIO Console | `minio-console.local` | `traefik.http.routers.minio-console` |

### Database Services (TCP)

Databases accessed through Traefik TCP routing:

| Database | Host Port | Traefik Router |
|----------|-----------|----------------|
| PostgreSQL | `localhost:${TRAEFIK_POSTGRES_PORT}` | `traefik.tcp.routers.postgres` |
| MongoDB | `localhost:${TRAEFIK_MONGODB_PORT}` | `traefik.tcp.routers.mongodb` |
| Neo4j | `localhost:${TRAEFIK_NEO4J_PORT}` | `traefik.tcp.routers.neo4j-bolt` |

### Model Containers (Internal Only)

**Model containers have NO external access:**

| Model | Access Method |
|-------|---------------|
| llama.cpp | Via LiteLLM only |
| Kokoro TTS | Via LiteLLM or Orchestrator |
| Faster-Whisper | Via LiteLLM or Orchestrator |
| All other TTS/ASR | Via LiteLLM or Orchestrator |

## Security Best Practices

### 1. Use Non-Standard Ports

Avoid common ports to reduce automated attacks:

```bash
# Instead of standard ports
TRAEFIK_HTTP_PORT=8880      # Not 80
TRAEFIK_HTTPS_PORT=8443     # Not 443
TRAEFIK_POSTGRES_PORT=5433  # Not 5432
TRAEFIK_MONGODB_PORT=27018  # Not 27017
```

### 2. Enable Authentication

```bash
# Traefik dashboard basic auth
TRAEFIK_DASHBOARD_AUTH_ENABLED=true

# Database passwords (generated by setup-secrets)
POSTGRES_PASSWORD=<strong-password>
MONGO_PASSWORD=<strong-password>
```

### 3. Limit Resource Access

```yaml
# Traefik resource limits
deploy:
  resources:
    limits:
      cpus: '0.5'
      memory: 256M
    reservations:
      cpus: '0.1'
      memory: 64M
```

### 4. Enable Logging

```bash
# Traefik logging
TRAEFIK_LOG_LEVEL=INFO
TRAEFIK_ACCESSLOG_ENABLED=true

# View logs
docker compose logs traefik
```

### 5. Use HTTPS in Production

```yaml
# Traefik HTTPS configuration
labels:
  - "traefik.http.routers.https.entrypoints=https"
  - "traefik.http.routers.https.tls=true"
  - "traefik.http.routers.https.tls.certresolver=letsencrypt"
```

## Verification

### Check No Direct Port Exposure

```bash
# Scan for exposed ports
docker compose ps --format "table {{.Name}}\t{{.Ports}}"

# Should only show Traefik with ports
# All other services should show "expose" not "ports"
```

### Verify Network Isolation

```bash
# Inspect network
docker network inspect shared-infra

# Check service connectivity
docker compose exec traefik ping litellm  # ✅ Should work
docker compose exec litellm ping traefik  # ✅ Should work
```

### Test Access Patterns

```bash
# ✅ Should work - via Traefik
curl http://litellm.local/health

# ❌ Should fail - direct access
curl http://localhost:8080/health  # If llama.cpp exposes 8080
```

## Migration Guide

### From Direct Port Exposure

If you currently expose services directly:

**Before (Insecure):**
```yaml
services:
  postgres:
    ports:
      - "5432:5432"
  
  llama-cpp:
    ports:
      - "8080:8080"
```

**After (Secure):**
```yaml
services:
  postgres:
    expose:
      - "5432"
    labels:
      - "traefik.tcp.routers.postgres.rule=HostSNI(`*`)"
      - "traefik.tcp.routers.postgres.entrypoints=postgres"
      - "traefik.tcp.services.postgres.loadbalancer.server.port=5432"
  
  llama-cpp:
    expose:
      - "8080"
    # No Traefik labels - accessed via LiteLLM only
```

**Update Connection Strings:**
```bash
# Old
postgresql://user:pass@localhost:5432/db

# New
postgresql://user:pass@localhost:${TRAEFIK_POSTGRES_PORT}/db
```

## Troubleshooting

### Can't Access Service

1. **Check Traefik is running:**
   ```bash
   docker compose ps traefik
   ```

2. **Verify Traefik labels:**
   ```bash
   docker compose exec traefik traefik version
   docker compose exec traefik cat /dev/null  # Test connectivity
   ```

3. **Check routing:**
   ```bash
   curl http://traefik.local:8081/api/http/routers
   ```

### Service Exposed Directly

If a service accidentally exposes ports:

```yaml
# ❌ Wrong
ports:
  - "8080:8080"

# ✅ Correct
expose:
  - "8080"
```

## Security Checklist

- [ ] Only Traefik has `ports:` sections
- [ ] All services use `shared-infra` network
- [ ] All web services have Traefik labels
- [ ] All databases use TCP routing
- [ ] Model containers have no external access
- [ ] Strong passwords generated
- [ ] Non-standard ports used
- [ ] Logging enabled
- [ ] Resource limits configured
- [ ] HTTPS configured (production)

## Next Steps

- **[Network Topology](network.md)** - Network configuration details
- **[Port Configuration](ports.md)** - Port management
- **[Operations Security](../operations/security.md)** - Hardening guide
