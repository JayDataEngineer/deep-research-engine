# Traefik

Traefik is the reverse proxy that handles all external access to the infrastructure.

## Overview

- **Purpose:** Reverse proxy and load balancer
- **Domain:** `traefik.local`
- **Dashboard:** `http://traefik.local:${TRAEFIK_DASHBOARD_PORT}`
- **Configuration:** `traefik/` directory

## Key Features

- ✅ **Single entry point** - Only service exposing ports to host
- ✅ **HTTP routing** - Routes to web services by domain
- ✅ **TCP routing** - Routes to databases by port
- ✅ **Health checks** - Monitors service health
- ✅ **Load balancing** - Distributes traffic across instances
- ✅ **SSL termination** - HTTPS support (optional)
- ✅ **Middleware** - Authentication, rate limiting, etc.

## Quick Start

```bash
# Start Traefik
make up-traefik

# Or using Docker Compose
docker compose -f docker-compose.traefik.yml up -d

# Access dashboard
open http://traefik.local:8081

# Check status
curl http://traefik.local:8081/ping
```

## Configuration

### Environment Variables

Located in `traefik/.env`:

```bash
# Ports
TRAEFIK_HTTP_PORT=8880
TRAEFIK_HTTPS_PORT=8443
TRAEFIK_DASHBOARD_PORT=8081

# Dashboard
TRAEFIK_DASHBOARD_ENABLED=true
TRAEFIK_DASHBOARD_INSECURE=true

# Resources
TRAEFIK_CPU_LIMIT=0.5
TRAEFIK_MEMORY_LIMIT=256M

# Logging
TRAEFIK_LOG_LEVEL=INFO
TRAEFIK_ACCESSLOG_ENABLED=true
```

### Docker Compose

```yaml
# docker-compose.traefik.yml
services:
  traefik:
    image: traefik:v3.0
    ports:
      - "${TRAEFIK_HTTP_PORT}:80"
      - "${TRAEFIK_HTTPS_PORT}:443"
      - "${TRAEFIK_DASHBOARD_PORT}:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./traefik/traefik.yml:/etc/traefik/traefik.yml:ro
    networks:
      - shared-infra
    labels:
      - "traefik.enable=true"
```

### Traefik Configuration

```yaml
# traefik/traefik.yml
entryPoints:
  http:
    address: ":80"
  https:
    address: ":443"
  postgres:
    address: ":5432"
  mongodb:
    address: ":27017"
  neo4j:
    address: ":7687"

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false
  file:
    filename: /etc/traefik/traefik.yml

api:
  dashboard: true
  insecure: true
```

## HTTP Routing

### Web Services

Services are routed by domain name:

```yaml
# Service labels
labels:
  - "traefik.enable=true"
  - "traefik.http.routers.myservice.rule=Host(`myservice.local`)"
  - "traefik.http.routers.myservice.entrypoints=http"
  - "traefik.http.services.myservice.loadbalancer.server.port=8080"
```

**Routed Services:**

| Service | Domain | Internal Port |
|---------|--------|--------------|
| LiteLLM | `litellm.local` | 8080 |
| Orchestrator | `orchestrator.local` | 8080 |
| Langfuse | `langfuse.local` | 3000 |
| Neo4j Browser | `neo4j.local` | 7474 |
| MinIO Console | `minio-console.local` | 9001 |
| ComfyUI | `comfyui.local` | 8188 |
| Mongo Express | `mongo-express.local` | 8081 |

## TCP Routing

### Database Services

Databases use TCP routing:

```yaml
# Service labels
labels:
  - "traefik.enable=true"
  - "traefik.tcp.routers.myservice.rule=HostSNI(`*`)"
  - "traefik.tcp.routers.myservice.entrypoints=postgres"
  - "traefik.tcp.services.myservice.loadbalancer.server.port=5432"
```

**Routed Databases:**

| Database | Entry Point | External Port | Internal Port |
|----------|-------------|--------------|---------------|
| PostgreSQL | `postgres` | 5433 | 5432 |
| MongoDB | `mongodb` | 27018 | 27017 |
| Neo4j Bolt | `neo4j` | 7688 | 7687 |

## Dashboard

### Access

```bash
# Open dashboard
open http://traefik.local:8081

# Or with custom port
open http://traefik.local:${TRAEFIK_DASHBOARD_PORT}
```

### Features

- **Overview** - System health and metrics
- **Routers** - HTTP and TCP routers
- **Services** - Backend services
- **Middlewares** - Authentication, rate limiting
- **Entrypoints** - Port configuration

### API Access

```bash
# Health check
curl http://traefik.local:8081/ping

# HTTP routers
curl http://traefik.local:8081/api/http/routers

# TCP routers
curl http://traefik.local:8081/api/tcp/routers

# Services
curl http://traefik.local:8081/api/http/services

# Overview
curl http://traefik.local:8081/api/overview
```

## Middleware

### Basic Authentication

```yaml
labels:
  - "traefik.http.middlewares.auth.basicauth.users=admin:$$apr1$$..."
  - "traefik.http.routers.dashboard.middlewares=auth"
```

### Rate Limiting

```yaml
labels:
  - "traefik.http.middlewares.rate-limit.ratelimit.average=100"
  - "traefik.http.middlewares.rate-limit.ratelimit.burst=50"
  - "traefik.http.routers.myservice.middlewares=rate-limit"
```

### Redirect HTTP to HTTPS

```yaml
labels:
  - "traefik.http.middlewares.redirect-https.redirectscheme.scheme=https"
  - "traefik.http.routers.myservice.middlewares=redirect-https"
```

## Monitoring

### Logs

```bash
# View Traefik logs
make logs-traefik

# Or using Docker Compose
docker compose logs traefik

# Follow logs
docker compose logs -f traefik
```

### Metrics

```bash
# Prometheus metrics (if enabled)
curl http://traefik.local:8080/metrics

# Health check
curl http://traefik.local:8081/ping
```

### Resource Usage

```bash
# Docker stats
docker stats traefik

# Resource limits
docker inspect traefik | grep -A 10 NanoCpus
```

## Troubleshooting

### Can't Access Dashboard

```bash
# Check Traefik is running
docker compose ps traefik

# Check port
cat .env | grep TRAEFIK_DASHBOARD_PORT

# Test connectivity
curl http://traefik.local:8081/ping

# Check logs
docker compose logs traefik
```

### Service Not Routed

```bash
# Check service labels
docker compose exec traefik traefik version

# View routers
curl http://traefik.local:8081/api/http/routers

# Check service discovery
curl http://traefik.local:8081/api/http/services
```

### TCP Routing Issues

```bash
# Check TCP routers
curl http://traefik.local:8081/api/tcp/routers

# Test connection
psql -h localhost -p 5433 -U postgres

# Check entrypoints
docker compose exec traefik netstat -tlnp
```

## Best Practices

1. **Keep Traefik updated** - Use latest stable version
2. **Enable access logging** - For debugging and auditing
3. **Use health checks** - Monitor service health
4. **Configure rate limiting** - Protect against abuse
5. **Enable authentication** - For dashboard and sensitive services
6. **Use non-standard ports** - Reduce automated attacks
7. **Monitor resource usage** - Set CPU/memory limits

## Next Steps

- **[Security Model](../architecture/security.md)** - Security architecture
- **[Port Configuration](../architecture/ports.md)** - Port management
- **[LiteLLM](litellm.md)** - Inference gateway
- **[Network Topology](../architecture/network.md)** - Network details
