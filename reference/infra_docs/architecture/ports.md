# Port Configuration

Complete guide to port configuration and management in the Shared Docker Infrastructure.

## Overview

All external ports are **configurable via environment variables** and managed centrally through Traefik.

```
┌─────────────────────────────────────────────────────────┐
│                    EXTERNAL PORTS                        │
│              (Configurable in .env files)                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │     TRAEFIK     │
                  │  (Port mapping) │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  INTERNAL PORTS │
                  │  (Fixed)        │
                  └─────────────────┘
```

## Configuration Files

### Main Configuration (`.env`)

```bash
# Copy from example
cp .env.example .env
```

**Key variables:**
```bash
# HTTP/HTTPS
TRAEFIK_HTTP_PORT=8880
TRAEFIK_HTTPS_PORT=8443

# Dashboard
TRAEFIK_DASHBOARD_PORT=8081

# Databases
TRAEFIK_POSTGRES_PORT=5433
TRAEFIK_MONGODB_PORT=27018
TRAEFIK_NEO4J_PORT=7688
```

### Traefik Configuration (`traefik/.env`)

```bash
# Copy from example
cp traefik/.env.example traefik/.env
```

**Key variables:**
```bash
TRAEFIK_HTTP_PORT=8880
TRAEFIK_HTTPS_PORT=8443
TRAEFIK_DASHBOARD_PORT=8081
```

## Default Port Assignments

### HTTP Entry Points

| Service | External Port | Internal Port | Variable |
|---------|--------------|---------------|----------|
| HTTP Traffic | 8880 | 80 | `TRAEFIK_HTTP_PORT` |
| HTTPS Traffic | 8443 | 443 | `TRAEFIK_HTTPS_PORT` |
| Traefik Dashboard | 8081 | 8080 | `TRAEFIK_DASHBOARD_PORT` |

### Web Services (HTTP Routing)

All web services share the HTTP port (8880) and are differentiated by domain:

| Service | Domain | External Port | Internal Port |
|---------|--------|--------------|---------------|
| LiteLLM | `litellm.local` | 8880 | 8080 |
| Orchestrator | `orchestrator.local` | 8880 | 8080 |
| Langfuse | `langfuse.local` | 8880 | 3000 |
| Neo4j Browser | `neo4j.local` | 8880 | 7474 |
| MinIO Console | `minio-console.local` | 8880 | 9001 |
| ComfyUI | `comfyui.local` | 8880 | 8188 |
| Mongo Express | `mongo-express.local` | 8880 | 8081 |

### Database Services (TCP Routing)

| Database | External Port | Internal Port | Variable |
|----------|--------------|---------------|----------|
| PostgreSQL | 5433 | 5432 | `TRAEFIK_POSTGRES_PORT` |
| MongoDB | 27018 | 27017 | `TRAEFIK_MONGODB_PORT` |
| Neo4j Bolt | 7688 | 7687 | `TRAEFIK_NEO4J_PORT` |

## Changing Ports

### Step 1: Update Environment Files

```bash
# Edit main .env
sed -i 's/TRAEFIK_HTTP_PORT=.*/TRAEFIK_HTTP_PORT=9999/' .env
sed -i 's/TRAEFIK_DASHBOARD_PORT=.*/TRAEFIK_DASHBOARD_PORT=9090/' .env

# Edit traefik/.env
sed -i 's/TRAEFIK_HTTP_PORT=.*/TRAEFIK_HTTP_PORT=9999/' traefik/.env
sed -i 's/TRAEFIK_DASHBOARD_PORT=.*/TRAEFIK_DASHBOARD_PORT=9090/' traefik/.env
```

### Step 2: Restart Traefik

```bash
# Stop Traefik
make down-traefik

# Start Traefik
make up-traefik

# Or using Docker Compose directly
docker compose -f docker-compose.traefik.yml down
docker compose -f docker-compose.traefik.yml up -d
```

### Step 3: Verify New Ports

```bash
# Test HTTP
curl http://litellm.local:9999/health

# Test Dashboard
curl http://traefik.local:9090/ping

# Check Traefik
docker compose ps traefik
```

### Step 4: Update Connection Strings

```bash
# PostgreSQL
postgresql://user:pass@localhost:5433/dbname  # Old port
postgresql://user:pass@localhost:NEW_PORT/dbname  # New port

# MongoDB
mongodb://localhost:27018  # Old port
mongodb://localhost:NEW_PORT  # New port

# Neo4j
neo4j://localhost:7688  # Old port
neo4j://localhost:NEW_PORT  # New port
```

## Port Conflict Resolution

### Check for Conflicts

```bash
# Linux
sudo lsof -i :8880
sudo netstat -tlnp | grep 8880

# Check multiple ports
for port in 8880 8443 8081 5433 27018 7688; do
    echo "Port $port:"
    sudo lsof -i :$port 2>/dev/null || echo "  Free"
done
```

### Resolve Conflicts

```bash
# If port 8880 is in use, change to 8881
sed -i 's/TRAEFIK_HTTP_PORT=.*/TRAEFIK_HTTP_PORT=8881/' .env
sed -i 's/TRAEFIK_HTTP_PORT=.*/TRAEFIK_HTTP_PORT=8881/' traefik/.env

# Restart Traefik
make down-traefik && make up-traefik
```

## Port Mapping Details

### HTTP Services

Traefik routes HTTP requests based on Host header:

```yaml
# Traefik configuration
labels:
  - "traefik.http.routers.litellm.rule=Host(`litellm.local`)"
  - "traefik.http.routers.litellm.entrypoints=http"
  - "traefik.http.services.litellm.loadbalancer.server.port=8080"
```

**Flow:**
```
Client:8880 → Traefik:80 → litellm.local → litellm:8080
```

### TCP Services

Traefik routes TCP connections based on entry point:

```yaml
# Traefik configuration
labels:
  - "traefik.tcp.routers.postgres.rule=HostSNI(`*`)"
  - "traefik.tcp.routers.postgres.entrypoints=postgres"
  - "traefik.tcp.services.postgres.loadbalancer.server.port=5432"
```

**Flow:**
```
Client:5433 → Traefik:5433 (postgres entrypoint) → postgres:5432
```

## Testing Port Configuration

### Static Test (No Services Running)

```bash
# Check configuration files
bash scripts/e2e-traefik-static-test.sh

# Or using CLI
infra test static
```

### Runtime Test (Services Running)

```bash
# Test all ports
bash scripts/e2e-traefik-test.sh

# Or using CLI
infra test runtime
```

### Port Switching Test

```bash
# Test port switching
bash scripts/e2e-port-switching-test.sh

# Or using CLI
infra test ports
```

## Port Security

### Firewall Configuration

Only allow Traefik ports:

```bash
# UFW (Ubuntu)
sudo ufw allow 8880/tcp   # HTTP
sudo ufw allow 8443/tcp   # HTTPS
sudo ufw allow 8081/tcp   # Dashboard
sudo ufw allow 5433/tcp   # PostgreSQL
sudo ufw allow 27018/tcp  # MongoDB
sudo ufw allow 7688/tcp   # Neo4j

# Or using range
sudo ufw allow 8880:8885/tcp
```

### Non-Standard Ports

Use non-standard ports to reduce automated attacks:

```bash
# Instead of standard ports
TRAEFIK_HTTP_PORT=8880      # Not 80
TRAEFIK_HTTPS_PORT=8443     # Not 443
TRAEFIK_POSTGRES_PORT=5433  # Not 5432
TRAEFIK_MONGODB_PORT=27018  # Not 27017
TRAEFIK_NEO4J_PORT=7688     # Not 7687
```

## Port Reference

### Complete Port List

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAEFIK_HTTP_PORT` | 8880 | HTTP traffic |
| `TRAEFIK_HTTPS_PORT` | 8443 | HTTPS traffic |
| `TRAEFIK_DASHBOARD_PORT` | 8081 | Traefik dashboard |
| `TRAEFIK_POSTGRES_PORT` | 5433 | PostgreSQL TCP |
| `TRAEFIK_MONGODB_PORT` | 27018 | MongoDB TCP |
| `TRAEFIK_NEO4J_PORT` | 7688 | Neo4j Bolt TCP |

### Internal Ports (Fixed)

| Service | Internal Port | Configurable |
|---------|--------------|--------------|
| LiteLLM | 8080 | No |
| Orchestrator | 8080 | No |
| llama.cpp | 8080 | No |
| Langfuse | 3000 | No |
| Neo4j HTTP | 7474 | No |
| Neo4j Bolt | 7687 | No |
| PostgreSQL | 5432 | No |
| MongoDB | 27017 | No |
| MinIO API | 9000 | No |
| MinIO Console | 9001 | No |

## Troubleshooting

### Can't Access Service

```bash
# 1. Check port configuration
cat .env | grep TRAEFIK
cat traefik/.env | grep TRAEFIK

# 2. Verify Traefik is listening
sudo netstat -tlnp | grep traefik

# 3. Test connectivity
curl -v http://localhost:8880/health

# 4. Check Traefik logs
docker compose logs traefik
```

### Wrong Port Mapping

```bash
# Check actual port mapping
docker compose ps --format "table {{.Name}}\t{{.Ports}}"

# Should show:
# traefik  0.0.0.0:8880->80/tcp
# NOT:
# traefik  0.0.0.0:8080->80/tcp (if you configured 8880)
```

### Environment Variables Not Applied

```bash
# Restart Traefik to apply changes
make down-traefik && make up-traefik

# Or recreate container
docker compose -f docker-compose.traefik.yml up -d --force-recreate
```

## Best Practices

1. **Document custom ports** - Update team documentation
2. **Use environment variables** - Never hardcode ports
3. **Test port changes** - Use `infra test ports`
4. **Avoid privileged ports** - Use ports > 1024
5. **Be consistent** - Use same ports across environments
6. **Backup configuration** - Version control `.env.example`

## Quick Reference

### Change All Ports

```bash
# Quick port change script
cat > change-ports.sh << 'EOF'
#!/bin/bash
NEW_HTTP=${1:-9999}
NEW_DASH=${2:-9090}

sed -i "s/TRAEFIK_HTTP_PORT=.*/TRAEFIK_HTTP_PORT=$NEW_HTTP/" .env
sed -i "s/TRAEFIK_DASHBOARD_PORT=.*/TRAEFIK_DASHBOARD_PORT=$NEW_DASH/" .env
sed -i "s/TRAEFIK_HTTP_PORT=.*/TRAEFIK_HTTP_PORT=$NEW_HTTP/" traefik/.env
sed -i "s/TRAEFIK_DASHBOARD_PORT=.*/TRAEFIK_DASHBOARD_PORT=$NEW_DASH/" traefik/.env

make down-traefik && make up-traefik
echo "Ports changed to HTTP=$NEW_HTTP, Dashboard=$NEW_DASH"
EOF

chmod +x change-ports.sh
./change-ports.sh 9999 9090
```

### Verify Port Configuration

```bash
# One-liner to check all ports
echo "Port Configuration:"
grep "TRAEFIK_.*_PORT" .env traefik/.env
```

## Next Steps

- **[Configuration Guide](../getting-started/configuration.md)** - Full configuration
- **[Security Model](security.md)** - Security architecture
- **[Operations](../operations/index.md)** - Maintenance tasks
