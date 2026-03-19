# Troubleshooting

Common issues and solutions for the Shared Docker Infrastructure.

## Troubleshooting Topics

### [Connection Issues](connection.md)

Network and port problems.

**Common Issues:**
- Can't access services
- Port conflicts
- DNS resolution failures
- Network connectivity problems

**Documentation:** [Connection Issues](connection.md)

### [VRAM Management](vram.md)

Out of memory errors and GPU issues.

**Common Issues:**
- Insufficient VRAM errors
- Model loading failures
- GPU not detected
- Memory leaks

**Documentation:** [VRAM Management](vram.md)

### [Service Failures](services.md)

Debugging failed services.

**Common Issues:**
- Service won't start
- Service crashes
- Health check failures
- Container restarts

**Documentation:** [Service Failures](services.md)

### [Traefik Issues](traefik.md)

Proxy and routing problems.

**Common Issues:**
- 404 errors
- 502 Bad Gateway
- Routing failures
- SSL/TLS issues

**Documentation:** [Traefik Issues](traefik.md)

### [FAQ](faq.md)

Frequently asked questions.

**Topics:**
- General questions
- Configuration questions
- Usage questions
- Best practices

**Documentation:** [FAQ](faq.md)

## Quick Troubleshooting

### Service Won't Start

```bash
# Check status
infra status

# View logs
infra logs <service-name>

# Check resources
docker stats

# Restart
infra restart <service-name>
```

### Can't Access Service

```bash
# 1. Check service is running
infra status

# 2. Check Traefik routing
curl http://traefik.local:8081/api/http/routers

# 3. Test health
curl http://litellm.local/health

# 4. Check logs
infra logs traefik
infra logs <service-name>
```

### Port Conflict

```bash
# Find what's using port
sudo lsof -i :8880

# Change port
sed -i 's/TRAEFIK_HTTP_PORT=.*/TRAEFIK_HTTP_PORT=8881/' .env

# Restart
infra restart traefik
```

### Out of VRAM

```bash
# Check VRAM
curl http://orchestrator.local/vram

# Check loaded models
curl http://orchestrator.local/status

# Unload models
curl -X POST http://orchestrator.local/unload/all

# Reload with smaller model
curl -X POST http://orchestrator.local/load/asr/whisper-cpu
```

## Diagnostic Commands

### System Status

```bash
# All services
infra status

# Docker containers
docker compose ps

# Docker network
docker network inspect shared-infra

# Resource usage
docker stats
```

### Network Diagnostics

```bash
# Check Traefik routing
curl http://traefik.local:8081/api/http/routers
curl http://traefik.local:8081/api/tcp/routers

# Test service health
curl http://litellm.local/health
curl http://orchestrator.local/health

# Check DNS resolution
docker compose exec traefik nslookup litellm
```

### Log Analysis

```bash
# View all logs
infra logs

# Specific service
infra logs litellm

# Follow logs
infra logs -f orchestrator

# Search for errors
infra logs | grep -i error
infra logs | grep -i exception
```

### Resource Monitoring

```bash
# VRAM usage
curl http://orchestrator.local/vram

# Full status
curl http://orchestrator.local/status

# Docker resources
docker stats

# GPU usage
nvidia-smi
```

## Common Error Messages

### "Connection refused"

**Cause:** Service not running or wrong port

**Solution:**
```bash
# Check service status
infra status

# Check port configuration
cat .env | grep TRAEFIK

# Restart service
infra restart <service-name>
```

### "502 Bad Gateway"

**Cause:** Backend service unhealthy

**Solution:**
```bash
# Check backend health
curl http://litellm.local/health

# Check logs
infra logs litellm

# Restart backend
infra restart litellm
```

### "Insufficient VRAM"

**Cause:** Not enough GPU memory

**Solution:**
```bash
# Check VRAM
curl http://orchestrator.local/vram

# Unload models
curl -X POST http://orchestrator.local/unload/all

# Use CPU fallback
curl -X POST http://orchestrator.local/load/asr/whisper-cpu
```

### "Model not found"

**Cause:** Model not configured or not loaded

**Solution:**
```bash
# List available models
curl http://litellm.local/v1/models

# Check Orchestrator status
curl http://orchestrator.local/status

# Load model
curl -X POST http://orchestrator.local/load/tts/kokoro
```

### "404 Not Found"

**Cause:** Wrong domain or routing issue

**Solution:**
```bash
# Check /etc/hosts
grep "local" /etc/hosts

# Check Traefik routers
curl http://traefik.local:8081/api/http/routers

# Verify domain
curl -v http://litellm.local/health
```

## Getting Help

### Before Asking for Help

1. Check the [FAQ](faq.md)
2. Review relevant documentation
3. Run diagnostic commands
4. Check logs for errors
5. Search existing issues

### Providing Information

When asking for help, include:

1. **Problem description** - What you're trying to do
2. **Error messages** - Full error output
3. **Logs** - Relevant log excerpts
4. **Configuration** - Relevant .env settings
5. **Steps taken** - What you've tried

### Resources

- **Documentation:** [Index](../index.md)
- **GitHub Issues:** [Open an issue](https://github.com/your-org/shared-docker-infra/issues)
- **Logs:** `infra logs`
- **Status:** `infra status`

## Next Steps

- **[Connection Issues](connection.md)** - Network problems
- **[VRAM Management](vram.md)** - Memory issues
- **[Service Failures](services.md)** - Service debugging
- **[Traefik Issues](traefik.md)** - Proxy problems
- **[FAQ](faq.md)** - Common questions
