# Operations

Operations and maintenance guide for running the Shared Docker Infrastructure.

## Operations Topics

### [CLI Reference](cli.md)

Python CLI tool for managing the infrastructure.

```bash
# Start services
infra core up
infra db up

# Check status
infra status

# View logs
infra logs
infra logs -f litellm

# Run tests
infra test all
```

**Documentation:** [CLI Reference](cli.md)

### [Makefile Commands](makefile.md)

Legacy Makefile commands (still supported).

```bash
make up              # Start all services
make down            # Stop all services
make status          # Check status
make logs            # View logs
make setup-secrets   # Generate secrets
make setup-hosts     # Configure /etc/hosts
```

**Documentation:** [Makefile Commands](makefile.md)

### [Monitoring](monitoring.md)

Health checks, metrics, and monitoring.

```bash
# Health checks
curl http://litellm.local/health
curl http://orchestrator.local/health
curl http://langfuse.local/api/health

# Resource usage
docker stats
curl http://orchestrator.local/vram

# Logs
infra logs -f
```

**Documentation:** [Monitoring](monitoring.md)

### [Backup & Recovery](backup.md)

Data backup strategies and recovery procedures.

```bash
# Backup databases
infra db backup

# Restore from backup
infra db restore
```

**Documentation:** [Backup & Recovery](backup.md)

### [Updates](updates.md)

Updating services and models.

```bash
# Update all services
git pull
docker compose pull
docker compose up -d

# Update specific service
docker compose pull litellm
docker compose up -d litellm
```

**Documentation:** [Updates](updates.md)

### [Security Best Practices](security.md)

Hardening and security guide.

- Use non-standard ports
- Enable authentication
- Configure firewall rules
- Regular updates
- Monitor logs
- Backup data

**Documentation:** [Security Best Practices](security.md)

## Daily Operations

### Morning Check

```bash
# Check all services
infra status

# Check resource usage
docker stats

# Check VRAM
curl http://orchestrator.local/vram

# Check errors in logs
infra logs | grep -i error
```

### Starting Services

```bash
# Start all
infra up

# Or start in order
infra core up
infra db up
infra up langfuse
infra up comfyui
```

### Stopping Services

```bash
# Stop all
infra down

# Stop specific category
infra core down
infra db down
```

### Viewing Logs

```bash
# All logs
infra logs

# Specific service
infra logs litellm

# Follow logs
infra logs -f orchestrator

# Recent logs
infra logs --tail 100
```

## Maintenance Tasks

### Weekly Tasks

- [ ] Check disk usage
- [ ] Review error logs
- [ ] Check VRAM usage patterns
- [ ] Review Langfuse analytics
- [ ] Backup databases

### Monthly Tasks

- [ ] Update services
- [ ] Review security configuration
- [ ] Clean up old logs
- [ ] Review resource allocation
- [ ] Test backup restoration

### Quarterly Tasks

- [ ] Full system audit
- [ ] Update documentation
- [ ] Review and optimize configuration
- [ ] Test disaster recovery
- [ ] Security review

## Troubleshooting Common Issues

### Service Not Starting

```bash
# Check status
infra status

# View logs
infra logs <service-name>

# Restart service
infra restart <service-name>

# Check resources
docker stats
```

### Out of VRAM

```bash
# Check VRAM usage
curl http://orchestrator.local/vram

# Check loaded models
curl http://orchestrator.local/status

# Unload models
curl -X POST http://orchestrator.local/unload/all
```

### Can't Access Service

```bash
# Check service is running
infra status

# Check Traefik routing
curl http://traefik.local:8081/api/http/routers

# Test connectivity
curl http://litellm.local/health
```

### Port Conflicts

```bash
# Check what's using port
sudo lsof -i :8880

# Change port in .env
sed -i 's/TRAEFIK_HTTP_PORT=.*/TRAEFIK_HTTP_PORT=8881/' .env

# Restart Traefik
infra restart traefik
```

## Performance Optimization

### Resource Allocation

```yaml
# Adjust in docker-compose files
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

### VRAM Management

```bash
# Monitor VRAM
watch -n 1 'curl -s http://orchestrator.local/vram | jq'

# Set thresholds
export ORCHESTRATOR_VRAM_THRESHOLD=2000
```

### Caching

- Keep frequently used models loaded
- Use CPU fallback for low-priority tasks
- Implement request caching in application

## Best Practices

1. **Monitor regularly** - Check status and logs daily
2. **Backup often** - Automated database backups
3. **Update carefully** - Test in staging first
4. **Document changes** - Keep runbooks updated
5. **Automate** - Use CLI for consistency
6. **Secure** - Follow security best practices
7. **Plan capacity** - Monitor resource trends

## Next Steps

- **[CLI Reference](cli.md)** - CLI commands
- **[Monitoring](monitoring.md)** - Monitoring guide
- **[Backup](backup.md)** - Backup procedures
- **[Security](security.md)** - Security practices
