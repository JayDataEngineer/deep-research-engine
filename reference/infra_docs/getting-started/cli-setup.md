# CLI Setup Guide

The Shared Docker Infrastructure includes a modern Python CLI tool built with UV for easy management of all services.

## Why Use the CLI?

The Python CLI (`infra`) provides:
- ✅ **Type-safe commands** - Better error messages and autocomplete
- ✅ **Cross-platform** - Works on Linux, macOS, and Windows
- ✅ **Modern interface** - Built with Typer and Rich for great UX
- ✅ **Extensible** - Easy to add new commands
- ✅ **Fast** - UV provides near-instant startup

## Installation

### Prerequisites

- **Python 3.11+**
- **UV** (fast Python package manager)

### Step 1: Install UV

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Step 2: Set Up CLI

```bash
cd ~/shared-docker-infra/scripts/cli

# Install dependencies and create virtual environment
uv sync

# Activate the environment
source .venv/bin/activate

# Or add to PATH
export PATH="$HOME/shared-docker-infra/scripts/cli/.venv/bin:$PATH"
```

### Step 3: Verify Installation

```bash
# Check CLI is working
infra --help

# See all commands
infra --help
```

## Usage

### Basic Commands

```bash
# Start all core services
infra core up

# Stop all services
infra down

# Check service status
infra status

# View logs
infra logs
infra logs litellm      # Specific service
infra logs -f llama     # Follow logs
```

### Service Management

```bash
# Core services (Traefik, LiteLLM, Orchestrator, Llama)
infra core up
infra core down

# Databases
infra db up
infra db down
infra db status

# TTS/ASR (via Orchestrator)
infra tts load kokoro
infra tts unload
infra asr load whisper
infra asr status

# Individual services
infra up traefik
infra up litellm
infra up orchestrator
```

### Testing

```bash
# Run all tests
infra test all

# Static analysis (no services needed)
infra test static

# Runtime tests (services must be running)
infra test runtime

# Port switching test
infra test ports
```

### Configuration

```bash
# Show current configuration
infra config show

# Show port configuration
infra config ports

# Set new ports
infra config set TRAEFIK_HTTP_PORT=9999

# Validate configuration
infra config validate
```

### Setup

```bash
# Generate secrets
infra setup secrets

# Configure /etc/hosts
infra setup hosts

# Create Docker network
infra setup network

# Full setup (all of the above)
infra setup all
```

## Command Reference

### `infra up`

Start services.

```bash
infra up                      # Start all
infra up traefik              # Start specific service
infra up traefik litellm      # Start multiple services
```

### `infra down`

Stop services.

```bash
infra down                    # Stop all
infra down traefik            # Stop specific service
```

### `infra status`

Show service status.

```bash
infra status                  # All services
infra status core             # Core services only
infra status --json           # JSON output
```

### `infra logs`

View service logs.

```bash
infra logs                    # All logs
infra logs litellm            # Specific service
infra logs -f                 # Follow logs
infra logs --tail 100         # Last 100 lines
infra logs --since 1h         # Last hour
```

### `infra core`

Manage core services.

```bash
infra core up                 # Start all core
infra core down               # Stop all core
infra core status             # Core service status
infra core restart            # Restart core services
```

### `infra db`

Manage databases.

```bash
infra db up                   # Start all databases
infra db down                 # Stop all databases
infra db status               # Database status
infra db backup               # Backup all databases
infra db restore              # Restore from backup
```

### `infra tts` / `infra asr`

Manage TTS/ASR models via Orchestrator.

```bash
infra tts load kokoro         # Load Kokoro TTS
infra tts unload              # Unload TTS
infra tts status              # Show TTS status

infra asr load whisper        # Load Whisper ASR
infra asr unload              # Unload ASR
infra asr status              # Show ASR status
```

### `infra test`

Run test suites.

```bash
infra test all                # Run all tests
infra test static             # Static analysis
infra test runtime            # Runtime tests
infra test ports              # Port switching test
```

### `infra config`

Manage configuration.

```bash
infra config show             # Show current config
infra config ports            # Show port config
infra config set KEY=value    # Set configuration
infra config validate         # Validate configuration
```

### `infra setup`

Setup infrastructure.

```bash
infra setup all               # Complete setup
infra setup secrets           # Generate secrets
infra setup hosts             # Configure /etc/hosts
infra setup network           # Create Docker network
```

## Advanced Usage

### Scripting with the CLI

```bash
#!/bin/bash

# Automated setup
infra setup all

# Start services
infra core up
infra db up

# Wait for services
sleep 10

# Verify
infra status

# Run tests
infra test static
```

### Using in Python Scripts

```python
import subprocess

# Run CLI commands
result = subprocess.run(
    ["infra", "status", "--json"],
    capture_output=True,
    text=True
)

status = json.loads(result.stdout)
print(status)
```

### Custom Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias infra-up='infra core up && infra db up'
alias infra-down='infra down'
alias infra-status='infra status'
alias infra-logs='infra logs -f'
```

## Troubleshooting

### Command Not Found

```bash
# Ensure virtual environment is activated
source ~/shared-docker-infra/scripts/cli/.venv/bin/activate

# Or add to PATH permanently
export PATH="$HOME/shared-docker-infra/scripts/cli/.venv/bin:$PATH"
```

### Permission Denied

```bash
# Ensure Docker permissions
sudo usermod -aG docker $USER
newgrp docker
```

### Dependencies Missing

```bash
# Reinstall dependencies
cd ~/shared-docker-infra/scripts/cli
uv sync --upgrade
```

## Migration from Makefile

| Makefile Command | CLI Equivalent |
|-----------------|----------------|
| `make up` | `infra up` |
| `make down` | `infra down` |
| `make status` | `infra status` |
| `make logs` | `infra logs` |
| `make up-traefik` | `infra up traefik` |
| `make setup-secrets` | `infra setup secrets` |
| `make setup-hosts` | `infra setup hosts` |
| `make test-traefik-static` | `infra test static` |

## Development

### Adding New Commands

Edit `scripts/cli/commands/`:

```python
# commands/mycommand.py
import typer

app = typer.Typer()

@app.command()
def my_feature():
    """My new feature"""
    typer.echo("Hello from my feature!")
```

Register in `main.py`:

```python
from commands.mycommand import app as mycommand_app

app.add_typer(mycommand_app, name="mycommand")
```

### Running Tests

```bash
cd ~/shared-docker-infra/scripts/cli

# Run CLI tests
pytest

# Run with coverage
pytest --cov=cli
```

## Next Steps

- **[Quick Start](quickstart.md)** - Start using the CLI
- **[Operations](../operations/index.md)** - Learn all operations
- **[API Reference](../api-reference/index.md)** - Explore the APIs

## Resources

- [Typer Documentation](https://typer.tiangolo.com/)
- [UV Documentation](https://docs.astral.sh/uv/)
- [Rich Documentation](https://rich.readthedocs.io/)
