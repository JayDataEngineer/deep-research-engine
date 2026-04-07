# Installation

## Prerequisites

- Python 3.11+
- Access to shared-docker-infra (for LiteLLM, databases)

## Install from PyPI (Recommended)

```bash
pip install deep-research-engine
```

### With Optional Dependencies

```bash
# All storage backends
pip install deep-research-engine[all-storage]

# Specific backends
pip install deep-research-engine[postgres]
pip install deep-research-engine[neo4j]
pip install deep-research-engine[mongodb]
pip install deep-research-engine[minio]

# Development
pip install deep-research-engine[dev]
```

## Install from Source

```bash
cd ~/Documents/programs/deep-research-engine

# With uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Verify Installation

```bash
# Check version
deep-research --version

# List available commands
deep-research --help
```

## Next Steps

- [Quick Start](quickstart.md) - Run your first workflow
- [Infrastructure Requirements](infrastructure.md) - Setup shared services
