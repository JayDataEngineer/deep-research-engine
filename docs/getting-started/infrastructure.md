# Infrastructure Requirements

This engine integrates with your existing [shared-docker-infra](../../shared-docker-infra/).

## Required Services

| Service | Purpose | Required For |
|---------|---------|--------------|
| **LiteLLM** | LLM inference gateway | All workflows |
| **Postgres** | Vector storage + embeddings | RAG, Telegram |
| **Neo4j** | Knowledge graph | RAG, Telegram |
| **MongoDB** | Media files | Telegram |
| **MinIO** | Large file storage | Telegram |

## Start Infrastructure

```bash
cd ~/Documents/programs/shared-docker-infra

# Start core services (required for all workflows)
make core-up

# Start databases (depends on workflow)
make db-up
```

## Service Endpoints

| Service | Domain | Purpose |
|---------|--------|---------|
| LiteLLM | `litellm.local` | All LLM inference |
| Postgres | `localhost:5433` | Vector storage |
| Neo4j | `localhost:7687` | Knowledge graph |
| MongoDB | `localhost:27017` | Media files |
| MinIO | `localhost:9000` | Object storage |

See [shared-docker-infra/docs/](../../shared-docker-infra/docs/) for full documentation.

## Environment Variables

Set these before running workflows:

```bash
# LiteLLM
export LITELLM_BASE_URL=http://litellm.local/v1
export LITELLM_API_KEY=sk-litellm-master

# Postgres
export CONTEXT_POSTGRES_URL=postgresql://postgres:password@localhost:5433/deep_research

# Neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=your_password

# MongoDB
export MONGO_URL=mongodb://localhost:27017/telegram

# MinIO
export MINIO_ENDPOINT=http://localhost:9000
export MINIO_ACCESS_KEY=minio
export MINIO_SECRET_KEY=minio_password
```

## Workflow-Specific Requirements

| Workflow | LiteLLM | Postgres | Neo4j | MongoDB | MinIO |
|----------|---------|----------|-------|---------|-------|
| RAG Research | ✓ | ✓ | ✓ | - | - |
| PDF Summarizer | ✓ | - | - | - | - |
| Telegram Ingest | ✓ | ✓ | ✓ | ✓ | ✓ |
| Skill Generator | ✓ | - | - | - | - |

## Verify Infrastructure

```bash
# Test LiteLLM
curl http://litellm.local/v1/models

# Test Postgres
psql -h localhost -p 5433 -U postgres -c "SELECT 1"

# Test Neo4j
cypher-shell -u neo4j -p your_password "RETURN 1"

# Test MongoDB
mongosh mongodb://localhost:27017 --eval "db.runCommand({ping: 1})"

# Test MinIO
curl http://localhost:9000/minio/health/live
```

## Troubleshooting

See [shared-docker-infra/docs/troubleshooting/](../../shared-docker-infra/docs/troubleshooting/) for infrastructure issues.
