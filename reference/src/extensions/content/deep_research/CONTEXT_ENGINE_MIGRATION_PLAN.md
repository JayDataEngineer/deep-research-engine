# Deep Research → Context Engine Migration Plan

**Principle**: Context logic should be handled in context_engine. Deep research sees only a clean set of tools and functions.

**Goal**: Eliminate all direct storage, database, and context logic from deep_research, delegating everything to context_engine.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DEEP RESEARCH                               │
│                    (Pure Orchestration Layer)                       │
│                                                                      │
│   - Research agents (worker, boss, lab, critic, synthesis)          │
│   - Graph flow and state management                                 │
│   - LLM orchestration                                               │
│   - Uses ONLY tools and functions from context_engine               │
│                                                                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            │ Clean API (tools + functions)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CONTEXT ENGINE                               │
│                    (All Storage & Context)                          │
│                                                                      │
│   Storage Backends:                                                  │
│   - PostgresVectorStore (artifacts, semantic search)                │
│   - Neo4jStore (knowledge graph)                                    │
│   - S3Backend/MinIO (large files, logs)                             │
│   - SQLiteBackend (grid search, local cache)                        │
│                                                                      │
│   Services:                                                          │
│   - SessionManager (research sessions)                              │
│   - ObservabilityService (logging, metrics)                         │
│   - ArtifactManager (findings, citations, entities)                 │
│   - ToolRegistry (centralized tool access)                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Migration Phases

### Phase 1: Observability System (HIGH PRIORITY)
**Estimated Effort**: 2-3 files, ~500 lines changed

#### Files to Modify/Delete

| File | Action | Risk |
|------|--------|------|
| `observability/core.py` | Refactor | Medium |
| `observability/file_handler.py` | Refactor | Medium |
| `observability/run_logger.py` | Refactor | Medium |
| `observability/metrics_tracker.py` | Keep (data models only) | Low |

#### What Changes

**Current Pattern (BAD):**
```python
# observability/file_handler.py
class FileHandler:
    def open_log_files(self):
        self._log_file = open(self.run_dir / "app.log", "a", encoding="utf-8")

    def write_to_log(self, message: str):
        self._log_file.write(f"[{timestamp}] {message}\n")
        self._log_file.flush()
```

**Target Pattern (GOOD):**
```python
# context_engine/tools/observability.py
@tool
def log_observation(
    session_id: str,
    log_type: str,  # "app", "llm_calls", "metrics"
    content: str,
    metadata: Optional[Dict] = None
) -> str:
    """Log an observation to context engine storage.

    Args:
        session_id: Research session ID
        log_type: Type of log (app, llm_calls, metrics, diagnostics)
        content: Log content
        metadata: Optional metadata

    Returns:
        Path where log was stored
    """
    path = f"/logs/{session_id}/{log_type}.log"
    context_engine.s3_backend.append(path, content)
    return path
```

**Deep Research Usage:**
```python
# observability/core.py - RunLogger
from src.shared.tools.registry import get_tool

log_observation = get_tool("log_observation")

class RunLogger:
    def _write_to_file(self, message: str):
        log_observation(
            session_id=self.session_id,
            log_type="app",
            content=message
        )
```

#### New Context Engine Tools to Create

1. **`log_observation`** - Write log entries
2. **`get_observations`** - Read log entries (for debugging)
3. **`save_metrics`** - Save run metrics as artifact
4. **`get_run_history`** - Get historical runs for a session

#### Migration Steps

1. Create `src/extensions/features/tools/context_engine/tools/observability.py`
2. Add tools to central registry
3. Refactor `FileHandler` to use context_engine tools
4. Update `RunLogger` to use `log_observation` tool
5. Keep file-based fallback for offline/dev mode

---

### Phase 2: Session Management (HIGH PRIORITY)
**Estimated Effort**: 1-2 files, ~300 lines changed

#### Files to Modify/Delete

| File | Action | Risk |
|------|--------|------|
| `sessions/research_sessions.py` | Refactor | High |
| `graph/context_engine_integration.py` | Update | Medium |

#### What Changes

**Current Pattern (BAD):**
```python
# sessions/research_sessions.py
import asyncpg

class SessionManager:
    async def _ensure_pool(self):
        self.pool = await asyncpg.create_pool(self.connection_string)

    async def create_session(self, ...):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO research_sessions (...)
                VALUES (...)
            """)
```

**Target Pattern (GOOD):**
```python
# context_engine/services/session_manager.py
class SessionManager:
    """Session management via context_engine."""

    async def create_session(
        self,
        query: str,
        use_case: str,
        config: Dict[str, Any]
    ) -> ResearchSession:
        session = ResearchSession(...)
        await self.context_engine.postgres_store.add_session(session)
        return session

# Deep research uses tool:
@tool
def create_research_session(
    query: str,
    use_case: str = "rag",
    variant: str = "default",
    config: Optional[Dict] = None
) -> str:
    """Create a new research session."""
    session = await session_manager.create_session(query, use_case, config)
    return session.session_id
```

#### New Context Engine Tools to Create

1. **`create_research_session`** - Start a new session
2. **`get_session_status`** - Get current session status
3. **`update_session_status`** - Update status (running, completed, failed)
4. **`list_sessions`** - List sessions for a user/query
5. **`rollback_session`** - Clean up a failed session

#### Migration Steps

1. Create `context_engine/services/session_manager.py`
2. Move session logic from `research_sessions.py`
3. Create session tools in context_engine
4. Update deep_research to use session tools
5. Keep `ResearchSession` dataclass in deep_research (it's just a model)

---

### Phase 3: Artifact Tools Refactoring (MEDIUM PRIORITY)
**Estimated Effort**: 1 file, ~200 lines changed

#### Files to Modify/Delete

| File | Action | Risk |
|------|--------|------|
| `tools/artifact_tools.py` | Refactor | Medium |

#### What Changes

**Current Pattern (BAD):**
```python
# tools/artifact_tools.py
@tool
def save_worker_artifact(...):
    from src.extensions.features.tools.context_engine.storage import S3Backend
    from src.extensions.features.tools.context_engine.artifacts import FindingArtifact

    # Creates its own S3Backend connection
    s3_backend = S3Backend(
        bucket=os.getenv("MINIO_BUCKET", "context-engine"),
        endpoint_url=os.getenv("MINIO_ENDPOINT", "..."),
        ...
    )
```

**Target Pattern (GOOD):**
```python
# tools/artifact_tools.py - Just wraps context_engine tools
@tool
def save_worker_artifact(
    content: str,
    filename: str,
    worker_id: Optional[str] = None,
    ...
) -> str:
    """Save research findings to Context Engine.

    This tool wraps context_engine's artifact storage.
    All actual storage logic is in context_engine.
    """
    # Get context_engine tool from registry
    from src.shared.tools.registry import get_tool
    store_artifact = get_tool("store_artifact")

    return store_artifact(
        artifact_type="finding",
        content=content,
        metadata={"worker_id": worker_id, "filename": filename, ...}
    )
```

#### Migration Steps

1. Remove direct `S3Backend` and `PostgresVectorStore` imports
2. Use context_engine tools via registry
3. Keep Pydantic validation schemas (good pattern)
4. Simplify to thin wrapper around context_engine

---

### Phase 4: Grid Search Storage (LOW PRIORITY)
**Estimated Effort**: 1 file, ~150 lines changed

#### Files to Modify/Delete

| File | Action | Risk |
|------|--------|------|
| `tools/grid_search.py` | Refactor | Low |

#### What Changes

**Current Pattern (BAD):**
```python
# tools/grid_search.py
import sqlite3

class GridSearchDB:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS grid_results (...)
        """)
```

**Target Pattern (GOOD):**
```python
# context_engine/storage/sqlite_backend.py
class SQLiteBackend(StorageBackend):
    """SQLite storage for local/grid data."""

    def execute(self, query: str, params: tuple = ()):
        ...

# tools/grid_search.py - Uses context_engine
@tool
def store_grid_result(
    run_id: str,
    config_hash: str,
    metrics: Dict[str, float]
) -> str:
    """Store grid search result via context_engine."""
    context_engine.sqlite_backend.execute(
        "INSERT INTO grid_results ...",
        (run_id, config_hash, json.dumps(metrics))
    )
```

#### Migration Steps

1. Create `SQLiteBackend` in context_engine (optional, low priority)
2. Refactor `grid_search.py` to use context_engine storage
3. Or: Keep SQLite local (it's genuinely local data, not context)

---

### Phase 5: Routes & Config Files (LOW PRIORITY)
**Estimated Effort**: 1 file, ~50 lines changed

#### Files to Modify/Delete

| File | Action | Risk |
|------|--------|------|
| `routes.py` | Refactor | Low |
| `config_diff.py` | Minor updates | Low |

#### What Changes

**Current Pattern (BAD):**
```python
# routes.py
@router.post("/config/save")
async def save_config(...):
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
```

**Target Pattern (GOOD):**
```python
# routes.py
@router.post("/config/save")
async def save_config(...):
    from src.shared.tools.registry import get_tool
    save_config_file = get_tool("save_config_file")
    await save_config_file(config_name, config)
```

#### Migration Steps

1. Create `save_config_file` tool in context_engine
2. Refactor routes to use tools
3. Keep for later (low priority)

---

## New Context Engine Components

### 1. ObservabilityService
Location: `src/extensions/features/tools/context_engine/services/observability.py`

```python
class ObservabilityService:
    """Centralized observability via context_engine."""

    def __init__(self, context_engine: ContextEngineBackends):
        self.context_engine = context_engine

    async def log(
        self,
        session_id: str,
        log_type: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Log an observation."""
        path = f"/logs/{session_id}/{log_type}.log"
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}] {content}\n"
        self.context_engine.s3_backend.append(path, entry)
        return path

    async def save_metrics(
        self,
        session_id: str,
        metrics: Dict[str, Any]
    ) -> str:
        """Save run metrics as artifact."""
        artifact = MetricsArtifact(
            id=f"metrics_{session_id}",
            content=json.dumps(metrics, indent=2),
            session_id=session_id,
            ...
        )
        await self.context_engine.add_artifact(artifact)
        return artifact.id
```

### 2. SessionService
Location: `src/extensions/features/tools/context_engine/services/session.py`

```python
class SessionService:
    """Session management via context_engine."""

    async def create_session(
        self,
        query: str,
        use_case: str,
        variant: str,
        config: Dict[str, Any]
    ) -> ResearchSession:
        """Create a new research session."""
        session = ResearchSession(
            session_id=str(uuid4()),
            query=query,
            use_case=use_case,
            variant=variant,
            config=config,
            status=SessionStatus.CREATED,
        )
        await self.context_engine.postgres_store.add_session(session)
        return session

    async def update_status(
        self,
        session_id: str,
        status: SessionStatus,
        error: Optional[str] = None
    ) -> None:
        """Update session status."""
        await self.context_engine.postgres_store.update_session(
            session_id,
            {"status": status.value, "error": error}
        )
```

### 3. New Tools in Central Registry

Add to `src/shared/tools/registry.py`:

```python
# Observability tools
TOOL_CATALOG["log_observation"] = log_observation
TOOL_CATALOG["save_metrics"] = save_metrics
TOOL_CATALOG["get_run_history"] = get_run_history

# Session tools
TOOL_CATALOG["create_research_session"] = create_research_session
TOOL_CATALOG["get_session_status"] = get_session_status
TOOL_CATALOG["update_session_status"] = update_session_status
```

---

## Migration Order (Recommended)

```
Week 1: Phase 1 - Observability System
├── Create observability tools in context_engine
├── Refactor FileHandler
└── Update RunLogger

Week 2: Phase 2 - Session Management
├── Create SessionService in context_engine
├── Create session tools
└── Update deep_research to use session tools

Week 3: Phase 3 - Artifact Tools
├── Remove direct backend imports
└── Use context_engine tools via registry

Week 4: Phase 4 & 5 - Cleanup
├── Grid search (optional)
└── Routes (optional)
```

---

## Files Summary

### Files to DELETE
| File | Phase | Reason |
|------|-------|--------|
| (none - all refactored) | - | - |

### Files to CREATE in context_engine
| File | Purpose |
|------|---------|
| `services/observability.py` | ObservabilityService class |
| `services/session.py` | SessionService class |
| `tools/observability.py` | log_observation, save_metrics tools |
| `tools/session.py` | create_session, update_status tools |
| `storage/sqlite_backend.py` | SQLite backend (optional) |

### Files to REFACTOR in deep_research
| File | Phase | Changes |
|------|-------|---------|
| `observability/core.py` | 1 | Use log_observation tool |
| `observability/file_handler.py` | 1 | Use context_engine S3Backend |
| `observability/run_logger.py` | 1 | Use observability tools |
| `sessions/research_sessions.py` | 2 | Use session tools |
| `tools/artifact_tools.py` | 3 | Remove direct backend imports |
| `tools/grid_search.py` | 4 | Use context_engine storage (optional) |
| `routes.py` | 5 | Use config tools (optional) |

---

## Verification Checklist

After each phase:

- [ ] All imports from `deep_research` to storage backends removed
- [ ] All storage operations go through context_engine tools
- [ ] Unit tests pass
- [ ] Integration test: `uv run backend/manage.py research "test" --use-case general`
- [ ] Verify artifacts in Postgres
- [ ] Verify logs in MinIO/S3Backend
- [ ] Verify session management works

---

## Risk Mitigation

1. **Keep fallbacks**: File-based logging as fallback when context_engine unavailable
2. **Gradual migration**: One phase at a time, verify after each
3. **Backward compatibility**: Keep old function signatures with deprecation warnings
4. **Testing**: Run full research flow after each phase

---

## Success Criteria

After complete migration:

1. **No direct storage imports** in deep_research
2. **All context via tools**: Deep research only uses tools from central registry
3. **Single source of truth**: Context engine owns all storage
4. **Clean separation**: Deep research = orchestration, Context engine = storage
5. **Observable**: All logs and metrics go through context_engine
