# Deep Research - Legacy Code Cleanup Report

**Date**: 2026-02-17
**Status**: Post Phase 1-5 Migration, identifying remaining legacy patterns

---

## Summary of Findings

After completing Phases 1-5 of the context_engine migration, the following legacy patterns remain:

### 1. **Session Management - ACTIVE USE** (HIGH PRIORITY)

**File**: `sessions/research_sessions.py`
- **Direct asyncpg connections**: Lines 35, 83, 99
- **Direct PostgresVectorStore usage**: Lines 644, 827
- **Direct S3Backend usage**: Lines 695, 697, 871, 873

**Active Usage**:
- `graph/graph.py:309`: `session_manager = await get_session_manager()`
- `graph/graph.py:311-318`: Uses `SessionContext` context manager

**Issue**: We created session tools in `context_engine/tools/session.py`, but `graph/graph.py` still uses the old `SessionManager` class.

**Resolution Options**:
1. **Recommended**: Refactor `graph/graph.py` to use context_engine session tools via registry
2. Alternative: Update `research_sessions.py` to use context_engine tools internally

---

### 2. **Observability Models - Direct File I/O** (MEDIUM PRIORITY)

**File**: `observability/models.py`
- **Line 304**: `with open(worker_outputs_file, 'w')` - Direct file write
- **Line 348**: `with open(raw_sources_file, 'w')` - Direct file write

**Functions**:
- `save_worker_outputs_json()` - Should use `save_output` tool
- `save_raw_sources_json()` - Should use `save_output` tool

---

### 3. **Flash Explorer Logger - Direct File I/O** (MEDIUM PRIORITY)

**File**: `agents/flash_explorer_logger.py`
- **Line 59**: `with open(self._log_file, "a", encoding="utf-8")` - Direct file write

**Issue**: This bypasses the refactored FileHandler

---

### 4. **Grid Search - Local SQLite** (ACCEPTABLE)

**File**: `tools/grid_search.py`
- **Lines 104, 155, 176, 198, 216**: Direct `sqlite3.connect()` calls

**Assessment**: **This is acceptable** - SQLite is appropriate for local grid search experiment data. This is NOT shared context, so it doesn't need to be in context_engine.

**Already Refactored**: Config and result storage now use context_engine S3Backend via `_save_to_context_engine()`

---

### 5. **Migration Scripts - One-time Use** (ACCEPTABLE)

**Files**:
- `observability/migrations/add_node_execution.py`
- `observability/migrations/add_llm_calls_postgres.py`

**Assessment**: **These are acceptable** - They're one-time migration scripts, not production code.

---

### 6. **Acceptable Direct File I/O**

These files use direct file I/O for appropriate reasons:

| File | Pattern | Reason |
|------|---------|--------|
| `logging_utils.py` | Config file reads | Internal config loading |
| `__init__.py` | Config file reads | Loading research configs |
| `storage/citations.py` | `Path(path).stem` | Path manipulation, not I/O |

---

## Recommended Next Steps

### Option A: Full Migration (Recommended)

1. **Refactor `graph/graph.py`** to use context_engine session tools
2. **Update `observability/models.py`** to use `save_output` tool
3. **Update `agents/flash_explorer_logger.py`** to use refactored FileHandler
4. **Deprecate or delete `sessions/research_sessions.py`**

### Option B: Minimal Migration

1. **Keep `sessions/research_sessions.py`** but update it to use context_engine tools internally
2. Update `observability/models.py` for file I/O
3. Update `agents/flash_explorer_logger.py` for file I/O

---

## Files to Modify

| Priority | File | Current | Target |
|----------|------|---------|--------|
| HIGH | `graph/graph.py` | Uses SessionManager | Use context_engine tools |
| HIGH | `sessions/research_sessions.py` | Direct DB connections | Deprecate or refactor |
| MEDIUM | `observability/models.py` | Direct file writes | Use save_output tool |
| MEDIUM | `agents/flash_explorer_logger.py` | Direct file writes | Use FileHandler |

---

## Detailed Analysis by File

### sessions/research_sessions.py

**Current State**:
- 988 lines of session management code
- Direct asyncpg database connections
- Direct PostgresVectorStore and S3Backend usage
- SessionManager class with lifecycle management

**Created in Phase 2**:
- `context_engine/tools/session.py` with `@tool` functions:
  - `create_research_session`
  - `get_session_status`
  - `update_session_status`
  - `list_sessions`
  - `cleanup_session`

**Gap**: The tools are for LLM invocation, but `graph/graph.py` needs synchronous/async class usage.

**Resolution**: Either:
1. Create a `SessionService` class in context_engine (non-tool)
2. Refactor `graph/graph.py` to invoke tools via registry
3. Update `research_sessions.py` to use context_engine tools internally

### observability/models.py

**Functions to refactor**:

```python
# CURRENT (bad)
def save_worker_outputs_json(run_dir, run_id, outputs_data, write_func):
    worker_outputs_file = run_dir / "worker_outputs.json"
    with open(worker_outputs_file, 'w') as f:
        json.dump({...}, f)

# TARGET (good)
def save_worker_outputs_json(session_id, outputs_data):
    from src.shared.tools.registry import get_tool
    save_output = get_tool("save_output")
    save_output.invoke({
        "session_id": session_id,
        "output_type": "worker_outputs",
        "content": json.dumps(outputs_data),
        "filename": "worker_outputs.json"
    })
```

### graph/graph.py

**Current code**:
```python
# Lines 309-318
session_manager = await get_session_manager()
async with SessionContext(
    manager=session_manager,
    query=query,
    use_case=use_case,
    variant=variant,
    run_id=run_logger.run_id,
    config=yaml_config,
) as session:
    session_id = session.session_id
    thread_id = session.thread_id
```

**Target code**:
```python
from src.shared.tools.registry import get_tool

create_session = get_tool("create_research_session")
result = await create_session.ainvoke({
    "query": query,
    "use_case": use_case,
    "variant": variant,
    "run_id": run_logger.run_id,
    "config": yaml_config
})
session_data = json.loads(result)
session_id = session_data["session_id"]
thread_id = session_data["thread_id"]
```

---

## Verification After Cleanup

Run these checks after cleanup:

```bash
# 1. Check for remaining direct DB connections
grep -r "import asyncpg" backend/src/extensions/content/deep_research --exclude-dir=".venv"

# 2. Check for direct S3Backend/PostgresVectorStore
grep -r "from.*storage import.*Backend" backend/src/extensions/content/deep_research --exclude-dir=".venv"

# 3. Check for direct file I/O (excluding acceptable cases)
grep -r "with open(" backend/src/extensions/content/deep_research --exclude-dir=".venv" | grep -v "__pycache__" | grep -v "migrations"

# 4. Run test
uv run backend/manage.py research "Test query" --use-case general --variant quick_test
```

---

## Conclusion

**Phases 1-5 completed successfully**:
- ✅ Observability tools created and FileHandler refactored
- ✅ Session tools created in context_engine
- ✅ Artifact tools use context_engine via registry
- ✅ Grid search configs use context_engine storage
- ✅ Routes use context_engine for config storage

**Remaining work**:
- ❌ `graph/graph.py` still uses old SessionManager
- ❌ `sessions/research_sessions.py` has direct DB connections
- ❌ `observability/models.py` has direct file I/O
- ❌ `agents/flash_explorer_logger.py` has direct file I/O

**Recommendation**: Complete the migration for `graph/graph.py` and deprecate `research_sessions.py` to fully achieve the architecture goal.
