# Logging System Refactor - Verification Report

**Date**: February 2026
**Extension**: deep_research
**Status**: Complete

## Summary

The logging system for the deep_research extension has been refactored to be well-organized, DRY (Don't Repeat Yourself), and comprehensive with proper timing/profiling support.

Additionally, the observability module (14 files, 6,000+ lines) has been consolidated into 4 focused modules with backward compatibility stubs.

## Part 1: Logging System Improvements

### 1. Centralized Logging Utilities (`logging_utils.py`)

**Added timing/profiling helpers:**
- `OperationTimer` - Dataclass for tracking operation duration with checkpoints
- `time_operation()` - Async context manager for timing operations
- `log_timing_breakdown()` - Function to log timing breakdowns
- `ProfilingLogger` - Class for profiling complex multi-step operations

**Example output:**
```
[web_research] completed in 2.50s (search_done: 1.00s, scrape_done: 1.50s)
```

### 2. Flash Explorer Refactor (`agents/flash_explorer.py`)

**Removed duplicate logging code:**
- Removed `_get_log_level()` - duplicated functionality in `logging_utils.py`
- Removed `_should_log()` - duplicated functionality
- Removed `_log_message()` - replaced with direct logger calls
- Removed `_run_logger_global` - unnecessary global state

**Now uses:**
```python
from ..logging_utils import get_logger, log_timing_breakdown
logger = get_logger(__name__, "flash_explorer")
```

### 3. FlashExplorerLogger Simplification (`agents/flash_explorer_logger.py`)

**Simplified to:**
- Delegates to central logging for console/file output
- Maintains `_write_to_file()` for run-specific observability logging
- Removed duplicate `_should_log()` and `_get_level()` methods

### 4. Web Agent Timing (`src/shared/tools/ulli_langchain/web/web_agent.py`)

**Added comprehensive timing:**
- **Simple research mode**: Tracks `search_searxng`, `scrape_firecrawl`, `persistence`, `total`
- **Investigative mode**: Tracks `graph_execution`, `total`
- **Individual steps**: Search and scrape timing logged per step

**Example log output:**
```
[WEB_AGENT] [simple_research] timing - total: 4.52s (search_searxng: 1.23s, scrape_firecrawl: 2.87s, persistence: 0.42s)
[WEB_AGENT] Search completed in 1.23s, found 10 URLs
[WEB_AGENT] Scraping completed in 2.87s, 5/5 successful
```

### 5. Print Statements Converted (`graph/state_models.py`)

**Before:**
```python
print(f"[STATE] Invalid citation data: {str(data)[:200]}... Error: {e}")
```

**After:**
```python
logger.error(f"[STATE] Invalid citation data: {str(data)[:200]}... Error: {e}")
```

## Logging Configuration

The logging system respects `config.yaml` settings:

```yaml
logging:
  level: normal  # Options: normal, quiet, debug, silent

  components:
    flash_explorer: normal  # Flash Explorer agent logs
    lab_workers: quiet      # Reduce noise from parallel workers
    web_search: quiet       # Web search tools
    deep_agent: quiet       # Deep agent sub-calls
    llm_calls: quiet        # LLM call logging
```

## Usage Examples

### Basic Logging
```python
from .logging_utils import get_logger

logger = get_logger(__name__)  # Uses global level
logger.info("Operation completed")
```

### Component-Specific Logging
```python
logger = get_logger(__name__, "flash_explorer")  # Uses component level
```

### Timing Operations
```python
from .logging_utils import time_operation

async with time_operation(logger, "web_research") as timer:
    results = await search(query)
    timer.add_checkpoint("search_done")
    content = await scrape(urls)
    timer.add_checkpoint("scrape_done")
# Logs: [web_research] completed in 2.50s (search_done: 1.00s, scrape_done: 1.50s)
```

### Timing Breakdown
```python
from .logging_utils import log_timing_breakdown

timings = {"search": 1.2, "scrape": 3.5, "synthesize": 0.8}
log_timing_breakdown(logger, "web_research", timings)
# Logs: [web_research] timing breakdown - total: 5.50s, search: 1.20s, scrape: 3.50s, synthesize: 0.80s
```

## Files Modified

| File | Changes |
|------|---------|
| `logging_utils.py` | Added timing helpers (OperationTimer, time_operation, ProfilingLogger) |
| `agents/flash_explorer.py` | Removed duplicate logging, uses centralized logger |
| `agents/flash_explorer_logger.py` | Simplified to use central logging |
| `graph/state_models.py` | Converted print() to logger.error() |
| `web/web_agent.py` | Added comprehensive timing for search/scrape/total |

## Acceptable print() Usage

The following files intentionally use `print()` for user-facing output:
- `cli.py` - CLI user interaction
- `config_diff.py` - CLI comparison output
- `scripts/visualize_graphs.py` - Script visualization output
- `observability/migrations/*.py` - Migration scripts
- `graph/graph_utilities.py` - Time-travel debugging utility

These are acceptable because they are CLI/script tools where print() is the intended output method.

## Part 2: Observability Module Consolidation

### Before (14 files, 6,000+ lines)
```
observability/
├── run_logger.py (971 lines)
├── run_logger_output.py (776 lines)
├── run_database.py (709 lines)
├── metrics_tracker.py (656 lines)
├── checkpointer.py (810 lines)
├── helpers.py (551 lines)
├── run_logger_models.py
├── run_logger_metadata.py
├── run_logger_serialization.py
├── llm_calls.py
├── callbacks.py
├── file_handler.py
├── middleware.py
├── routes.py
└── migrations/
```

### After (5 focused modules, 2,500 lines)

```
observability/
├── models.py            # Data models, metadata extraction, serialization (360 lines)
├── core.py              # RunLogger, FileHandler, MetricsTracker (643 lines)
├── callbacks.py         # SpawnerCallbackHandler, ObservabilityCallback (771 lines)
├── run_logger_output.py # Markdown output generation (581 lines)
├── routes.py            # FastAPI endpoints (23 lines)
├── __init__.py          # Public API (74 lines)
├── extension.py         # Extension definition (48 lines)
└── migrations/          # Database migrations
```

| Module | Purpose | Lines | Content |
|--------|---------|-------|---------|
| `models.py` | Data models | 360 | RunMetrics, WorkerMetrics, metadata extraction, serialization |
| `core.py` | Core logging | 643 | RunLogger, MetricsTracker, FileHandler, FileHandlerFactory |
| `callbacks.py` | Callback handlers | 771 | SpawnerCallbackHandler, ObservabilityCallback |
| `run_logger_output.py` | Output | 581 | Markdown generation (charts removed) |
| `routes.py` | API routes | 23 | FastAPI endpoints |
| `__init__.py` | Public API | 74 | Clean exports, documentation |

### Public API (from `__init__.py`)

```python
from deep_research.observability import (
    # Core logging
    RunLogger,
    get_run_logger,
    with_run_logger,
    # Data models
    RunMetrics,
    WorkerMetrics,
    # Metadata extraction
    extract_provider_from_base_url,
    extract_llm_metadata,
    # Callbacks
    ObservabilityCallback,
    create_observability_callback,
    SpawnerCallbackHandler,
    create_callback_handler,
)
```

### Files Removed

- `run_logger.py` - Consolidated into core.py and models.py
- `run_logger_models.py` - Consolidated into models.py
- `run_logger_metadata.py` - Consolidated into models.py
- `run_logger_serialization.py` - Consolidated into models.py
- `file_handler.py` - Consolidated into core.py
- `metrics_tracker.py` - Consolidated into core.py
- `middleware.py` - Merged into callbacks.py
- `run_database.py` - Removed (unused)
- `output_md.py` - Removed (unused)
- `helpers.py` - Removed (unused)
- `common.py` - Removed (unused)
- `llm_calls.py` - Removed (unused)
- `checkpointer.py` - Removed (unused)

### New Files Created

- `models.py` - Split from core.py to keep file sizes manageable (360 lines)

### Chart Generation Removed

Chart generation (matplotlib) was removed from `run_logger_output.py` for simplicity:
- No more `include_images` parameter (ignored if passed)
- Faster output generation
- No matplotlib dependency required

## Verification

Run a deep research operation and verify:
1. Component-specific log levels are respected
2. Timing breakdowns appear for web research operations
3. Search (SearXNG) timing is logged
4. Scrape (Firecrawl) timing is logged
5. Total operation timing is logged
6. Import from `deep_research.observability` works correctly

## Part 3: DRY Code Fixes (February 2026)

### 3.1 Centralized Connection Strings

**Problem:** Connection strings were hardcoded in 10+ files across the extension.

**Solution:** Added centralized constants to `constants.py`:

```python
# constants.py
DEFAULT_POSTGRES_CONNECTION_STRING = os.getenv(
    "CONTEXT_ENGINE_POSTGRES_CONNECTION_STRING",
    "postgresql://ulli:ulli_password@localhost:5433/ulli_db"
)
DEFAULT_NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
DEFAULT_NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j_test_password")
DEFAULT_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
DEFAULT_MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
DEFAULT_MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
DEFAULT_MINIO_BUCKET = os.getenv("MINIO_BUCKET", "context-engine")
```

**Files Updated:**
- `graph/context_engine_integration.py`
- `observability/checkpointer.py`
- `observability/migrations/add_llm_calls_postgres.py`
- `observability/migrations/add_node_execution.py`
- `sessions/research_sessions.py` (3 instances)

### 3.2 Standardized Brief Type (TypedDict Pattern)

**Problem:** ResearchBrief was handled as either dict OR Pydantic model, leading to repetitive `isinstance(brief, dict)` checks throughout boss.py (5+ locations).

**Solution:** Use LangGraph's official TypedDict pattern:
1. State schema: `research_brief: Optional[Dict[str, Any]]` (not Pydantic)
2. Validate with Pydantic at output boundary only (in `_synthesize_results`)
3. Return dict for state (via `.model_dump()`)
4. Access via `.get()` throughout: `brief.get("original_query")`

**Before (NON-DRY):**
```python
original_query = brief.get('original_query') if isinstance(brief, dict) else (brief.original_query if brief else state.get('query', ''))
complexity = brief.get('complexity_estimate') if isinstance(brief, dict) else (brief.complexity_estimate if brief else None)
```

**After (TypedDict Pattern):**
```python
# State definition (TypedDict)
class GlobalState(TypedDict):
    research_brief: Optional[Dict[str, Any]]  # Dict, not Pydantic

# Node access (pure dict flow)
brief = state.get("research_brief") or {}
original_query = brief.get("original_query", state.get("query", ""))
complexity = brief.get("complexity_estimate", 0.5)
```

**Why this is the pro pattern:**
- LangGraph checkpoints serialize to dict anyway
- TypedDict provides type safety without runtime overhead
- No `isinstance()` checks needed anywhere
- Pydantic validation only at output boundaries

**Files Updated:**
- `graph/state.py` - Changed `research_brief: Optional[ResearchBrief]` to `Optional[Dict[str, Any]]`
- `graph/nodes/common.py` - Removed `ensure_research_brief()`, `_synthesize_results()` returns dict
- `graph/nodes/boss.py` - Pure dict access via `.get()`
- `graph/nodes/critic.py` - Pure dict access via `.get()`
- `graph/nodes/synthesis.py` - Pure dict access via `.get()`

### 3.3 Configurable Recursion Limit

**Problem:** `recursion_limit` was hardcoded to 600 in `flash_explorer.py`.

**Solution:**
1. Added `flash_recursion_limit` to `config.yaml` (under `spawner_config`)
2. Updated default in `flash_explorer.py` to match config default (50)

**config.yaml:**
```yaml
spawner_config:
  # Flash Explorer Recursion Limit
  # Maximum LLM calls per Flash Explorer worker before stopping
  flash_recursion_limit: 50
```

## Part 4: Bugfixes (February 2026)

### 4.1 Docker Helper Path Fix

**Problem:** `docker_helper.py` used a fragile `.parent` chain that went one level too far, looking for `docker-compose.lang-tools.yml` in the project root instead of `backend/`.

**Solution:** Use centralized `BACKEND_DIR` constant from `constants.py`.

**Files Updated:**
- `utils/docker_helper.py` - Changed from 7 `.parent` calls to use `BACKEND_DIR` constant

### 4.2 Missing RunLogger Methods

**Problem:** Decorators in `graph/decorators.py` called `log_phase_start()`, `log_phase_end()`, and `log_error()` on `RunLogger`, but these methods didn't exist after the observability module consolidation.

**Solution:** Added the three missing methods to `RunLogger` class in `observability/core.py`:
- `log_phase_start(phase, details)` - Log phase start with optional metadata
- `log_phase_end(phase, elapsed, details)` - Log phase completion with timing
- `log_error(phase, error, context)` - Log errors with context

### 4.3 Model Slot Debug Logging Level

**Problem:** Model slot resolution debug info was logged at WARNING level, causing noise in logs.

**Solution:** Changed from `logger.warning()` to `logger.debug()` in `graph/helpers/llm.py`.

### 4.4 RabbitMQ Healthcheck for Khepri Migration

**Problem:** Firecrawl container repeatedly restarted because RabbitMQ's `check_running` healthcheck passed before the internal Mnesia→Khepri migration completed.

**Solution:** Updated `docker-compose.lang-tools.yml`:
- Changed healthcheck from `check_running` to `status` (more comprehensive)
- Increased `start_period` from 5s to 30s
- Increased `interval` from 5s to 10s
- Increased `retries` from 3 to 5

### Summary of Part 4 Changes

| Issue | Files Changed |
|-------|---------------|
| Docker path fix | `utils/docker_helper.py`, `constants.py` |
| Missing RunLogger methods | `observability/core.py` |
| Model slot log level | `graph/helpers/llm.py` |
| RabbitMQ healthcheck | `docker-compose.lang-tools.yml` |

### Summary of Changes

| Issue | Files Changed | Lines Removed | Lines Added |
|-------|---------------|---------------|-------------|
| Connection strings | 6 files | ~30 | ~15 |
| Brief type handling | 3 files | ~50 | ~20 |
| Recursion limit | 2 files | 2 | 10 |
| **Total** | **11 files** | **~82** | **~45** |

## Part 5: Config Module Refactor (February 2026)

### 5.1 Modular Config Package

**Problem:** `helpers/config.py` (691 lines) handled too many concerns, violating Single Responsibility Principle:
- Context variables (run_logger, yaml_config, context_engine)
- Config caching with file modification tracking
- YAML loading with multi-level hierarchy
- Deep merge utilities
- Environment variable expansion
- Validation wrapper
- Overdrive mode transformation
- Runtime config conversion

**Solution:** Split into focused modules:

```
helpers/config/
├── __init__.py     # Re-exports all public functions (backward compatible)
├── constants.py    # Env var caches (~20 lines)
├── context.py      # Context variables (~120 lines)
├── loader.py       # YAML loading, caching, merging (~340 lines)
├── modes.py        # Overdrive mode transformation (~70 lines)
└── runtime.py      # ResearchRuntimeConfig conversion (~120 lines)
```

### Module Responsibilities

| Module | Purpose | Lines |
|--------|---------|-------|
| `constants.py` | Environment variable caches (`_LLM_MODEL_NAME`, etc.) | 20 |
| `context.py` | Context variables for thread-safe data passing | 120 |
| `loader.py` | YAML loading, caching, merging, validation | 340 |
| `modes.py` | Overdrive mode transformation | 70 |
| `runtime.py` | Runtime config utilities (yaml_to_runtime_config) | 120 |
| `__init__.py` | Public API re-exports | 75 |

### Public API Preserved

All existing imports continue to work without modification:

```python
from ...helpers.config import (
    # Constants
    _LLM_MODEL_NAME,
    # Context functions
    get_run_logger_from_config,
    set_run_logger_context,
    reset_run_logger_context,
    set_yaml_config_context,
    get_yaml_config_from_context,
    reset_yaml_config_context,
    set_context_engine_context,
    get_context_engine_from_context,
    reset_context_engine_context,
    # Loader functions
    _load_yaml_config,
    _apply_config_overrides,
    _expand_env_vars,
    # Mode transformations
    _apply_overdrive_mode,
    # Runtime utilities
    yaml_to_runtime_config,
    get_runtime_config,
    get_config_from_runnable_config,
)
```

### Files Modified

| File | Action |
|------|--------|
| `helpers/config.py` | Deleted (replaced with package) |
| `helpers/config/__init__.py` | Created - re-export all public API |
| `helpers/config/constants.py` | Created - env var caches |
| `helpers/config/context.py` | Created - context variables |
| `helpers/config/loader.py` | Created - YAML loading, caching, merging |
| `helpers/config/modes.py` | Created - overdrive mode |
| `helpers/config/runtime.py` | Created - runtime config utilities |

### No Changes Required

These files continue to work without modification:
- `helpers/__init__.py` - imports `config` as module reference
- `helpers/llm.py` - imports `_LLM_MODEL_NAME` (re-exported by `__init__.py`)
- `graph/graph.py` - imports from `.helpers.config`
- `graph/nodes/common.py` - imports from `.helpers.config`
- All test files - imports remain valid

### Summary of Part 5 Changes

| Issue | Files Changed | Lines Before | Lines After |
|-------|---------------|--------------|-------------|
| Config module split | 6 new + 1 deleted | 691 | ~755 (better organized) |

## Part 6: Logging Gaps and Consistency Fixes (February 2026)

### 6.1 Logging Constants and Naming Conventions

**Problem:** Logging used inconsistent naming patterns across files.

**Solution:** Added centralized constants to `logging_utils.py`:

```python
# Component names - use these for get_logger(name, COMPONENT_X)
COMPONENT_SCOUT = "worker"
COMPONENT_BOSS = "boss"
COMPONENT_LAB = "lab"
COMPONENT_LAB_ARCHITECT = "lab_architect"
COMPONENT_CRITIC = "critic"
COMPONENT_SYNTHESIS = "synthesis"
COMPONENT_FLASH_EXPLORER = "flash_explorer"
COMPONENT_WEB_AGENT = "web_agent"
COMPONENT_CONTEXT_ENGINE = "context_engine"

# Log message prefixes - ALWAYS uppercase with brackets
PREFIX_SCOUT = "[SCOUT]"
PREFIX_BOSS = "[BOSS]"
# ... etc

# Helper functions
def format_timing(elapsed: float) -> str
def format_timing_breakdown(elapsed: float, breakdown: Dict) -> str
def log_phase_start(logger, prefix, phase_details)
def log_phase_end(logger, prefix, elapsed, details)
def log_error_with_context(logger, prefix, error, context)
```

### 6.2 LLM Call Timing

**Problem:** LLM calls lacked timing information, making performance analysis difficult.

**Solution:** Added timing around all LLM invocations:

**boss.py:**
- Guidance LLM calls (2 locations)
- Main decision LLM call
- Log format: `[BOSS] Guidance LLM call: 2.35s`

**synthesis.py:**
- First LLM call with tools
- Second LLM call (with tool results)
- Log format: `[SYNTHESIS] First LLM call: 3.12s`

### 6.3 Web Operation Timing

**Problem:** Web deep agent calls lacked timing breakdowns.

**Solution:** Added timing to flash_explorer.py:

```python
web_start = time.time()
web_findings, web_raw_sources, web_artifact_ids = await _simple_research(...)
web_elapsed = time.time() - web_start

logger.info(f"[EXPLORER] web_deep_agent completed: {len(web_findings)} chars, {len(web_raw_sources)} sources in {web_elapsed:.2f}s")
```

Also added timing data to `deep_agent_result`:
```python
"timing": {"web": web_elapsed, "pdf": pdf_elapsed, "total": total_elapsed}
```

### 6.4 Error Handling Improvements

**Problem:** Exception handlers didn't log exception type, making debugging harder.

**Solution:** Updated all exception handlers to include `type(e).__name__`:

**Before:**
```python
except Exception as e:
    logger.error(f"[BOSS] Guidance generation failed: {e}")
```

**After:**
```python
except Exception as e:
    logger.error(f"[BOSS] Guidance generation failed: {type(e).__name__}: {e}")
```

**Files Updated:**
- `graph/nodes/boss.py` (2 handlers)
- `graph/nodes/synthesis.py` (1 handler)
- `agents/flash_explorer.py` (4 handlers)

### 6.5 Progress Logging for Long Operations

**Problem:** Long-running operations lacked progress indicators.

**Solution:** Added detailed timing breakdown to `unified_flash_explorer_spawner`:

```python
[UNIFIED] Planned 3 avenues in 0.45s
[UNIFIED] Flash Explorers: 3/3 successful in 12.34s
[UNIFIED] Spawner complete: total=15.67s (planning=0.45s, explorers=12.34s, synthesis=2.88s)
```

### Summary of Part 6 Changes

| Issue | Files Changed | Impact |
|-------|---------------|--------|
| Logging constants | `logging_utils.py` | Centralized naming |
| LLM timing | `boss.py`, `synthesis.py` | Performance visibility |
| Web timing | `flash_explorer.py` | Operation breakdown |
| Error handling | 3 files, 7 handlers | Better debugging |
| Progress logging | `common.py` | Long operation visibility |

## Part 7: Print Statement Removal from Graph Nodes (February 2026)

### 7.1 Problem: Mixed Logging in Research Nodes

**Problem:** Graph nodes used both `run_logger._write_to_file()` AND `print()` for the same messages, causing:
- Duplicated output to console
- Inconsistent logging patterns
- Violation of DRY observability principles

**Affected Files:**
- `graph/nodes/unified/critic.py` - 1 print statement
- `graph/nodes/unified/synthesis.py` - 4 print statements
- `graph/graph_utilities.py` - 5 print statements in `print_execution_tree()`
- `config_startup_validator.py` - 1 print statement

### 7.2 Solution: Use DRY Observability System

All graph nodes now use the centralized observability system:

**critic.py - Before:**
```python
run_logger._write_to_file(f"[UNIFIED_CRITIC] Max iterations ({max_iterations}) reached, proceeding to formatter")
print(f"[CRITIC] Max iterations ({max_iterations}) reached, returning to formatter")
```

**critic.py - After:**
```python
run_logger._write_to_file(f"[UNIFIED_CRITIC] Max iterations ({max_iterations}) reached, proceeding to formatter")
```

**synthesis.py - Before:**
```python
print(f"[SYNTHESIS] output_config: {output_config}, output_mode: {output_mode}")
run_logger._write_to_file(f"[SYNTHESIS] output_config: {output_config}, output_mode: {output_mode}")
```

**synthesis.py - After:**
```python
run_logger._write_to_file(f"[SYNTHESIS] output_config: {output_config}, output_mode: {output_mode}")
```

**graph_utilities.py - Before:**
```python
print(f"\n[TIME_TRAVEL] Execution Tree for thread_id=...")
print("=" * 60)
```

**graph_utilities.py - After:**
```python
logger.info(f"[TIME_TRAVEL] Execution Tree for thread_id=...")
logger.info("=" * 60)
```

**config_startup_validator.py - Before:**
```python
print(f"[CONFIG] Validated {config_name} successfully")
```

**config_startup_validator.py - After:**
```python
logger.info(f"[CONFIG] Validated {config_name} successfully")
```

### 7.3 Acceptable Print Usage Preserved

The following files intentionally retain `print()` for user-facing CLI output:
- `cli.py` - CLI user interaction and results display
- `config_diff.py` - CLI comparison output
- `observability/migrations/*.py` - One-time migration scripts

These are acceptable because they are CLI/script tools where `print()` is the intended output method.

### Summary of Part 7 Changes

| File | Print Statements Removed | Replacement |
|------|--------------------------|-------------|
| `graph/nodes/unified/critic.py` | 1 | `run_logger._write_to_file()` (already present) |
| `graph/nodes/unified/synthesis.py` | 4 | `run_logger._write_to_file()` (already present) |
| `graph/graph_utilities.py` | 5 | `logger.info()` |
| `config_startup_validator.py` | 1 | `logger.info()` |
| **Total** | **11** | **Observability system** |
