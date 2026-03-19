# Observability System Fixes Applied

**Date:** 2026-02-16
**Component:** deep_research/observability

## Summary of Changes

### Critical Fixes

#### 1. Added Missing `set_tool_calls` Method (`run_logger.py`)
- **Issue:** `callbacks.py:451` called `self.run_logger.set_tool_calls()` but the method didn't exist
- **Fix:** Added `set_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None` method to RunLogger class

#### 2. Fixed `from_config` Classmethod Indentation (`run_logger.py`)
- **Issue:** The `@classmethod` decorator and `from_config` method were defined outside the class (lines 430-443)
- **Fix:** Properly indented the method to be inside the RunLogger class

#### 3. Fixed Route Path Conflict (`output_md.py`)
- **Issue:** `/output_md/runs` endpoint would never match because `/{session_id}` pattern captures it first
- **Fix:** Changed route from `/runs` to `/list/runs`

#### 4. Fixed Backward Compatibility Imports (`__init__.py`)
- **Issue:** `get_run_logger` and `with_run_logger` were only imported in fallback mode
- **Fix:** Always import factory functions from `core.py` for backward compatibility

### Medium Priority Fixes

#### 5. Thread Safety for File Writes (`file_handler.py`)
- **Issue:** Concurrent writes could cause data corruption
- **Fix:** Added `threading.Lock()` and wrapped `write_to_log()` and `write_to_llm_calls_log()` with lock acquisition

#### 6. SSE Reconnection Support (`routes.py`)
- **Issue:** SSE stream had no event IDs for reconnection recovery
- **Fix:** Added `event_id` counter and `format_sse()` helper to include `id:` field in SSE messages
- **Enhanced:** Improved error events to include `error_type`, `thread_id`, and `recoverable` flag

#### 7. Removed Hardcoded Paths (`output_md.py`)
- **Issue:** Used hardcoded `"data/outputs/logs/server"` path
- **Fix:** Now uses `LOGS_DIR` constant from `src.constants` with fallback

#### 8. Improved Exception Handling (`output_md.py`)
- **Issue:** Silent exception swallowing in `_find_run_dir()` and `list_runs()`
- **Fix:** Added logging for `JSONDecodeError` and other exceptions

## Files Modified

| File | Changes |
|------|---------|
| `observability/run_logger.py` | Added `set_tool_calls()`, fixed `from_config` indentation |
| `observability/file_handler.py` | Added thread lock for concurrent writes |
| `observability/output_md.py` | Fixed route path, use constants, improved error handling |
| `observability/__init__.py` | Fixed backward compatibility imports |
| `routes.py` | SSE event IDs, enhanced error responses |

## Verification

All fixes verified with Python imports and assertions:
```
✓ All observability imports successful
✓ RunLogger.set_tool_calls works
✓ RunLogger.from_config works
✓ FileHandler has _write_lock
✓ DiagnosticLogger.track_llm_response works
```

## API Changes

### Breaking Changes
- **Route changed:** `/api/v1/output_md/runs` → `/api/v1/output_md/list/runs`
  - Clients listing runs should update to use new endpoint

### New SSE Event Format
```
id: 0
event: start
data: {"thread_id": "research_abc123", "status": "starting"}
```

Error events now include more context:
```json
{
  "error": "Exception message",
  "error_type": "ValueError",
  "thread_id": "research_abc123",
  "recoverable": true
}
```
