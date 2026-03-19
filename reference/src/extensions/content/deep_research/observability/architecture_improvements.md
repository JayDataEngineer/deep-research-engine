# Architecture Improvements Log

**Date**: 2026-02-14
**Summary**: Implemented 5 architectural improvements for deep_research, context_engine, and content_formatter extensions.

## Changes Implemented

### 1. Model Selection Config (Improvement 4)

**Files Modified:**
- `backend/src/extensions/content/deep_research/config.yaml`
- `backend/src/extensions/content/deep_research/graph/context_engine_entities.py`
- `backend/src/extensions/content/deep_research/graph/context_engine_integration.py`
- `backend/src/extensions/features/tools/context_engine/extension.py`

**Changes:**
- Added `entity_extraction_model` and `graph_query_model` config options to `context_engine` section in config.yaml
- Models can be overridden via environment variables:
  - `CONTEXT_ENGINE_ENTITY_MODEL` - Model for entity extraction
  - `CONTEXT_ENGINE_GRAPH_MODEL` - Model for graph queries
- Updated `extract_and_store_entities()` to accept `yaml_config` parameter and read model from config
- Updated `get_context_for_synthesis()` to read model from config
- Updated `natural_language_query()` in extension.py to read model from config

**Default Model:** `GLM-4.7-Flash-UD-Q4_K_XL`

### 2. Extension DI for content_formatter (Improvement 5)

**Files Modified:**
- `backend/src/extensions/content/deep_research/graph/output.py`

**Changes:**
- Updated `_format_via_content_formatter()` to use extension manager for dependency injection
- Added proper error handling with crash-only philosophy (no silent fallbacks)
- Raises `RuntimeError` if extension not available instead of returning original response

### 3. Retry Logic in content_formatter (Improvement 2)

**Files Modified:**
- `backend/src/extensions/content/content_formatter/base.py`

**Changes:**
- Implemented actual retry loop with `max_retries` parameter
- Added `_format_validation_errors()` helper to extract readable error details from Pydantic ValidationError
- Added `_add_retry_feedback()` helper to append validation feedback to prompt
- Retry loop includes simple backoff (0.5s * attempt)
- Crash-only: Raises `RuntimeError` with detailed error info after all retries fail

### 4. Deduplicate Context Engine Initialization (Improvement 1)

**Files Modified:**
- `backend/src/extensions/content/deep_research/graph/context_engine_integration.py`

**Changes:**
- Removed inline `ContextEngineBackends` class
- Updated `initialize_context_engine_for_research()` to get extension from extension manager
- Extension manager provides the initialized `ContextEngineExtension` instance
- Reduced code duplication by reusing extension infrastructure

### 5. Fix Synthesis Tool Execution (Improvement 3)

**Files Modified:**
- `backend/src/extensions/content/deep_research/graph/nodes/synthesis.py`

**Changes:**
- Fixed MockRuntime to get context_engine from context variable (not from state)
- Added proper crash-only error if context engine not available
- The tools expect `runtime.state.get("_context_engine")` which now correctly receives the context engine from the context variable

## Verification

All imports verified successfully:
```
✓ context_engine_entities imports correctly
✓ context_engine_integration imports correctly
✓ output.py imports correctly
✓ content_formatter/base.py imports correctly
✓ synthesis.py imports correctly
✓ context_engine/extension.py imports correctly
```

## Configuration Example

```yaml
# In deep_research/config.yaml
context_engine:
  enabled: true
  mode: research
  isolation: session

  # Model selection (with environment variable override)
  entity_extraction_model: ${CONTEXT_ENGINE_ENTITY_MODEL-GLM-4.7-Flash-UD-Q4_K_XL}
  graph_query_model: ${CONTEXT_ENGINE_GRAPH_MODEL-GLM-4.7-Flash-UD-Q4_K_XL}
```

## Crash-Only Philosophy Maintained

All changes follow the crash-only philosophy:
- No silent fallbacks
- Errors are propagated immediately
- RuntimeError raised with detailed error messages
- Validation errors are surfaced, not hidden
