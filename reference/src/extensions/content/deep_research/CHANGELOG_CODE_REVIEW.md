# Deep Research Extension - Code Review Changes

**Date**: 2026-02-14
**Reviewer**: Claude Code

## Summary

This document tracks the changes made based on the code review plan for the deep_research extension.

---

## P0 - Critical Bug Fixes

### 1. Fixed Scout Bug - Research Avenues Irrelevant to Query

**Problem**: Scout was generating completely irrelevant research avenues because the prompt was too generic.

**Root Cause**: In `graph/nodes/common.py:_plan_avenues()`, the prompt was:
```python
planning_prompt = f"""You are planning research avenues.

Input:
{input_data}

Generate 3-5 high-level RESEARCH AVENUES.
...
"""
```

**Fix**:
1. Updated `_plan_avenues()` to have a more explicit prompt that emphasizes the input IS the user's query
2. Added `planning_context` parameter to pass brand/use_case context
3. Updated `unified_flash_explorer_spawner()` to accept and pass `planning_context`
4. Updated `worker.py` to build and pass planning context from variant config

**Files Modified**:
- `graph/nodes/common.py` - Updated `_plan_avenues()` and `unified_flash_explorer_spawner()`
- `graph/nodes/worker.py` - Added `_build_planning_context()` helper and updated `worker_node()`

**Verification**: Query "Create 5 tweets about gym discipline" should now generate research avenues related to:
- Gym discipline topics
- Fitness motivation
- Content creation for fitness brand
NOT generic ML/research methodology topics

---

### 2. Integrated content_formatter into Synthesis Workflow

**Problem**: Synthesis produces raw output but content_formatter is never called, so `formatted_response` is always `None`.

**Root Cause**: Graph flow ended at critic → END without calling formatter.

**Fix**:
1. Created new `graph/nodes/formatter.py` node that:
   - Takes `final_response` (raw synthesis)
   - Gets output format from config (twitter, podcast, etc.)
   - Calls `content_formatter` extension
   - Stores result in `formatted_response`

2. Updated graph routing:
   - Changed `route_from_critic()` to route to `formatter` instead of END
   - Formatter routes to END after formatting

3. Updated graph factory:
   - Added `formatter_node` to graph
   - Added retry policy for formatter
   - Updated conditional edges to include "formatter"

**Files Created**:
- `graph/nodes/formatter.py` - New formatter node

**Files Modified**:
- `graph/nodes/__init__.py` - Export `formatter_node`
- `graph/graph_routing.py` - Route critic → formatter instead of critic → END
- `graph/graph_factory.py` - Add formatter node and edges
- `graph/graph.py` - Import `formatter_node`

**New Graph Flow**:
```
START → worker → boss → lab_architect → synthesis → critic → formatter → END
                                                            ↓
                                                          (iteration)
                                                            ↓
                                                          boss
```

**Verification**: After running with `output.mode: twitter`, `formatted_response` should contain the formatted tweets.

---

## Additional Bug Fixes Discovered

### 3. Fixed incorrect import path for get_extension_manager

**Problem**: Two files were importing `get_extension_manager` from wrong path.

**Wrong**: `from src.extensions.core.manager.loader import get_extension_manager`
**Correct**: `from src.api.core.extensions import get_extension_manager`

**Files Modified**:
- `graph/context_engine_integration.py` - Fixed import path
- `graph/output.py` - Fixed import path

---

## Known Infrastructure Limitation

**CLI Context Engine Issue**: Running deep research from CLI (`uv run manage.py deep_research research ...`) fails because the extension manager is not initialized. This is a pre-existing infrastructure limitation - the CLI doesn't initialize the extension manager, so Context Engine can't be accessed.

**Workaround**: Run research through the running server instead of CLI.

---

## P1 - High Priority

### 4. Split boss.py into smaller modules

**Status**: COMPLETED

Split 836-line boss.py into modular package:

```
boss/
├── __init__.py          # Re-exports boss_node
├── node.py              # Main boss_node function (~150 lines, orchestrator)
├── recursion.py         # Recursion limit handling
├── preset_graphs.py     # Profile preset handling + custom flows
├── fixed_graph.py       # Fixed worker graph mode
├── llm_decision.py      # LLM-based fallback decision logic
└── routing.py           # Command API routing decisions
```

Benefits:
- Each module has single responsibility
- Easier to test and maintain
- Clearer code organization
- Better separation of concerns

### 5. Move hardcoded values to config.yaml

**Status**: COMPLETED

Added Pydantic config models and updated code to use config values:

**New Pydantic Models** (config_validation.py):
- `RecursionHandlingConfig`: low_steps_threshold, critical_steps_threshold, force_completion_iterations
- `SynthesisConfig`: min_length, placeholder_patterns
- `CriticConfig`: synthesis_issue_keywords, research_issue_keywords

**Code Updates**:
- `boss.py`: Uses `spawner_config.recursion_handling.*` thresholds
- `synthesis.py`: Uses `spawner_config.synthesis.min_length` and `placeholder_patterns`
- `critic.py`: Uses `spawner_config.critic.synthesis_issue_keywords` and `research_issue_keywords`

**Example config.yaml usage**:
```yaml
spawner_config:
  recursion_handling:
    low_steps_threshold: 5
    critical_steps_threshold: 3
    force_completion_iterations: 2
  synthesis:
    min_length: 500
    placeholder_patterns:
      - "i'll help you curate"
      - "i'll search for"
  critic:
    synthesis_issue_keywords:
      - "structure"
      - "organization"
    research_issue_keywords:
      - "missing"
      - "incomplete"
```

---

## Files Changed Summary

| File | Change Type |
|------|-------------|
| `graph/nodes/common.py` | Modified - Scout bug fix |
| `graph/nodes/worker.py` | Modified - Added planning context |
| `graph/nodes/formatter.py` | Created - New formatter node |
| `graph/nodes/__init__.py` | Modified - Export formatter_node |
| `graph/graph_routing.py` | Modified - Route critic → formatter |
| `graph/graph_factory.py` | Modified - Add formatter node |
| `graph/graph.py` | Modified - Import formatter_node |
| `graph/context_engine_integration.py` | Modified - Fixed import path |
| `graph/output.py` | Modified - Fixed import path |
