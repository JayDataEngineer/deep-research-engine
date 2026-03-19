# DRY Refactor Plan - Deep Research

**Goal:** Reduce unnecessary complexity by consolidating duplicated patterns.

**Principle:** Only fix what's actually causing pain. Don't over-engineer solutions.

---

## Boss Prompt Fix (2026-03-08)

### Problem
The Boss prompts still referenced the old "deep" vs "shallow" decision pattern, but the code had been updated to only spawn workers. The `BossDecision` schema only has:
- `workers_to_spawn: List[str]` - which workers to spawn
- `reasoning: str` - why these workers

### Root Cause
The `boss.guidance` prompt was being used for TWO different purposes:
1. **Main boss node (`boss.py`)**: Deciding which workers to spawn
2. **Preset mode (`profiles/base.py`)**: Providing contextual guidance to workers

These are fundamentally different operations!

### Solution
Split the prompts into two distinct purposes:
- **`boss.spawn`**: For main boss node to decide which workers to spawn
- **`boss.guidance`**: For preset mode to provide contextual guidance to workers

**Files Updated:**
- `prompts/base.yaml`:
  - `boss` prompt - now says "decide which workers to spawn"
  - `boss.structured` prompt - removed "decision: standard/deep" language
  - `boss.spawn` - NEW: For main boss node to decide which workers to spawn
  - `boss.guidance` - RESTORED: For preset mode to provide guidance to workers
  - `extraction.boss_decision` - removed "standard" vs "deep" language

- `config.yaml`:
  - `boss.structured` prompt - removed "decision: standard/deep" language
  - `boss.spawn` - NEW: For main boss node
  - `boss.guidance` - RESTORED: For preset mode

- `graph/nodes/unified/boss.py`:
  - Changed from `boss.guidance` to `boss.spawn`
  - Updated fallback prompt to match new logic

- `graph/helpers/prompt_registry.py`:
  - Added `boss.spawn` variable declaration

- `config_schema_strict.py`:
  - Added `boss.spawn` to required prompts

### Key Changes
- Boss ALWAYS spawns workers (default: ["worker"])
- Boss does NOT decide between "deep" and "shallow"
- Critic decides when research is sufficient
- For simple queries: spawn 1 worker
- For complex queries: spawn 2-3 workers
- `boss.spawn` and `boss.guidance` are now separate prompts with different purposes

---

## ZAI Provider Routing Fix (2026-03-09) - ARCHIVED

> **Note (2026-03-14):** This fix is now obsolete. All model routing is handled by LiteLLM proxy.
> See `src/shared/litellm_factory.py` for the current LLM configuration.

### Original Problem
The `zai` provider was not in the `pattern_routed_providers` set, causing "Provider 'zai' not found in API key database" error. The ZAI credentials are stored in `.env` (ZAI_API_KEY) and routed via `backends.yaml`, not the API key database.

### Original Solution
Added `zai` to `pattern_routed_providers` in `graph/helpers/llm.py`:
```python
pattern_routed_providers = {"llamacpp", "localhost", "zai"}
provider_param = None if provider in pattern_routed_providers else provider
```

### Current Solution (2026-03-14)
All LLM calls now go through LiteLLM proxy at `http://litellm.local/v1`. Model routing is configured in LiteLLM's `config.yaml`, not in the application code.

```python
from src.shared import get_llm

# All LLM calls are automatically routed by LiteLLM
llm = await get_llm(model="zai-glm-4-7")  # Routes to ZAI Cloud
llm = await get_llm(model="qwen-35-27")   # Routes to local llama.cpp
```

---

## Phase 1: Logging Helpers (HIGH IMPACT, LOW EFFORT)

### Problem
30+ occurrences of `run_logger._write_to_file(f"[COMPONENT] message")` pattern.

### Solution
Add component-aware logging helper to `logging_utils.py`:

```python
def log_component(run_logger, component: str, message: str) -> None:
    """Log with automatic component prefix."""
    if run_logger:
        run_logger._write_to_file(f"[{component.upper()}] {message}")
```

### Files to Update
- `graph/nodes/unified/boss.py` (~8 calls)
- `graph/nodes/unified/critic.py` (~6 calls)
- `graph/nodes/unified/synthesis.py` (~10 calls)
- `tools/sub_worker.py` (~15 calls)

### Migration Pattern
```python
# Before
run_logger._write_to_file(f"[UNIFIED_BOSS] Query: {query[:100]}...")

# After
log_component(run_logger, "BOSS", f"Query: {query[:100]}...")
```

---

## Phase 2: Extract Pydantic Helper (MEDIUM IMPACT, LOW EFFORT)

### Problem
All 3 unified nodes use `result["extracted"]` after `draft_extract()`.

### Solution
Add helper to `graph/nodes/unified/base.py`:

```python
async def draft_extract_pydantic(
    query: str,
    system_prompt: str,
    llm: Any,
    output_schema: Type[BaseModel],
    **kwargs
) -> BaseModel:
    """Returns extracted Pydantic model directly (not dict)."""
    result = await draft_extract(query, system_prompt, llm, output_schema, **kwargs)
    return result["extracted"]
```

### Files to Update
- `graph/nodes/unified/boss.py:147` - `decision = result["extracted"]`
- `graph/nodes/unified/critic.py:204` - `evaluation = result["extracted"]`
- `graph/nodes/unified/synthesis.py:209` - `output = result["extracted"]`

### Migration Pattern
```python
# Before
result = await draft_extract(query, prompt, llm, BossDecision)
decision = result["extracted"]

# After
decision = await draft_extract_pydantic(query, prompt, llm, BossDecision)
```

---

## Phase 3: Prompt Fallback Helper (MEDIUM IMPACT, MEDIUM EFFORT)

### Problem
Duplicated pattern for "template not found, using fallback":

```python
template = get_prompt_from_config(yaml_config, "template_name")
if not template or contains_placeholder(template):
    run_logger._write_to_file("[COMPONENT] WARNING: template not found")
    template = "fallback hardcoded..."
```

### Solution
Add to `graph/helpers/prompt_registry.py`:

```python
def get_prompt_with_fallback(
    yaml_config: Dict[str, Any],
    template_name: str,
    fallback: str,
    run_logger: Any = None,
    component: str = "UNKNOWN"
) -> str:
    """Get prompt template with fallback if not found or invalid."""
    template = get_prompt_from_config(yaml_config, template_name)

    if not template or contains_placeholder(template):
        if run_logger:
            log_component(run_logger, component,
                f"WARNING: {template_name} template not found, using fallback")
        return fallback

    return template
```

### Files to Update
- `graph/nodes/unified/boss.py`
- `graph/nodes/unified/critic.py`
- `graph/nodes/unified/synthesis.py`

---

## NOT DOING (Intentional Exclusions)

### 1. Local Imports for Circular Dependencies
The pattern `from ..helpers.llm import create_llm_from_config` inside functions is **intentional** - it avoids circular imports. Don't change this.

### 2. Streaming Events
The streaming module is already well-organized with consistent `emit_*` functions. No changes needed.

### 3. State Access Patterns
Mix of `state.get()` and `state["key"]` is acceptable - TypedDict provides type hints but doesn't enforce runtime access.

### 4. Unified Node Decorator
Already implemented well with `@research_node(Phase.X)`.

### 5. Command API
Already using consistent `Command(update={}, goto="")` pattern.

---

## Implementation Order

1. **Phase 1** (logging helper) - ✅ COMPLETE
2. **Phase 2** (pydantic helper) - Small change, cleaner API
3. **Phase 3** (prompt fallback) - Optional, lower priority

---

## Success Metrics

| Metric | Before | After |
|--------|--------|-------|
| `run_logger._write_to_file(f"[COMPONENT]` occurrences | 30+ | ~15 |
| `result["extracted"]` pattern | 3 | 0 |
| Prompt fallback duplication | 3 | 0 |

---

## Testing

After each phase:
1. Run: `uv run backend/manage.py research "test query" --use-case base`
2. Verify output.md logs show correct component prefixes
3. Verify no regressions in research quality
