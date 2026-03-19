# Web Research Agent Refactor Plan

## Issues Identified

### A. Fallbacks to Remove (9 locations)

1. **Line 425-430**: URL selection LLM fallback → pick top 3
2. **Line 483-485**: URL selection parse fallback → use top results
3. **Line 567-573**: Findings extraction LLM fallback → store raw content
4. **Line 609-614**: Findings extraction exception fallback → store raw content
5. **Line 692-698**: Synthesis LLM fallback → concatenate findings
6. **Line 729-735**: Synthesis exception fallback → concatenate findings
7. **Line 794-796**: LangGraph not available fallback → simple research
8. **Line 829-833**: Synthesis missing fallback → manual concatenation
9. **Line 837-840**: Investigative exception fallback → simple research

**Why remove?** Fallbacks hide failures, making the complex agent system even harder to debug.

### B. Legacy Implementations to Remove

1. **Lines 1033-1058**: Legacy function wrapper `web_research_agent()`
   - Singleton pattern with module-level cache
   - Maintains backward compatibility with old call sites
   - Should be removed entirely

2. **Class-based tool pattern** → Consider moving to pure functions for LangGraph V1

### C. LangGraph V1 Alignment Issues

1. **Node signatures**: `_scrape_urls(state, raw_state, worker_id)` → Should only receive `state`
2. **Parameter passing**: Uses closure for `raw_state` and `worker_id`
3. **State structure**: Uses `InvestigativeState` (Dict subclass) instead of TypedDict
4. **Context management**: Should use contextvars for non-state data

### D. Profiling Requirements

Currently **NO timing metrics** are tracked. Need to add:

1. **Per-node timing**:
   - `_plan_research`: LLM call time
   - `_search_step`: web_search time
   - `_select_urls`: LLM call time
   - `_scrape_urls`: web_get times (parallel)
   - `_extract_findings`: LLM call time
   - `_synthesize_findings`: LLM call time

2. **Tool-level timing**:
   - Total investigative research time
   - Total simple research time
   - Time per search refinement
   - Aggregated tool usage times

3. **Output format**: Return structured timing data with content

## Refactor Strategy

### Phase 1: Add Profiling Infrastructure

```python
class ResearchProfiler:
    """Tracks timing metrics for web research operations."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self._start_times: Dict[str, float] = {}

    def start(self, operation: str):
        self._start_times[operation] = time.time()

    def end(self, operation: str) -> float:
        elapsed = time.time() - self._start_times.pop(operation)
        self.timings[operation].append(elapsed)
        return elapsed

    def get_report(self) -> Dict[str, Any]:
        return {
            "operations": {op: {
                "count": len(times),
                "total": sum(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            } for op, times in self.timings.items()},
            "total_time": sum(sum(times) for times in self.timings.values()),
        }
```

### Phase 2: Remove Fallbacks

Replace each fallback with explicit error handling:

```python
# Before (with fallback):
llm = _get_llm()
if not llm:
    # Fallback: pick top 3 URLs
    search_data["selected_urls"] = search_data["urls"][:3]
    return state

# After (no fallback):
llm = _get_llm()
if not llm:
    raise RuntimeError("LLM required for URL selection in investigative mode")
```

### Phase 3: Remove Legacy Code

1. Delete lines 1033-1058 (legacy function wrapper)
2. Remove singleton pattern usage
3. Update all call sites to use class-based API

### Phase 4: LangGraph V1 Alignment

```python
# Use contextvars for extra data
_research_state_context = contextvars.ContextVar('research_state', default=None)
_worker_id_context = contextvars.ContextVar('worker_id', default=None)

# Node functions receive only state
async def _scrape_urls(state: InvestigativeState) -> InvestigativeState:
    # Get raw_state from context
    raw_state = _research_state_context.get()
    worker_id = _worker_id_context.get()
    ...
```

### Phase 5: Add Structured Timing Output

```python
# Return type becomes:
async def _investigative_research(
    task: str,
    ...
) -> tuple[str, List[str], Dict[str, Any]]:  # content, source_ids, timing_report
    profiler = ResearchProfiler()
    ...
    return content, source_ids, profiler.get_report()
```

## Dependencies

Files that reference `web_research_agent`:
- `config.yaml`: Lists tool as enabled
- `graph/state.py`: Documents tool usage
- `graph/nodes/lab.py`: Uses tool in worker nodes
- `graph/nodes/boss.py`: Maps to web_search

## Testing Strategy

1. Run existing test with new profiling
2. Verify no fallbacks are hit
3. Check timing output format
4. Verify LangGraph V1 compatibility
