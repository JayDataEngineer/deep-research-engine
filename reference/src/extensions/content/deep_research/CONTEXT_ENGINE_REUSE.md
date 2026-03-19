# Context Engine Reuse and Incremental Improvement

## Concept

The Deep Research framework produces **two valuable outputs**:

1. **The synthesized result** - The final report/answer the user requested
2. **The Context Engine** - All artifacts, sources, embeddings, and knowledge graph built during the Lab phase

Currently, we only return #1. The Context Engine is discarded after each run, even though it represents significant accumulated research value.

## Feature: Context Engine Persistence and Reuse

### Core Idea

After a research session completes, the Context Engine should be **persisted** and **reusable** for:

1. **Different output formats** - Reuse the same research, generate different outputs
2. **Follow-up queries** - Ask new questions within the same research context
3. **Incremental improvement** - Add more research to an existing Context Engine

### Use Case Example

**First Run - Build Context Engine:**
```bash
uv run manage.py research "Hitler's motivations in WWII" \
    --sources books:"./hitler_books/*.pdf" \
    --mode standard_report
```

- Scout analyzes query
- Boss creates worker tasks
- Lab workers build Context Engine from:
  - Local books (PDFs)
  - Web searches (Wikipedia, historical sources)
- Synthesizer produces markdown report

**Second Run - Reuse Context Engine, Different Output:**
```bash
uv run manage.py research "What did Hitler think of Churchill?" \
    --resume-from session_20250110_143000 \
    --mode character_profile
```

- Skip Scout → Boss → Lab (Context Engine exists)
- Use existing `raw_sources`, `worker_outputs`, artifacts
- Run NEW Synthesizer variant (character_profile format)
- Output: Character profile for WWII drama

**Third Run - Incremental Improvement:**
```bash
uv run manage.py research "Add more about Hitler's relationship with Stalin" \
    --resume-from session_20250110_143000 \
    --mode incremental
```

- Scout → Boss → Lab run ONLY for new research
- Findings are ADDED to existing Context Engine
- Existing artifacts are preserved and enhanced
- Synthesizer uses full Context Engine (old + new)

## Architecture

### Storage Layer (Already Exists)

The Context Engine is already stored persistently:

| Storage        | Purpose                                   | Location                |
|----------------|-------------------------------------------|-------------------------|
| PostgresSaver  | Checkpoints (conversation state)          | POSTGRES_URL            |
| PostgresStore  | Artifacts (semantic search with pgvector) | CONTEXT_ENGINE_POSTGRES |
| MinIO          | Large files (>5000 chars)                 | MINIO_ENDPOINT          |
| Neo4j          | Knowledge graph (entities, relationships) | NEO4J_URL               |

### Session Management (Needs Implementation)

```python
# List existing research sessions
def list_sessions(
    limit: int = 10,
    query: Optional[str] = None  # Filter by topic
) -> List[ResearchSession]:
    """Query existing research sessions for reuse."""
    pass

# Get session metadata
def get_session_info(research_id: str) -> SessionInfo:
    """Get info about a session before resuming."""
    return SessionInfo(
        research_id="session_20250110_143000",
        original_query="Hitler's motivations in WWII",
        created_at="2025-01-10T14:30:00",
        source_count=42,
        tier_a_count=5,
        tier_b_count=10,
        worker_count=3,
        artifacts_count=15
    )

# Resume from existing session
async def resume_from_session(
    research_id: str,
    new_query: str,
    output_mode: str = "standard_report"
) -> ResearchResult:
    """Skip to Critic/Synthesis using existing Context Engine."""
    pass
```

### Output Mode Variants (Needs Implementation)

```python
# Synthesizer variants for different outputs
SYNTHESIZER_MODES = {
    "standard_report": StandardReportSynthesizer,
    "character_profile": CharacterProfileSynthesizer,
    "briefing": ExecutiveBriefingSynthesizer,
    "timeline": TimelineSynthesizer,
    "comparison": ComparisonSynthesizer,
}

# Critic variants for different quality checks
CRITIC_MODES = {
    "standard": StandardCritic,
    "factual": FactualAccuracyCritic,
    "narrative": NarrativeFlowCritic,
    "historical": HistoricalAccuracyCritic,
}
```

### Incremental Improvement Mode

```python
# Lab workers should support "append" mode
async def lab_workers_incremental(
    state: GlobalState,
    existing_research_id: str,
) -> Dict[str, Any]:
    """Run additional research and append to existing Context Engine.

    - Existing artifacts are preserved
    - New worker findings are added
    - InformationFunnel re-tiers (old artifacts keep priority)
    - Knowledge graph is enhanced (new entities/relations merged)
    """
    # Load existing state
    existing_state = await load_checkpoint(existing_research_id)

    # Run new Scout → Boss → Lab for NEW query only
    # ...

    # Merge with existing Context Engine
    await merge_context_engines(
        existing=existing_research_id,
        new=state["research_id"]
    )
```

## CLI Interface

```bash
# Standard research (builds new Context Engine)
uv run manage.py research "query here" --mode standard_report

# List existing sessions
uv run manage.py research --list-sessions

# Resume with different output
uv run manage.py research "new question or same" \
    --resume-from session_20250110_143000 \
    --mode character_profile

# Incremental improvement
uv run manage.py research "additional research topic" \
    --resume-from session_20250110_143000 \
    --mode incremental

# Show session info
uv run manage.py research --session-info session_20250110_143000
```

## Implementation Checklist

- [ ] Session listing/query (`list_sessions`)
- [ ] Session metadata retrieval (`get_session_info`)
- [ ] Resume entry point (skip Scout → Boss → Lab)
- [ ] Synthesizer mode variants (character_profile, briefing, etc.)
- [ ] Critic mode variants (factual, narrative, etc.)
- [ ] Incremental improvement mode (merge Context Engines)
- [ ] CLI flags for `--resume-from`, `--list-sessions`, `--session-info`
- [ ] Context Engine query tool (semantic search across sessions)
- [ ] Migration utilities (upgrade old sessions to new schema)

## Benefits

1. **Faster follow-up queries** - No need to re-research the same topic
2. **Multiple output formats** - Same research, different presentations
3. **Incremental knowledge building** - Context Engine improves over time
4. **Cost efficiency** - Reuse existing embeddings, summaries, analysis
5. **Better research quality** - Build on previous work instead of starting fresh

## Technical Notes

### State Isolation

Each `research_id` creates isolated namespaces:
- Postgres namespaces: `("research", research_id, "tier_a")`
- MinIO paths: `context-engine/{research_id}/artifacts/`
- Neo4j nodes: `EntityArtifact(research_id=...)`

This allows multiple sessions to coexist without interference.

### Checkpoint Loading

LangGraph's PostgresSaver stores full state snapshots. Resuming requires:
1. Loading checkpoint by thread_id (research_id)
2. Restoring GlobalState
3. Jumping to Critic/Synthesis phase directly
4. Replacing Critic/Synthesizer with variant if specified

### Graph Merging (Incremental Mode)

When adding to an existing Context Engine:
- Entities: Merge if name/type match, add new if unique
- Relationships: Merge if (source, relation, target) exists
- Artifacts: Append new findings, preserve old
- Tiers: Old artifacts keep tier priority, new ones added below

## Related

- `src/shared/checkpointer/factory.py` - PostgresSaver configuration
- `src/extensions/content/deep_research/helpers/information_funnel.py` - Tier management
- `src/extensions/content/deep_research/storage/` - Context Engine storage helpers
- `src/extensions/content/deep_research/tools/artifact_tools.py` - Artifact save/read tools
