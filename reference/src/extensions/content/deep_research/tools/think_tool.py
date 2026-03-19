"""think_tool - Strategic reflection tool for deep research agents.

This tool enables agents to pause and assess progress strategically during
research workflows. Following the official LangChain deepagents pattern.

Usage in agents:
    agent = create_deep_agent(
        model=model,
        tools=[think_tool, query_postgres, get_artifact, ...],
        system_prompt="Use think_tool to assess research progress..."
    )

Example agent usage:
    # Agent can call think_tool like this:
    {
        "name": "think_tool",
        "arguments": {
            "reflection": "I've gathered sufficient information from web sources. 
                          The research covers philosophical and psychological perspectives.
                          However, I'm missing marketing insights. I should spawn a marketer 
                          architect to explore the emotional hooks and target audience angles."
        }
    }
"""

import logging
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection during deep research.

    Use this tool to pause and assess research progress strategically.

    Args:
        reflection: Your strategic reflection on the research state.

    Returns:
        Confirmation that reflection was recorded with timestamp.
    """
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()

    # Log reflection for observability
    logger.info(f"[THINK_TOOL] Reflection recorded at {timestamp}")
    logger.debug(f"[THINK_TOOL] Content: {reflection[:500]}...")
    
    return f"Reflection recorded at {timestamp}"


# =============================================================================
# ADVANCED THINK TOOL WITH CONTEXT ENGINE INTEGRATION
# =============================================================================

async def think_tool_with_persistence(
    reflection: str,
    session_id: str,
    store = None,
) -> str:
    """Enhanced think_tool that persists reflections to Context Engine.
    
    This version saves reflections to the Context Engine for:
    - Later retrieval during synthesis
    - Observability and debugging
    - Audit trail of reasoning
    
    Args:
        reflection: Strategic reflection text
        session_id: Research session UUID
        store: LangGraph Store instance (optional, for persistence)
    
    Returns:
        Confirmation with reflection ID
    
    Example:
        >>> result = await think_tool_with_persistence(
        ...     reflection="Research progress update...",
        ...     session_id="uuid-123",
        ...     store=store
        ... )
        'Reflection recorded (ID: think_abc123) at 2026-03-11T10:30:00Z'
    """
    from datetime import datetime, timezone
    from ..storage import save_artifact, compute_content_hash

    timestamp = datetime.now(timezone.utc).isoformat()
    reflection_id = f"think_{compute_content_hash(reflection)[:8]}"
    
    # Log reflection
    logger.info(f"[THINK_TOOL] {reflection_id} recorded at {timestamp}")
    
    # Persist to Context Engine if store provided
    if store:
        try:
            await save_artifact(
                store=store,
                session_id=session_id,
                artifact_type="think_reflection",
                content={
                    "reflection_id": reflection_id,
                    "content": reflection,
                    "timestamp": timestamp,
                },
                key=reflection_id,
            )
            logger.debug(f"[THINK_TOOL] Persisted {reflection_id} to Context Engine")
        except Exception as e:
            logger.warning(f"[THINK_TOOL] Failed to persist reflection: {e}")
    
    return f"Reflection recorded (ID: {reflection_id}) at {timestamp}"


# =============================================================================
# THINK TOOL FACTORY
# =============================================================================

def create_think_tool(
    persist: bool = False,
    session_id: str = None,
    store = None,
):
    """Factory to create think_tool with optional persistence.
    
    Args:
        persist: Whether to persist reflections to Context Engine
        session_id: Session UUID (required if persist=True)
        store: LangGraph Store (required if persist=True)
    
    Returns:
        Configured think_tool
    
    Example:
        # Simple version (no persistence)
        think = create_think_tool()
        
        # With persistence
        think = create_think_tool(
            persist=True,
            session_id=session_id,
            store=store
        )
    """
    if persist:
        if not session_id or not store:
            raise ValueError(
                "session_id and store are required when persist=True"
            )
        
        # Return async version with persistence
        from functools import partial
        return partial(
            think_tool_with_persistence,
            session_id=session_id,
            store=store
        )
    else:
        # Return simple synchronous version
        return think_tool


# =============================================================================
# THINK TOOL INSTRUCTIONS FOR SYSTEM PROMPTS
# =============================================================================

THINK_TOOL_INSTRUCTIONS = """
## Strategic Reflection with think_tool

Use the `think_tool` to pause and assess your research progress strategically.

### When to Use think_tool

1. **After Initial Research**: Summarize what you've learned and identify gaps
2. **Before Major Decisions**: Explain your reasoning for spawning architects or passing to synthesis
3. **When Uncertain**: Work through confusion and plan next steps
4. **Quality Check**: Evaluate if research is sufficient for synthesis

### What to Include in Reflections

- **Progress Summary**: What have you accomplished so far?
- **Key Findings**: What are the most important discoveries?
- **Gaps Identified**: What's missing or unclear?
- **Confidence Level**: How confident are you (0.0-1.0)?
- **Next Steps**: What should happen next and why?

### Example Usage

```
think_tool(
    reflection="'''
    Progress: Completed web research on topic X.
    
    Key Findings:
    - Found 3 major perspectives from authoritative sources
    - Identified consensus on points A and B
    - Discovered ongoing debate about point C
    
    Gaps:
    - Missing practical implementation examples
    - Need more recent sources (most are 2+ years old)
    - No industry perspective yet
    
    Confidence: 0.6 - Need implementation examples before synthesis
    
    Next Steps:
    Spawn industry analyst architect to find real-world case studies
    and practical implementation patterns.
    '''"
)
```

### Reflections Are Persistent

Your reflections are saved and can be retrieved later during synthesis.
This creates an audit trail of your reasoning and decision-making.
"""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "think_tool",
    "think_tool_with_persistence",
    "create_think_tool",
    "THINK_TOOL_INSTRUCTIONS",
]
