"""Critic Deep Agent - Quality Gate.

Read-only StoreBackend access.
Uses SMART model (Sonnet) via LiteLLM.
Returns structured JSON evaluation.
"""

from typing import Any

from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from loguru import logger

from context_engine.config import get_model
from context_engine.store.tools import create_search_vectors_tool


# Structured output schema for critic response
CRITIC_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "critic_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "passed": {
                    "type": "boolean",
                    "description": "Whether context quality passes the threshold",
                },
                "score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Quality score from 0.0 to 1.0",
                },
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of identified issues",
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of improvement suggestions",
                },
            },
            "required": ["passed", "score", "issues", "suggestions"],
            "additionalProperties": False,
        },
    },
}


@tool
def list_memories(store: BaseStore, session_id: str) -> str:
    """List all stored memories for review.

    Args:
        store: The LangGraph store
        session_id: Session ID to list memories for

    Returns:
        Formatted list of memory files
    """
    import asyncio

    async def _list():
        items = await store.alist_namespaces(prefix=(session_id,))
        if not items:
            return "No memories found."

        lines = ["## Stored Memories\n"]
        for item in items:
            lines.append(f"- {item.key}: {item.value.get('content', '')[:100]}...")

        return "\n".join(lines)

    return asyncio.run(_list())


def create_critic_agent(
    store: BaseStore,
    session_id: str,
    prompts: dict[str, str],
) -> Any:
    """Create critic agent with read-only access.

    Uses SMART model (Sonnet) via LiteLLM.
    Can only read and search, NOT write.

    Args:
        store: LangGraph BaseStore for reading
        session_id: Current session ID for namespacing
        prompts: Dict of prompts from PromptManager

    Returns:
        Compiled StateGraph agent
    """
    # Import deepagents here to avoid import issues if not installed
    try:
        from deepagents import create_deep_agent
    except ImportError as e:
        logger.error(f"deepagents not installed: {e}")
        msg = "deepagents package required. Install with: uv add deepagents"
        raise ImportError(msg) from e

    # Get SMART model (Sonnet) via LiteLLM
    model = get_model("smart")

    # Read-only tools
    search_vectors = create_search_vectors_tool(store, session_id)

    # Get system prompt
    system_prompt = prompts.get("critic_agent", "")

    # Create agent WITHOUT store parameter
    # This means no StoreBackend, so no write_file tool
    # Only built-in read_file (from StateBackend) available
    agent = create_deep_agent(
        model=model,
        tools=[search_vectors],
        system_prompt=system_prompt,
        subagents=[],  # No subagents for critic
        # NO store parameter = read-only access
        response_format=CRITIC_RESPONSE_FORMAT,
    )

    logger.info(f"Created critic agent for session {session_id}")
    return agent


def parse_critic_response(response: dict[str, Any]) -> dict[str, Any]:
    """Parse critic agent response into structured format.

    Args:
        response: Raw agent response dict

    Returns:
        Structured evaluation dict with keys:
        - passed: bool
        - score: float
        - issues: list[str]
        - suggestions: list[str]
    """
    import json

    # Get last message content
    messages = response.get("messages", [])
    if not messages:
        return {
            "passed": False,
            "score": 0.0,
            "issues": ["No response from critic"],
            "suggestions": ["Check critic agent configuration"],
        }

    last_message = messages[-1]
    content = getattr(last_message, "content", str(last_message))

    # Try to parse JSON
    try:
        # Handle both string and dict content
        if isinstance(content, str):
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
        else:
            result = content

        # Validate required fields
        return {
            "passed": bool(result.get("passed", False)),
            "score": float(result.get("score", 0.0)),
            "issues": list(result.get("issues", [])),
            "suggestions": list(result.get("suggestions", [])),
        }

    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"Failed to parse critic response: {e}")
        return {
            "passed": False,
            "score": 0.0,
            "issues": [f"Failed to parse critic response: {e}"],
            "suggestions": ["Ensure critic returns valid JSON"],
        }
