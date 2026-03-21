"""Shared callback handlers for LangChain/LangGraph agent tracking.

This package provides reusable callback handlers for tracking agent execution
metrics across all extensions. This keeps logging DRY and consistent.

Usage:
    from .callbacks import AgentTracker, ChainTracker

    # Track LLM calls, tool calls, and tokens
    tracker = AgentTracker()
    result = await agent.ainvoke(input_data, config={"callbacks": [tracker]})

    # Get metrics
    summary = tracker.get_summary()
    print(f"LLM calls: {summary['llm_calls']}")
    print(f"Tool calls: {summary['tool_calls']}")

    # Track chain/graph execution
    chain_tracker = ChainTracker()
    result = await chain.ainvoke(input_data, config={"callbacks": [chain_tracker]})
"""

from .agent_tracker import AgentTracker, ChainTracker, create_agent_tracker

__all__ = ["AgentTracker", "ChainTracker", "create_agent_tracker"]
