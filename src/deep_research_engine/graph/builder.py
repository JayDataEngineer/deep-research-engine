"""StateGraph builder for Deep Research Engine workflow.

Constructs the workflow graph with proper routing between phases:
    START → ingestion → drafter → critic ⇄ drafter (retry)
                              ↓
                        neo4j_builder → synthesizer → END
"""

from typing import Any, Literal

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.pregel import Pregel

from ..config import WorkflowConfig
from ..state import EngineState
from ..utils.logging import get_loguru_logger

logger = get_loguru_logger("graph.builder")


def build_workflow_graph(config: WorkflowConfig) -> StateGraph:
    """Build the StateGraph for the deep research workflow.

    Args:
        config: Workflow configuration

    Returns:
        Uncompiled StateGraph ready for execution
    """
    builder = StateGraph(EngineState)

    # Import nodes (lazy to avoid circular imports)
    from ..nodes import (
        critic_node,
        drafter_node,
        ingestion_node,
        neo4j_builder_node,
        synthesizer_node,
    )

    # Add nodes
    builder.add_node("ingestion", ingestion_node)
    builder.add_node("drafter", drafter_node)
    builder.add_node("critic", critic_node)
    builder.add_node("neo4j_builder", neo4j_builder_node)
    builder.add_node("synthesizer", synthesizer_node)

    # Entry point
    builder.add_edge(START, "ingestion")

    # Conditional routing functions
    def route_ingestion(state: EngineState) -> Literal["drafter", "error"]:
        """Route after ingestion phase."""
        if state.get("fatal_error"):
            return "error"
        return "drafter"

    def route_drafter(state: EngineState) -> Literal["critic", "error"]:
        """Route after drafter phase."""
        if state.get("fatal_error") or not state.get("worker_results"):
            return "error"
        return "critic"

    def route_critic(state: EngineState) -> Literal["drafter", "neo4j_builder", "error"]:
        """Route after critic phase - supports retry loop."""
        if state.get("fatal_error"):
            return "error"

        # Get critic config for max retries
        config_data = state.get("context", {}).get("config", {})
        max_retries = 3  # default

        if isinstance(config_data.get("phases"), list):
            for p in config_data["phases"]:
                if p.get("name") == "critic":
                    max_retries = p.get("config", {}).get("max_retries", 3)
                    break
        elif isinstance(config_data.get("phases"), dict):
            max_retries = (
                config_data.get("phases", {})
                .get("critic", {})
                .get("config", {})
                .get("max_retries", 3)
            )

        # Count critic iterations
        iteration = len([p for p in state.get("phase_history", []) if p == "critic"])

        # Check if passed or max retries reached
        if state.get("critic_passed", False) or iteration >= max_retries:
            return "neo4j_builder"

        return "drafter"

    def route_neo4j_builder(state: EngineState) -> Literal["synthesizer", "error"]:
        """Route after neo4j builder phase."""
        if state.get("fatal_error"):
            return "error"
        return "synthesizer"

    def route_synthesizer(state: EngineState) -> Literal[END, "error"]:
        """Route after synthesizer phase."""
        if state.get("fatal_error"):
            return "error"
        return END

    # Add conditional edges
    builder.add_conditional_edges("ingestion", route_ingestion, ["drafter", "error"])
    builder.add_conditional_edges("drafter", route_drafter, ["critic", "error"])
    builder.add_conditional_edges("critic", route_critic, ["drafter", "neo4j_builder", "error"])
    builder.add_conditional_edges("neo4j_builder", route_neo4j_builder, ["synthesizer", "error"])
    builder.add_conditional_edges("synthesizer", route_synthesizer, [END, "error"])

    logger.info("Workflow graph built successfully")
    return builder


def compile_workflow_graph(
    config: WorkflowConfig,
    checkpointer: Any = None,
    store: Any = None,
) -> Pregel:
    """Build and compile the workflow graph with optional persistence.

    Args:
        config: Workflow configuration
        checkpointer: Optional AsyncPostgresSaver for checkpointing
        store: Optional LangGraph Store for vectorization

    Returns:
        Compiled StateGraph (Pregel) ready for execution
    """
    builder = build_workflow_graph(config)

    compiled = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )

    logger.info("Workflow graph compiled successfully")
    return compiled
