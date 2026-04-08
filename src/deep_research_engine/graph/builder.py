"""StateGraph builder for Deep Research Engine workflow.

Constructs the workflow graph with proper routing between phases:
    START → ingestion → drafter → critic ⇄ drafter (retry)
                              ↓
                  content_clusterer ←── NEW
                         ↓
                  neo4j_builder → artifact_maker → END
"""

from typing import Any, Literal

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.pregel import Pregel

from ..config import WorkflowConfig, get_phase_config
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
        artifact_maker_node,
        content_clusterer_node,
        critic_node,
        drafter_node,
        ingestion_node,
        neo4j_builder_node,
    )

    # Add nodes
    builder.add_node("ingestion", ingestion_node)
    builder.add_node("drafter", drafter_node)
    builder.add_node("critic", critic_node)
    builder.add_node("content_clusterer", content_clusterer_node)
    builder.add_node("neo4j_builder", neo4j_builder_node)
    builder.add_node("artifact_maker", artifact_maker_node)

    # Entry point
    builder.add_edge(START, "ingestion")

    # Conditional routing functions
    def route_ingestion(state: EngineState) -> Literal["drafter", END]:
        """Route after ingestion phase."""
        if state.get("fatal_error"):
            return END
        return "drafter"

    def route_drafter(state: EngineState) -> Literal["critic", END]:
        """Route after drafter phase."""
        if state.get("fatal_error") or not state.get("worker_results"):
            return END
        return "critic"

    def route_critic(state: EngineState) -> Literal["drafter", "content_clusterer", END]:
        """Route after critic phase - supports retry loop."""
        if state.get("fatal_error"):
            return END

        # The critic node sets critic_passed and critique_history directly.
        # Use critique_history length to count iterations (each critic run appends one entry).
        critique_history = state.get("critique_history", [])
        iteration = len(critique_history)

        # Get critic config for max retries
        config_data = state.get("context", {}).get("config", {})
        phase_config = get_phase_config(config_data, "critic")
        max_retries = phase_config.get("max_retries", 3)

        # Check if passed or max retries reached
        if state.get("critic_passed", False) or iteration >= max_retries:
            return "content_clusterer"

        return "drafter"

    def route_content_clusterer(state: EngineState) -> Literal["neo4j_builder", END]:
        """Route after content clusterer phase."""
        if state.get("fatal_error"):
            return END
        return "neo4j_builder"

    def route_neo4j_builder(state: EngineState) -> Literal["artifact_maker", END]:
        """Route after neo4j builder phase."""
        if state.get("fatal_error"):
            return END
        return "artifact_maker"

    def route_artifact_maker(state: EngineState) -> Literal[END]:
        """Route after artifact maker phase."""
        if state.get("fatal_error"):
            return END
        return END

    # Add conditional edges
    builder.add_conditional_edges("ingestion", route_ingestion, ["drafter", END])
    builder.add_conditional_edges("drafter", route_drafter, ["critic", END])
    builder.add_conditional_edges("critic", route_critic, ["drafter", "content_clusterer", END])
    builder.add_conditional_edges("content_clusterer", route_content_clusterer, ["neo4j_builder", END])
    builder.add_conditional_edges("neo4j_builder", route_neo4j_builder, ["artifact_maker", END])
    builder.add_conditional_edges("artifact_maker", route_artifact_maker, [END])

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
