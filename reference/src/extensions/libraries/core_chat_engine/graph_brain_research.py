"""Deep Research Brain - Spawner Multi-Agent Integration

This module wraps the Spawner multi-agent system for integration with the chat application.
It provides the same interface as CompanionBrainGraphV2 but uses the Spawner pipeline.
"""

import logging
import time
import json
from typing import TypedDict, Annotated, Optional, Dict, Any, List, Literal
from datetime import datetime
import sys
from pathlib import Path

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

# Add src to path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import Spawner components (from deep_research extension)
from src.extensions.content.deep_research.graph import build_v4_graph, run_v4_spawner
from src.extensions.content.deep_research.graph.state import (
    GlobalState,
    ResearchBrief,
    WorkerResult,
    CritiqueResult,
    create_global_state,
)
from src.extensions.content.deep_research import load_research_config, create_tool_registry

# Import companion brain components
from ..state.typed_state import ProcessedResponse, IntentType, IntentResult

# Import shared reducer
from src.shared.state.base import add_messages_reducer

logger = logging.getLogger(__name__)


# =============================================================================
# LangGraph State Definition for Research Mode
# =============================================================================

class ResearchBrainState(TypedDict):
    """State for the research brain using Spawner multi-agent pipeline.

    Compatible with assistant_brain interface but uses Spawner internally.
    """
    # Input messages (conversation history) - with reducer
    messages: Annotated[
        list[BaseMessage],
        "The conversation messages in LangChain format",
        add_messages_reducer,  # Reducer function from shared.state.base
    ]

    # User input for this turn
    user_input: str

    # Mode (should be "deep_research")
    mode: str

    # Research configuration
    use_case: str  # "rag", "podcast", "trading", "osint", "news", "content"
    variant: str  # "default" or custom

    # Spawner outputs (for streaming and final response)
    research_brief: Optional[ResearchBrief]
    worker_results: List[WorkerResult]
    synthesis: Optional[str]
    critique: Optional[CritiqueResult]

    # Thread ID for persistence
    thread_id: str

    # Timestamps
    start_time: float
    processing_time: Optional[float]

    # Final response (output of post_process node)
    response: Optional[ProcessedResponse]


# =============================================================================
# Graph Nodes
# =============================================================================

def create_input_node_research():
    """Create the input processing node for research mode."""
    def input_node(state: ResearchBrainState) -> Dict[str, Any]:
        """Process user input and add to messages."""
        logger.debug("Node: input_processor_research")

        user_input = state.get("user_input", "")
        if not user_input:
            return {}

        # Create HumanMessage - this will be appended by the reducer
        user_message = HumanMessage(content=user_input)

        return {"messages": [user_message]}

    return input_node


def create_research_node(api_key: str, base_url: str, model: str):
    """Create the main research node that runs Spawner with streaming."""
    async def research_node(state: ResearchBrainState, writer) -> Dict[str, Any]:
        """Execute Spawner pipeline with progress streaming."""
        logger.info(f"[RESEARCH NODE] Starting Spawner pipeline for: {state.get('user_input', '')[:100]}...")

        # Emit starting phase
        writer({
            "type": "phase_change",
            "phase": "scout",
            "phase_name": "Analysis",
            "message": "Analyzing query complexity...",
            "status": "active"
        })

        # Load config
        use_case = state.get("use_case", "rag")
        variant = state.get("variant", "default")
        config = load_research_config(use_case, variant)

        # Update config with API credentials
        config["model_configs"]["spawner"]["api_key"] = api_key
        config["model_configs"]["spawner"]["base_url"] = base_url
        config["model_configs"]["spawner"]["model_name"] = model

        # Create phase callback for streaming
        def phase_callback(phase_info: Dict[str, Any]):
            """Callback to emit phase changes to frontend."""
            phase_name = phase_info.get("phase", "unknown")
            phase_display = {
                "scout": "Query Analysis",
                "boss": "Task Planning",
                "workers": "Research",
                "synthesis": "Synthesis",
                "critic": "Quality Review",
            }.get(phase_name, phase_name.title())

            writer({
                "type": "phase_change",
                "phase": phase_name,
                "phase_name": phase_display,
                "message": phase_info.get("message", f"Running {phase_display}..."),
                "status": "active"
            })

        try:
            # Run Spawner pipeline
            # Note: run_v4_spawner is a generator, we need to consume it
            result = None
            for chunk in run_v4_spawner(
                query=state.get("user_input", ""),
                config=config,
                phase_callback=phase_callback,
            ):
                result = chunk

                # Stream intermediate results if available
                if "worker_results" in chunk:
                    for worker_result in chunk["worker_results"]:
                        writer({
                            "type": "worker_progress",
                            "worker_id": worker_result.get("worker_id", "unknown"),
                            "message": f"{worker_result.get('task', 'Working...')}",
                            "status": "active"
                        })

            # Extract results
            synthesis = result.get("synthesis", "") if result else ""
            worker_results = result.get("worker_results", []) if result else []
            brief = result.get("brief") if result else None

            # Stream synthesis as content chunks
            if synthesis:
                writer({
                    "type": "content_chunk",
                    "content": synthesis
                })

            return {
                "synthesis": synthesis,
                "worker_results": worker_results,
                "research_brief": brief,
            }

        except Exception as e:
            logger.error(f"[RESEARCH NODE] Error: {e}")
            import traceback
            traceback.print_exc()

            # Emit error phase
            writer({
                "type": "phase_change",
                "phase": "error",
                "phase_name": "Error",
                "message": f"Research failed: {str(e)}",
                "status": "error"
            })

            return {
                "synthesis": f"I apologize, but the research pipeline encountered an error: {str(e)}",
                "worker_results": [],
                "research_brief": None,
            }

    return research_node


def create_post_process_node_research():
    """Create the post-processing node for research mode."""
    def post_process_node(state: ResearchBrainState) -> Dict[str, Any]:
        """Finalize response and update processing time."""
        logger.debug("Node: post_process_research")

        start_time = state.get("start_time", time.time())
        synthesis = state.get("synthesis", "")

        # Build ProcessedResponse
        processed = ProcessedResponse(
            text=synthesis,
            intent=IntentResult(
                primary=IntentType.CASUAL_CHAT,
                confidence=1.0,
            ),
            source="deep_research",
            processing_time=time.time() - start_time,
            was_routed=True,
        )

        return {"response": processed, "processing_time": processed.processing_time}

    return post_process_node_research


# =============================================================================
# Graph Builder
# =============================================================================

def create_research_graph(
    api_key: str,
    base_url: str,
    model: str,
    use_case: str = "rag",
    variant: str = "default",
    checkpointer=None,
) -> StateGraph:
    """Create the research brain as a LangGraph wrapping Spawner.

    Args:
        api_key: API key for the LLM service
        base_url: Base URL for the LLM API
        model: Model name to use
        use_case: Research domain (rag, podcast, trading, osint, news, content)
        variant: Config variant
        checkpointer: Optional checkpointer for persistence

    Returns:
        Compiled StateGraph ready for invocation
    """
    logger.info(f"Creating Research Graph with use_case={use_case}, variant={variant}")

    # Create nodes
    input_node = create_input_node_research()
    research_node = create_research_node(api_key, base_url, model)
    post_process_node = create_post_process_node_research()

    # Build the graph
    builder = StateGraph(ResearchBrainState)

    # Add nodes
    builder.add_node("input_processor", input_node)
    builder.add_node("research", research_node)
    builder.add_node("post_process", post_process_node)

    # Define the flow: START → input → research → post_process → END
    builder.add_edge(START, "input_processor")
    builder.add_edge("input_processor", "research")
    builder.add_edge("research", "post_process")
    builder.add_edge("post_process", END)

    # Compile with checkpointer if provided
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Research brain StateGraph created with Spawner integration")
    return graph


# =============================================================================
# Wrapper Class
# =============================================================================

class ResearchBrainGraph:
    """
    Wrapper for the Spawner research graph.

    This provides the same interface as CompanionBrainGraphV2 but uses
    the Spawner multi-agent pipeline for deep research.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        use_case: str = "rag",
        variant: str = "default",
        checkpointer=None,
        thread_id: Optional[str] = None,
    ):
        """Initialize the research brain graph.

        Args:
            api_key: API key for the LLM service
            base_url: Base URL for the LLM API
            model: Model name to use
            use_case: Research domain (rag, podcast, trading, osint, news, content)
            variant: Config variant
            checkpointer: Optional checkpointer for persistence
            thread_id: Thread ID for this conversation
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.use_case = use_case
        self.variant = variant

        # Create thread_id
        self.thread_id = thread_id or f"research_thread_{time.time()}"

        # Create the StateGraph
        self.graph = create_research_graph(
            api_key=api_key,
            base_url=base_url,
            model=model,
            use_case=use_case,
            variant=variant,
            checkpointer=checkpointer,
        )

        # Local message storage
        self._messages: List[BaseMessage] = []

    def process(self, user_input: str) -> ProcessedResponse:
        """
        Process user input through the research brain.

        Args:
            user_input: Raw user input text

        Returns:
            ProcessedResponse with the research output
        """
        config = {"configurable": {"thread_id": self.thread_id}}

        # Prepare initial state
        initial_state = {
            "messages": self._messages.copy(),
            "user_input": user_input,
            "mode": "deep_research",
            "use_case": self.use_case,
            "variant": self.variant,
            "thread_id": self.thread_id,
            "start_time": time.time(),
            "research_brief": None,
            "worker_results": [],
            "synthesis": None,
            "critique": None,
        }

        # Invoke the graph
        result = self.graph.invoke(initial_state, config)

        # Update local storage from result
        if result.get("messages"):
            self._messages = result["messages"].copy()

        # Return the response
        response = result.get("response")
        if response:
            return response
        else:
            # Fallback
            return ProcessedResponse(
                text="I apologize, but I couldn't generate a research response.",
                intent=IntentResult(
                    primary=IntentType.UNKNOWN,
                    confidence=0.0,
                ),
                source="error",
            )

    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        self.thread_id = f"research_thread_{time.time()}"
        self._messages = []

    def get_conversation_history(self) -> List[BaseMessage]:
        """Get conversation history."""
        return self._messages.copy()


# =============================================================================
# Factory Functions
# =============================================================================

def create_research_brain(
    api_key: str,
    base_url: str,
    model: str,
    use_case: str = "rag",
    variant: str = "default",
    checkpointer=None,
    thread_id: Optional[str] = None,
) -> ResearchBrainGraph:
    """
    Factory function to create ResearchBrainGraph.

    Args:
        api_key: API key for the LLM service
        base_url: Base URL for the LLM API
        model: Model name to use
        use_case: Research domain (rag, podcast, trading, osint, news, content)
        variant: Config variant
        checkpointer: Optional checkpointer for persistence
        thread_id: Thread ID for conversation persistence

    Returns:
        Initialized ResearchBrainGraph
    """
    return ResearchBrainGraph(
        api_key=api_key,
        base_url=base_url,
        model=model,
        use_case=use_case,
        variant=variant,
        checkpointer=checkpointer,
        thread_id=thread_id,
    )
