"""Main engine entry point."""

from typing import AsyncIterator, Dict, Any, Optional, Union
from pathlib import Path
import asyncio
import uuid

from .config import WorkflowConfig, load_config
from .state import EngineState, create_initial_state
from .storage import StorageManager
from .utils.logging import setup_logging, get_logger


async def run_workflow(
    input_data: Union[str, Dict[str, Any]],
    config: WorkflowConfig,
    session_id: Optional[str] = None,
    stream: bool = True,
    storage_backends: Optional[list] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Run a workflow with the given configuration.
    
    Args:
        input_data: Query string or dict with workflow input
        config: Workflow configuration
        session_id: Optional session ID for tracing
        stream: If True, yield events as they happen
        storage_backends: List of backends to enable
        
    Yields:
        Events (if streaming) or final result
    """
    # Setup logging
    setup_logging(level=config.execution.get("logging", "info"))
    logger = get_logger("engine")
    
    # Generate IDs
    run_id = str(uuid.uuid4())
    session_id = session_id or run_id
    
    logger.info(
        f"Starting workflow: {config.workflow['name']}",
        run_id=run_id,
        session_id=session_id,
    )
    
    # Initialize storage
    storage = StorageManager()
    backends_to_enable = storage_backends or list(config.storage.keys())
    
    for backend_name in backends_to_enable:
        backend_config = config.storage.get(backend_name, {})
        if backend_config.get("enabled", False):
            try:
                await storage.initialize(backend_name, backend_config)
                logger.info(f"Initialized storage: {backend_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize {backend_name}: {e}")
    
    # Create initial state
    state = create_initial_state(
        workflow_name=config.workflow["name"],
        workflow_type=config.workflow["type"],
        input_data={"query": input_data} if isinstance(input_data, str) else input_data,
        session_id=session_id,
        run_id=run_id,
        max_iterations=config.phases.get("boss", {}).get("config", {}).get("workers", {}).get("max", 3),
    )
    
    # Build and run graph
    try:
        if stream:
            async for event in _run_graph_streaming(state, config):
                yield event
        else:
            final_state = await _run_graph_sync(state, config)
            yield {
                "type": "complete",
                "success": True,
                "output": final_state.get("synthesis_output"),
                "state": final_state,
            }
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        yield {
            "type": "error",
            "error": str(e),
            "phase": "execution",
        }
    finally:
        await storage.cleanup()


async def _run_graph_streaming(state: EngineState, config: WorkflowConfig):
    """Run graph with streaming."""
    # TODO: Implement LangGraph streaming
    yield {"type": "phase_start", "phase": "boss"}
    yield {"type": "phase_complete", "phase": "boss", "duration": 0.0}
    yield {"type": "synthesis_complete", "output": "TODO: Implement graph execution"}


async def _run_graph_sync(state: EngineState, config: WorkflowConfig) -> EngineState:
    """Run graph synchronously."""
    # TODO: Implement LangGraph execution
    return state
