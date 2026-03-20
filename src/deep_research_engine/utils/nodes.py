"""Utilities for LangGraph nodes."""

import functools
import time
from typing import Any, Callable, Literal

from langgraph.types import Command

from ..state import EngineState
from .logging import get_loguru_logger, log_phase_complete, log_phase_start


def node_context(phase_name: str):
    """Decorator for LangGraph nodes to handle common tasks.

    - Logs phase start and completion
    - Updates phase_history and current_phase
    - Handles exceptions and updates fatal_error

    Args:
        phase_name: Name of the phase
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(state: EngineState, *args, **kwargs) -> Any:
            logger = get_loguru_logger(f"nodes.{phase_name}")
            log_phase_start(phase_name)
            
            start_time = time.time()
            
            # Update history and current phase
            history = state.get("phase_history", [])
            if not history or history[-1] != phase_name:
                history = history + [phase_name]
            
            try:
                # Call the actual node function
                result = await func(state, *args, **kwargs)
                
                duration = time.time() - start_time
                log_phase_complete(phase_name, duration=duration)
                
                # If result is already a Command, we can't easily modify its internal 'update' dict
                # without knowing its structure. But we can wrap it or ensure nodes return 
                # what we want. 
                # Actually, LangGraph 0.2 Command is quite flexible.
                
                if isinstance(result, Command):
                    # We assume nodes that return Command already handled their updates
                    # but we can try to merge our state updates if possible.
                    # Since we can't easily merge into Command after creation, 
                    # we'll encourage nodes to return dicts or we'll wrap the logic.
                    return result
                
                # If it's a dict (legacy state), update and return as Command
                if isinstance(result, dict):
                    updates = result
                    updates["current_phase"] = phase_name
                    updates["phase_history"] = history
                    
                    # Determine next step if not provided
                    goto = updates.pop("__goto__", None)
                    
                    return Command(
                        update=updates,
                        goto=goto
                    )
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {phase_name} node: {e}", exc_info=True)
                error_msg = str(e)
                
                # Return Command with error
                return Command(
                    update={
                        "fatal_error": error_msg,
                        "current_phase": phase_name,
                        "phase_history": history,
                    },
                    goto=None # Let the graph builder handle error routing
                )

        return wrapper

    return decorator
