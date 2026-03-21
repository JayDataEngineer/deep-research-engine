"""Parallel tool execution wrappers with semaphore protection.

This module wraps async tools to add semaphore-based concurrency control,
preventing system overload when tools are called in parallel.

Usage:
    from .tools.parallel_tools import wrap_tools_with_semaphores

    # Wrap all tools before passing to agent
    safe_tools = wrap_tools_with_semaphores(worker_tools)
"""

import asyncio
import uuid
from typing import Any, Awaitable, Callable
from langchain_core.tools import StructuredTool
from langchain_core.runnables import Runnable

from .concurrency import (
    get_semaphore_for_tool,
    execute_with_semaphore,
)

# =============================================================================
# MODULE-LEVEL REGISTRY FOR WRAPPED TOOLS
# =============================================================================
# Pydantic models (StructuredTool) don't properly handle instance attributes
# set before super().__init__(). We use a module-level dict to store the
# original tools, keyed by a unique ID stored on the wrapper.
_ORIGINAL_TOOLS: dict[str, StructuredTool] = {}


class SemaphoreProtectedTool:
    """Wraps a LangChain tool to execute with semaphore protection.

    This wrapper intercepts tool calls and acquires the appropriate
    semaphore before executing, preventing resource exhaustion.
    """

    def __init__(self, tool: StructuredTool):
        """Initialize the wrapper.

        Args:
            tool: The LangChain StructuredTool to wrap
        """
        self._tool = tool
        self.name = tool.name
        self.description = tool.description
        self.args_schema = tool.args_schema
        self.return_direct = tool.return_direct
        self.metadata = tool.metadata or {}
        self._coroutine = getattr(tool, '_coroutine', None)
        self.func = getattr(tool, 'func', None)

    async def ainvoke(self, *args, **kwargs) -> Any:
        """Execute the tool with semaphore protection."""
        semaphore = get_semaphore_for_tool(self.name)

        if semaphore is None:
            # No limit - execute directly
            return await self._tool.ainvoke(*args, **kwargs)
        else:
            # Acquire semaphore before executing
            async with semaphore:
                return await self._tool.ainvoke(*args, **kwargs)

    def __getattr__(self, name):
        """Forward all other attributes to the wrapped tool."""
        return getattr(self._tool, name)


def wrap_tool_with_semaphore(tool: StructuredTool) -> StructuredTool:
    """Wrap a single LangChain tool with semaphore protection.

    Args:
        tool: The LangChain StructuredTool to wrap

    Returns:
        A new tool with semaphore-protected execution
    """
    # Create a wrapper class that behaves like the original tool
    # but with semaphore-protected ainvoke method

    class SemaphoreWrappedTool(StructuredTool):
        def __init__(self, original_tool, wrapper_id: str):
            # First call super().__init__() to initialize the Pydantic model
            super().__init__(
                name=original_tool.name,
                description=original_tool.description,
                func=original_tool.func,
                args_schema=original_tool.args_schema,
                return_direct=original_tool.return_direct,
                metadata=original_tool.metadata or {},
            )
            # Use object.__setattr__ to bypass Pydantic's attribute validation
            # This ensures the private attribute is actually set on the instance
            object.__setattr__(self, '_wrapper_id', wrapper_id)
            # Preserve coroutine if present
            if hasattr(original_tool, '_coroutine'):
                object.__setattr__(self, '_coroutine', original_tool._coroutine)

        async def ainvoke(self, *args, **kwargs):
            """Execute with semaphore protection."""
            # Look up the original tool from the module registry
            original_tool = _ORIGINAL_TOOLS.get(self._wrapper_id)
            if original_tool is None:
                raise RuntimeError(f"Wrapped tool {self.name} has lost its original tool reference")

            semaphore = get_semaphore_for_tool(self.name)

            if semaphore is None:
                return await original_tool.ainvoke(*args, **kwargs)
            else:
                async with semaphore:
                    return await original_tool.ainvoke(*args, **kwargs)

    # Generate unique ID and store original tool in registry
    wrapper_id = str(uuid.uuid4())
    _ORIGINAL_TOOLS[wrapper_id] = tool

    return SemaphoreWrappedTool(tool, wrapper_id)


def wrap_tools_with_semaphores(tools: list) -> list:
    """Wrap multiple tools with semaphore protection.

    Args:
        tools: List of LangChain tools (StructuredTool or similar)

    Returns:
        List of tools with semaphore protection

    Example:
        from .tools.parallel_tools import wrap_tools_with_semaphores

        # Original tools (no protection)
        worker_tools = [web_search, web_reader, community_scrape]

        # Wrap with semaphore protection
        safe_tools = wrap_tools_with_semaphores(worker_tools)

        # Now can be used safely with parallel tool calling
        agent = create_agent(model=model, tools=safe_tools)
    """
    wrapped_tools = []

    for tool in tools:
        # Only wrap StructuredTool with ainvoke method
        if isinstance(tool, StructuredTool) and hasattr(tool, 'ainvoke'):
            wrapped = wrap_tool_with_semaphore(tool)
            wrapped_tools.append(wrapped)
        else:
            # Non-LangChain tool - pass through
            wrapped_tools.append(tool)

    return wrapped_tools


# =============================================================================
# PARALLEL TOOL EXECUTOR
# =============================================================================

async def execute_tool_calls_parallel(
    tool_calls: list[dict],
    available_tools: dict,
    max_concurrency: int = 10
) -> list[dict]:
    """Execute multiple tool calls in parallel with semaphore protection.

    This function is used when the LLM returns multiple tool_calls in one response.
    It executes them in parallel while respecting per-tool semaphore limits.

    Args:
        tool_calls: List of tool call dicts from LLM response
            Each dict should have: {"id": "...", "name": "...", "args": {...}}
        available_tools: Dict mapping tool names to tool functions/objects
        max_concurrency: Global limit on concurrent executions (default: 10)

    Returns:
        List of results in the same order as tool_calls

    Example:
        # LLM returned 3 tool calls
        tool_calls = [
            {"id": "call_1", "name": "web_search", "args": {"query": "Nietzsche"}},
            {"id": "call_2", "name": "web_search", "args": {"query": "Stoicism"}},
            {"id": "call_3", "name": "community_scrape", "args": {"sub": "r/philosophy"}},
        ]

        results = await execute_tool_calls_parallel(tool_calls, tool_registry)

        # Results are in the same order as inputs
        for tool_call, result in zip(tool_calls, results):
            print(f"{tool_call['name']}: {result}")
    """
    async def execute_single_call(tool_call: dict) -> dict:
        """Execute a single tool call with all safety measures."""
        tool_name = tool_call.get("name", "")
        tool_id = tool_call.get("id", "")
        tool_args = tool_call.get("args", {})

        # Find the tool function
        tool_obj = available_tools.get(tool_name)
        if tool_obj is None:
            return {
                "tool_call_id": tool_id,
                "error": f"Tool '{tool_name}' not found",
                "result": None
            }

        # Execute with semaphore protection
        try:
            result = await execute_with_semaphore(
                tool_obj.ainvoke if hasattr(tool_obj, 'ainvoke') else tool_obj,
                tool_name,
                tool_args
            )

            return {
                "tool_call_id": tool_id,
                "result": result
            }
        except Exception as e:
            return {
                "tool_call_id": tool_id,
                "error": str(e),
                "result": None
            }

    # Create tasks for all tool calls
    tasks = [
        execute_single_call(tool_call)
        for tool_call in tool_calls
    ]

    # Execute all tasks in parallel (respecting individual tool semaphores)
    results = await asyncio.gather(*tasks)

    return results
