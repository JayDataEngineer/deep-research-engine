"""Consolidated LangChain/LangGraph callback handlers for logging.

.. deprecated:: 2.0.0
    Use LangFuse callback handlers instead. See:
    src/extensions/integrations/langfuse/callbacks.py

    Migration guide:
    - create_callback_handler(run_logger) -> create_langfuse_callback(session_id)
    - SpawnerCallbackHandler -> LangFuse CallbackHandler (automatic)
    - ObservabilityCallback -> LangFuse CallbackHandler (automatic)

    Example:
        # OLD
        from deep_research.observability import create_callback_handler
        handler = create_callback_handler(run_logger)

        # NEW
        from src.extensions.integrations.langfuse import create_langfuse_callback
        handler = create_langfuse_callback(session_id="research-123")

This module contains all callback handlers for observability:
- SpawnerCallbackHandler: For LangGraph execution with detailed phase tracking
- ObservabilityCallback: Modern BaseCallbackHandler for general agent logging

Consolidated from:
- callbacks.py (original)
- middleware.py (ObservabilityCallback)

Usage:
    # For LangGraph StateGraph
    from deep_research.observability import create_callback_handler
    handler = create_callback_handler(run_logger)
    result = graph.invoke(state, config={"callbacks": [handler]})

    # For LangChain agents
    from deep_research.observability import create_observability_callback
    callback = create_observability_callback(run_name="my_agent")
    result = graph.invoke(state, callbacks=[callback])
"""

import warnings

warnings.warn(
    "SpawnerCallbackHandler and ObservabilityCallback are deprecated. "
    "Use LangFuse callback handlers instead. "
    "See: src/extensions/integrations/langfuse/callbacks.py",
    DeprecationWarning,
    stacklevel=2,
)

from pathlib import Path
import time
import json
import logging
import sys

from typing import Any, Dict, List, Optional, Sequence
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult, ChatGenerationChunk
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.agents import AgentAction, AgentFinish

from .run_logger import RunLogger
from ..logging_utils import safe_get_logger

logger = safe_get_logger(__name__, "callbacks")


class SpawnerCallbackHandler(BaseCallbackHandler):
    """Callback handler for logging LangGraph execution.

    Use with:
        graph.invoke(
            state,
            config={"callbacks": [SpawnerCallbackHandler(run_logger)]}
        )

    Logs:
    - LLM calls (model invocations)
    - Tool calls (including MCP tools)
    - Agent actions and decisions
    - Chain starts/ends (phases)
    """

    def __init__(
        self,
        run_logger: Optional[RunLogger] = None,
        verbose: bool = True
    ):
        """Initialize the callback handler.

        Args:
            run_logger: RunLogger instance for structured logging
            verbose: Whether to log detailed info
        """
        super().__init__()
        self.run_logger = run_logger
        self.verbose = verbose
        self._llm_calls: List[Dict[str, Any]] = []
        self._tool_calls: List[Dict[str, Any]] = []
        self._phase_stack: List[str] = []
        self._start_time: float = time.time()
        self._current_phase_start: float = time.time()
        self._tool_start_time: float = 0.0
        self._current_tool_name: str = ""
        # For prompt breakdown tracking
        self._prompt_breakdown: Dict[str, Any] = {
            "worker_to_boss": [],
            "boss_to_workers": []
        }

    # ========== LLM CALLBACKS ==========

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts processing."""
        phase = self._phase_stack[-1] if self._phase_stack else "unknown"

        # Handle prompts that could be string or dict (depending on LangChain version)
        if prompts:
            first_prompt = prompts[0]
            if isinstance(first_prompt, dict):
                # For dict prompts, extract 'content' or stringify
                prompt_preview = str(first_prompt.get('content', first_prompt))[:100]
            else:
                prompt_preview = str(first_prompt)[:100]
        else:
            prompt_preview = ""
        # Handle None serialized
        if serialized is None or not isinstance(serialized, dict):
            model_name = "unknown"
        else:
            model_name = serialized.get("name", "unknown")

        self._write_log(f"[LLM START] Phase: {phase}, Model: {model_name}")
        self._write_log(f"[LLM START] Prompt: {prompt_preview}...")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called on each new token (streaming)."""
        # Skip individual token logs for performance
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM finishes processing."""
        phase = self._phase_stack[-1] if self._phase_stack else "unknown"

        # Extract token usage - try multiple locations for different LangChain versions
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        content_text = ""

        # Extract response text for estimation
        if hasattr(response, "generations") and response.generations and response.generations[0]:
            gen = response.generations[0][0]
            if hasattr(gen, "text"):
                content_text = str(gen.text)

        # Method 1: Check response.llm_output for token_usage (older LangChain)
        if response.llm_output and hasattr(response.llm_output, "token_usage"):
            usage = response.llm_output.token_usage or {}
            total_tokens = usage.get("total_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

        # Method 2: Check response object directly for usage_metadata (newer LangChain)
        if total_tokens == 0 and hasattr(response, "usage_metadata"):
            usage = response.usage_metadata or {}
            total_tokens = usage.get("total_tokens", 0)
            prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

        # Method 3: Check response_metadata for token usage
        if total_tokens == 0 and hasattr(response, "response_metadata"):
            metadata = response.response_metadata or {}
            if "token_usage" in metadata:
                usage = metadata["token_usage"] or {}
                total_tokens = usage.get("total_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

        # Method 4: Check generations for generation_info (older LangChain)
        if total_tokens == 0:
            generations = response.generations or []
            if generations and generations[0]:
                gen = generations[0][0]
                if hasattr(gen, "generation_info"):
                    info = gen.generation_info or {}
                    prompt_tokens = info.get("prompt_tokens", 0)
                    completion_tokens = info.get("completion_tokens", 0)
                    total_tokens = prompt_tokens + completion_tokens

        # Fallback: Estimate tokens from text length if no token usage found
        # This handles cases where LLM provider doesn't return token metadata
        if total_tokens == 0 and content_text:
            # Rough estimation: ~4 characters per token (English text average)
            completion_tokens = len(content_text) // 4
            total_tokens = completion_tokens
            # Mark as estimated for clarity
            self._write_log(f"[LLM END] Phase: {phase}, Tokens: {total_tokens} (estimated, in: {prompt_tokens}, out: {completion_tokens})")

            if self.run_logger:
                # Skip automatic logging for Boss phase - Boss handles its own logging
                if phase.lower() != "boss":
                    self.run_logger.log_phase_call(
                        phase=phase,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        time_seconds=time.time() - self._current_phase_start
                    )

            self._llm_calls.append({
                "phase": phase,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            })
            return

        # Log content preview for debugging
        content_preview = content_text[:100] if content_text else ""

        self._write_log(
            f"[LLM END] Phase: {phase}, "
            f"Tokens: {total_tokens} (in: {prompt_tokens}, out: {completion_tokens})"
        )

        if self.run_logger:
            # Skip automatic logging for Boss phase - Boss handles its own logging via log_phase_call
            # This prevents double-logging when Boss uses preset graphs (0 tokens) + guidance LLM (actual tokens)
            if phase.lower() != "boss":
                # Always log the call (even if tokens are 0 - timing is still useful)
                self.run_logger.log_phase_call(
                    phase=phase,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    time_seconds=time.time() - self._current_phase_start
                )

        self._llm_calls.append({
            "phase": phase,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        })

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM encounters an error."""
        self._write_log(f"[LLM ERROR] {type(error).__name__}: {error}")

    # ========== CHAIN CALLBACKS (for phases) ==========

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when a chain (phase) starts."""
        # Handle case where serialized is None or not a dict
        if serialized is None:
            return
        if isinstance(serialized, dict):
            chain_name = serialized.get("name", "unknown")
        else:
            chain_name = str(serialized) if serialized else "unknown"

        # Track phase if it's one of our known phases
        # Use strict matching to avoid misattributing chains like "worker_boss_helper" to "boss"
        known_phases = ["worker", "boss", "researcher", "critic", "synthesis", "spawner"]
        chain_name_lower = chain_name.lower()
        # Match only if chain_name starts with a known phase or is exactly a known phase
        # This prevents "boss_helper" from being logged as "boss"
        is_known_phase = any(
            chain_name_lower == phase or chain_name_lower.startswith(phase + "_")
            for phase in known_phases
        )
        if is_known_phase:
            self._phase_stack.append(chain_name)
            self._current_phase_start = time.time()

            if self.run_logger:
                self.run_logger.log_phase_start(chain_name)

            self._write_log(f"[PHASE START] {chain_name}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when a chain (phase) ends."""
        if self._phase_stack:
            chain_name = self._phase_stack.pop()
            elapsed = time.time() - self._current_phase_start

            # Suppress automatic log_phase_end for Boss phase
            # Boss manually logs via log_phase_call for its guidance LLM
            # The callback would log an empty wrapper (0.00s, 0 tokens) before guidance completes
            # Boss always does manual logging, so we skip callback's automatic log_phase_end
            if self.run_logger:
                phase_lower = chain_name.lower()
                if phase_lower != "boss":
                    self.run_logger.log_phase_end(chain_name, elapsed)
                # Boss phase: skip automatic log_phase_end entirely (manual logging only)

            self._write_log(f"[PHASE END] {chain_name} ({elapsed:.2f}s)")

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when a chain encounters an error."""
        if self._phase_stack:
            chain_name = self._phase_stack.pop()
            self._write_log(f"[PHASE ERROR] {chain_name}: {error}")

    # ========== TOOL CALLBACKS ==========

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts executing."""
        # Handle None serialized
        if serialized is None or not isinstance(serialized, dict):
            tool_name = kwargs.get("name", "unknown")
        else:
            tool_name = serialized.get("name", "unknown")

        self._current_tool_name = tool_name
        self._tool_start_time = time.time()

        # Truncate long inputs
        input_preview = input_str[:200]
        if len(input_str) > 200:
            input_preview += "..."

        self._write_log(f"[TOOL START] {tool_name}")
        self._write_log(f"[TOOL INPUT] {input_preview}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool finishes executing."""
        # Extract tool name from kwargs if available
        tool_name = kwargs.get("name", self._current_tool_name)

        # Calculate duration
        duration = time.time() - self._tool_start_time

        # Handle different output types (string or ToolMessage object)
        output_str = str(output) if not isinstance(output, str) else output

        # Truncate long outputs
        output_preview = output_str[:200]
        if len(output_str) > 200:
            output_preview += "..."

        self._write_log(f"[TOOL END] {tool_name} ({duration:.3f}s)")
        self._write_log(f"[TOOL OUTPUT] {output_preview}")

        # Extract researcher/worker ID from phase stack
        researcher_id = "unknown"
        if self._phase_stack:
            current_phase = self._phase_stack[-1]
            # Check if phase contains researcher ID (e.g., "researcher_0", "researcher_1")
            for phase in self._phase_stack:
                if "researcher" in phase.lower():
                    researcher_id = phase
                    break
            # Fallback: check if current phase is a researcher
            if researcher_id == "unknown" and "researcher" in current_phase.lower():
                researcher_id = current_phase

        self._tool_calls.append({
            "tool": tool_name,
            "output_length": len(output_str),
            "duration": duration,
            "researcher": researcher_id
        })

        # Log to RunLogger for metrics
        if self.run_logger:
            self.run_logger.log_tool_call(tool_name, duration, len(output_str))

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when a tool encounters an error."""
        tool_name = kwargs.get("name", "unknown")
        self._write_log(f"[TOOL ERROR] {tool_name}: {error}")

    # ========== AGENT CALLBACKS ==========

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Called when agent takes an action."""
        self._write_log(f"[AGENT ACTION] {action.tool}: {action.tool_input}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Called when agent finishes."""
        output = finish.return_values.get("output", "")
        # Convert to string safely in case output is a dict
        output_str = str(output)[:100] if output else ""
        self._write_log(f"[AGENT FINISH] {output_str}")

    # ========== RETRIEVER CALLBACKS ==========

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        **kwargs: Any,
    ) -> None:
        """Called when retriever starts."""
        self._write_log(f"[RETRIEVER START] Query: {query[:100]}")

    def on_retriever_end(self, documents: List[Any], **kwargs: Any) -> None:
        """Called when retriever finishes."""
        self._write_log(f"[RETRIEVER END] Found {len(documents)} documents")

    # ========== UTILITIES ==========

    def _write_log(self, message: str) -> None:
        """Write log message to both handlers."""
        if self.verbose:
            logger.info(message)

        if self.run_logger:
            self.run_logger._write_to_file(message)

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary.

        Returns:
            Dict with llm_calls, tool_calls, phases
        """
        return {
            "llm_calls": self._llm_calls,
            "tool_calls": self._tool_calls,
            "phases": self._phase_stack,
            "total_time": time.time() - self._start_time
        }

    def finalize(
        self,
        final_response: str,
        iteration_count: int = 1,
        sources: Optional[List[str]] = None,
        citations: Optional[List[Any]] = None
    ) -> None:
        """Finalize logging and save.

        Args:
            final_response: Final agent response
            iteration_count: Number of research loop iterations (from graph state)
            sources: Optional list of source names (e.g., ["BBC", "CNN"])
            citations: Optional list of full citation objects with URLs, excerpts, etc.
        """
        if self.run_logger:
            summary = self.get_summary()
            total_llm_tokens = sum(c.get("total_tokens", 0) for c in self._llm_calls)

            # Set start_time and end_time on run_logger metrics for calculation
            if self.run_logger.metrics.start_time == 0.0:
                self.run_logger.metrics.start_time = self._start_time
            self.run_logger.metrics.end_time = time.time()

            # Use actual iteration_count from graph state, not LLM call count
            self.run_logger.set_response(
                response=final_response,
                iterations=iteration_count,
                source_count=len(self._tool_calls)
            )

            # Set sources and citations if provided
            if sources is not None or citations is not None:
                self.run_logger.set_sources(
                    sources=sources or [],
                    citations=citations or []
                )

            # Set tool calls breakdown
            self.run_logger.set_tool_calls(self._tool_calls)

            # Note: prompt_breakdown is populated by nodes (worker, boss) via log_worker_to_boss/log_boss_to_workers
            # We don't set it here to avoid overwriting the data logged by nodes

            self.run_logger.log_phase_end("spawner", summary["total_time"])

            self.run_logger.save()

        # Log final summary with clear labels
        self._write_log(f"[FINALIZE] Total time: {summary['total_time']:.2f}s")
        self._write_log(f"[FINALIZE] Research iterations: {iteration_count}")
        self._write_log(f"[FINALIZE] Direct LLM calls: {len(self._llm_calls)} (Boss, Architect, Critic, Synthesis)")
        self._write_log(f"[FINALIZE] Worker tool calls tracked separately in researcher_tool_stats")


def create_callback_handler(
    run_logger: Optional[RunLogger] = None,
    verbose: bool = True
) -> SpawnerCallbackHandler:
    """Factory to create a callback handler.

    Args:
        run_logger: RunLogger instance
        verbose: Enable verbose logging

    Returns:
        SpawnerCallbackHandler instance
    """
    return SpawnerCallbackHandler(run_logger=run_logger, verbose=verbose)


# =============================================================================
# OBSERVABILITY CALLBACK (from middleware.py)
# =============================================================================

class ObservabilityCallback(BaseCallbackHandler):
    """Callback handler for logging agent execution to files.

    This is a simpler callback for general agent logging, suitable for
    use with LangChain LLMs and simple agents.

    Logs:
    - app.log: Structured log with timestamps
    - metrics.json: Token usage, timing, tool calls
    - output.md: Human-readable run summary

    Callbacks:
    - on_llm_start: Log each model call start
    - on_llm_end: Log each model call completion with timing
    - on_tool_start: Log each tool call start
    - on_tool_end: Log each tool call completion
    - on_chain_start: Initialize logging for a new run
    - on_chain_end: Finalize and save logs
    """

    def __init__(
        self,
        run_logger: Optional[RunLogger] = None,
        run_name: str = "agent",
        use_case: str = "rag",
        variant: str = "default",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the observability callback.

        Args:
            run_logger: Existing RunLogger instance (optional)
            run_name: Name for this run
            use_case: Use case name
            variant: Config variant
            config: Configuration dict
        """
        super().__init__()
        self.run_logger = run_logger or RunLogger(
            run_name=run_name,
            config=config,
            use_case=use_case,
            variant=variant
        )
        self._model_call_count = 0
        self._tool_call_count = 0
        self._start_time: Optional[float] = None
        self._current_llm_start: Optional[float] = None
        self._current_tool_start: Optional[float] = None
        self._query: Optional[str] = None

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        run_id: str,
        **kwargs
    ) -> None:
        """Called when the chain/agent starts execution."""
        self._start_time = time.time()

        # Extract query from inputs/messages
        messages = inputs.get("messages", inputs.get("input", []))
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get("role") == "user":
                        self._query = msg.get("content", "")
                        break
                elif hasattr(msg, "type"):
                    if msg.type == "human":
                        self._query = msg.content
                        break

        if self._query:
            self.run_logger.log_query(self._query)

        self.run_logger.log_phase_start("agent", details={
            "messages_count": len(messages) if isinstance(messages, list) else 0,
            "query_preview": (self._query or "")[:100],
            "run_id": run_id
        })

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: str,
        **kwargs
    ) -> None:
        """Called when LLM starts processing."""
        self._model_call_count += 1
        self._current_llm_start = time.time()

        model_name = serialized.get("name", "unknown") if serialized else "unknown"
        prompt_text = "\n".join(prompts) if prompts else ""

        self.run_logger._write_to_file(
            f"[MODEL CALL {self._model_call_count}] Starting call with model: {model_name}"
        )

    def on_llm_end(
        self,
        response: LLMResult,
        run_id: str,
        **kwargs
    ) -> None:
        """Called when LLM finishes processing."""
        if self._current_llm_start is None:
            return

        elapsed = time.time() - self._current_llm_start
        self._current_llm_start = None

        generations = response.generations or []
        if generations and generations[0]:
            generation = generations[0][0]
            content = generation.text if hasattr(generation, "text") else str(generation)
            completion_tokens = len(content) // 4

            self.run_logger.log_phase_call(
                phase=f"model_call_{self._model_call_count}",
                prompt_tokens=0,
                completion_tokens=completion_tokens,
                time_seconds=elapsed
            )

            self.run_logger._write_to_file(
                f"[MODEL CALL {self._model_call_count}] Completed in {elapsed:.2f}s"
            )

    def on_llm_error(
        self,
        error: Exception,
        run_id: str,
        **kwargs
    ) -> None:
        """Called when LLM encounters an error."""
        if self._current_llm_start is None:
            return

        elapsed = time.time() - self._current_llm_start
        self._current_llm_start = None

        self.run_logger._write_to_file(
            f"[MODEL CALL {self._model_call_count}] ERROR after {elapsed:.2f}s: {type(error).__name__}: {error}"
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: str,
        **kwargs
    ) -> None:
        """Called when a tool starts executing."""
        self._tool_call_count += 1
        self._current_tool_start = time.time()

        tool_name = serialized.get("name", "unknown") if serialized else "unknown"

        self.run_logger._write_to_file(
            f"[TOOL CALL {self._tool_call_count}] {tool_name}()"
        )

    def on_tool_end(
        self,
        output: str,
        run_id: str,
        **kwargs
    ) -> None:
        """Called when a tool finishes executing."""
        if self._current_tool_start is None:
            return

        elapsed = time.time() - self._current_tool_start
        self._current_tool_start = None

        self.run_logger._write_to_file(
            f"[TOOL CALL {self._tool_call_count}] Completed in {elapsed:.2f}s"
        )

    def on_tool_error(
        self,
        error: Exception,
        run_id: str,
        **kwargs
    ) -> None:
        """Called when a tool encounters an error."""
        if self._current_tool_start is None:
            return

        elapsed = time.time() - self._current_tool_start
        self._current_tool_start = None

        tool_name = kwargs.get("name", "unknown")

        self.run_logger._write_to_file(
            f"[TOOL CALL {self._tool_call_count}] {tool_name} FAILED after {elapsed:.2f}s: {error}"
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        run_id: str,
        **kwargs
    ) -> None:
        """Called when the chain/agent completes execution."""
        elapsed = time.time() - self._start_time if self._start_time else 0

        # Extract final response
        final_response = ""
        outputs_messages = outputs.get("outputs", outputs).get("messages", outputs.get("messages", []))
        if isinstance(outputs_messages, list):
            for msg in reversed(outputs_messages):
                if isinstance(msg, dict):
                    if msg.get("role") == "assistant":
                        final_response = msg.get("content", "")
                        break
                elif hasattr(msg, "type"):
                    if msg.type == "ai":
                        final_response = msg.content
                        break

        self.run_logger.set_response(
            response=final_response,
            iterations=self._model_call_count,
            source_count=self._tool_call_count
        )

        self.run_logger.log_phase_end("agent", elapsed, details={
            "total_model_calls": self._model_call_count,
            "total_tool_calls": self._tool_call_count,
            "final_response_length": len(final_response),
            "run_id": run_id
        })

        self.run_logger.save()

    def on_chain_error(
        self,
        error: Exception,
        run_id: str,
        **kwargs
    ) -> None:
        """Called when the chain/agent encounters an error."""
        elapsed = time.time() - self._start_time if self._start_time else 0

        self.run_logger._write_to_file(
            f"[AGENT ERROR] Failed after {elapsed:.2f}s: {type(error).__name__}: {error}"
        )
        self.run_logger.save()


def create_observability_callback(
    run_name: str = "agent",
    use_case: str = "rag",
    variant: str = "default",
    config: Optional[Dict[str, Any]] = None
) -> ObservabilityCallback:
    """Factory function to create observability callback.

    This is the recommended modern pattern for logging in LangChain/LangGraph.

    Args:
        run_name: Name for this run
        use_case: Use case name
        variant: Config variant
        config: Configuration dict

    Returns:
        ObservabilityCallback instance

    Example:
        from deep_research.observability import create_observability_callback

        callback = create_observability_callback(run_name="my_agent")
        result = graph.invoke(state, callbacks=[callback])
    """
    return ObservabilityCallback(
        run_name=run_name,
        use_case=use_case,
        variant=variant,
        config=config
    )
