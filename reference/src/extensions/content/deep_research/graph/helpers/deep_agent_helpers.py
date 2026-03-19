"""Deep Agent wrapper helpers for structured LLM outputs.

Provides reusable functions for creating and invoking deep_agent with proper
configuration, middleware, and error handling.

This module encapsulates:
- Deep agent creation with middleware
- Tool registry setup
- System prompt construction
- Agent invocation with error handling
- Structured extraction from agent outputs (LLM-based, not manual JSON parsing)
"""

from typing import Dict, Any, List, Optional, Tuple

# Add backend to path for absolute imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=5)

# Import prompt helper
from .prompts import get_prompt_from_config


def get_llm_response_content(response: Any, run_logger: Any = None) -> str:
    """Extract content from LLM response, handling thinking mode.

    When thinking mode is enabled (GLM-4.7-Flash, etc.), the actual content
    is often in 'reasoning_content' or 'reasoning' field, not 'content'.

    Args:
        response: LLM response object (LLMChunk or similar)
        run_logger: Optional logger for debugging

    Returns:
        String content from the response
    """
    # DEBUG: Log all response attributes
    if run_logger:
        attrs = {attr: type(getattr(response, attr, None)).__name__
                 for attr in dir(response) if not attr.startswith('_')}
        run_logger._write_to_file(f"[CONTENT_EXTRACTOR] Response type: {type(response).__name__}")
        run_logger._write_to_file(f"[CONTENT_EXTRACTOR] Response attrs: {list(attrs.keys())[:20]}")

    # Try content first (normal mode)
    content = getattr(response, 'content', '') or ''

    # DEBUG: Log initial content
    if run_logger:
        run_logger._write_to_file(f"[CONTENT_EXTRACTOR] Initial content: {len(content)} chars")

    # If content is empty or very short, check reasoning fields
    if not content or len(content.strip()) < 10:
        # Check reasoning attribute (our LLMChunk format)
        reasoning = getattr(response, 'reasoning', None)
        if run_logger:
            run_logger._write_to_file(f"[CONTENT_EXTRACTOR] reasoning attr: {bool(reasoning)}, len={len(str(reasoning)) if reasoning else 0}")
        if reasoning and len(str(reasoning).strip()) > len(content.strip()):
            content = str(reasoning)

        # Check additional_kwargs for reasoning_content (LangChain format)
        if (not content or len(content.strip()) < 10) and hasattr(response, 'additional_kwargs'):
            reasoning_content = response.additional_kwargs.get('reasoning_content')
            if run_logger:
                run_logger._write_to_file(f"[CONTENT_EXTRACTOR] additional_kwargs keys: {list(response.additional_kwargs.keys())}")
                run_logger._write_to_file(f"[CONTENT_EXTRACTOR] reasoning_content: {bool(reasoning_content)}, len={len(str(reasoning_content)) if reasoning_content else 0}")
            if reasoning_content and len(str(reasoning_content).strip()) > len(content.strip()):
                content = str(reasoning_content)

        # Check response_metadata for reasoning_content
        if (not content or len(content.strip()) < 10) and hasattr(response, 'response_metadata'):
            reasoning_content = response.response_metadata.get('reasoning_content')
            if run_logger:
                run_logger._write_to_file(f"[CONTENT_EXTRACTOR] response_metadata keys: {list(response.response_metadata.keys()) if response.response_metadata else 'None'}")
            if reasoning_content and len(str(reasoning_content).strip()) > len(content.strip()):
                content = str(reasoning_content)

    if run_logger:
        run_logger._write_to_file(f"[CONTENT_EXTRACTOR] Final content: {len(content)} chars, preview: {content[:200]}...")

    return content


def create_critic_deep_agent(
    llm: Any,
    tools: List,
    system_prompt: str,
    run_logger: Optional[Any] = None,
    yaml_config: Optional[Dict[str, Any]] = None,
    pydantic_model: Optional[Any] = None,
) -> Any:
    """Create deep_agent for critic node with middleware.

    Args:
        llm: The LLM instance to use
        tools: List of tools for the agent
        system_prompt: System prompt for the critic
        run_logger: Optional RunLogger for logging
        yaml_config: Optional YAML config dict with middleware settings
        pydantic_model: Optional Pydantic model (NOT used with create_deep_agent,
            used separately in extract_structured_output instead)

    Returns:
        Configured deep_agent instance

    Raises:
        RuntimeError: If deep_agent creation fails
    """
    from deepagents import create_deep_agent
    from langchain.agents.middleware import ToolRetryMiddleware, ToolCallLimitMiddleware

    # NOTE: Do NOT use with_structured_output() with create_deep_agent!
    # The deepagents package expects string messages for its internal reasoning loop.
    # Structured extraction happens AFTER agent completes via extract_structured_output().
    model = llm

    # Get middleware config from YAML or use defaults
    middleware_config = yaml_config.get("middleware", {}) if yaml_config else {}
    retry_config = middleware_config.get("tool_retry", {})
    call_limit_config = middleware_config.get("tool_call_limit", {})

    critic_middleware = [
        ToolRetryMiddleware(
            max_retries=retry_config.get("max_retries", 2),
            backoff_factor=retry_config.get("backoff_factor", 2.0),
            initial_delay=retry_config.get("initial_delay_seconds", 1.0),
            max_delay=retry_config.get("max_delay_seconds", 60.0),
            jitter=retry_config.get("jitter", True),
            on_failure=retry_config.get("on_failure", "raise"),
        ),
        ToolCallLimitMiddleware(
            tool_name="fact_check",
            thread_limit=call_limit_config.get("fact_check_thread_limit", 5),
        ),
    ]

    try:
        critic_agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            middleware=critic_middleware,
        )

        if run_logger:
            run_logger._write_to_file("[DEEP_AGENT] Created critic deep_agent with middleware")

        return critic_agent

    except Exception as e:
        raise RuntimeError(
            "\n"
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║     DEEP_AGENT ERROR: Failed to Create Critic Agent                 ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            f"Failed to create deep_agent: {type(e).__name__}: {e}\n"
            "\n"
            "FIX:\n"
            "  1. Ensure deepagents package is installed\n"
            "  2. Check LLM supports tool calling\n"
            "  3. Verify tools are properly configured\n"
            "\n"
        ) from e


async def invoke_deep_agent(
    agent: Any,
    prompt: str,
    tracker: Any,
    run_logger: Optional[Any] = None,
    role: str = "critic"
) -> Tuple[str, int, int, List[str]]:
    """Invoke deep_agent and extract response with metrics.

    Args:
        agent: The deep_agent instance
        prompt: The user prompt to send
        tracker: AgentTracker for monitoring
        run_logger: Optional RunLogger for logging
        role: Role name for logging (default "critic")

    Returns:
        Tuple of (response_text, llm_count, tool_count, tools_used)

    Raises:
        RuntimeError: If agent invocation fails or returns empty response
    """
    try:
        if run_logger:
            run_logger._write_to_file(f"[{role.upper()}] Invoking deep_agent with prompt ({len(prompt)} chars)...")

        # Invoke agent with callbacks
        result = await agent.ainvoke(
            {"messages": [{"role": "system", "content": prompt}]},
            config={"debug": True},
            callbacks=[tracker]  # Callbacks passed as separate parameter
        )

        # Extract response from agent result
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]

            if run_logger:
                run_logger._write_to_file(f"[{role.upper()}] Agent returned {len(messages)} messages")

            # Manual tracking of LLM calls and tool usage
            llm_count = 0
            tool_count = 0
            tools_used = []

            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                if run_logger:
                    run_logger._write_to_file(f"[{role.upper()}] Message {i}: type={msg_type}")

                # Check for AIMessage or LLMChunk (LLM responses)
                if msg_type in ("AIMessage", "LLMChunk", "ChatMessage"):
                    llm_count += 1
                    # Check for tool_calls attribute
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_count += 1
                            tool_name = tc.get("name", "unknown")
                            tools_used.append(tool_name)
                            if run_logger:
                                run_logger._write_to_file(f"[{role.upper()}]   Tool call: {tool_name}")

                # Check for ToolMessage (tool results)
                elif msg_type == "ToolMessage" or (hasattr(msg, "name") and msg.name):
                    tool_count += 1
                    tool_name = getattr(msg, "name", "unknown")
                    tools_used.append(tool_name)
                    if run_logger:
                        run_logger._write_to_file(f"[{role.upper()}]   Tool result: {tool_name}")

            if run_logger:
                run_logger._write_to_file(f"[{role.upper()}] Execution: {llm_count} LLM calls, {tool_count} tool invocations")
                if tools_used:
                    run_logger._write_to_file(f"[{role.upper()}] Tools: {', '.join(set(tools_used))}")

            # Extract response text
            if messages:
                last_msg = messages[-1]
                response_text = str(last_msg.content) if hasattr(last_msg, 'content') else str(last_msg)
                if run_logger:
                    run_logger._write_to_file(f"[{role.upper()}] Response length: {len(response_text)} chars")
                    run_logger._write_to_file(f"[{role.upper()}] Response preview: {response_text[:500]}")

                # Check for empty response
                if not response_text or len(response_text.strip()) == 0:
                    # DIAGNOSTIC: Track empty response for debugging
                    from ...observability.diagnostic_logging import track_critic_response
                    track_critic_response(
                        messages=messages,
                        response_text=response_text,
                        run_logger=run_logger,
                    )
                    raise RuntimeError(
                        f"[{role.upper()}] Deep_agent returned empty response"
                    )

                return response_text, llm_count, tool_count, tools_used
            else:
                raise RuntimeError(
                    f"[{role.upper()}] No messages in agent result"
                )
        else:
            # Result is not dict with messages
            response_text = str(result)
            if run_logger:
                run_logger._write_to_file(f"[{role.upper()}] Result is not dict with messages, type={type(result)}")
            return response_text, 0, 0, []

    except Exception as e:
        # Check if this is an empty response error we raised
        if isinstance(e, RuntimeError) and "empty response" in str(e):
            raise
        # Other errors - wrap with context
        raise RuntimeError(
            "\n"
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║         DEEP_AGENT ERROR: Agent Invocation Failed                     ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            f"Deep agent encountered error: {type(e).__name__}: {e}\n"
            "\n"
            "FIX:\n"
            "  1. Check LLM is available and configured correctly\n"
            "  2. Verify tools are properly registered\n"
            "  3. Check prompt is valid\n"
            "\n"
        ) from e


async def extract_structured_output(
    llm: Any,
    agent_output: str,
    pydantic_model: Any,
    extraction_description: Optional[str] = None,
    extraction_prompt_key: Optional[str] = None,
    run_logger: Optional[Any] = None,
    role: str = "extractor",
    yaml_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Extract structured data from agent output using LLM with structured output.

    This implements the "Draft then Extract" pattern:
    1. Deep agent runs freely and returns natural language text
    2. Separate LLM call with with_structured_output() extracts validated data

    This is more reliable than manual JSON parsing because:
    - Handles conversational noise in agent response
    - Uses provider-native structured output (constrained decoding)
    - Built-in Pydantic validation with automatic retry on failure

    CRITICAL: The extraction LLM uses enable_thinking=False because:
    - Extraction is a simple parsing/structuring task, not reasoning
    - The deep agent already did the thinking - extraction just parses output
    - Thinking mode would waste tokens and slow down extraction

    Args:
        llm: The LLM instance to use (will be cloned with thinking disabled)
        agent_output: Raw text output from deep agent
        pydantic_model: Pydantic model class for structured output
        extraction_description: Optional description of what to extract (hardcoded)
        extraction_prompt_key: Optional key to load description from YAML config
            (e.g., "extraction.boss_decision" loads from prompts yaml)
            Takes precedence over extraction_description if both provided
        run_logger: Optional RunLogger for logging
        role: Role name for logging (default "extractor")
        yaml_config: Optional YAML config for creating extraction LLM with thinking disabled

    Returns:
        Validated Pydantic model instance

    Raises:
        RuntimeError: If extraction LLM call fails
        pydantic.ValidationError: If validation fails (caught by LangGraph retry)
    """
    from pydantic import BaseModel

    # Build extraction description - prefer loading from config
    schema_desc = _get_schema_description(pydantic_model)

    if extraction_prompt_key and yaml_config:
        # Load description from YAML config (preferred)
        full_extraction_description = get_prompt_from_config(
            yaml_config,
            extraction_prompt_key,
        )
    elif extraction_description:
        # Use provided description
        full_extraction_description = extraction_description
    else:
        # Fallback to schema description
        full_extraction_description = f"Extract into this exact JSON schema:\n{schema_desc}"

    # Get extraction prompt from config (CRASH-ONLY - no fallbacks)
    if yaml_config:
        prompt = get_prompt_from_config(
            yaml_config,
            "extraction.json",
            agent_output=agent_output,
            extraction_description=full_extraction_description,
        )
    else:
        # Fallback if no config - build prompt inline (shouldn't happen in production)
        prompt = f"""You are a JSON extraction assistant. Extract ONLY from the agent response below.

AGENT RESPONSE:
{agent_output}

{full_extraction_description}

CRITICAL REQUIREMENTS:
- Output ONLY a valid JSON object
- Do NOT include any explanatory text before or after the JSON
- Do NOT repeat the agent response
- Do NOT include labels like "AGENT RESPONSE:"

Your response must be ONLY the JSON object, nothing else."""

    try:
        if run_logger:
            run_logger._write_to_file(f"[{role.upper()}] Starting structured extraction (input: {len(agent_output)} chars)")

        # CRITICAL: Create extraction LLM with tool model (fast, no thinking)
        # Extraction is a simple parsing task - no complex reasoning needed
        # Use the tool model (Flash) which is faster and cheaper than the main Pro model
        # The deep agent already did the thinking - extraction just parses the output
        if yaml_config:
            from .llm import create_llm_from_config
            extraction_llm_base = await create_llm_from_config(
                yaml_config,
                role="worker",  # CRITICAL: Use worker role → tool model (Flash)
                enable_thinking=False  # CRITICAL: Disable thinking for extraction
            )
            extraction_llm = extraction_llm_base.with_structured_output(pydantic_model)
            if run_logger:
                run_logger._write_to_file(f"[{role.upper()}] Created extraction LLM with tool model (Flash, thinking=False)")
        else:
            # Fallback: use provided LLM (may have thinking enabled)
            extraction_llm = llm.with_structured_output(pydantic_model)
            if run_logger:
                run_logger._write_to_file(f"[{role.upper()}] Using provided LLM for extraction (WARNING: may use Pro model with thinking)")

        # Invoke extraction
        result = await extraction_llm.ainvoke(prompt)

        if run_logger:
            run_logger._write_to_file(f"[{role.upper()}] Extraction succeeded, got {type(result).__name__}")

        return result

    except Exception as e:
        if run_logger:
            run_logger._write_to_file(f"[{role.upper()}] Extraction failed: {e}")
        raise RuntimeError(
            f"[{role.upper()}] Structured extraction failed: {e}"
        ) from e


def extract_synthesis_content(response: Any, run_logger: Optional[Any] = None) -> str:
    """Extract clean synthesis content from LLM response.

    This implements deterministic content extraction:
    - Strips out JSON tool calls that LLM might output
    - Removes markdown code blocks containing tool definitions
    - Returns clean text for synthesis output

    This ensures the system works regardless of prompt content - even if
    the prompt mentions tools, we strip any tool-like output.

    Args:
        response: LLM response (AIMessage or similar)
        run_logger: Optional RunLogger for logging

    Returns:
        Clean synthesis text content
    """
    import re

    # Get content from response
    if hasattr(response, 'content'):
        content = response.content
    elif isinstance(response, str):
        content = response
    else:
        content = str(response)

    if not content:
        return ""

    # Strip out JSON code blocks that look like tool calls
    # Pattern matches ```json ... "name": "query_postgres" ... ```
    tool_call_pattern = r'```json\s*\{[^`]*"name"\s*:\s*"(query_postgres|get_artifact|read_vfs_raw|query_neo4j)"[^`]*\}\s*```'
    content = re.sub(tool_call_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

    # Also strip inline JSON tool calls (without markdown)
    # Pattern matches {"name": "some_tool", ...}
    inline_tool_pattern = r'\{\s*"name"\s*:\s*"(query_postgres|get_artifact|read_vfs_raw|query_neo4j)"[^}]*\}'
    content = re.sub(inline_tool_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

    # Strip "Example tool call:" sections
    example_pattern = r'Example tool call:\s*\{[^}]*\}'
    content = re.sub(example_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

    # Clean up excessive whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()

    if run_logger:
        run_logger._write_to_file(f"[SYNTHESIS_EXTRACTOR] Extracted {len(content)} chars of clean content")

    return content


def _get_schema_description(pydantic_model: Any) -> str:
    """Generate a human-readable description from Pydantic model with TYPE information.

    Args:
        pydantic_model: Pydantic model class

    Returns:
        Description of model fields with type hints
    """
    from pydantic import BaseModel
    from typing import get_origin, get_args
    import typing

    if not issubclass(pydantic_model, BaseModel):
        return f"data matching schema: {pydantic_model.__name__}"

    def get_type_hint(annotation) -> str:
        """Convert type annotation to human-readable string."""
        origin = get_origin(annotation)

        if origin is list or origin is typing.List:
            args = get_args(annotation)
            if args:
                inner_type = get_type_hint(args[0])
                return f"list of {inner_type}"
            return "list"
        elif origin is dict or origin is typing.Dict:
            return "object/dict"
        elif annotation is str:
            return "string"
        elif annotation is int:
            return "integer"
        elif annotation is float:
            return "number"
        elif annotation is bool:
            return "boolean"
        elif hasattr(annotation, '__name__'):
            return annotation.__name__
        else:
            return str(annotation)

    descriptions = []
    for field_name, field_info in pydantic_model.model_fields.items():
        field_desc = field_info.description or ""
        type_hint = get_type_hint(field_info.annotation)

        if field_desc:
            descriptions.append(f"- {field_name} ({type_hint}): {field_desc}")
        else:
            descriptions.append(f"- {field_name} ({type_hint})")

    return "\n".join(descriptions)
