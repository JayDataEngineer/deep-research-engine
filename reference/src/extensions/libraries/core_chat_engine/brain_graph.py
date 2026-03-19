"""Companion Brain Graph v2 - Full LangChain Tool Calling Integration

This version uses LangChain's tool calling with LangGraph's ToolNode.
The LLM decides when to call tools based on the conversation context.

Uses Pydantic models for state validation.
"""

import logging
import time
import json
import uuid
import asyncio
from typing import TypedDict, Annotated, Optional, Dict, Any, List, Literal, NotRequired
from datetime import datetime
import sys
from pathlib import Path

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import StreamWriter
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.extensions.libraries.core_chat_engine.config.brain_config import BrainConfig, get_preset_config
from src.extensions.libraries.core_chat_engine.config.personality_config import PersonalityConfig, get_preset
from src.extensions.libraries.core_chat_engine.state.brain_state import (
    ConversationState,
    UserProfile,
)
from src.extensions.libraries.core_chat_engine.state.typed_state import (
    ProcessedResponse,
    IntentType,
    IntentResult,
)
from src.extensions.libraries.core_chat_engine.conversation.manager import ConversationManager, create_conversation_manager
from src.extensions.libraries.core_chat_engine.personality.layer import PersonalityLayer, create_personality_layer
from src.extensions.libraries.core_chat_engine.memory.context_injection import build_memory_system_message
from src.shared.tools.registry import load_tools

# Mode to tools mapping - reads from extension configs
# This maps mode names to their enabled_tools lists
_MODE_TOOLS = {
    "standard": ["web_search", "web_get"],
    "companion": ["web_search", "web_get", "vector_search", "add_documents"],
    "roleplay": ["web_search", "edit_character_image", "set_character_reference", "clear_character_reference"],
    "deep_research": ["web_search", "web_get", "web_map", "web_crawl", "deep_agent"],
    "rp": ["web_search", "edit_character_image", "set_character_reference", "clear_character_reference"],  # alias for roleplay
}

def get_companion_tools(
    mode: str = "companion",
    tool_names: Optional[List[str]] = None,
    include_experimental: bool = False,
    include_workflow_tools: bool = True,
):
    """Get tools for a specific mode using the central registry.

    This is a compatibility wrapper for the old get_companion_tools function.

    Args:
        mode: Operating mode (used to lookup default tools if tool_names not provided)
        tool_names: Optional list of tool names to load (takes precedence over mode defaults)
        include_experimental: Ignored (for compatibility)
        include_workflow_tools: Ignored (for compatibility)

    Returns:
        List of tool instances for the given mode
    """
    # Use provided tool_names, or fallback to mode defaults
    if tool_names is None:
        tool_names = _MODE_TOOLS.get(mode, _MODE_TOOLS["companion"])
    return load_tools(tool_names)

# Import new LLM abstraction
from src.shared import get_llm, BaseLLM
from src.shared.state.base import add_messages_reducer

logger = logging.getLogger(__name__)


# =============================================================================
# LangGraph State Definition v2
# =============================================================================


class ChatModeState(TypedDict):
    """State for chat mode graphs with tool calling and validation.

    Uses Pydantic models for validation and LangGraph's reducer system.
    """
    # Input messages (conversation history) - with reducer
    # PERSISTENT: Only User input + Final AI response stored here
    messages: Annotated[
        list[BaseMessage],
        "The conversation messages in LangChain format",
        add_messages_reducer,  # Reducer function from shared.state.base
    ]

    # User input for this turn
    user_input: str

    # EPHEMERAL STATE: Tool context cleared every turn
    # Stores raw tool results (web articles, deep_agent outputs, etc.)
    # This is NOT persisted to conversation history
    tool_context: NotRequired[List[str]]  # Raw tool outputs for current turn only

    # Core state (persistent across turns)
    conversation: ConversationState
    user_profile: UserProfile

    # Config
    mode: str
    personality_enabled: bool

    # Tool configuration (from extension config)
    enabled_tools: NotRequired[List[str]]  # Tool names enabled for this session

    # Model information for tracking and persistence
    model: Optional[str]  # Model name (e.g., "glm-4.7", "gpt-4")
    provider: Optional[str]  # Provider name (e.g., "zai", "openai", "local")

    # Temperature settings for LLM nodes
    main_temperature: NotRequired[float]  # Temperature for main chat LLM
    router_temperature: NotRequired[float]  # Temperature for tool router LLM
    tool_model: NotRequired[str]  # Separate model for tool routing

    # Timestamps
    start_time: float
    processing_time: Optional[float]

    # Thread ID for persistence
    thread_id: str

    # Final response (output of post_process node)
    response: Optional[ProcessedResponse]

    # Tool router results
    router_tool_calls: list[dict]  # Tools to call as decided by router
    router_reasoning: str  # Router's explanation of tool selection

    # Executed tool info for displaying tool badges
    executed_tools: list[dict]  # Tools that were executed with results


# =============================================================================
# Tool Router System
# =============================================================================

TOOL_ROUTER_SYSTEM_PROMPT = """You are a TOOL ROUTER. Your ONLY job is to decide which tools to call based on the user's request.

Available tools:
- generate_image: Create images (generate, draw, create, picture, photo)
- web_search: Search web for current information (search, latest, what's, find, current events)
- web_get: Read specific URLs (when URL is provided or user asks to read a webpage)
- edit_character_image: Modify existing character image (use prompt param for edit instructions like "make her smile")
- save_memory: Save important information to long-term memory (preferences, facts, events, goals, relationships)
- search_memory: Search through stored long-term memories
- save_character: Save character to storage
- load_character: Load character from storage
- list_characters: Show available characters

Rules:
1. If user asks for current events, news, weather, or recent information → web_search
2. If user provides a URL or asks to read/scrape a webpage → web_get
3. If user asks for any image, picture, drawing, or visual content → generate_image
4. If user wants to modify a character's expression, pose, or appearance → edit_character_image
   - IMPORTANT: Use "prompt" parameter for edit instructions (e.g., {"prompt": "make her smile"})
5. If user shares personal information (preferences, facts, goals, events) → save_memory
6. If user asks about their past information or what you remember → search_memory
7. If user asks about saving/loading/listing characters → use character management tools
8. If user just wants to chat with NO tools needed → return empty tools list

Output ONLY valid JSON in this exact format:
{
  "tools": [
    {"name": "tool_name", "params": {"param_name": "param_value"}}
  ],
  "reasoning": "Brief explanation of why these tools were selected"
}

Do NOT include any text outside the JSON. Do NOT make up tools that don't exist.
"""


def create_tool_router_node(api_key: str, base_url: str, model: str):
    """Create the tool router node that decides which tools to call.

    This is a specialized node that ONLY makes tool routing decisions.
    It does NOT generate responses - that's the main LLM's job.

    Uses Shared Kernel LLM factory (no legacy code).
    """
    async def tool_router_node(state: ChatModeState) -> Dict[str, Any]:
        """Decide which tools to call based on user message."""
        messages: list[BaseMessage] = state.get("messages", [])
        if not messages:
            return {"router_tool_calls": []}

        # EARLY EXIT: If no tools are enabled, skip routing entirely (direct mode)
        enabled_tools = state.get("enabled_tools", None)
        if enabled_tools is not None and len(enabled_tools) == 0:
            logger.info("[TOOL ROUTER] No tools enabled (direct mode), skipping router")
            return {"router_tool_calls": [], "router_reasoning": "Direct mode - no tools enabled"}

        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = str(msg.content)
                break

        # Import from Shared Kernel - NO LEGACY CODE
        try:
            from src.shared import get_tool_router_llm
        except ImportError:
            logger.error("Shared LLM factory not available for tool router")
            return {"router_tool_calls": []}

        try:
            # NO FALLBACKS - model must be in state
            runtime_tool_model = state.get("tool_model") or state.get("model")
            if not runtime_tool_model:
                raise ValueError("model not found in state - required for tool router")

            # NO FALLBACKS - temperature must be in state
            router_temperature = state.get("router_temperature")
            if router_temperature is None:
                raise ValueError("router_temperature not found in state - required for tool router")

            # Get LLM for tool router (thinking disabled - fast routing only)
            router_llm = await get_tool_router_llm(model=runtime_tool_model, temperature=router_temperature, enable_thinking=False)

            # Build dynamic system prompt with actual tool schemas
            # This ensures the router knows about required parameters
            enabled_tools = state.get("enabled_tools", None)
            tools = get_companion_tools(mode=state.get("mode", "companion"), tool_names=enabled_tools)

            # Build tool schema documentation
            tool_schemas = []
            for tool in tools:
                schema = tool.args_schema.model_json_schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}
                required = schema.get("required", [])
                props = schema.get("properties", {})

                param_docs = []
                for param_name, param_info in props.items():
                    required_mark = "REQUIRED" if param_name in required else f"optional (default: {param_info.get('default', 'none')})"
                    param_docs.append(f"  - {param_name}: {required_mark} - {param_info.get('description', '')}")

                tool_schemas.append(f"- {tool.name}: {tool.description or ''}\n  Parameters:\n" + "\n".join(param_docs))

            dynamic_prompt = f"""You are a TOOL ROUTER. Your ONLY job is to decide which tools to call based on the user's request.

Available tools with their REQUIRED parameters:

{chr(10).join(tool_schemas)}

CRITICAL: You MUST include ALL required parameters when calling a tool.
- If a parameter is marked REQUIRED, you MUST provide it
- For web_search: ALWAYS provide time_range (day/week/month/year/noLimit) and category (general/news/images/videos/etc)
- For news queries, use category="news"
- For general searches, use category="general"

Output ONLY valid JSON in this exact format:
{{
  "tools": [
    {{"name": "tool_name", "params": {{"param_name": "param_value"}}}}
  ],
  "reasoning": "Brief explanation of why these tools were selected"
}}

Do NOT include any text outside the JSON. Do NOT make up tools that don't exist.
"""

            # Build router messages in dict format for BaseLLM
            router_messages = [
                {"role": "system", "content": dynamic_prompt},
                {"role": "user", "content": f"User request: {user_message}"}
            ]

            # Call LLM (non-streaming for router)
            response = await router_llm.ainvoke(router_messages)
            response_text = response.content.strip()

            # Parse JSON response
            import re
            # Extract JSON from response (handle potential markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                response_text = json_match.group(0)

            router_decision = json.loads(response_text)
            tool_calls = router_decision.get("tools", [])

            logger.info(f"[TOOL ROUTER] Selected {len(tool_calls)} tools: {[tc.get('name') for tc in tool_calls]}")

            # Convert to LangChain tool call format
            langchain_tool_calls = []
            for i, tc in enumerate(tool_calls):
                langchain_tool_calls.append({
                    "name": tc["name"],
                    "args": tc.get("params", {}),
                    "id": f"tool_{tc['name']}_{int(time.time() * 1000)}"
                })

            # CRITICAL FIX: Create AIMessage with tool_calls so it's saved to checkpoint
            # This ensures the AI message (with tool decision) is persisted
            from langchain_core.messages import AIMessage
            ai_msg = AIMessage(content="")
            ai_msg.tool_calls = langchain_tool_calls

            return {
                "messages": [ai_msg],  # Add AIMessage to messages so it's saved to checkpoint
                "router_tool_calls": langchain_tool_calls,
                "router_reasoning": router_decision.get("reasoning", "")
            }

        except json.JSONDecodeError as e:
            logger.error(f"[TOOL ROUTER] Failed to parse JSON: {e}")
            return {"router_tool_calls": []}
        except Exception as e:
            logger.error(f"[TOOL ROUTER] Error: {e}")
            return {"router_tool_calls": []}

    return tool_router_node


def should_call_tools(state: ChatModeState) -> Literal["tool_execution", "main_llm"]:
    """Check if router decided to call tools or if we should go straight to main LLM."""
    tool_calls = state.get("router_tool_calls", [])
    has_tools = bool(tool_calls and len(tool_calls) > 0)

    return "tool_execution" if has_tools else "main_llm"


# =============================================================================
# Graph Nodes v2
# =============================================================================

def create_input_node_v2():
    """Create the input processing node."""
    def input_node(state: ChatModeState) -> Dict[str, Any]:
        """Process user input and add to messages."""
        logger.debug("Node: input_processor_v2")

        user_input = state.get("user_input", "")
        if not user_input:
            return {}

        messages = state.get("messages", [])

        # Check if the user_input is already represented in the last message
        # This prevents duplicates when initial_state already contains the human message
        if messages:
            last_message = messages[-1]
            # Check if last message is a HumanMessage with the same content
            if (isinstance(last_message, HumanMessage) and
                last_message.content == user_input):
                logger.debug("[INPUT] User input already in messages, skipping duplicate")
                return {}

        # Create HumanMessage - this will be appended by the reducer
        user_message = HumanMessage(content=user_input)

        return {"messages": [user_message]}

    return input_node


def create_llm_node_v2(api_key: str, base_url: str, model: str, personality: PersonalityLayer):
    """Create the LLM node using LangChain ChatOpenAI (Shared Kernel approach)."""
    async def llm_node(state: ChatModeState, writer: StreamWriter) -> Dict[str, Any]:
        """Call LLM with tools using LangChain ChatOpenAI from Shared Kernel."""
        # Collect CoT steps for persistence (stored in additional_kwargs)
        cot_steps = []

        def collect_and_emit(cot_event: dict) -> None:
            """Collect CoT step for storage and emit via writer.

            Adds required fields (id, label, status, icon) if missing.
            """
            # Ensure CoT event has required fields for persistence
            if "id" not in cot_event:
                cot_event["id"] = str(uuid.uuid4())
            if "label" not in cot_event:
                cot_event["label"] = cot_event.get("message", cot_event.get("type", "Thinking..."))
            if "status" not in cot_event:
                cot_event["status"] = "active"
            if "icon" not in cot_event:
                cot_event["icon"] = "sparkle"
            cot_steps.append(cot_event)
            writer(cot_event)

        # Emit status event - starting to think
        collect_and_emit({
            "type": "status",
            "message": "Thinking..."
        })

        messages: list[BaseMessage] = state.get("messages", [])

        # Import from Shared Kernel - NO LEGACY CODE
        try:
            from src.shared import get_llm
        except ImportError:
            logger.error("Shared LLM factory not available")
            return {
                "messages": messages + [AIMessage(
                    content="I apologize, but the LLM factory is not available.",
                )]
            }

        # Get mode for tool selection
        mode = state.get("mode", "companion")
        enabled_tools = state.get("enabled_tools", None)
        tools = get_companion_tools(mode=mode, tool_names=enabled_tools)

        # Get LLM for tool router (thinking disabled for fast decisions)
        from src.shared import get_tool_router_llm

        # NO FALLBACKS - model must be in state
        runtime_model = state.get("model")
        if not runtime_model:
            raise ValueError("model not found in state - required for tool router")

        # NO FALLBACKS - temperature must be in state
        router_temperature = state.get("router_temperature")
        if router_temperature is None:
            raise ValueError("router_temperature not found in state - required for tool router")

        llm = await get_tool_router_llm(
            model=runtime_model,
            temperature=router_temperature
        )

        # Convert LangChain tools to OpenAI format for BaseLLM
        openai_tools = None
        if tools:
            openai_tools = []
            for tool in tools:
                # Convert LangChain tool to OpenAI function format
                # Use model_json_schema() for Pydantic v2 compatibility
                schema = tool.args_schema.model_json_schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or f"Tool: {tool.name}",
                        "parameters": schema
                    }
                })

        # Add system prompt using prompt_engine library
        # This replaces the old prompt_loader system with configurable Jinja2 templates
        if state.get("personality_enabled"):
            from src.extensions.libraries.core_chat_engine.prompts import build_system_messages

            # Get memory context if available
            memory_context = None
            try:
                from ..memory.context_injection import build_memory_system_message
                memory_msg = build_memory_system_message(
                    current_input=state.get("user_input", ""),
                    user_id=state.get("user_id", "default"),
                )
                memory_context = memory_msg.content if memory_msg else None
            except Exception as e:
                logger.debug(f"[LLM NODE] Could not build memory context: {e}")

            # Build system messages using prompt_engine
            system_messages = build_system_messages(
                mode=state.get("mode", "direct"),
                character=state.get("character"),
                user_profile=state.get("user_profile"),
                memory_context=memory_context,
                personality_config=personality.config if personality else None,
            )

            # Prepend system messages to the conversation
            messages = system_messages + messages
        else:
            # Even without personality, add a minimal system prompt
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content="You are a helpful AI assistant.")] + messages

        # Convert LangChain messages to dict format for BaseLLM
        llm_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                llm_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                llm_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                llm_messages.append({"role": "system", "content": msg.content})

        # Collect response data
        response_content = ""
        reasoning_content = ""
        tool_calls_emitted = set()
        last_reasoning_emit_len = 0

        try:
            # Stream from BaseLLM with tools
            async for chunk in llm.astream(llm_messages, tools=openai_tools):
                # Handle reasoning from LLMChunk
                if chunk.reasoning:
                    reasoning_content += chunk.reasoning

                    # Stream reasoning chunks immediately as CoT events (real-time)
                    collect_and_emit({
                        "type": "status",
                        "message": chunk.reasoning
                    })
                    last_reasoning_emit_len = len(reasoning_content)

                # Handle content from LLMChunk
                if chunk.content:
                    response_content += chunk.content

                # Handle tool calls from LLMChunk
                if chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        tool_id = tool_call.get("id", "")
                        if tool_id and tool_id not in tool_calls_emitted:
                            tool_name = tool_call.get("name", "")
                            # Emit tool_start event
                            collect_and_emit({
                                "type": "tool_start",
                                "toolCallId": tool_id,
                                "toolName": tool_name,
                                "input": tool_call.get("args", {})
                            })
                            tool_calls_emitted.add(tool_id)

            # Get final message with complete tool_calls
            final_chunk = await llm.ainvoke(llm_messages, tools=openai_tools)

            # Emit final reasoning chunk if there's remaining content
            if reasoning_content and last_reasoning_emit_len < len(reasoning_content):
                final_reasoning = reasoning_content[last_reasoning_emit_len:]
                collect_and_emit({
                    "type": "status",
                    "message": final_reasoning
                })

            # Emit tool_end events for any tool calls
            if final_chunk.tool_calls:
                for tc in final_chunk.tool_calls:
                    tool_id = tc.get("id", "")
                    tool_name = tc.get("name", "")
                    collect_and_emit({
                        "type": "tool_end",
                        "toolCallId": tool_id,
                        "toolName": tool_name,
                        "result": "Tool call initiated"
                    })

            # Create additional_kwargs for CoT steps and reasoning_content
            additional_kwargs = {}
            if cot_steps:
                additional_kwargs["cot_steps"] = cot_steps

            # Store reasoning_content for persistence (retrieved on chat reload)
            if reasoning_content:
                additional_kwargs["reasoning_content"] = reasoning_content
                logger.info(f"[MAIN LLM] Storing {len(reasoning_content)} chars of reasoning_content for persistence (tool_router)")

                # NOTE: NOT re-emitting the complete reasoning as a status event here
                # because it was already streamed in chunks during the astream loop.
                # Re-emitting would create duplicate reasoning content in the frontend.

            # Store model information for persistence
            if state.get("model"):
                additional_kwargs["model"] = state["model"]
            if state.get("provider"):
                additional_kwargs["provider"] = state["provider"]

            # Build response message from LLMChunk
            ai_msg = AIMessage(
                content=final_chunk.content,
                additional_kwargs=additional_kwargs,
            )

            # Convert tool_calls from LLMChunk format to LangChain format
            if final_chunk.tool_calls:
                from langchain_core.messages.tool import ToolCall
                langchain_tool_calls = []
                for tc in final_chunk.tool_calls:
                    langchain_tool_calls.append(ToolCall(
                        name=tc.get("name", ""),
                        args=tc.get("args", {}),
                        id=tc.get("id", "")
                    ))
                ai_msg.tool_calls = langchain_tool_calls

            # Emit completion status
            collect_and_emit({
                "type": "status",
                "message": "Complete"
            })

            # Return only the new message - reducer will append it
            return {"messages": [ai_msg]}

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "messages": messages + [AIMessage(
                    content=f"I apologize, but I encountered an error: {str(e)}",
                )]
            }

    return llm_node


def create_tool_execution_node():
    """Create the tool execution node as an async function.

    LangGraph nodes must be callable. This function creates an async
    function that handles tool execution with streaming support.

    For the Tool Router pattern, this executes tools decided by the router.

    EPHEMERAL STATE: Tool results are stored in tool_context, not messages.
    This prevents raw tool outputs from polluting the conversation history.
    """
    # Note: We build tools_dict inside the node function to use state's enabled_tools

    async def tool_execution_node(state: ChatModeState, writer: StreamWriter) -> Dict[str, Any]:
        """Execute tool calls from the router.

        Populates tool_context with raw tool results (ephemeral).
        Does NOT add tool messages to the persistent conversation history.
        """
        # Get tool calls from router decision
        router_tool_calls = state.get("router_tool_calls", [])

        if not router_tool_calls:
            return {"executed_tools": [], "tool_context": []}

        # Build tools_dict from state's enabled_tools
        enabled_tools = state.get("enabled_tools", None)
        mode = state.get("mode", "companion")
        tools = get_companion_tools(mode=mode, tool_names=enabled_tools)
        tools_dict = {tool.name: tool for tool in tools}

        executed_tools = []
        tool_contexts = []  # EPHEMERAL: Raw tool results for current turn only
        # Get thread_id from state for session_id injection
        thread_id = state.get("thread_id", "default")

        for tool_call in router_tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {}).copy()  # Copy to avoid mutation
            tool_id = tool_call.get("id")

            # Inject session_id for image tools that need it
            if tool_name in ["generate_image", "edit_character_image"]:
                if "session_id" not in tool_args:
                    tool_args["session_id"] = thread_id

            # Emit tool start event (with input)
            writer({
                "type": "tool_start",
                "toolCallId": tool_id,
                "toolName": tool_name,
                "input": tool_args
            })

            # Initialize variables for error tracking (used outside the if-else block)
            is_error = False
            error_message = None
            result = ""  # Default fallback
            result_dict = None  # Preserve original dict for image events
            sub_tool_calls = []  # Track sub-tool calls (e.g., web_get from web_search)

            # Execute tool
            if tool_name not in tools_dict:
                result = f"Error: Tool '{tool_name}' not found"
                is_error = True
                error_message = result
                logger.error(f"[TOOL EXECUTION] {result}")

                # Emit tool error event
                writer({
                    "type": "tool_end",
                    "toolCallId": tool_id,
                    "toolName": tool_name,
                    "error": result
                })
            else:
                try:
                    tool = tools_dict[tool_name]
                    # Use LangChain's ainvoke for proper async/sync handling
                    # This handles both @tool decorated async functions and sync functions
                    result = await tool.ainvoke(tool_args)

                    # DEBUG: Check if result is a coroutine (should never happen after await)
                    import inspect
                    if inspect.iscoroutine(result):
                        logger.error(f"[TOOL EXECUTION] ERROR: result is a coroutine! {result}")
                        result = str(result)  # Convert to string to prevent JSON serialization error

                    # Check if result is an error (string or dict format)
                    if isinstance(result, str):
                        # Check for error patterns in strings
                        is_error = (
                            result.startswith("Error:") or
                            result.startswith("Failed") or
                            "Failed" in result or
                            "error" in result.lower()
                        )
                        if is_error:
                            error_message = result
                        # For image tools that return JSON strings, parse them immediately
                        # so result_dict is available for the image event emission logic
                        elif tool_name in ["generate_image", "edit_character_image"] and result.startswith('{'):
                            try:
                                result_dict = json.loads(result)
                            except json.JSONDecodeError:
                                result_dict = None
                    elif isinstance(result, dict):
                        # Check for success field in dict results
                        is_error = not result.get("success", True)
                        if is_error:
                            error_message = result.get("error", "Tool execution failed")
                        # Preserve original dict for image events
                        result_dict = result.copy()

                    # Always emit as tool_output (Vercel AI SDK compatible)
                    # Include error status in the data for custom components
                    output_data = {"result": result, "success": not is_error}

                    # DEBUG: Check if output_data contains any coroutines before JSON serialization
                    def check_for_coroutines(obj, path="root"):
                        import inspect
                        if inspect.iscoroutine(obj):
                            logger.error(f"[TOOL EXECUTION] Found coroutine at {path}: {type(obj)}")
                            return True
                        elif isinstance(obj, dict):
                            for k, v in obj.items():
                                if check_for_coroutines(v, f"{path}.{k}"):
                                    return True
                        elif isinstance(obj, (list, tuple)):
                            for i, item in enumerate(obj):
                                if check_for_coroutines(item, f"{path}[{i}]"):
                                    return True
                        return False

                    if check_for_coroutines(output_data, "output_data"):
                        logger.error(f"[TOOL EXECUTION] output_data contains coroutines, this will cause JSON serialization error!")
                        output_data = {"result": "Error: Tool returned a coroutine instead of a result", "success": False}

                    if is_error:
                        output_data["error"] = error_message or result
                        logger.error(f"[TOOL EXECUTION] {tool_name} returned error: {error_message or result}")
                        writer({
                            "type": "tool_end",
                            "toolCallId": tool_id,
                            "toolName": tool_name,
                            "output": output_data
                        })
                    else:
                        writer({
                            "type": "tool_end",
                            "toolCallId": tool_id,
                            "toolName": tool_name,
                            "output": output_data
                        })

                        # ===================================================================
                        # SEARCH + SELECTIVE SCRAPE: Auto-scrape relevant articles
                        # ===================================================================
                        # After web_search completes, use FlashRank to select 3-5 most relevant articles
                        # FlashRank is a cross-encoder reranker that scores query-document relevance
                        # Much faster and cheaper than using an LLM for selection
                        if tool_name == "web_search" and isinstance(result, str):
                            try:
                                # Parse search results to extract articles
                                articles = []
                                import re
                                # Match pattern: **Title**\nURL: https://...\nSnippet: ...
                                article_pattern = r'\*\*([^\*]+)\*\*\nURL: (https?://[^\s\n]+)\n([^\n]*?(?:\n[^\n]*?)*?)(?=\n\*\*|$)'
                                matches = re.findall(article_pattern, result)

                                for idx, match in enumerate(matches):
                                    title = match[0].strip()
                                    url = match[1].strip()
                                    # Get everything from URL line until next ** or end
                                    full_match = re.search(r'\*\*' + re.escape(title) + r'\*\*\nURL: ' + re.escape(url) + r'\n(.*?)(?=\n\*\*|$)', result, re.DOTALL)
                                    if full_match:
                                        snippet = full_match.group(1).strip()[:500]  # First 500 chars for reranking
                                        articles.append({
                                            "index": idx,
                                            "title": title,
                                            "url": url,
                                            "snippet": snippet
                                        })

                                if articles and len(articles) > 0:
                                    logger.info(f"[SEARCH+SCRAPE] Found {len(articles)} articles, reranking with FlashRank...")

                                    # Get user's original query for context
                                    user_query = tool_args.get("query", "")

                                    # Use FlashRank to rerank articles by relevance
                                    # FlashRank uses cross-encoder to score query-document pairs
                                    from flashrank import Ranker, RerankRequest

                                    # Initialize FlashRank reranker (uses ms-marco-TinyBERT by default)
                                    reranker = Ranker()

                                    # Prepare documents for FlashRank
                                    # Format: [{"id": "0", "text": "title + snippet"}, ...]
                                    docs = []
                                    for article in articles:
                                        # Combine title and snippet for better relevance scoring
                                        doc_text = f"{article['title']}\n{article['snippet']}"
                                        docs.append({
                                            "id": str(article["index"]),
                                            "text": doc_text
                                        })

                                    # Rerank articles using FlashRank
                                    # Returns sorted list of documents with relevance scores
                                    rerank_request = RerankRequest(query=user_query, passages=docs)
                                    reranked = reranker.rerank(rerank_request)

                                    logger.info(f"[SEARCH+SCRAPE] FlashRank reranking complete")

                                    # Select top 3-5 articles based on reranking scores
                                    top_k = min(5, len(reranked))
                                    selected_indices = [int(doc["id"]) for doc in reranked[:top_k]]

                                    logger.info(f"[SEARCH+SCRAPE] Selected {len(selected_indices)} articles: {selected_indices}")

                                    # Scrape selected articles using web_get
                                    web_get_tool = tools_dict.get("web_get")
                                    scraped_content = []

                                    if web_get_tool and selected_indices:
                                        for idx in selected_indices:
                                            if idx < len(articles):
                                                article = articles[idx]
                                                url = article["url"]

                                                logger.info(f"[SEARCH+SCRAPE] Scraping article {idx + 1}: {article['title'][:50]}...")

                                                # Generate a unique tool_call_id for this web_get call
                                                web_get_id = f"web_get_scrape_{idx}_{int(time.time() * 1000)}"

                                                # Emit tool_start event for web_get (visible in frontend)
                                                writer({
                                                    "type": "tool_start",
                                                    "toolCallId": web_get_id,
                                                    "toolName": "web_get",
                                                    "input": {"url": url}
                                                })

                                                try:
                                                    # Call web_get tool
                                                    scrape_result = await web_get_tool.ainvoke({"url": url})

                                                    # Store scraped content
                                                    scraped_content.append({
                                                        "title": article["title"],
                                                        "url": url,
                                                        "content": scrape_result[:3000]  # First 3000 chars
                                                    })

                                                    # Track for executed_tools (using outer scope variable)
                                                    sub_tool_calls.append({
                                                        "name": "web_get",
                                                        "args": {"url": url},
                                                        "result": f"Read {len(scrape_result)} chars from {article['title'][:50]}...",
                                                        "success": True
                                                    })

                                                    logger.info(f"[SEARCH+SCRAPE] Scraped {len(scrape_result)} chars from {url}")

                                                    # Emit tool_end event for web_get (visible in frontend)
                                                    writer({
                                                        "type": "tool_end",
                                                        "toolCallId": web_get_id,
                                                        "toolName": "web_get",
                                                        "output": {"result": f"Read {len(scrape_result)} chars from {article['title'][:50]}...", "success": True}
                                                    })

                                                except Exception as scrape_error:
                                                    logger.warning(f"[SEARCH+SCRAPE] Failed to scrape {url}: {scrape_error}")

                                                    # Track failed call
                                                    sub_tool_calls.append({
                                                        "name": "web_get",
                                                        "args": {"url": url},
                                                        "result": f"Failed to read article",
                                                        "success": False
                                                    })

                                                    # Emit tool_end error event
                                                    writer({
                                                        "type": "tool_end",
                                                        "toolCallId": web_get_id,
                                                        "toolName": "web_get",
                                                        "output": {"result": f"Failed to read article", "success": False, "error": str(scrape_error)}
                                                    })

                                    # Append scraped content to the original result
                                    if scraped_content:
                                        scrape_summary = f"\n\n--- Full Article Content ({len(scraped_content)} articles) ---\n"
                                        for idx, sc in enumerate(scraped_content):
                                            scrape_summary += f"\n## {sc['title']}\nSource: {sc['url']}\n\n{sc['content']}\n"

                                        result = result + scrape_summary
                                        logger.info(f"[SEARCH+SCRAPE] Added {len(scrape_summary)} chars of scraped content to results")

                                        # Emit completion status
                                        writer({
                                            "type": "status",
                                            "message": f"Read {len(scraped_content)} articles"
                                        })

                            except Exception as scrape_error:
                                logger.warning(f"[SEARCH+SCRAPE] Auto-scrape failed: {scrape_error}")
                                # Don't fail the entire tool execution if scraping fails
                                pass

                        # For image tools, emit data-image event for frontend rendering
                        # This will be picked up by the Vercel AI SDK and rendered as an <Image> component
                        if tool_name in ["generate_image", "edit_character_image"]:
                            try:
                                # Use result_dict if available (preserves original dict structure)
                                # Otherwise try parsing the result string
                                if result_dict and isinstance(result_dict, dict):
                                    result_data = result_dict
                                elif isinstance(result, str) and result.startswith('{'):
                                    result_data = json.loads(result)
                                else:
                                    result_data = {}

                                if result_data.get("success") and result_data.get("image_base64"):
                                    image_base64 = result_data["image_base64"]
                                    # Create proper data URL for the image
                                    # The AI SDK Image component expects base64 data (without data:image prefix)
                                    image_source = image_base64

                                    # Emit as transient data-image event
                                    writer({
                                        "type": "data-image",
                                        "image": image_source,
                                        "transient": True  # Don't persist in message history
                                    })
                            except (json.JSONDecodeError, Exception):
                                pass  # Image streaming failed, but tool execution succeeded
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    result = f"Error: {str(e)}"
                    is_error = True
                    error_message = result

                    # Emit as tool_output with error data (Vercel AI SDK compatible)
                    writer({
                        "type": "tool_end",
                        "toolCallId": tool_id,
                        "toolName": tool_name,
                        "output": {"result": result, "success": False, "error": result}
                    })

            # Track executed tool for displaying tool info
            # Use the is_error variable that was already computed above
            executed_tools.append({
                "name": tool_name,
                "args": tool_args,
                "result": result[:200] if isinstance(result, str) else str(result)[:200],  # First 200 chars for display
                "success": not is_error
            })

            # Also add any sub-tool calls (e.g., web_get from web_search)
            # This ensures nested tool calls are visible in the UI
            if sub_tool_calls:
                executed_tools.extend(sub_tool_calls)

            # Create tool message with metadata for proper display
            # The main LLM needs context about what tools were used and their results
            # to generate an appropriate response
            from langchain_core.messages import ToolMessage

            # Build informative tool content for the main LLM
            # This gives the LLM context about tool execution without overwhelming it
            if is_error:
                # For errors, provide brief context
                tool_content = f"The {tool_name} tool encountered an error. Details are shown in the tool output above."
            elif tool_name in ["generate_image", "edit_character_image"]:
                # For image tools, parse JSON and extract only the path to avoid polluting LLM context
                # with massive base64 strings (1.3MB+ images = ~425K tokens!)
                # Store image_base64 in content for frontend persistence (not passed to LLM due to format)
                try:
                    # Use result_dict if available (preserves original dict structure)
                    # Otherwise try parsing the result string
                    if result_dict and isinstance(result_dict, dict):
                        result_data = result_dict
                    elif isinstance(result, str) and result.startswith('{'):
                        result_data = json.loads(result)
                    else:
                        result_data = {}

                    if result_data.get("success") and result_data.get("image_path"):
                        # Pass only the image path to LLM, not the full base64
                        image_path = result_data["image_path"]
                        # Provide context about tool usage
                        llm_message = f"I successfully generated an image using the {tool_name} tool. The image has been saved to: {image_path}"

                        # Store structured content with image data for frontend persistence
                        # The frontend can parse this JSON to extract the image
                        # Format: {"text": "...", "image_base64": "...", "image_path": "..."}
                        tool_content = json.dumps({
                            "text": llm_message,
                            "image_base64": result_data.get("image_base64", ""),
                            "image_path": image_path,
                        })
                        logger.info(f"[TOOL EXECUTION] {tool_name}: Stored structured content with image ({len(result_data.get('image_base64', ''))} chars)")
                    else:
                        # If no success/path, pass a brief message
                        tool_content = f"I used the {tool_name} tool. Check the tool output above for the result."
                except (json.JSONDecodeError, Exception) as e:
                    # If not valid JSON, pass as-is (shouldn't happen for image tools)
                    logger.warning(f"[TOOL EXECUTION] {tool_name}: Could not parse result for LLM: {e}")
                    tool_content = f"I used the {tool_name} tool. Check the tool output above for the result."
            else:
                # For non-image tools, provide structured context
                # Format: "I used the {tool_name} tool with {args}. Result: {result}"
                # This gives the LLM full context about what was done
                args_summary = ", ".join(f"{k}={v}" for k, v in tool_args.items()) if tool_args else "no arguments"

                # Truncate very long results to avoid overwhelming the LLM
                result_summary = result
                if isinstance(result, str) and len(result) > 1000:
                    result_summary = result[:1000] + "... (truncated)"

                tool_content = f"I used the {tool_name} tool (with {args_summary}). Result: {result_summary}"

            # EPHEMERAL: Store raw tool result in tool_context instead of messages
            # This prevents tool results from polluting the conversation history
            tool_contexts.append(tool_content)

        # Return ephemeral tool_context instead of persistent messages
        # executed_tools is still returned for UI badge display
        return {"tool_context": tool_contexts, "executed_tools": executed_tools}

    return tool_execution_node


def create_main_llm_node(api_key: str, base_url: str, model: str, personality: PersonalityLayer):
    """Create the main LLM node that generates the final response.

    This node does NOT make tool decisions - those are handled by the tool_router.
    This node ONLY generates high-quality responses based on user input and tool results.

    For companion mode, this also injects relevant long-term memories into context.
    """
    async def main_llm_node(state: ChatModeState, writer: StreamWriter) -> Dict[str, Any]:
        """Generate response using main LLM (no tool decisions here)."""
        logger.info(f"[MAIN LLM] Generating response with {len(state.get('messages', []))} messages")

        messages: list[BaseMessage] = state.get("messages", [])
        executed_tools = state.get("executed_tools", [])

        # Get user_id from state for memory context
        thread_id = state.get("thread_id", "")
        # Extract user_id from thread_id (format: "companion-thread-{timestamp}" or "thread_{timestamp}")
        user_id = "default"
        if thread_id:
            # Try to extract user_id from thread_id or use thread_id as user_id
            user_id = thread_id.split("-thread-")[0] if "-thread-" in thread_id else thread_id

        # Collect CoT steps for persistence (stored in additional_kwargs)
        cot_steps = []

        def collect_and_emit(cot_event: dict) -> None:
            """Collect CoT step for storage and emit via writer.

            Adds required fields (id, label, status, icon) if missing.
            """
            # Ensure CoT event has required fields for persistence
            if "id" not in cot_event:
                cot_event["id"] = str(uuid.uuid4())
            if "label" not in cot_event:
                cot_event["label"] = cot_event.get("message", cot_event.get("type", "Thinking..."))
            if "status" not in cot_event:
                cot_event["status"] = "active"
            if "icon" not in cot_event:
                cot_event["icon"] = "sparkle"
            cot_steps.append(cot_event)
            writer(cot_event)

        # NEVER emit router reasoning to frontend - only tool calls
        # Router reasoning stays in backend logs only
        router_reasoning = state.get("router_reasoning")
        if router_reasoning:
            logger.info(f"[MAIN LLM] Router reasoning (NOT sent to frontend): {router_reasoning[:100]}...")
            # Router reasoning is logged but NEVER emitted to frontend

        # Emit tool output events for frontend badge display
        # Using DynamicToolUIPart format for tool router pattern
        # The SDK expects 'tool-output-available' type with toolCallId
        for tool_info in executed_tools:
            tool_name = tool_info["name"]
            tool_result = tool_info["result"]
            tool_success = tool_info.get("success", True)

            # Generate a synthetic toolCallId for the tool router pattern
            # This allows the SDK to create a proper ToolUIPart
            tool_call_id = f"tool_{tool_name}_{int(time.time() * 1000)}"

            if tool_success:
                # Emit tool-output-available event for successful tools
                # Format output for Vercel AI SDK: wrap in object with "text" field
                writer({
                    "type": "tool-output-available",
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "output": {"text": tool_result} if tool_result else "",
                })
                # Also emit status for thinking display (no colon for cleaner UI)
                collect_and_emit({
                    "type": "status",
                    "message": f"Used {tool_name}"
                })
            else:
                # Emit tool-output-error event for failed tools
                writer({
                    "type": "tool-output-error",
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "errorText": tool_result,
                })
                # Also emit status for thinking display (no colon for cleaner UI)
                collect_and_emit({
                    "type": "status",
                    "message": f"Tool failed: {tool_name}"
                })

        # Import from Shared Kernel
        try:
            from src.shared import get_llm
            from langchain_core.messages import SystemMessage
        except ImportError:
            logger.error("Shared LLM factory not available")
            return {"messages": messages + [AIMessage(content="Error: LLM factory not available")]}

        # Convert LangChain messages to dict format for BaseLLM
        llm_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                # Handle multimodal content - extract text only for main LLM
                # The tool already handled the image, so we don't need to send it again
                content = msg.content
                if isinstance(content, list):
                    # Extract text from multimodal content
                    text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
                    content = "".join(text_parts)
                llm_messages.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                # CRITICAL: Skip AIMessages with empty content AND no tool_calls
                # These are meaningless and will cause the API to reject the request
                has_content = bool(msg.content and str(msg.content).strip())
                has_tool_calls = hasattr(msg, 'tool_calls') and bool(msg.tool_calls)

                if not has_content and not has_tool_calls:
                    logger.debug(f"[MAIN LLM] Skipping empty AIMessage (no content, no tool_calls)")
                    continue

                # CRITICAL: Include tool_calls if present so LLM knows about tool invocations
                # This enables the LLM to associate tool results with its original tool calls
                ai_msg_dict = {"role": "assistant", "content": msg.content}
                if has_tool_calls:
                    # Convert LangChain tool_calls format to OpenAI API format
                    # LangChain: [{"name": "...", "args": {...}, "id": "..."}]
                    # OpenAI: [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...}"}}]
                    openai_tool_calls = []
                    for tc in msg.tool_calls:
                        if isinstance(tc, dict):
                            # LangChain dict-style ToolCall
                            tool_id = tc.get('id', '')
                            tool_name = tc.get('name', '')
                            tool_args = tc.get('args', {})

                            # Convert args to JSON string
                            import json
                            args_json = json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)

                            openai_tool_calls.append({
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": args_json
                                }
                            })

                    if openai_tool_calls:
                        ai_msg_dict["tool_calls"] = openai_tool_calls
                        logger.debug(f"[MAIN LLM] AIMessage has {len(openai_tool_calls)} tool_calls (OpenAI format)")
                llm_messages.append(ai_msg_dict)
            elif isinstance(msg, ToolMessage):
                # CRITICAL: Pass tool_call_id so LLM knows which tool call this result belongs to
                # This enables the LLM to understand tool execution context and respond appropriately
                llm_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id
                })
                logger.debug(f"[MAIN LLM] Converted ToolMessage: tool_call_id={msg.tool_call_id}, content={msg.content[:50]}...")

        # Add system prompt using prompt_engine library
        if state.get("personality_enabled"):
            from src.extensions.libraries.core_chat_engine.prompts import build_system_messages

            # Get memory context if available
            memory_context = None
            try:
                from ..memory.context_injection import build_memory_system_message
                memory_msg = build_memory_system_message(
                    current_input=state.get("user_input", ""),
                    user_id=state.get("user_id", "default"),
                )
                memory_context = memory_msg.content if memory_msg else None
            except Exception as e:
                logger.debug(f"[MAIN LLM] Could not build memory context: {e}")

            # Build system messages using prompt_engine
            system_messages = build_system_messages(
                mode=state.get("mode", "direct"),
                character=state.get("character"),
                user_profile=state.get("user_profile"),
                memory_context=memory_context,
                personality_config=personality.config if personality else None,
            )

            # Convert to dict format and insert at beginning
            for msg in reversed(system_messages):
                if isinstance(msg, SystemMessage):
                    llm_messages.insert(0, {"role": "system", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    llm_messages.insert(0, {"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    llm_messages.insert(0, {"role": "assistant", "content": msg.content})

        # Add memory context for companion mode (injects relevant memories)
        # Only for companion mode, NOT for roleplay or assistant modes
        mode = state.get("mode", "companion")
        if mode == "companion":
            try:
                # Get current user input for memory search
                current_input = ""
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        current_input = str(msg.content) if hasattr(msg, 'content') else ""
                        break

                # Build memory context (non-blocking, uses cached memories)
                memory_context = build_memory_system_message(
                    current_input=current_input,
                    user_id=user_id,
                    max_tokens=500,  # Limit memory context to ~500 tokens
                )

                if memory_context:
                    # Inject memory context after personality prompt
                    llm_messages.insert(0, {"role": "system", "content": memory_context.content})
                    logger.info(f"[MAIN LLM] Injected memory context for user={user_id}")
                else:
                    logger.debug(f"[MAIN LLM] No relevant memories found for user={user_id}")

            except Exception as e:
                # Don't fail the conversation if memory injection fails
                logger.warning(f"[MAIN LLM] Memory context injection failed: {e}")
                logger.warning(f"[MAIN LLM] Continuing without memory context...")

        # Add router reasoning as context if available
        router_reasoning = state.get("router_reasoning", "")
        if router_reasoning:
            logger.info(f"[MAIN LLM] Router reasoning: {router_reasoning}")

        # EPHEMERAL STATE: Inject tool_context into system prompt
        # This contains raw tool results from the current turn only
        tool_context = state.get("tool_context", [])
        if tool_context:
            # Format tool context as a structured system message
            tool_context_message = (
                "The following tools were executed and their results are available for context:\n\n" +
                "\n\n".join(tool_context) +
                "\n\nPlease use this tool context to inform your response to the user."
            )
            llm_messages.insert(0, {"role": "system", "content": tool_context_message})
            logger.info(f"[MAIN LLM] Injected tool_context with {len(tool_context)} tool results")

        # Get LLM from Shared Kernel (BaseLLM interface)
        # Main chat uses get_main_chat_llm which has thinking enabled
        from src.shared import get_main_chat_llm

        # NO FALLBACKS - model must be in state
        runtime_model = state.get("model")
        if not runtime_model:
            raise ValueError("model not found in state - required for main chat")

        # NO FALLBACKS - temperature must be in state
        main_temperature = state.get("main_temperature")
        if main_temperature is None:
            raise ValueError("main_temperature not found in state - required for main chat")

        logger.info(f"[MAIN LLM] Using model: {runtime_model}, temperature: {main_temperature}")

        llm = await get_main_chat_llm(
            model=runtime_model,
            temperature=main_temperature
        )

        response_content = ""
        reasoning_content = ""
        tool_calls_emitted = set()

        # DEBUG: Log summary of messages being sent to LLM (consolidated to reduce log spam)
        logger.info(f"[MAIN LLM] Calling LLM with {len(llm_messages)} messages")
        # Build summary: count by role and flag unusual content
        role_counts = {}
        unusual_msgs = []
        for i, msg in enumerate(llm_messages):
            if isinstance(msg, dict):
                role = msg.get('role', '?')
                role_counts[role] = role_counts.get(role, 0) + 1
                content_len = len(str(msg.get('content', '')))
                # Flag suspicious messages (very short but not tool messages)
                if content_len < 5 and role != 'tool' and content_len > 0:
                    unusual_msgs.append(f"Msg {i}: role={role}, content_len={content_len}, content_preview={repr(msg.get('content', ''))[:50]}")
        logger.info(f"[MAIN LLM] Message breakdown: {role_counts}")
        if unusual_msgs:
            logger.warning(f"[MAIN LLM] Found {len(unusual_msgs)} suspicious messages (very short content): {unusual_msgs[:5]}...")

        try:
            # Stream using BaseLLM interface
            async for chunk in llm.astream(llm_messages):
                # DEBUG: Log chunk structure
                logger.debug(f"[MAIN LLM] LLMChunk: content={chunk.content[:50] if chunk.content else ''}..., reasoning={chunk.reasoning[:50] if chunk.reasoning else ''}...")

                # Handle reasoning from LLMChunk
                # Stream reasoning chunks immediately via writer for real-time display
                logger.debug(f"[MAIN LLM] chunk.reasoning value: repr={repr(chunk.reasoning)[:100]}, len={len(chunk.reasoning) if chunk.reasoning else 0}")
                if chunk.reasoning:
                    reasoning_content += chunk.reasoning
                    # Stream reasoning chunk via writer (will be emitted as reasoning events)
                    logger.info(f"[MAIN LLM] Streaming reasoning chunk: {len(chunk.reasoning)} chars, writer={writer is not None}")
                    try:
                        writer({
                            "type": "status",
                            "message": chunk.reasoning
                        })
                        logger.debug(f"[MAIN LLM] Successfully emitted reasoning chunk via writer")
                    except Exception as e:
                        logger.error(f"[MAIN LLM] Failed to emit reasoning chunk via writer: {e}")
                        import traceback
                        traceback.print_exc()

                # Handle content from LLMChunk
                if chunk.content:
                    response_content += chunk.content
                    # Stream content chunks immediately via writer
                    writer({
                        "type": "content_chunk",
                        "content": chunk.content
                    })
                # Note: reasoning is handled separately above (emitted as status events)
                # We do NOT emit reasoning as content_chunk to avoid duplication

            # Log reasoning summary
            if reasoning_content:
                logger.info(f"[MAIN LLM] Generated {len(reasoning_content)} chars of reasoning_content")

            # Create AIMessage with reasoning_content for persistence
            additional_kwargs = {}
            if cot_steps:
                additional_kwargs["cot_steps"] = cot_steps
                logger.info(f"[MAIN LLM] Storing {len(cot_steps)} CoT steps in additional_kwargs for persistence")
                logger.debug(f"[MAIN LLM] CoT steps: {cot_steps}")

            # Store reasoning_content for persistence (retrieved on chat reload)
            if reasoning_content:
                additional_kwargs["reasoning_content"] = reasoning_content
                logger.info(f"[MAIN LLM] Storing {len(reasoning_content)} chars of reasoning_content for persistence")

                # NOTE: NOT re-emitting the complete reasoning as a status event here
                # because it was already streamed in chunks during the astream loop.
                # Re-emitting would create duplicate reasoning content in the frontend.

            # Determine final content for AIMessage persistence
            # Note: Content is already streamed in real-time during the astream loop above.
            # We do NOT re-emit content here to avoid duplication.
            final_content = response_content if response_content else reasoning_content
            if not response_content and reasoning_content:
                logger.info(f"[MAIN LLM] Using reasoning_content ({len(reasoning_content)} chars) as final content (thinking mode)")

            return {"messages": [AIMessage(content=final_content, additional_kwargs=additional_kwargs)]}

        except Exception as e:
            logger.error(f"[MAIN LLM] Error: {e}")
            import traceback
            traceback.print_exc()
            return {"messages": [AIMessage(content=f"I apologize, but I encountered an error: {str(e)}")]}

    return main_llm_node


def create_post_process_node_v2():
    """Create the post-processing node.

    For companion mode, this also triggers background memory extraction
    and trait evolution after each conversation turn.
    """
    def post_process_node(state: ChatModeState) -> Dict[str, Any]:
        """Finalize response and update conversation state."""
        logger.debug("Node: post_process_v2")

        # Trigger background memory extraction for companion mode
        mode = state.get("mode", "companion")
        if mode == "companion":
            try:
                # Get conversation summary for memory extraction
                messages = state.get("messages", [])
                if len(messages) >= 2:  # Need at least user + assistant message
                    # Build conversation summary
                    conversation_turns = []
                    i = 0
                    while i < len(messages):
                        if isinstance(messages[i], HumanMessage):
                            user_msg = str(messages[i].content) if hasattr(messages[i], 'content') else ""
                            # Look for assistant response
                            assistant_msg = ""
                            if i + 1 < len(messages) and isinstance(messages[i + 1], AIMessage):
                                assistant_msg = str(messages[i + 1].content) if hasattr(messages[i + 1], 'content') else ""
                            conversation_turns.append(f"User: {user_msg}\nAssistant: {assistant_msg}")
                            i += 2
                        else:
                            i += 1

                    # Create summary from last few turns
                    summary = "\n\n".join(conversation_turns[-3:])  # Last 3 turns

                    if summary:
                        # Get user_id and assistant_id from state
                        thread_id = state.get("thread_id", "")
                        # Extract user_id from thread_id (format: "companion-thread-{timestamp}" or "{user_id}_{name}_{id}")
                        parts = thread_id.split("_")
                        user_id = parts[0] if parts else "default"

                        # For companion mode with companion profiles, the thread_id should be the assistant_id
                        # Format: {user_id}_{assistant_name}_{timestamp}
                        # So we can use thread_id as both thread_id and assistant_id lookup
                        assistant_id = thread_id if "_thread-" not in thread_id else f"{user_id}_companion"

                        # Detect interaction signals from conversation (simple heuristic)
                        interaction_signals = {}
                        summary_lower = summary.lower()
                        if any(word in summary_lower for word in ["laugh", "lol", "haha", "😂", "🤣"]):
                            interaction_signals["user_laughed"] = True
                        if any(word in summary_lower for word in ["thank", "thanks", "thx"]):
                            interaction_signals["user_thanked"] = True
                        if any(word in summary_lower for word in ["love", "❤️", "heart"]):
                            interaction_signals["user_flirted"] = True

                        # Trigger background extraction with trait evolution (enhanced version)
                        try:
                            import asyncio
                            from .memory.extraction_worker import extract_memories_with_trait_evolution

                            # Schedule extraction in background (doesn't block)
                            loop = asyncio.get_event_loop()
                            asyncio.create_task(
                                extract_memories_with_trait_evolution(
                                    conversation_summary=summary,
                                    user_id=user_id,
                                    assistant_id=assistant_id,
                                    interaction_signals=interaction_signals if interaction_signals else None,
                                )
                            )
                            logger.info(f"[POST_PROCESS] Scheduled background memory + trait evolution for user={user_id}, companion={assistant_id}")

                        except Exception as e:
                            # Don't fail if extraction scheduling fails
                            logger.warning(f"[POST_PROCESS] Failed to schedule memory extraction: {e}")

            except Exception as e:
                # Don't fail the post-processing if memory extraction fails
                logger.warning(f"[POST_PROCESS] Memory extraction setup failed: {e}")

        messages = state.get("messages", [])
        start_time = state.get("start_time", time.time())

        # Get the last assistant message
        last_message: Optional[AIMessage] = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_message = msg
                break

        if not last_message:
            return {"processing_time": time.time() - start_time}

        # Extract response text from LangChain AIMessage
        response_text = str(last_message.content) if hasattr(last_message, 'content') else ""
        has_tool_calls = bool(last_message.tool_calls and len(last_message.tool_calls) > 0)

        # Note: Images are now streamed immediately via custom events in assistant_brain.py
        # No need to create separate image messages here

        # Build ProcessedResponse
        processed = ProcessedResponse(
            text=response_text,
            intent=IntentResult(
                primary=IntentType.CASUAL_CHAT,
                confidence=1.0,
            ),
            source="llm",
            processing_time=time.time() - start_time,
            was_routed=has_tool_calls,
        )

        return {
            "response": processed,
            "processing_time": processed.processing_time,
            "tool_context": [],  # EPHEMERAL: Clear tool context after response (don't persist to next turn)
        }

    return post_process_node


def create_update_state_node_v2(conversation_manager: ConversationManager):
    """Create the state update node."""
    def update_state_node(state: ChatModeState) -> Dict[str, Any]:
        """Update conversation state with the new turn."""
        logger.debug("Node: update_state_v2")

        messages: list[BaseMessage] = state.get("messages", [])
        conversation = state.get("conversation")

        # Find last user and assistant messages
        last_user = None
        last_assistant = None

        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and last_user is None:
                last_user = str(msg.content) if hasattr(msg, 'content') else ""
            elif isinstance(msg, AIMessage) and last_assistant is None:
                last_assistant = str(msg.content) if hasattr(msg, 'content') else ""

            # Stop if we found both
            if last_user and last_assistant:
                break

        # Only update conversation state if it exists (Companion mode)
        # Brain modes (Direct, Tool_call, Deep_agent) don't use conversation object
        if conversation is not None and last_user and last_assistant:
            conversation_manager.add_turn(
                conversation,
                last_user,
                last_assistant,
                IntentType.CASUAL_CHAT,
                was_routed=False,
                tool_used=None,
            )

        return {}

    return update_state_node


# =============================================================================
# Conditional Edge Functions v2
# =============================================================================

def should_continue_tools(state: ChatModeState) -> Literal["tools", "post_process"]:
    """Check if LLM wants to call more tools or is done."""
    messages: list[BaseMessage] = state.get("messages", [])
    logger.info(f"[SHOULD_CONTINUE] Checking {len(messages)} messages")

    if not messages:
        return "post_process"

    # Check last assistant message for tool calls
    for msg in reversed(messages):
        # LangChain AIMessage check
        if isinstance(msg, AIMessage):
            has_tools = bool(msg.tool_calls and len(msg.tool_calls) > 0)
            logger.info(f"[SHOULD_CONTINUE] Last AIMessage has tool_calls: {has_tools}")
            if has_tools:
                logger.info("[SHOULD_CONTINUE] Returning 'tools'")
                return "tools"
            break

    logger.info("[SHOULD_CONTINUE] Returning 'post_process'")
    return "post_process"


# =============================================================================
# Graph Builder v2
# =============================================================================

def create_chat_mode_graph(
    api_key: str,
    base_url: str,
    model: str,
    config: Optional[BrainConfig] = None,
    mode: str = "direct",
    personality: Optional[PersonalityConfig] = None,
    personality_enabled: bool = True,
    enabled_tools: Optional[List[str]] = None,
    checkpointer=None,
) -> StateGraph:
    """Create a chat mode graph using Shared Kernel LLM factory.

    Args:
        api_key: API key for the LLM service
        base_url: Base URL for the LLM API
        model: Model name to use
        config: Brain configuration
        mode: Operating mode (direct, tool_call, deep_agent, deep_research)
        personality: Personality configuration
        personality_enabled: Whether to include personality prompts
        enabled_tools: Optional list of tool names to enable (from extension config)
        checkpointer: Optional checkpointer for persistence

    Returns:
        Compiled StateGraph ready for invocation
    """
    config = config or get_preset_config(mode)

    if personality:
        config.personality = personality

    # Initialize components
    conversation_manager = create_conversation_manager(
        max_context_turns=config.max_context_turns,
        summarize_interval=config.summarize_interval,
        context_window_size=config.context_window_size,
    )
    personality_layer = create_personality_layer(
        config=config.personality,
        mode=mode,
        personality_enabled=personality_enabled,
    )

    # Get tools
    tools = get_companion_tools()

    # Create nodes - Tool Router Pattern
    input_node = create_input_node_v2()
    tool_router_node = create_tool_router_node(api_key, base_url, model)
    main_llm_node = create_main_llm_node(api_key, base_url, model, personality_layer)
    tool_execution_node = create_tool_execution_node()
    post_process_node = create_post_process_node_v2()
    update_state_node = create_update_state_node_v2(conversation_manager)

    # Build the graph
    builder = StateGraph(ChatModeState)

    # Add nodes
    builder.add_node("input_processor", input_node)
    builder.add_node("tool_router", tool_router_node)
    builder.add_node("main_llm", main_llm_node)
    builder.add_node("tool_execution", tool_execution_node)
    builder.add_node("post_process", post_process_node)
    builder.add_node("update_conversation", update_state_node)

    # Define the flow - Tool Router Pattern
    # START → input → tool_router → [has_tools?] → tool_execution → main_llm → post_process → update → END
    #                                     → [no_tools?] ↗

    # Entry point
    builder.add_edge(START, "input_processor")
    builder.add_edge("input_processor", "tool_router")

    # After tool_router, either execute tools or go straight to main_llm
    builder.add_conditional_edges(
        "tool_router",
        should_call_tools,
        {
            "tool_execution": "tool_execution",
            "main_llm": "main_llm",
        }
    )

    # After tool execution, go to main_llm
    builder.add_edge("tool_execution", "main_llm")

    # After main_llm, go to post_process
    builder.add_edge("main_llm", "post_process")

    # After post_process, update conversation or end
    builder.add_edge("post_process", "update_conversation")
    builder.add_edge("update_conversation", END)

    # Compile with checkpointer if provided
    # Unwrap wrapped checkpointer if needed to get the actual BaseCheckpointSaver
    actual_checkpointer = checkpointer
    if checkpointer is not None and hasattr(checkpointer, 'checkpointer'):
        actual_checkpointer = checkpointer.checkpointer

    graph = builder.compile(checkpointer=actual_checkpointer)

    logger.info("Companion brain StateGraph v2 created with TOOL ROUTER pattern")
    return graph


# =============================================================================
# Wrapper Class v2
# =============================================================================

class CompanionBrainGraphV2:
    """
    Wrapper for the LangGraph StateGraph with tool calling.

    This version uses LangChain's native tool calling - the LLM decides
    when to call tools instead of using a separate intent classifier.
    """

    def __init__(
        self,
        llm,
        config: Optional[BrainConfig] = None,
        checkpointer=None,
        thread_id: Optional[str] = None,
        personality_enabled: bool = True,
        mode: str = "companion",
        personality: Optional[PersonalityConfig] = None,
    ):
        """Initialize the companion brain graph v2.

        Args:
            llm: LangChain LLM instance (ChatOpenAI or compatible)
            config: Brain configuration
            checkpointer: Optional checkpointer for persistence
            thread_id: Thread ID for this conversation
            personality_enabled: Whether to include personality prompts
            mode: Operating mode
            personality: Optional personality configuration override
        """
        self.config = config or get_preset_config(mode)
        if personality:
            self.config.personality = personality

        self.mode = mode
        self._personality_enabled = personality_enabled
        self._llm = llm

        # Create thread_id
        self.thread_id = thread_id or f"thread_{time.time()}"

        # Create conversation manager
        self._conversation_manager = create_conversation_manager(
            max_context_turns=self.config.max_context_turns,
            summarize_interval=self.config.summarize_interval,
            context_window_size=self.config.context_window_size,
        )

        # Use LiteLLM constants directly
        from src.shared.litellm_factory import LITELLM_API_KEY, LITELLM_BASE_URL
        
        # Store model name for state initialization
        self._model = llm.config.model

        # Create the StateGraph
        self.graph = create_companion_graph_v2(
            api_key=LITELLM_API_KEY,
            base_url=LITELLM_BASE_URL,
            model=llm.config.model,
            config=self.config,
            mode=mode,
            personality=self.config.personality,
            personality_enabled=personality_enabled,
            checkpointer=checkpointer,
        )

        # Personality layer for direct access
        self.personality = create_personality_layer(
            config=self.config.personality,
            mode=mode,
            personality_enabled=personality_enabled,
        )

        # Local message storage for conversation history (fallback without checkpointer)
        self._messages: List[Dict[str, Any]] = []
        self._conversation_state = self._conversation_manager.create_state()
        self._user_profile = UserProfile()

    async def process(self, user_input: str) -> ProcessedResponse:
        """
        Process user input through the brain graph.

        Args:
            user_input: Raw user input text

        Returns:
            ProcessedResponse with the brain's output
        """
        config = {"configurable": {"thread_id": self.thread_id}}

        # Check for existing state (from checkpointer or local storage)
        try:
            current_state = self.graph.get_state(config)
            if current_state and current_state.values.get("messages"):
                initial_state = current_state.values
                # Add new input
                initial_state["user_input"] = user_input
                initial_state["start_time"] = time.time()
                # Ensure required state values are present (may not be in old checkpoints)
                initial_state.setdefault("model", self._model)
                initial_state.setdefault("main_temperature", self.config.temperature)
                initial_state.setdefault("router_temperature", 0.3)  # Lower temperature for more deterministic tool routing
                # Update local storage
                self._messages = initial_state["messages"].copy()
            else:
                # Use local storage if available
                initial_state = {
                    "messages": self._messages.copy(),
                    "conversation": self._conversation_state,
                    "user_profile": self._user_profile,
                    "mode": self.mode,
                    "personality_enabled": self._personality_enabled,
                    "user_input": user_input,
                    "start_time": time.time(),
                    "thread_id": self.thread_id,
                    "router_tool_calls": [],
                    "router_reasoning": "",
                    "executed_tools": [],
                    # Required state values for LLM nodes
                    "model": self._model,
                    "main_temperature": self.config.temperature,
                    "router_temperature": 0.3,  # Lower temperature for more deterministic tool routing
                }
        except Exception:
            # New conversation - use local storage
            initial_state = {
                "messages": self._messages.copy(),
                "conversation": self._conversation_state,
                "user_profile": self._user_profile,
                "mode": self.mode,
                "personality_enabled": self._personality_enabled,
                "user_input": user_input,
                "start_time": time.time(),
                "thread_id": self.thread_id,
                "router_tool_calls": [],
                "router_reasoning": "",
                "executed_tools": [],
                # Required state values for LLM nodes
                "model": self._model,
                "main_temperature": self.config.temperature,
                "router_temperature": 0.3,  # Lower temperature for more deterministic tool routing
            }

        # Invoke the graph
        result = await self.graph.ainvoke(initial_state, config)

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
                text="I apologize, but I couldn't generate a response.",
                intent=IntentResult(
                    primary=IntentType.UNKNOWN,
                    confidence=0.0,
                ),
                source="error",
            )

    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        self._conversation_manager = create_conversation_manager(
            max_context_turns=self.config.max_context_turns,
            summarize_interval=self.config.summarize_interval,
            context_window_size=self.config.context_window_size,
        )
        self.thread_id = f"thread_{time.time()}"
        # Reset local storage
        self._messages = []
        self._conversation_state = self._conversation_manager.create_state()
        self._user_profile = UserProfile()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        # Return local message storage (always maintained)
        return self._messages.copy()

    def get_graph_visualization(self) -> str:
        """Get ASCII visualization of the graph structure."""
        from io import StringIO
        import sys

        # Capture print_ascii output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            self.graph.get_graph().print_ascii()
            return sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

    def set_personality_enabled(self, enabled: bool) -> None:
        """Enable or disable personality prompts at runtime."""
        self._personality_enabled = enabled
        self.personality.set_personality_enabled(enabled)

    def set_personality(self, personality: PersonalityConfig) -> None:
        """Update personality configuration."""
        self.config.personality = personality
        # Rebuild graph with new personality
        from src.shared.litellm_factory import LITELLM_API_KEY, LITELLM_BASE_URL
        self.graph = create_companion_graph_v2(
            api_key=LITELLM_API_KEY,
            base_url=LITELLM_BASE_URL,
            model=self._llm.config.model,
            config=self.config,
            mode=self.mode,
            personality=personality,
            personality_enabled=self._personality_enabled,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_companion_brain_graph_v2(
    llm,
    config: Optional[BrainConfig] = None,
    mode: str = "companion",
    personality: str = None,
    checkpointer=None,
    thread_id: Optional[str] = None,
    personality_enabled: bool = True,
) -> CompanionBrainGraphV2:
    """
    Factory function to create CompanionBrainGraphV2.

    Args:
        llm: LangChain LLM instance
        config: Optional brain configuration
        mode: Operating mode ("companion" or "rp")
        personality: Personality preset name
        checkpointer: Optional checkpointer for persistence
        thread_id: Thread ID for conversation persistence
        personality_enabled: Whether to include personality prompts

    Returns:
        Initialized CompanionBrainGraphV2
    """
    if config is None:
        config = get_preset_config(mode)

    if personality:
        from ..config.personality_config import get_preset
        config.personality = get_preset(personality)

    return CompanionBrainGraphV2(
        llm=llm,
        config=config,
        checkpointer=checkpointer,
        thread_id=thread_id,
        personality_enabled=personality_enabled,
        mode=mode,
        personality=config.personality,
    )
