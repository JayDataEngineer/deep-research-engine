"""Context management strategies for LLM conversations.

Hot-swappable strategies like lang_tools backends:
- basic: Simple FIFO clipping
- sliding_window: Keep recent + system
- keep_recent: Keep N messages
- summarize: Summarize old messages using LLM

Usage:
    from src.extensions.libraries.core_chat_engine.memory.context_strategies import manage_context, count_tokens

    # Apply context management before LLM call
    messages = manage_context(
        messages,
        max_tokens=200000,
        output_buffer=20000,
        strategy="basic"
    )
"""

import os
import sys
import time
from typing import List, Dict, Any, Literal, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ContextStrategy(Enum):
    """Available context management strategies."""
    BASIC = "basic"  # Simple FIFO - clip oldest messages first
    SLIDING_WINDOW = "sliding_window"  # Keep recent window + system prompt
    KEEP_RECENT = "keep_recent"  # Keep N most recent messages
    SUMMARIZE = "summarize"  # Summarize old messages using LLM


# Strategy aliases for compatibility
BASIC_STRATEGY = ContextStrategy.BASIC
SLIDING_WINDOW_STRATEGY = ContextStrategy.SLIDING_WINDOW
KEEP_RECENT_STRATEGY = ContextStrategy.KEEP_RECENT
SUMMARIZE_STRATEGY = ContextStrategy.SUMMARIZE


# Callback function type for logging summarization calls
# Args: summary_text, original_tokens, summary_tokens, messages_summarized, duration_seconds
SummaryLogCallback = Optional[Callable[[str, int, int, int, float], None]]


# Default log callback - prints to stderr
def _default_summary_log(summary: str, original_tokens: int, summary_tokens: int,
                         num_messages: int, duration: float) -> None:
    """Default logging callback for summarization."""
    print(f"[Context Summarization] Generated summary in {duration:.2f}s", file=sys.stderr)
    print(f"[Context Summarization] Summarized {num_messages} messages ({original_tokens} → {summary_tokens} tokens)", file=sys.stderr)
    print(f"[Context Summarization] Summary: {summary[:200]}{'...' if len(summary) > 200 else ''}", file=sys.stderr)


def count_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate token count for messages.

    Uses a simple heuristic: ~4 chars per token for English text.
    For production, you might want to use tiktoken or the LLM's tokenizer.

    Args:
        messages: List of message dicts with 'content' key

    Returns:
        Estimated token count
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            # Multimodal content
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        total_chars += len(item.get("text", ""))
                elif isinstance(item, str):
                    total_chars += len(item)

    # Rough estimate: 4 chars per token (works well for English)
    return total_chars // 4


def _basic_strategy(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    output_buffer: int
) -> List[Dict[str, Any]]:
    """Basic FIFO context management.

    Clips oldest non-system messages first until within token limit.
    Preserves system message at index 0.

    Strategy:
    1. Keep system message (index 0) always
    2. Clip oldest messages until token count is safe
    3. Reserve output_buffer space for LLM response

    Args:
        messages: Message list (may be mutated!)
        max_tokens: Model's context window (e.g., 200000 for glm-4.7)
        output_buffer: Tokens to reserve for output (e.g., 20000)

    Returns:
        Clipped message list
    """
    # Make a copy to avoid mutating original
    messages = list(messages)

    # Calculate safe limit (reserve space for output)
    safe_limit = max_tokens - output_buffer

    # Keep clipping until we're safe
    while len(messages) > 1 and count_tokens(messages) > safe_limit:
        # Remove oldest non-system message
        # Assume index 0 is system, index 1 is oldest user/assistant message
        if len(messages) > 1:
            messages.pop(1)  # Remove message after system prompt
        else:
            # Only system message left - can't clip further
            break

    return messages


def _sliding_window_strategy(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    output_buffer: int,
    window_size: int = 50
) -> List[Dict[str, Any]]:
    """Sliding window context management.

    Keeps system prompt + recent N messages (sliding window).
    Good for maintaining recent context while dropping older history.

    Args:
        messages: Message list
        max_tokens: Model's context window
        output_buffer: Tokens to reserve for output
        window_size: Number of recent messages to keep (excluding system)

    Returns:
        Clipped message list
    """
    # Always keep system message (index 0)
    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None

    # Get recent messages (reverse to get newest first)
    recent_messages = []
    token_count = 0
    safe_limit = max_tokens - output_buffer

    # Start from end and collect messages until we hit the limit
    for msg in reversed(messages):
        # Skip system message in reverse iteration
        if msg.get("role") == "system":
            continue

        msg_tokens = count_tokens([msg])
        if token_count + msg_tokens > safe_limit:
            break

        recent_messages.insert(0, msg)  # Insert at beginning (maintains order)
        token_count += msg_tokens

    # Rebuild with system message + recent window
    result = []
    if system_msg:
        result.append(system_msg)
    result.extend(recent_messages[:window_size])

    return result


def _keep_recent_strategy(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    output_buffer: int,
    keep_count: int = 20
) -> List[Dict[str, Any]]:
    """Keep N most recent messages (simple version).

    Similar to sliding_window but with explicit count instead of token limit.
    Faster but less precise about token limits.

    Args:
        messages: Message list
        max_tokens: Model's context window
        output_buffer: Tokens to reserve for output
        keep_count: Number of recent messages to keep

    Returns:
        Clipped message list
    """
    # Always keep system message
    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None

    # Get messages after system
    if system_msg:
        non_system = messages[1:]
    else:
        non_system = messages

    # Keep last N messages
    recent = non_system[-keep_count:] if len(non_system) > keep_count else non_system

    # Rebuild
    result = []
    if system_msg:
        result.append(system_msg)
    result.extend(recent)

    return result


def _summarize_strategy(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    output_buffer: int,
    llm: Any = None,
    keep_count: int = 10,
    summary_log_callback: SummaryLogCallback = None
) -> List[Dict[str, Any]]:
    """Summarize old messages using LLM.

    Summarizes older messages (beyond keep_count) using an LLM call.
    Keeps recent messages verbatim for context.

    Strategy:
    1. Keep system message always
    2. Keep last N messages verbatim (most recent context)
    3. Summarize older messages using LLM
    4. Return: [system, summary, recent_messages]

    Args:
        messages: Message list
        max_tokens: Model's context window
        output_buffer: Tokens to reserve for output
        llm: LLM instance to use for summarization (must support .ainvoke() or .invoke())
        keep_count: Number of recent messages to keep verbatim
        summary_log_callback: Optional callback for logging summarization calls

    Returns:
        Message list with summary replacing old messages
    """
    # Always keep system message
    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None

    # Get messages after system
    if system_msg:
        non_system = messages[1:]
    else:
        non_system = messages

    # If we have few messages, no need to summarize
    if len(non_system) <= keep_count:
        return messages

    # Split into old messages (to summarize) and recent (to keep verbatim)
    old_messages = non_system[:-keep_count]
    recent_messages = non_system[-keep_count:]

    # Build text from old messages for summarization
    conversation_text = ""
    for msg in old_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        conversation_text += f"{role.upper()}: {content}\n\n"

    # Default summary prompt
    summary_prompt = f"""Summarize the following conversation concisely.

Focus on:
1. Key topics discussed
2. Important decisions or conclusions
3. Action items or next steps
4. Critical context needed to continue

Keep the summary under 500 words.

CONVERSATION:
{conversation_text}

SUMMARY:"""

    # If no LLM provided, fall back to basic clipping
    if llm is None:
        print("[Context Summarization] No LLM provided, falling back to basic clipping", file=sys.stderr)
        return _basic_strategy(messages, max_tokens, output_buffer)

    # Call LLM to generate summary
    start_time = time.time()
    try:
        # Try async first
        if hasattr(llm, 'ainvoke'):
            import asyncio
            # Check if we're in an async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, need to create task
                    summary_result = asyncio.run_coroutine_threadsafe(
                        llm.ainvoke(summary_prompt),
                        loop
                    ).result(timeout=30)
                else:
                    summary_result = asyncio.run(llm.ainvoke(summary_prompt))
            except (RuntimeError, asyncio.TimeoutError):
                # Fall back to sync
                summary_result = llm.invoke(summary_prompt)
        else:
            # Use sync invoke
            summary_result = llm.invoke(summary_prompt)

        # Extract summary text from response
        if hasattr(summary_result, 'content'):
            summary_text = str(summary_result.content)
        else:
            summary_text = str(summary_result)

        duration = time.time() - start_time

        # Log the summarization call
        original_tokens = count_tokens(old_messages)
        summary_tokens = count_tokens([{"role": "user", "content": summary_text}])

        if summary_log_callback:
            summary_log_callback(summary_text, original_tokens, summary_tokens, len(old_messages), duration)
        else:
            _default_summary_log(summary_text, original_tokens, summary_tokens, len(old_messages), duration)

        # Build result: [system, summary, recent_messages]
        result = []
        if system_msg:
            result.append(system_msg)

        # Add summary as a system message
        result.append({
            "role": "system",
            "content": f"[Previous conversation summary]\n{summary_text}"
        })

        # Add recent messages verbatim
        result.extend(recent_messages)

        return result

    except Exception as e:
        print(f"[Context Summarization] Error during summarization: {e}", file=sys.stderr)
        print(f"[Context Summarization] Falling back to basic clipping", file=sys.stderr)
        # Fall back to basic clipping on error
        return _basic_strategy(messages, max_tokens, output_buffer)


def manage_context(
    messages: List[Dict[str, Any]],
    max_tokens: int = 200000,
    output_buffer: int = 20000,
    strategy: str | ContextStrategy = ContextStrategy.BASIC,
    **kwargs
) -> List[Dict[str, Any]]:
    """Manage context window to prevent token limit errors.

    Hot-swappable strategy interface. Select strategy via:
    - strategy parameter (this function)
    - CONTEXT_STRATEGY env var (fallback)
    - Config file (via env var)

    Args:
        messages: Message list from conversation
        max_tokens: Model's context window (default: 200k for glm-4.7)
        output_buffer: Tokens to reserve for LLM output (default: 20k)
        strategy: Strategy name ("basic", "sliding_window", "keep_recent", "summarize")
        **kwargs: Additional strategy-specific params:
            - window_size: For sliding_window (default: 50)
            - keep_count: For keep_recent/summarize (default: 20 for keep_recent, 10 for summarize)
            - llm: For summarize - LLM instance to use for summarization
            - summary_log_callback: For summarize - Optional callback for logging

    Returns:
        Clipped message list safe to send to LLM

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Old question"},
        ...     {"role": "assistant", "content": "Old answer"},
        ...     {"role": "user", "content": "New question"},
        ... ]
        >>> safe_messages = manage_context(messages, max_tokens=10000, strategy="basic")
    """
    # Allow env var override (like SEARCH_BACKEND for lang_tools)
    env_strategy = os.getenv("CONTEXT_STRATEGY")
    if env_strategy:
        strategy = env_strategy

    # Normalize to enum
    if isinstance(strategy, str):
        strategy = ContextStrategy(strategy.lower())

    # Dispatch to strategy implementation
    if strategy == ContextStrategy.SUMMARIZE:
        return _summarize_strategy(messages, max_tokens, output_buffer, **kwargs)
    elif strategy == ContextStrategy.SLIDING_WINDOW:
        return _sliding_window_strategy(messages, max_tokens, output_buffer, **kwargs)
    elif strategy == ContextStrategy.KEEP_RECENT:
        return _keep_recent_strategy(messages, max_tokens, output_buffer, **kwargs)
    else:  # BASIC is default
        return _basic_strategy(messages, max_tokens, output_buffer)
