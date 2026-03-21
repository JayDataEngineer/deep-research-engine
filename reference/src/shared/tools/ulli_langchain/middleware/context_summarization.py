"""Token Limit Middleware for LLM calls.

Prevents "Context size has been exceeded" errors by enforcing token limits
before LLM calls. Uses summarization and truncation strategies to reduce
context size when approaching limits.

Usage:
    from .tools.ulli_langchain.middleware import TokenLimitMiddleware

    middleware = [
        TokenLimitMiddleware(max_tokens_ratio=0.8),  # 80% of model context
        ToolRetryMiddleware(max_retries=2),
    ]
"""

from typing import Any, Dict, List, Optional, Callable
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from .logging_config import get_logger

logger = get_logger(__name__)


def count_tokens_estimate(messages: List[Any]) -> int:
    """Estimate token count for messages.

    Uses simple heuristic: ~4 characters per token.
    """
    total = 0
    for msg in messages:
        if hasattr(msg, 'content'):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get('content', '')
        else:
            content = str(msg)

        # Rough estimate: 4 chars per token
        total += len(content) // 4

        # Add overhead for message structure
        total += 10

    return max(total, 1)


def get_model_context_limit(model_name: str) -> int:
    """Get context limit for a model."""
    limits = {
        "glm-4.7": 128000,
        "glm-4": 128000,
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16384,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
    }

    for key, limit in limits.items():
        if key.lower() in model_name.lower():
            return limit

    # Default to 32k for unknown models
    return 32000


def summarize_messages(
    messages: List[BaseMessage],
    max_tokens: int,
    keep_recent: int = 4
) -> List[BaseMessage]:
    """Summarize messages to fit within token limit.

    Strategy:
    1. Keep system messages
    2. Keep most recent N messages
    3. Summarize older conversation into a single message

    Args:
        messages: Original message list
        max_tokens: Target token limit
        keep_recent: Number of recent messages to keep intact

    Returns:
        Summarized message list
    """
    if not messages:
        return messages

    # Separate system messages
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    conversation = [m for m in messages if not isinstance(m, SystemMessage)]

    # Check if we already fit
    current_tokens = count_tokens_estimate(system_msgs + conversation)
    if current_tokens <= max_tokens:
        return messages

    logger.info(f"[TOKEN_LIMIT] Summarizing {len(messages)} messages ({current_tokens} tokens -> {max_tokens} target)")

    # Keep recent messages
    if len(conversation) > keep_recent:
        recent_msgs = conversation[-keep_recent:]
        older_msgs = conversation[:-keep_recent]

        # Create summary of older messages
        summary_parts = []
        for msg in older_msgs:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = str(msg.content)[:500]  # Truncate long content
            summary_parts.append(f"{role}: {content}")

        summary = "[Earlier conversation summarized]:\n" + "\n".join(summary_parts)

        # Create summary message
        summary_msg = HumanMessage(content=summary)

        result = system_msgs + [summary_msg] + recent_msgs
    else:
        # Just truncate all messages
        result = []
        remaining_tokens = max_tokens - count_tokens_estimate(system_msgs)
        tokens_per_msg = remaining_tokens // max(len(conversation), 1)

        for msg in conversation:
            content = str(msg.content)
            max_chars = tokens_per_msg * 4 - 100  # Leave buffer

            if len(content) > max_chars:
                content = content[:max_chars] + "\n[...truncated...]"

            if isinstance(msg, HumanMessage):
                result.append(HumanMessage(content=content))
            elif isinstance(msg, AIMessage):
                result.append(AIMessage(content=content))
            else:
                result.append(HumanMessage(content=content))

        result = system_msgs + result

    new_tokens = count_tokens_estimate(result)
    logger.info(f"[TOKEN_LIMIT] Summarized to {len(result)} messages ({new_tokens} tokens)")

    return result


class TokenLimitMiddleware:
    """Middleware that enforces token limits before LLM calls.

    This wraps the LLM to check and reduce message size before each call.
    """

    def __init__(
        self,
        max_tokens_ratio: float = 0.75,
        model_name: str = "glm-4.7",
        keep_recent: int = 6
    ):
        """Initialize middleware.

        Args:
            max_tokens_ratio: Ratio of model context to use (default 0.75 = 75%)
            model_name: Model name to determine context limit
            keep_recent: Number of recent messages to keep intact
        """
        self.max_tokens_ratio = max_tokens_ratio
        self.model_name = model_name
        self.keep_recent = keep_recent
        self.context_limit = get_model_context_limit(model_name)
        self.max_tokens = int(self.context_limit * max_tokens_ratio)

        logger.info(
            f"[TOKEN_LIMIT] Initialized with max_tokens={self.max_tokens} "
            f"({max_tokens_ratio*100}% of {self.context_limit})"
        )

    def __call__(self, func: Callable) -> Callable:
        """Wrap a function with token limiting."""
        async def wrapper(*args, **kwargs):
            # Extract messages from args/kwargs
            messages = None

            if args and isinstance(args[0], dict) and 'messages' in args[0]:
                messages = args[0]['messages']
            elif 'messages' in kwargs:
                messages = kwargs['messages']
            elif 'input' in kwargs:
                # Some agents use 'input' instead of 'messages'
                messages = kwargs['input']

            if messages:
                # Check token count
                token_count = count_tokens_estimate(messages)

                if token_count > self.max_tokens:
                    logger.warning(
                        f"[TOKEN_LIMIT] Context ({token_count} tokens) exceeds limit "
                        f"({self.max_tokens}). Applying summarization."
                    )

                    # Summarize messages
                    summarized = summarize_messages(
                        messages,
                        self.max_tokens,
                        self.keep_recent
                    )

                    # Update kwargs with summarized messages
                    if 'messages' in kwargs:
                        kwargs['messages'] = summarized
                    elif 'input' in kwargs:
                        kwargs['input'] = summarized
                    elif args and isinstance(args[0], dict) and 'messages' in args[0]:
                        new_args = list(args)
                        new_args[0] = {**args[0], 'messages': summarized}
                        args = tuple(new_args)

            return await func(*args, **kwargs)

        return wrapper


def create_token_limited_agent(
    agent: Any,
    model_name: str = "glm-4.7",
    max_tokens_ratio: float = 0.75
) -> Any:
    """Wrap an agent with token limiting.

    Args:
        agent: The agent to wrap
        model_name: Model name for context limit
        max_tokens_ratio: Ratio of context to use

    Returns:
        Wrapped agent with token limiting
    """
    middleware = TokenLimitMiddleware(
        max_tokens_ratio=max_tokens_ratio,
        model_name=model_name
    )

    # Wrap the ainvoke method
    original_ainvoke = agent.ainvoke

    async def limited_ainvoke(*args, **kwargs):
        return await middleware(original_ainvoke)(*args, **kwargs)

    agent.ainvoke = limited_ainvoke

    return agent
