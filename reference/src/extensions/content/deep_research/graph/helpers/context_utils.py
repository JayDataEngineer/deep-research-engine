"""Context management utilities for Deep Research.

This module consolidates repeated patterns for:
- Token budget calculation
- Lazy loading instruction generation
- Model context detection

DRY principle: Don't Repeat Yourself - use these utilities instead of
duplicating the same logic across lab.py, synthesis.py, and critic.py.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path


def calculate_token_budget(
    yaml_config: Dict[str, Any],
    role: str,
    estimated_overhead: int = 5000
) -> Dict[str, int]:
    """Calculate available token budget for a given role.

    This consolidates the repeated pattern of:
    1. Get model name from config
    2. Get model's context window
    3. Get role-specific buffer percentage
    4. Calculate available tokens

    Args:
        yaml_config: Full configuration dict
        role: Role name ("lab", "synthesis", "critic")
        estimated_overhead: Estimated tokens for system/template overhead (default: 5000)

    Returns:
        Dict with:
            - model_max_tokens: Model's full context window
            - buffer_tokens: Reserved buffer for output
            - available_tokens: Tokens available for content
            - buffer_percent: Buffer percentage used
    """
    from src.shared.model_context import get_model_max_tokens, get_recommended_buffer_for_role

    model_name = yaml_config.get("model", "glm-4.7")
    model_max_tokens = get_model_max_tokens(model_name, default=128000)
    buffer_percent = get_recommended_buffer_for_role(role)
    buffer_tokens = int(model_max_tokens * (buffer_percent / 100.0))
    available_tokens = model_max_tokens - buffer_tokens - estimated_overhead

    return {
        "model_max_tokens": model_max_tokens,
        "buffer_tokens": buffer_tokens,
        "available_tokens": available_tokens,
        "buffer_percent": buffer_percent,
    }


def calculate_percentage_limit(
    yaml_config: Dict[str, Any],
    percentage: float
) -> int:
    """Calculate token limit as percentage of model context.

    Args:
        yaml_config: Full configuration dict
        percentage: Percentage as float (e.g., 0.10 for 10%, 0.15 for 15%)

    Returns:
        Token limit as integer
    """
    from src.shared.model_context import get_model_max_tokens

    model_name = yaml_config.get("model", "glm-4.7")
    model_max_tokens = get_model_max_tokens(model_name, default=128000)
    return int(model_max_tokens * percentage)


def count_tokens_safe(text: str, fallback_ratio: int = 4) -> int:
    """Count tokens with fallback estimation.

    Args:
        text: Text to count
        fallback_ratio: Character-to-token ratio (UNUSED - removed fallback)

    Returns:
        Actual token count

    Raises:
        RuntimeError: If token counting fails
    """
    try:
        from src.extensions.libraries.core_chat_engine.memory.context_strategies import count_tokens
        return count_tokens([{"role": "user", "content": text}])
    except Exception as e:
        # CRITICAL: Token counting failure - log and re-raise (no fallback!)
        import logging
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from ...logging_utils import get_logger
        logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level
        logger.error(f"[CONTEXT_UTILS] Token counting failed: {e}")
        raise RuntimeError(f"Token counting failed: {e}") from e


def build_lazy_loading_instruction(
    artifact_path: str,
    token_count: int,
    manifest_abstract: str = "",
    manifest_themes: List[str] = None,
    instruction_type: str = "worker"
) -> str:
    """Build lazy loading instruction for overflow scenarios.

    This consolidates the repeated pattern of building lazy loading messages
    across lab.py (3 locations) into a single utility.

    Args:
        artifact_path: Path to the artifact file
        token_count: Number of tokens that were overflowed
        manifest_abstract: Optional manifest abstract
        manifest_themes: Optional list of key themes
        instruction_type: Type of instruction ("worker", "synthesis", "combined")

    Returns:
        Formatted lazy loading instruction string
    """
    if instruction_type == "worker":
        # Individual worker overflow (lab.py lines 122-157, 2122-2140)
        themes_str = ""
        if manifest_themes:
            themes_str = "\nKEY THEMES:\n" + "\n".join(f"- {theme}" for theme in manifest_themes)

        return f"""=== WORKER FINDINGS (OVERFLOW) ===

This worker generated too much content to include directly ({token_count} tokens).
Full findings have been saved to an artifact file.

ARTIFACT PATH: {artifact_path}

MANIFEST:
{manifest_abstract or "No manifest available"}
{themes_str}

To view the full findings, use the read_artifact tool:
- Call read_artifact with artifact_path: "{artifact_path}"
- You can specify max_length or section parameters if needed

=== END WORKER FINDINGS ===
"""

    elif instruction_type == "synthesis":
        # Synthesis overflow (synthesis.py lines 342-351)
        return f"""=== ON-DEMAND RETRIEVAL (MANDATORY) ===
CRITICAL: You will receive inputs from multiple workers. Some may be provided
as direct text, others as ARTIFACT PATH references.

You are FORBIDDEN from ignoring Artifact Paths. If you see a path, you MUST
call read_artifact to ingest that data BEFORE writing your response.

Ignoring artifact paths will result in incomplete research outputs.

To retrieve full findings:
- Call read_artifact with the artifact_path (e.g., '{artifact_path}')
- The tool will retrieve the full content from that artifact file
- You MUST do this for ALL artifact paths before synthesizing
"""

    elif instruction_type == "combined":
        # Combined worker overflow (lab.py lines 231-248)
        parts = [
            "=== PREVIOUS WORKER OUTPUTS ===\n",
            f"Combined findings are too large to include directly ({token_count} tokens).",
            "Full findings have been saved to artifact files.\n",
        ]
        if manifest_abstract:
            parts.append(f"ABSTRACT: {manifest_abstract}\n")
        if manifest_themes:
            parts.append("THEMES:")
            for theme in manifest_themes:
                parts.append(f"  - {theme}")
            parts.append("")
        parts.extend([
            f"ARTIFACT: {artifact_path}",
            "",
            "To view any worker's full findings, use the read_artifact tool with the artifact_path shown above.",
            "=== END PREVIOUS WORKER OUTPUTS ===",
        ])
        return "\n".join(parts)

    else:
        raise ValueError(f"Unknown instruction_type: {instruction_type}")


def check_token_overflow(
    text: str,
    yaml_config: Dict[str, Any],
    max_percent: float,
    run_logger=None
) -> tuple[bool, int, int]:
    """Check if text exceeds percentage-based token budget.

    Args:
        text: Text to check
        yaml_config: Full configuration dict
        max_percent: Maximum percentage as float (e.g., 0.10 for 10%)
        run_logger: Optional run_logger for logging

    Returns:
        Tuple of (is_overflow, token_count, max_tokens)
    """
    token_count = count_tokens_safe(text)
    max_tokens = calculate_percentage_limit(yaml_config, max_percent)

    if run_logger and token_count > max_tokens:
        from src.shared.model_context import get_model_max_tokens
        model_name = yaml_config.get("model", "glm-4.7")
        model_max_tokens = get_model_max_tokens(model_name, default=128000)
        run_logger._write_to_file(
            f"Token count ({token_count} / {model_max_tokens} = {token_count/model_max_tokens*100:.1f}%) "
            f"exceeds budget ({max_percent*100:.0f}% = {max_tokens} tokens)"
        )

    return token_count > max_tokens, token_count, max_tokens


def enforce_token_limit(
    messages: List[Dict[str, str]],
    yaml_config: Dict[str, Any],
    role: str = "worker",
    run_logger: Optional[Any] = None,
    phase: str = "LLM"
) -> List[Dict[str, str]]:
    """Enforce token limit before LLM call to prevent timeout/hang.

    GLM-4.7 and other models will timeout or hang if sent too much context.
    This function ENFORCES the limit by truncating if necessary.

    Percentage-based with YAML override support:
    - Reads percentages from YAML config (token_limits section)
    - Falls back to role-based defaults (70% workers, 80% synthesis/critic)
    - Auto-detects model max context from registry

    Args:
        messages: Message list to send to LLM
        yaml_config: Full configuration dict
        role: Role name (worker, synthesis, critic, worker, boss)
        run_logger: Optional run_logger for logging
        phase: Phase name for logging (default: "LLM")

    Returns:
        Safe messages list (truncated if necessary)

    Raises:
        RuntimeError: If messages would exceed 90% of model context (critical overflow)

    YAML Override Format:
        token_limits:
          worker_percent: 0.70
          synthesis_percent: 0.80
          critic_percent: 0.80
          default_percent: 0.75
    """
    from src.shared.model_context import get_model_max_tokens, get_safe_token_limit

    model_name = yaml_config.get("model", "glm-4.7")
    model_max_tokens = get_model_max_tokens(model_name, default=128000)

    # Get safe limit based on role (with YAML overrides)
    warning_limit = get_safe_token_limit(model_name, role, yaml_config)
    critical_limit = int(model_max_tokens * 0.90)

    # Calculate actual token count
    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    estimated_tokens = count_tokens_safe(str(messages))

    if estimated_tokens > critical_limit:
        # CRITICAL: Would exceed 90% - use summarization middleware instead of raising error
        if run_logger:
            run_logger._write_to_file(
                f"[{phase}] CRITICAL: Token count ({estimated_tokens} / {model_max_tokens} = "
                f"{estimated_tokens/model_max_tokens*100:.1f}%) exceeds critical threshold "
                f"(90%). Applying summarization middleware to reduce context."
            )

        # Apply summarization to reduce context size
        messages = summarize_messages(messages, critical_limit - 2000, run_logger, phase)

    if estimated_tokens > warning_limit and run_logger:
        # WARNING: Exceeds warning limit - log and truncate user message
        run_logger._write_to_file(
            f"[{phase}] WARNING: Token count ({estimated_tokens} / {model_max_tokens} = "
            f"{estimated_tokens/model_max_tokens*100:.1f}%) exceeds safety threshold "
            f"({warning_limit/model_max_tokens*100:.1f}%). Truncating last user message."
        )

        # Truncate the last user message to fit
        safe_tokens = warning_limit - 1000  # Leave buffer
        messages = truncate_messages_to_fit(messages, safe_tokens, run_logger, phase)

    return messages


def truncate_messages_to_fit(
    messages: List[Dict[str, str]],
    max_tokens: int,
    run_logger: Optional[Any] = None,
    phase: str = "LLM"
) -> List[Dict[str, str]]:
    """Truncate messages to fit within token limit.

    Preserves system messages and truncates the last user message.

    Args:
        messages: Original message list
        max_tokens: Maximum tokens allowed
        run_logger: Optional run_logger for logging
        phase: Phase name for logging

    Returns:
        Truncated message list
    """
    # Keep system messages intact
    system_msgs = [m for m in messages if m.get("role") == "system"]

    # Find last user message (the one to truncate)
    user_msgs = []
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user_idx = i
            user_msgs.append(msg)
        elif msg.get("role") == "assistant":
            user_msgs.append(msg)

    if not user_msgs:
        return system_msgs

    # Truncate last user message
    last_msg = user_msgs[-1]
    content = last_msg.get("content", "")

    # Estimate safe length (chars = tokens * 4)
    safe_chars = max_tokens * 4

    if len(content) > safe_chars:
        truncated = content[:safe_chars]
        truncated += "\n\n[Content truncated due to token limit...]"
        last_msg["content"] = truncated

        if run_logger:
            run_logger._write_to_file(
                f"[{phase}] Truncated content from {len(content)} to {len(truncated)} chars"
            )

    user_msgs[-1] = last_msg
    return system_msgs + user_msgs


def summarize_messages(
    messages: List[Dict[str, str]],
    max_tokens: int,
    run_logger: Optional[Any] = None,
    phase: str = "LLM"
) -> List[Dict[str, str]]:
    """Summarize messages to fit within token limit using smart compression.
    
    This function applies several strategies to reduce context size:
    1. Summarize older assistant messages
    2. Truncate long user messages
    3. Keep recent messages intact
    
    Args:
        messages: Original message list
        max_tokens: Maximum tokens allowed
        run_logger: Optional run_logger for logging
        phase: Phase name for logging
        
    Returns:
        Compressed message list that fits within token limit
    """
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    
    # Calculate current token estimate
    estimated_tokens = count_tokens_safe(str(messages))
    
    # If already under limit, return as-is
    if estimated_tokens <= max_tokens:
        return messages
    
    if run_logger:
        run_logger._write_to_file(
            f"[{phase}] Summarizing messages: {estimated_tokens} -> {max_tokens} tokens"
        )
    
    # Strategy 1: Remove middle assistant messages (keep first and last)
    system_msgs = [m for m in messages if m.get("role") == "system"]
    
    # Get conversation messages (user + assistant)
    conversation = [m for m in messages if m.get("role") in ("user", "assistant")]
    
    # Keep first 2 and last 2 messages, summarize the middle
    if len(conversation) > 4:
        first_messages = conversation[:2]
        last_messages = conversation[-2:]
        middle_messages = conversation[2:-2]
        
        # Create a summary of middle messages
        summary_content = f"[... {len(middle_messages)} turns of conversation summarized ...]"
        
        # Build compressed conversation
        compressed = system_msgs + first_messages + [
            {"role": "assistant", "content": summary_content}
        ] + last_messages
        
        if run_logger:
            run_logger._write_to_file(
                f"[{phase}] Compressed conversation from {len(messages)} to {len(compressed)} messages"
            )
        
        return compressed
    
    # Strategy 2: If still too long, truncate long contents
    safe_tokens_per_message = max_tokens // len(messages)
    safe_chars_per_message = safe_tokens_per_message * 4
    
    compressed = []
    for msg in messages:
        content = msg.get("content", "")
        if len(content) > safe_chars_per_message:
            compressed_content = content[:safe_chars_per_message] + "\n\n[...truncated...]"
            compressed.append({**msg, "content": compressed_content})
        else:
            compressed.append(msg)
    
    if run_logger:
        new_estimate = count_tokens_safe(str(compressed))
        run_logger._write_to_file(
            f"[{phase}] Truncated messages: {estimated_tokens} -> {new_estimate} tokens"
        )
    
    return compressed
