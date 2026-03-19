"""Proper message structure for LLM calls.

CRITICAL INSIGHT: The human user ONLY provides the initial query.
ALL subsequent content (findings, context, tool results, feedback) is SYSTEM-GENERATED.

Therefore:
- `role: user` = ONLY the original human query (from state["query"])
- `role: system` = Everything else (prompts, context, findings, tool results)

This module enforces this pattern throughout deep research.

MESSAGE STRUCTURE PATTERN:
==========================
1. System message: Role definition and instructions (from config prompts)
2. System message: Context data (findings, tool results, etc.) - SYSTEM-GENERATED
3. User message: The original human query - THE ONLY USER MESSAGE

Example:
    [
        {"role": "system", "content": "You are a research agent..."},
        {"role": "system", "content": "Context: [findings from workers]"},
        {"role": "user", "content": "Create 5 tweets about gym discipline"}
    ]

NOT:
    [
        {"role": "system", "content": "You are a research agent..."},
        {"role": "user", "content": "Results from 3 explorers: [...]"}  # WRONG!
    ]
"""

from typing import List, Dict, Any, Optional, Union


# =============================================================================
# CORE MESSAGE BUILDERS
# =============================================================================

def build_research_messages(
    system_prompt: str,
    original_query: str,
    context_data: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build messages for research LLM calls with proper role attribution.

    CRITICAL: The original_query is the ONLY user message. All other content
    (context_data, findings, tool results) is SYSTEM-generated and goes in
    system messages.

    Args:
        system_prompt: The system instructions (from config prompts)
        original_query: The ORIGINAL human query (the only user input)
        context_data: System-generated context (findings, tool results, etc.)

    Returns:
        Properly structured message list with correct role attribution
    """
    messages = []

    # 1. System message - defines role and behavior
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 2. System message - context/data (SYSTEM-GENERATED, not user!)
    if context_data:
        messages.append({"role": "system", "content": context_data})

    # 3. User message - the ORIGINAL human query (THE ONLY USER MESSAGE)
    if original_query:
        messages.append({"role": "user", "content": original_query})

    return messages


def build_initial_query_messages(
    system_prompt: str,
    human_query: str,
) -> List[Dict[str, str]]:
    """Build messages for the initial query (before any research).

    This is used when we have the original human query but no system-generated
    context yet.

    Args:
        system_prompt: The system instructions
        human_query: The original human query

    Returns:
        Message list with system prompt and user query
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_query},
    ]


def build_context_messages(
    system_prompt: str,
    human_query: str,
    context_sections: List[str],
) -> List[Dict[str, str]]:
    """Build messages with multiple context sections.

    All context_sections are SYSTEM-GENERATED content (findings, tool results,
    feedback, etc.) and should be in system messages, NOT user messages.

    Args:
        system_prompt: The system instructions
        human_query: The original human query (THE ONLY USER MESSAGE)
        context_sections: List of system-generated context sections

    Returns:
        Message list with proper role attribution
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Add all context as system messages (SYSTEM-GENERATED)
    for section in context_sections:
        if section:
            messages.append({"role": "system", "content": section})

    # Add the human query (THE ONLY USER MESSAGE)
    messages.append({"role": "user", "content": human_query})

    return messages


# =============================================================================
# SPECIALIZED BUILDERS FOR DEEP RESEARCH NODES
# =============================================================================

def build_planning_messages(
    system_instructions: str,
    research_avenue: str,
    context_section: str = "",
) -> List[Dict[str, str]]:
    """Build messages for planning/research avenue generation.

    Args:
        system_instructions: The system prompt from config
        research_avenue: The research avenue to investigate
        context_section: Additional context (SYSTEM-GENERATED)

    Returns:
        Properly structured message list
    """
    # Build context as system message
    context_parts = []
    if research_avenue:
        context_parts.append(f"Research avenue to investigate: {research_avenue}")
    if context_section:
        context_parts.append(context_section)

    context_data = "\n".join(context_parts) if context_parts else None

    # The "research avenue" is derived from the original query, so it's still
    # part of the system's interpretation, not a new user message.
    # We include the original avenue as context in a system message.
    return build_research_messages(
        system_prompt=system_instructions,
        original_query=research_avenue,  # This IS the user's intent
        context_data=context_data,
    )


def build_synthesis_messages(
    system_instructions: str,
    findings_summary: str,
    num_explorers: int,
    original_query: str,
) -> List[Dict[str, str]]:
    """Build messages for synthesis.

    CRITICAL: Findings are SYSTEM-GENERATED, not user-generated!

    Args:
        system_instructions: The system prompt from config
        findings_summary: Summary of findings (SYSTEM-GENERATED)
        num_explorers: Number of explorers that contributed
        original_query: The original human query

    Returns:
        Properly structured message list
    """
    # Context is SYSTEM-GENERATED
    context_data = f"Results from {num_explorers} explorers:\n{findings_summary}"

    return build_research_messages(
        system_prompt=system_instructions,
        original_query=original_query,  # THE ONLY USER MESSAGE
        context_data=context_data,  # SYSTEM-GENERATED
    )


def build_critic_messages(
    system_instructions: str,
    original_query: str,
    draft_content: str,
    research_brief: str = "",
    history: str = "",
) -> List[Dict[str, str]]:
    """Build messages for critic evaluation.

    CRITICAL: Draft, brief, and history are ALL SYSTEM-GENERATED!

    Args:
        system_instructions: The system prompt from config
        original_query: The original human query (THE ONLY USER MESSAGE)
        draft_content: Draft content to evaluate (SYSTEM-GENERATED)
        research_brief: Research brief summary (SYSTEM-GENERATED)
        history: Previous critique history (SYSTEM-GENERATED)

    Returns:
        Properly structured message list
    """
    context_sections = []

    if research_brief:
        context_sections.append(f"RESEARCH BRIEF:\n{research_brief}")

    if draft_content:
        context_sections.append(f"DRAFT TO EVALUATE:\n{draft_content}")

    if history:
        context_sections.append(f"PREVIOUS CRITIQUES:\n{history}")

    return build_context_messages(
        system_prompt=system_instructions,
        human_query=original_query,
        context_sections=context_sections,
    )


def build_boss_messages(
    system_instructions: str,
    original_query: str,
    previous_findings: str = "",
    critic_feedback: str = "",
    iteration: int = 1,
) -> List[Dict[str, str]]:
    """Build messages for boss decision-making.

    CRITICAL: Findings and feedback are SYSTEM-GENERATED!

    Args:
        system_instructions: The system prompt from config
        original_query: The original human query (THE ONLY USER MESSAGE)
        previous_findings: Findings from previous iteration (SYSTEM-GENERATED)
        critic_feedback: Feedback from critic (SYSTEM-GENERATED)
        iteration: Current iteration number

    Returns:
        Properly structured message list
    """
    context_sections = []

    context_sections.append(f"Iteration: {iteration}")

    if previous_findings:
        context_sections.append(f"PREVIOUS FINDINGS:\n{previous_findings}")

    if critic_feedback:
        context_sections.append(f"CRITIC FEEDBACK:\n{critic_feedback}")

    return build_context_messages(
        system_prompt=system_instructions,
        human_query=original_query,
        context_sections=context_sections,
    )


def build_system_only(system_prompt: str) -> List[Dict[str, str]]:
    """Build a message list with only a system message.

    Used when the system prompt already contains everything needed.
    """
    return [{"role": "system", "content": system_prompt}]


def build_user_only(user_content: str) -> List[Dict[str, str]]:
    """Build a message list with only a user message.

    WARNING: Only use this for the ORIGINAL human query!
    System-generated content should NOT be in user messages.
    """
    return [{"role": "user", "content": user_content}]


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================

def convert_prompt_to_messages(
    prompt: Union[str, List[Dict[str, str]], List[Any]],
    original_query: Optional[str] = None,
    context_data: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Convert various prompt formats to properly structured messages.

    This is the main entry point for ensuring proper message structure.

    CRITICAL: If you have system-generated content (findings, context, etc.),
    pass it as context_data, NOT as part of a "user" message!

    Args:
        prompt: The prompt - can be:
            - str: A system prompt (will be converted to system message)
            - List[Dict]: Already structured messages (validated and returned)
            - List[BaseMessage]: LangChain messages (converted to dicts)
        original_query: The ORIGINAL human query (for the user message)
        context_data: System-generated context (goes in system message)

    Returns:
        Properly structured message list
    """
    # Already a list - validate and return
    if isinstance(prompt, list):
        messages = []
        for msg in prompt:
            if isinstance(msg, dict):
                # Validate role
                if "role" not in msg:
                    raise ValueError(f"Message dict missing 'role': {msg}")
                if "content" not in msg:
                    raise ValueError(f"Message dict missing 'content': {msg}")
                messages.append(msg)
            else:
                # LangChain BaseMessage - convert
                from langchain_core.messages import BaseMessage
                if isinstance(msg, BaseMessage):
                    role = msg.type
                    if role == "human":
                        role = "user"
                    elif role == "ai":
                        role = "assistant"
                    messages.append({"role": role, "content": str(msg.content)})
                else:
                    raise ValueError(f"Unknown message type: {type(msg)}")

        # Add context and query if provided
        if context_data:
            # Insert before any user messages
            user_idx = next(
                (i for i, m in enumerate(messages) if m.get("role") == "user"),
                len(messages)
            )
            messages.insert(user_idx, {"role": "system", "content": context_data})

        if original_query:
            messages.append({"role": "user", "content": original_query})

        return messages

    # String - treat as system prompt
    if isinstance(prompt, str):
        return build_research_messages(
            system_prompt=prompt,
            original_query=original_query or "",
            context_data=context_data,
        )

    raise ValueError(f"Unknown prompt type: {type(prompt)}")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_message_roles(messages: List[Dict[str, str]]) -> List[str]:
    """Validate that message roles are correctly attributed.

    Returns a list of warnings/issues found. Empty list means valid.

    Checks:
    1. Only ONE user message (the original query)
    2. All system-generated content is in system messages
    """
    issues = []

    user_count = sum(1 for m in messages if m.get("role") == "user")

    if user_count == 0:
        issues.append("No user message found - missing original query?")
    elif user_count > 1:
        issues.append(
            f"Multiple user messages ({user_count}) found. "
            "Only the original query should be a user message. "
            "All other content (findings, context, etc.) should be system messages."
        )

    return issues


def get_message_summary(messages: List[Dict[str, str]]) -> str:
    """Get a summary of message structure for debugging."""
    roles = [m.get("role", "unknown") for m in messages]
    counts = {}
    for r in roles:
        counts[r] = counts.get(r, 0) + 1

    parts = [f"{role}: {count}" for role, count in counts.items()]
    return f"Messages ({len(messages)} total): {', '.join(parts)}"
