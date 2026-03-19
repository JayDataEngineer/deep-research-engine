"""Twitter/X Thread Synthesis - Generates engaging Twitter threads from context.

Reads from LangGraph Store and generates a thread of tweets.
This is an OPTIONAL separate CLI command, NOT part of the main pipeline.

Usage:
    uv run context-engine synthesize --session-id abc123 --format twitter

Based on content_formatter from unified-llm-interface.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field
from loguru import logger


def _get_model(tier: str):
    """Lazy import for model."""
    from context_engine.config import get_model
    return get_model(tier)


async def _get_store():
    """Lazy import for store."""
    from context_engine.store.setup import get_store
    return await get_store()


# =============================================================================
# SCHEMAS
# =============================================================================


class Tweet(BaseModel):
    """A single tweet in a thread."""

    content: str = Field(
        description="Tweet content (max 280 characters)",
        max_length=280,
    )
    hashtags: list[str] = Field(
        default_factory=list,
        description="Hashtags for this tweet",
    )
    media_hint: Optional[str] = Field(
        default=None,
        description="Suggested media type (image, video, None)",
    )


class TweetThread(BaseModel):
    """A thread of tweets."""

    tweets: list[Tweet] = Field(
        description="List of tweets in the thread",
        min_length=1,
        max_length=10,
    )
    hook_strength: str = Field(
        description="Assessment of first tweet's hook strength (weak/okay/strong)",
        pattern="^(weak|okay|strong)$",
    )
    call_to_action: str = Field(
        description="Call to action or closing thought",
        max_length=500,
    )
    total_char_count: int = Field(
        description="Total character count across all tweets",
    )


# =============================================================================
# PROMPT
# =============================================================================

TWITTER_SYNTHESIS_PROMPT = """You are an expert at creating engaging Twitter/X threads from research content.

Create a compelling Twitter thread based on the following context. The thread should:
1. Have a strong hook in the first tweet
2. Break down complex ideas into digestible tweets
3. Use engaging language and proper formatting
4. Include relevant hashtags (2-3 per tweet max)
5. End with a call to action

CONTEXT:
{context}

OUTPUT FORMAT (JSON):
{{
  "tweets": [
    {{
      "content": "Tweet text (max 280 chars)",
      "hashtags": ["tag1", "tag2"],
      "media_hint": "image|video|null"
    }}
  ],
  "hook_strength": "weak|okay|strong",
  "call_to_action": "Final call to action",
  "total_char_count": 1234
}}

REQUIREMENTS:
- Maximum 10 tweets
- Each tweet max 280 characters
- First tweet must hook the reader
- Use line breaks for readability
- Include relevant hashtags
- End with engagement prompt

JSON OUTPUT:"""


# =============================================================================
# SYNTHESIS FUNCTION
# =============================================================================


async def synthesize_twitter_thread(
    session_id: str,
    max_items: int = 20,
    tone: str = "professional",
) -> TweetThread:
    """Synthesize a Twitter thread from session context.

    Args:
        session_id: Session ID to read context from
        max_items: Maximum items to include from store
        tone: Tone for the thread (professional, casual, educational)

    Returns:
        TweetThread with synthesized content

    Raises:
        ValueError: If no content found or synthesis fails
    """
    logger.info(f"Synthesizing Twitter thread for session: {session_id}")

    # Get store and read context (lazy imports)
    store = await _get_store()
    context = await _get_context_for_synthesis(store, session_id, max_items)

    if not context:
        raise ValueError(f"No content found for session {session_id}")

    # Get model (lazy import)
    model = _get_model("smart")

    # Build prompt
    prompt = TWITTER_SYNTHESIS_PROMPT.format(
        context=context,
        tone=tone,
    )

    # Invoke model
    response = await model.ainvoke(prompt)
    content = response.content

    # Parse JSON
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        raise ValueError("No JSON found in synthesis response")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse synthesis JSON: {e}")

    # Create TweetThread
    thread = TweetThread(
        tweets=[Tweet(**t) for t in data.get("tweets", [])],
        hook_strength=data.get("hook_strength", "okay"),
        call_to_action=data.get("call_to_action", ""),
        total_char_count=sum(len(t.content) for t in data.get("tweets", [])),
    )

    logger.info(
        f"Synthesized Twitter thread: {len(thread.tweets)} tweets, "
        f"{thread.total_char_count} chars"
    )

    return thread


async def _get_context_for_synthesis(
    store: Any,
    session_id: str,
    max_items: int,
) -> str:
    """Get context from store for synthesis.

    Args:
        store: LangGraph store
        session_id: Session ID
        max_items: Max items to include

    Returns:
        Concatenated context string
    """
    # Try different namespaces
    namespaces = [
        (session_id, "memories"),
        (session_id, "input"),
        (session_id, "summary"),
    ]

    all_content = []

    for namespace in namespaces:
        try:
            results = await store.asearch(namespace, query="", limit=max_items)
            for result in results:
                content = result.value.get("content", "")
                if content:
                    all_content.append(f"[{result.key}]\n{content}\n")
        except Exception as e:
            logger.debug(f"No content in {namespace}: {e}")

    if not all_content:
        return ""

    # Limit total context length
    context = "\n---\n".join(all_content)
    max_chars = 15000
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[...content truncated...]"

    return context


def format_thread_output(thread: TweetThread) -> str:
    """Format thread for display.

    Args:
        thread: TweetThread to format

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("TWITTER THREAD")
    lines.append(f"Hook Strength: {thread.hook_strength}")
    lines.append(f"Total Tweets: {len(thread.tweets)}")
    lines.append(f"Total Characters: {thread.total_char_count}")
    lines.append("=" * 50)
    lines.append("")

    for i, tweet in enumerate(thread.tweets, 1):
        lines.append(f"TWEET {i}/{len(thread.tweets)}")
        lines.append("-" * 30)
        lines.append(tweet.content)
        if tweet.hashtags:
            lines.append(" ".join(f"#{tag}" for tag in tweet.hashtags))
        if tweet.media_hint:
            lines.append(f"[Media: {tweet.media_hint}]")
        lines.append("")

    lines.append("=" * 50)
    lines.append("CALL TO ACTION:")
    lines.append(thread.call_to_action)
    lines.append("=" * 50)

    return "\n".join(lines)
