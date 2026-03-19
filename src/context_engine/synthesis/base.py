"""Base synthesizer with shared functionality.

Eliminates code duplication across Twitter, Podcast, and Report synthesis modules.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger


def _get_model(tier: str):
    """Lazy import for model."""
    from context_engine.config import get_model

    return get_model(tier)


async def _get_store():
    """Lazy import for store."""
    from context_engine.store.setup import get_store

    return await get_store()


async def get_context_for_synthesis(
    session_id: str,
    max_items: int = 20,
    max_chars: int = 15000,
) -> str:
    """Get context from store for synthesis.

    Shared across all synthesis modules.

    Args:
        session_id: Session ID
        max_items: Max items per namespace
        max_chars: Maximum total characters

    Returns:
        Concatenated context string
    """
    store = await _get_store()

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
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[...content truncated...]"

    return context


def parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response.

    Shared across all synthesis modules.

    Args:
        content: LLM response content

    Returns:
        Parsed JSON dict

    Raises:
        ValueError: If no valid JSON found
    """
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        raise ValueError("No JSON found in synthesis response")

    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse synthesis JSON: {e}")


class BaseSynthesizer(ABC):
    """Base class for all synthesizers.

    Provides shared functionality for:
    - Context retrieval
    - JSON parsing
    - Model access
    """

    # Override in subclasses
    context_max_chars: int = 15000
    context_max_items: int = 20

    @property
    @abstractmethod
    def name(self) -> str:
        """Synthesizer name."""
        pass

    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get the prompt template for this synthesizer.

        Returns:
            Prompt template string with {context} placeholder
        """
        pass

    @abstractmethod
    def parse_result(self, data: dict) -> Any:
        """Parse the synthesis result into the appropriate model.

        Args:
            data: Parsed JSON dict

        Returns:
            Synthesized result (TweetThread, PodcastScript, Report, etc.)
        """
        pass

    async def synthesize(self, session_id: str, **kwargs: Any) -> Any:
        """Run synthesis for a session.

        Args:
            session_id: Session ID to synthesize
            **kwargs: Additional arguments for prompt rendering

        Returns:
            Synthesized result

        Raises:
            ValueError: If no content found or synthesis fails
        """
        logger.info(f"Synthesizing {self.name} for session: {session_id}")

        # Get context
        context = await get_context_for_synthesis(
            session_id=session_id,
            max_items=self.context_max_items,
            max_chars=self.context_max_chars,
        )

        if not context:
            raise ValueError(f"No content found for session {session_id}")

        # Build prompt
        prompt = self.get_prompt_template().format(context=context, **kwargs)

        # Get model and invoke
        model = _get_model("smart")
        response = await model.ainvoke(prompt)

        # Parse response
        data = parse_json_response(response.content)

        return self.parse_result(data)
