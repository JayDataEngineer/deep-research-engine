"""NLP utilities for entity extraction and text processing."""

import asyncio
import json
import re
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from ..shared.config import DEFAULT_MODEL
from ..shared.infra_client import InfraClient

EXTRACTION_PROMPT = """Extract all named entities from the following text.

Return a JSON object with exactly these fields:
- "people": list of person names mentioned
- "topics": list of topics, subjects, or themes
- "locations": list of locations mentioned
- "dates": list of dates or time references

Text:
{content}

Respond with ONLY the JSON object, no other text."""


class ExtractionResult(BaseModel):
    people: list[str] = Field(default_factory=list, description="Names of people mentioned or detected")
    topics: list[str] = Field(
        default_factory=list,
        description="Topics, subjects, or themes identified",
    )
    locations: list[str] = Field(
        default_factory=list,
        description="Locations mentioned",
    )
    dates: list[str] = Field(
        default_factory=list,
        description="Dates or time references",
    )


# InfraClient singleton
_infra: InfraClient | None = None


def _get_infra() -> InfraClient:
    global _infra
    if _infra is None:
        _infra = InfraClient()
    return _infra


def _parse_json_response(text: str) -> dict[str, Any]:
    """Try to extract JSON from LLM response text."""
    # Direct parse
    try:
        result = json.loads(text)
        validated = ExtractionResult(**result)
        return validated.model_dump()
    except (json.JSONDecodeError, Exception):
        pass

    # Try extracting from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            validated = ExtractionResult(**result)
            return validated.model_dump()
        except (json.JSONDecodeError, Exception):
            pass

    # Try finding first { ... } block
    brace_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            validated = ExtractionResult(**result)
            return validated.model_dump()
        except (json.JSONDecodeError, Exception):
            pass

    logger.warning("Could not parse entity extraction response as JSON")
    return {"people": [], "topics": [], "locations": [], "dates": []}


async def extract_entities(content: str, model: str | None = None) -> dict[str, Any]:
    """Extract named entities from text content using LLM via InfraClient.

    Args:
        content: Text to extract entities from.
        model: Model to use. Defaults to DEFAULT_MODEL.
    """
    if not content or len(content.strip()) < 10:
        return {
            "people": [],
            "topics": [],
            "locations": [],
            "dates": [],
        }

    model_name = model or DEFAULT_MODEL

    try:
        logger.debug(f"Extracting entities from {len(content)} chars using {model_name}")

        prompt = EXTRACTION_PROMPT.format(content=content[:8000])
        response = await asyncio.to_thread(
            _get_infra().chat,
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.1,
            max_tokens=10000,
        )

        # Extract text from response
        text = response["choices"][0]["message"]["content"]
        return _parse_json_response(text)

    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return {
            "people": [],
            "topics": [],
            "locations": [],
            "dates": [],
            "error": str(e),
        }
