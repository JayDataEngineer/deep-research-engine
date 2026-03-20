"""NLP utilities for entity extraction and text processing."""

import os
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from loguru import logger

# LiteLLM
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "fake-key")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")


class ExtractionResult(BaseModel):
    """Result of entity extraction."""
    people: List[str] = Field(
        default_factory=list,
        description="Names of people mentioned or detected"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Topics, subjects, or themes identified"
    )
    locations: List[str] = Field(
        default_factory=list,
        description="Locations mentioned"
    )
    dates: List[str] = Field(
        default_factory=list,
        description="Dates or time references"
    )


EXTRACTION_PROMPT = """Extract entities from the following content.

Content:
{content}

Extract:
1. People: Names of individuals mentioned (first names, full names, nicknames)
2. Topics: Main subjects, themes, or topics discussed
3. Locations: Places mentioned (cities, countries, venues)
4. Dates: Time references (dates, days, months, years)

Be thorough but only include clearly identifiable entities."""


async def extract_entities(content: str) -> Dict[str, Any]:
    """Extract people, topics, locations, and dates from content.

    Args:
        content: Text content to analyze (message, transcription, or description)

    Returns:
        Dictionary with people, topics, locations, and dates lists
    """
    if not content or len(content.strip()) < 10:
        return {
            "people": [],
            "topics": [],
            "locations": [],
            "dates": [],
        }

    logger.debug(f"Extracting entities from {len(content)} chars")

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            temperature=0.1,
        )

        structured_llm = llm.with_structured_output(ExtractionResult)

        prompt = EXTRACTION_PROMPT.format(content=content)
        result = await structured_llm.ainvoke(prompt)

        logger.info(
            f"Extraction complete: {len(result.people)} people, {len(result.topics)} topics"
        )

        return result.model_dump()

    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return {
            "people": [],
            "topics": [],
            "locations": [],
            "dates": [],
            "error": str(e),
        }
