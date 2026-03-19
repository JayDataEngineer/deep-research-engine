"""Title Generator Library.

Fast, non-blocking title generation using the LLM factory.
Runs in parallel with chat response - triggered immediately on user message.
"""

from .title_generator import (
    TitleGenerator,
    get_title_generator,
    MAX_TITLE_LENGTH,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    DEFAULT_TITLE_MODEL,
)

__all__ = [
    'TitleGenerator',
    'get_title_generator',
    'MAX_TITLE_LENGTH',
    'MAX_NEW_TOKENS',
    'TEMPERATURE',
    'DEFAULT_TITLE_MODEL',
]
