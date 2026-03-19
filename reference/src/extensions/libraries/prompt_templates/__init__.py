"""Shared Prompt Template Library

Provides Jinja2-based prompt templating for all chat modes.
"""

from .template_engine import (
    PromptTemplateEngine,
    PromptContext,
    TemplateRegistry,
    get_template_engine,
    get_template_registry,
    render_prompt,
    render_character,
)

__all__ = [
    "PromptTemplateEngine",
    "PromptContext",
    "TemplateRegistry",
    "get_template_engine",
    "get_template_registry",
    "render_prompt",
    "render_character",
]
