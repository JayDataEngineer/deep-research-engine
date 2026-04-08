"""LangFuse client for prompt management.

LangFuse is the single source of truth for all prompts.

Usage:
    from deep_research_engine.prompts import get_prompt
    prompt = await get_prompt("research-planner")
    compiled = prompt.compile(strategy="complexity_based", min_workers=1)
"""

import os
from typing import Any, Union

from langfuse import Langfuse
from langfuse.model import ChatPromptClient, TextPromptClient

from ..utils.logging import get_logger

logger = get_logger("prompts.client")


PromptClient = Union[TextPromptClient, ChatPromptClient, dict[str, Any]]


class LangFuseError(Exception):
    """Base exception for LangFuse errors."""

    pass


def get_langfuse_client() -> Langfuse:
    """Get LangFuse client instance.

    Raises:
        LangFuseError: If credentials not configured
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://langfuse.local")

    if not public_key or not secret_key:
        raise LangFuseError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY required.\n"
            "Get keys from: http://langfuse.local → Settings → API Keys"
        )

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )


class PromptWrapper:
    """Wrapper for LangFuse or Local prompts."""

    def __init__(self, prompt: PromptClient, name: str | None = None):
        if isinstance(prompt, dict):
            # Local prompt from templates.py
            self._template = prompt["prompt"]
            self._config = prompt.get("config", {})
            self._name = name or "local-prompt"
            self._version = "local"
        else:
            # LangFuse prompt
            self._prompt = prompt
            self._template = prompt.prompt
            self._config = prompt.config or {}
            self._name = prompt.name
            self._version = prompt.version

    @property
    def prompt(self) -> str:
        return self._template

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str | int:
        return self._version

    def compile(self, **kwargs: Any) -> str:
        """Compile prompt with variables using Jinja2."""
        from jinja2 import StrictUndefined, Template, UndefinedError

        try:
            template = Template(self._template, undefined=StrictUndefined)
            return template.render(**kwargs)
        except UndefinedError as e:
            required = self.get_required_variables()
            provided = list(kwargs.keys())
            missing = set(required) - set(provided)
            raise LangFuseError(
                f"Missing variables for prompt '{self._name}': {sorted(missing)}. "
                f"Provided: {sorted(provided)}. Required: {sorted(required)}"
            ) from e

    def get_langchain_prompt(self) -> str:
        """Convert to LangChain format ({{ var }} → { var })."""
        import re

        return re.sub(r"\{\{\s*(\w+)\s*\}\}", r"{\1}", self._template)

    def get_required_variables(self) -> list[str]:
        """Extract variable names from prompt template."""
        import re

        matches = re.findall(r"\{\{\s*(\w+)\s*\}\}", self._template)
        return list(set(matches))

    def get_model_config(self) -> dict[str, Any]:
        """Get model config from prompt config."""
        return {
            "model": self._config.get("model", "qwen-35-27"),
            "temperature": self._config.get("temperature", 0.7),
            "enable_thinking": self._config.get("enable_thinking", False),
        }


class PromptManager:
    """Manages prompts from LangFuse or Local fallback."""

    def __init__(self, client: Langfuse | None = None):
        self._client = client
        self._cache: dict[str, PromptWrapper] = {}
        # Auto-detect source: if LangFuse keys aren't set, use local
        explicit_source = os.getenv("PROMPT_SOURCE", "").lower()
        if explicit_source:
            self._source = explicit_source
        elif os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            self._source = "langfuse"
        else:
            self._source = "local"
            logger.info("LangFuse keys not set — using local prompt templates")

    @property
    def client(self) -> Langfuse:
        if self._client is None:
            self._client = get_langfuse_client()
        return self._client

    async def get_prompt(
        self,
        name: str,
        version: str | None = None,
        label: str = "production",
    ) -> PromptWrapper:
        """Get prompt from LangFuse or Local templates."""
        cache_key = f"{name}:{version or 'latest'}:{label}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._source == "local":
            logger.debug(f"Loading local prompt: {name}")
            return self._get_local_prompt(name)

        try:
            prompt = self.client.get_prompt(name=name, version=version, label=label)
            wrapper = PromptWrapper(prompt)
            self._cache[cache_key] = wrapper
            return wrapper
        except Exception as e:
            if self._source == "langfuse":
                logger.warning(f"LangFuse prompt '{name}' failed, falling back to local: {e}")
            return self._get_local_prompt(name)

    def _get_local_prompt(self, name: str) -> PromptWrapper:
        """Helper to get a local prompt wrapper."""
        from .templates import PROMPTS

        if name not in PROMPTS:
            raise LangFuseError(f"Prompt '{name}' not found in local templates or LangFuse.")

        wrapper = PromptWrapper(PROMPTS[name], name=name)
        self._cache[name] = wrapper
        return wrapper


_prompt_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    """Get global prompt manager."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


async def get_prompt(
    name: str,
    version: str | None = None,
    label: str = "production",
) -> PromptWrapper:
    """Get prompt from LangFuse.

    Example:
        prompt = await get_prompt("research-planner")
        compiled = prompt.compile(
            strategy="complexity_based",
            min_workers=1,
            max_workers=3,
            query="AI trends"
        )
    """
    return await get_prompt_manager().get_prompt(name, version=version, label=label)


def clear_prompt_cache():
    """Clear the prompt cache."""
    get_prompt_manager()._cache.clear()
