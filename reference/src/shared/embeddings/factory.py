"""Embedding Factory - Centralized embedding model instantiation.

All apps use this factory to get embedding instances.

Supports multiple embedding providers:
- OpenAI: text-embedding-3-small, text-embedding-3-large
- Local: Local embedding servers (llama.cpp, Ollama, etc.)
- ZAI: ZAI embedding models
- HuggingFace: Local sentence-transformer models
- Other: Any OpenAI-compatible API

Pattern:
    from src.shared.embeddings.factory import get_embeddings

    embeddings = get_embeddings()  # Uses default from config
    embeddings = get_embeddings(provider="openai", model="text-embedding-3-small")
    embeddings = get_embeddings(provider="huggingface", model="sentence-transformers/all-MiniLM-L6-v2")

COMPLIANCE: All returns are BaseEmbeddings interface.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List, TYPE_CHECKING

# Get the backend directory (parent of src/)
backend_root = Path(__file__).parent.parent.parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

logger = logging.getLogger(__name__)


# =============================================================================
# PROVIDER DETECTION
# =============================================================================

def _detect_provider(base_url: str, explicit_provider: Optional[str] = None) -> str:
    """Detect embedding provider from base URL or explicit provider.

    Args:
        base_url: The base URL of the embedding API (for auto-detection)
        explicit_provider: Explicitly specified provider (takes precedence)

    Returns:
        Provider name: "openai", "zai", "local", "huggingface"
    """
    # If explicitly provided, validate and return it
    if explicit_provider:
        valid_providers = {"openai", "zai", "local", "huggingface"}
        if explicit_provider not in valid_providers:
            raise ValueError(f"Invalid provider: {explicit_provider}. Must be one of {valid_providers}")
        return explicit_provider

    # Detect local server (for OpenAI-compatible APIs)
    is_local = any(
        indicator in base_url.lower()
        for indicator in ["localhost", "127.0.0.1", "0.0.0.0", "::1", "192.168.", "10.", "/v1/"]
    )

    if is_local:
        return "local"

    # Detect ZAI
    if "z.ai" in base_url:
        return "zai"

    # Default to OpenAI-compatible
    return "openai"


# =============================================================================
# RATE-LIMITED EMBEDDINGS WRAPPER
# =============================================================================

class RateLimitedEmbeddings:
    """Generic wrapper that applies rate limiting to any embeddings instance.

    This wrapper is provider-agnostic and works with any LangChain embeddings
    class that implements embed_documents() and embed_query() methods.

    The rate limiter is applied synchronously - calls will block until
    the rate limiter allows the request to proceed.

    Args:
        embeddings: The underlying embeddings instance to wrap
        rate_limiter: Rate limiter instance (e.g., InMemoryRateLimiter)

    Example:
        >>> from langchain_openai import OpenAIEmbeddings
        >>> from langchain_core.rate_limiters import InMemoryRateLimiter
        >>> base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        >>> rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)
        >>> embeddings = RateLimitedEmbeddings(base_embeddings, rate_limiter)
    """

    def __init__(self, embeddings: Any, rate_limiter: Any):
        """Initialize the rate-limited embeddings wrapper.

        Args:
            embeddings: The underlying embeddings instance
            rate_limiter: Rate limiter with acquire() method
        """
        self._embeddings = embeddings
        self._rate_limiter = rate_limiter

    def _acquire(self) -> None:
        """Acquire permission from the rate limiter.

        This blocks until the rate limiter allows the request.
        """
        if self._rate_limiter is not None:
            try:
                # InMemoryRateLimiter uses acquire() method
                if hasattr(self._rate_limiter, 'acquire'):
                    self._rate_limiter.acquire()
            except Exception as e:
                logger.warning(f"[RateLimitedEmbeddings] Rate limiter acquire failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with rate limiting.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        self._acquire()
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text with rate limiting.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        self._acquire()
        return self._embeddings.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of documents with rate limiting.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        # For async, we run the rate limiter acquire in a thread pool
        if self._rate_limiter is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._acquire)

        # Check if underlying embeddings supports async
        if hasattr(self._embeddings, 'aembed_documents'):
            return await self._embeddings.aembed_documents(texts)
        else:
            # Fallback to sync
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._embeddings.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query text with rate limiting.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        # For async, we run the rate limiter acquire in a thread pool
        if self._rate_limiter is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._acquire)

        # Check if underlying embeddings supports async
        if hasattr(self._embeddings, 'aembed_query'):
            return await self._embeddings.aembed_query(text)
        else:
            # Fallback to sync
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._embeddings.embed_query, text)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying embeddings.

        This ensures that wrapper behaves like the original embeddings
        for any attributes or methods not explicitly overridden.
        """
        return getattr(self._embeddings, name)

    def __repr__(self) -> str:
        """String representation showing the wrapper and underlying embeddings."""
        return f"RateLimitedEmbeddings({self._embeddings!r})"


# =============================================================================
# EMBEDDINGS CACHE (per-model singleton)
# =============================================================================

_embeddings_cache: dict[str, Any] = {}


def get_embeddings(
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    rate_limiter: Optional[Any] = None,
    **kwargs
) -> Any:
    """Get embedding instance for model name.

    Centralized embeddings creation. All apps use this.

    Args:
        model: Model name (uses default from config if None)
        provider: Provider name ("openai", "zai", "local"). Auto-detected from base_url if None.
        base_url: API base URL (uses default from config if None)
        api_key: API key (uses default from config if None)
        rate_limiter: Optional rate limiter for API calls
        **kwargs: Additional args passed to embeddings implementation

    Returns:
        Embeddings instance (langchain_core.embeddings.Embeddings)

    Example:
        >>> # With defaults
        >>> embeddings = get_embeddings()
        >>> # With specific model
        >>> embeddings = get_embeddings(model="text-embedding-3-large")
        >>> # With local provider
        >>> embeddings = get_embeddings(provider="local", base_url="http://localhost:8080")
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.rate_limiters import InMemoryRateLimiter

    # Get settings for defaults
    try:
        from src.config import settings
        config_settings = settings
    except ImportError:
        class _DummySettings:
            def get_backend_for_model(self, model):
                return {"base_url": "https://api.openai.com/v1", "api_key": ""}
            llm_model = "text-embedding-3-small"
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            embedding_provider = "huggingface"
        config_settings = _DummySettings()

    # Use provided model or default from Settings.embedding_model
    if model is None:
        model = getattr(config_settings, 'embedding_model', "sentence-transformers/all-MiniLM-L6-v2")

    # Detect provider if not specified
    if provider is None:
        provider = getattr(config_settings, 'embedding_provider', "huggingface")

    # VALIDATE: Ensure we're not trying to use an LLM model for embeddings
    # Common LLM models that should NOT be used for embeddings:
    _LLM_MODELS = {
        "glm-4.7", "glm-4", "glm-4-flash", "glm-4-flashx",
        "qwen-omni-turbo", "qwen-omni", "qwen-plus", "qwen-max", "qwen",
        "gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini",
        "claude-3", "claude-3.5-sonnet", "claude-3-opus",
        "llama", "llama2", "llama3",
        "mistral", "mixtral",
    }
    _model_lower = model.lower()
    if any(llm in _model_lower for llm in _LLM_MODELS):
        raise ValueError(
            f"Invalid embedding model: '{model}'. This appears to be an LLM model, not an embedding model.\n"
            f"LLM models like glm-4.7, qwen-omni-turbo, gpt-4, etc. CANNOT be used for embeddings.\n\n"
            f"Please set EMBEDDING_MODEL environment variable to a valid embedding model like:\n"
            f"  - sentence-transformers/all-MiniLM-L6-v2 (local HuggingFace)\n"
            f"  - text-embedding-3-small (OpenAI)\n"
            f"  - text-embedding-3-large (OpenAI)\n\n"
            f"Current settings:\n"
            f"  config_settings.embedding_model = {getattr(config_settings, 'embedding_model', 'NOT SET')}\n"
            f"  config_settings.embedding_provider = {getattr(config_settings, 'embedding_provider', 'NOT SET')}\n"
        )

    # Get base_url (simplified for LiteLLM - no pattern routing needed)
    if base_url is None:
        # Default to HuggingFace for local embeddings
        # For OpenAI embeddings, base_url should be explicitly provided
        if "openai" in model.lower() or "text-embedding" in model.lower():
            base_url = "https://api.openai.com/v1"
        else:
            # Local HuggingFace - no base_url needed
            base_url = ""

    # Detect provider if not specified (only needed for non-HuggingFace providers)
    if provider is None:
        provider = _detect_provider(base_url, provider)

    # Get API key from environment or settings
    if api_key is None:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
        elif provider == "huggingface":
            api_key = ""  # HuggingFace doesn't need API key for local models
        else:
            api_key = os.getenv("LLM_API_KEY", "")

    # Create cache key
    cache_key = f"{provider}:{base_url}:{model}"

    # Check cache first
    if cache_key in _embeddings_cache:
        logger.debug(f"[EmbeddingFactory] Using cached embeddings: {cache_key}")
        return _embeddings_cache[cache_key]

    # Create embeddings based on provider
    logger.info(f"[EmbeddingFactory] Creating embeddings: provider={provider}, model={model}, base_url={base_url}")

    if provider == "huggingface":
        # HuggingFace embeddings - local sentence-transformer models
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            # Default model for HuggingFace
            if model is None or "text-embedding" in model:
                model = kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

            # Build model kwargs (device selection, trust_remote_code, etc.)
            # Default to CPU to avoid CUDA out of memory errors
            # User can override by passing model_kwargs={"device": "cuda"} explicitly
            model_kwargs = kwargs.get("model_kwargs", {"device": "cpu"}).copy()
            model_kwargs["trust_remote_code"] = True

            # Build encode kwargs (normalization, batch size, etc.)
            encode_kwargs = kwargs.get("encode_kwargs", {"normalize_embeddings": True})

            embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                show_progress=kwargs.get("show_progress", False)
            )

        except ImportError:
            raise ImportError(
                "[EmbeddingFactory] HuggingFace embeddings requested but langchain-huggingface not installed. "
                "Install with: uv pip install langchain-huggingface"
            )

    elif provider == "local":
        # For local servers, we still use OpenAIEmbeddings but with local base_url
        # This works with llama.cpp embedding server and similar
        embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_base=base_url,
            openai_api_key="dummy",  # Local servers often don't need real API key
            **kwargs
        )

    elif provider == "zai":
        # ZAI provider (if they have embeddings API)
        # For now, use OpenAI-compatible interface
        embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_base=base_url,
            openai_api_key=api_key,
            **kwargs
        )

    else:  # openai or compatible
        embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_base=base_url,
            openai_api_key=api_key,
            **kwargs
        )

    # Apply rate limiter if provided using generic wrapper
    if rate_limiter is not None:
        logger.info(f"[EmbeddingFactory] Applying rate limiter to embeddings: {type(rate_limiter).__name__}")
        embeddings = RateLimitedEmbeddings(embeddings, rate_limiter)

    # Cache and return
    _embeddings_cache[cache_key] = embeddings
    return embeddings


def clear_embeddings_cache():
    """Clear all cached embeddings.

    Useful for testing or when embeddings configuration changes.

    Example:
        >>> clear_embeddings_cache()
        >>> embeddings = get_embeddings()  # Fresh instance
    """
    global _embeddings_cache
    _embeddings_cache.clear()
    logger.info("[EmbeddingFactory] Cleared embeddings cache")


def get_embeddings_from_config(config: Dict[str, Any]) -> Any:
    """Get embedding instance from configuration dict.

    This function accepts the standard config format used by RAG and other tools,
    mapping it to the factory parameters.

    Config format:
        {
            "tool_settings": {
                "embeddings": {
                    "provider": "huggingface" | "openai" | "local" | "zai",
                    "model_name": "model-name",  # optional
                    "model_kwargs": {...},       # HuggingFace-specific
                    "encode_kwargs": {...},      # HuggingFace-specific
                    "api_key": "...",            # OpenAI-specific
                }
            }
        }

    Args:
        config: Configuration dict with tool_settings.embeddings

    Returns:
        Embeddings instance from factory

    Example:
        >>> config = {
        ...     "tool_settings": {
        ...         "embeddings": {
        ...             "provider": "huggingface",
        ...             "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        ...         }
        ...     }
        ... }
        >>> embeddings = get_embeddings_from_config(config)
    """
    embedding_config = config.get("tool_settings", {}).get("embeddings", {})
    provider = embedding_config.get("provider", "huggingface")
    model_name = embedding_config.get("model_name")

    # Map config format to factory parameters
    factory_kwargs = {}

    # Handle HuggingFace-specific kwargs
    if provider == "huggingface":
        if "model_kwargs" in embedding_config:
            factory_kwargs["model_kwargs"] = embedding_config["model_kwargs"]
        if "encode_kwargs" in embedding_config:
            factory_kwargs["encode_kwargs"] = embedding_config["encode_kwargs"]
        factory_kwargs["show_progress"] = embedding_config.get("show_progress", False)

    # Handle OpenAI API key from config
    elif provider == "openai":
        api_key = embedding_config.get("api_key", os.getenv("OPENAI_API_KEY"))
        if api_key:
            factory_kwargs["api_key"] = api_key

    # Handle base_url if specified
    if "base_url" in embedding_config:
        factory_kwargs["base_url"] = embedding_config["base_url"]

    return get_embeddings(
        model=model_name,
        provider=provider,
        **factory_kwargs
    )


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def get_embedding_model(model: str = "text-embedding-3-small") -> Any:
    """Legacy function for backward compatibility.

    Deprecated: Use get_embeddings() instead.

    Args:
        model: Model name

    Returns:
        Embeddings instance
    """
    logger.warning("[EmbeddingFactory] get_embedding_model() is deprecated, use get_embeddings()")
    return get_embeddings(model=model)
