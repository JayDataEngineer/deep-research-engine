"""Configuration Validation Schema.

Provides Pydantic models for validating YAML configs at load time.
This enables early error detection and better debugging.

Usage:
    from deep_research.config_validation import validate_config

    config = yaml.safe_load(file)
    validated = validate_config(config)
"""

from typing import Dict, List, Any, Optional, Literal, Union, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import os
import logging

# Get logger for this module
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL AVAILABILITY CHECKING
# =============================================================================

class ModelAvailabilityChecker:
    """Checks if required models and providers are available.

    Dynamically detects available providers based on:
    - Environment variables (API keys)
    - Running services (llama.cpp, ollama)
    - Config files
    """

    # Known models that should be available per provider
    KNOWN_MODELS_BY_PROVIDER = {
        "zai": {
            # Zhipu AI / GLM models
            "glm-4.7", "glm-4-plus", "glm-4-air", "glm-4", "glm-3-turbo",
            "glm-4-flash", "glm-4.7-flash", "GLM-4.7-Flash-UD-Q4_K_XL",
        },
        "openai": {
            # OpenAI models
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
            "gpt-4-turbo-preview", "gpt-4-vision-preview",
        },
        "anthropic": {
            # Anthropic Claude models
            "claude-opus-4-20250514", "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229",
        },
        "google": {
            # Google models
            "gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash",
        },
        "deepseek": {
            # DeepSeek models
            "deepseek-chat", "deepseek-coder", "deepseek-reasoner",
        },
        "mistral": {
            # Mistral models
            "mistral-large", "mistral-medium", "mistral-small",
            "mistral-7b", "mixtral-8x7b", "mixtral-8x22b",
        },
        "llamacpp": {
            # Local llama.cpp models (GGUF files)
            # These are loaded dynamically, so we accept any model name
            # but check if llama.cpp server is running
        },
        "ollama": {
            # Ollama models - loaded dynamically
        },
    }

    # All known models for backward compatibility
    KNOWN_MODELS = set()
    for models in KNOWN_MODELS_BY_PROVIDER.values():
        KNOWN_MODELS.update(models)

    @classmethod
    def get_available_providers(cls) -> dict:
        """Get dict of available providers and their status.

        Checks multiple sources in order of priority:
        1. API key database (SQLite/PostgreSQL)
        2. Environment variables
        3. Running services (llama.cpp, ollama)
        4. Installed libraries (HuggingFace)

        Returns:
            Dict mapping provider name to (is_available, message)
        """
        available = {}

        # Try to get keys from API key database first
        db_keys = cls._get_keys_from_database()

        # Check ZAI (Zhipu AI) - via settings or database
        zai_key = db_keys.get("zai") or db_keys.get("glm")
        if zai_key:
            available["zai"] = (True, "API key found in database")
        else:
            try:
                from src.config import settings
                if hasattr(settings, 'llm_base_url') and hasattr(settings, 'llm_api_key'):
                    if settings.llm_base_url and settings.llm_api_key:
                        available["zai"] = (True, "API credentials configured")
                    else:
                        available["zai"] = (False, "No API credentials in settings")
                else:
                    available["zai"] = (False, "Settings not available")
            except Exception as e:
                available["zai"] = (False, f"Error checking: {e}")

        # Check OpenAI - database or env
        openai_key = db_keys.get("openai") or os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            available["openai"] = (True, "API key found" + (" in database" if db_keys.get("openai") else ""))
        else:
            available["openai"] = (False, "No OPENAI_API_KEY environment variable or database key")

        # Check Anthropic - database or env
        anthropic_key = db_keys.get("anthropic") or os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            available["anthropic"] = (True, "API key found" + (" in database" if db_keys.get("anthropic") else ""))
        else:
            available["anthropic"] = (False, "No ANTHROPIC_API_KEY environment variable or database key")

        # Check Google - database or env
        google_key = db_keys.get("google") or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        if google_key:
            available["google"] = (True, "API key found" + (" in database" if db_keys.get("google") else ""))
        else:
            available["google"] = (False, "No GOOGLE_API_KEY/GEMINI_API_KEY environment variable or database key")

        # Check Cohere - database only (not commonly in env)
        cohere_key = db_keys.get("cohere")
        if cohere_key:
            available["cohere"] = (True, "API key found in database")
        else:
            available["cohere"] = (False, "No cohere key in database")

        # Check OpenRouter - database only
        openrouter_key = db_keys.get("openrouter")
        if openrouter_key:
            available["openrouter"] = (True, "API key found in database")
        else:
            available["openrouter"] = (False, "No openrouter key in database")

        # Check llama.cpp server
        llamacpp_port = os.environ.get("LLAMACPP_PORT", "18734")
        try:
            import urllib.request
            import socket
            socket.setdefaulttimeout(2)  # 2 second timeout
            req = urllib.request.urlopen(f"http://127.0.0.1:{llamacpp_port}/health")
            if req.status == 200:
                available["llamacpp"] = (True, f"Server running on port {llamacpp_port}")
            else:
                available["llamacpp"] = (False, f"Server returned status {req.status}")
        except Exception as e:
            available["llamacpp"] = (False, f"Server not reachable: {str(e)[:50]}")

        # Check Ollama
        try:
            import urllib.request
            import socket
            socket.setdefaulttimeout(2)
            req = urllib.request.urlopen("http://127.0.0.1:11434/api/tags")
            if req.status == 200:
                available["ollama"] = (True, "Ollama running on port 11434")
            else:
                available["ollama"] = (False, f"Ollama returned status {req.status}")
        except Exception as e:
            available["ollama"] = (False, f"Ollama not reachable: {str(e)[:50]}")

        # Check HuggingFace (always available via transformers library)
        try:
            import transformers
            available["huggingface"] = (True, "Transformers library installed")
        except ImportError:
            available["huggingface"] = (False, "Transformers library not installed")

        return available

    @classmethod
    def _get_keys_from_database(cls) -> Dict[str, str]:
        """Get API keys from the database.

        Delegates to centralized provider_checks utility.

        Returns:
            Dict mapping key_type to "exists" for available keys
        """
        from src.services.provider_checks import _get_keys_from_database
        return _get_keys_from_database()

    @classmethod
    def get_available_models(cls) -> set:
        """Get set of models that are currently available based on environment.

        Returns:
            Set of available model names
        """
        available = set()

        # Check ZAI/LLM settings
        try:
            from src.config import settings
            base_url = getattr(settings, 'llm_base_url', '')
            api_key = getattr(settings, 'llm_api_key', '')

            if base_url and api_key:
                # If we have API credentials, assume models are available
                # In production, you might want to actually query the API
                available.update(cls.KNOWN_MODELS)
        except ImportError:
            pass

        # Check for local llama.cpp
        if os.path.exists(os.path.expanduser("~/.llama.cpp")) or \
           os.path.exists("/models") or \
           os.path.exists("./models"):
            available.add("llama.cpp")

        return available

    @classmethod
    def check_model(cls, model_name: str) -> Tuple[bool, str]:
        """Check if a specific model is available.

        Args:
            model_name: Name of the model to check

        Returns:
            Tuple of (is_available, message)
        """
        if not model_name:
            return False, "Model name is empty"

        # Check if it's a known model
        available = cls.get_available_models()

        # For now, if it's in KNOWN_MODELS, we consider it potentially available
        # In production, you'd query the actual API
        if model_name in cls.KNOWN_MODELS:
            from src.config import settings
            if hasattr(settings, 'llm_base_url') and hasattr(settings, 'llm_api_key'):
                if settings.llm_base_url and settings.llm_api_key:
                    return True, f"Model {model_name} is available"

        # Check for local models
        if "llama" in model_name.lower() or "local" in model_name.lower():
            if os.path.exists(os.path.expanduser("~/.llama.cpp")):
                return True, f"Local model {model_name} may be available"

        # Unknown model - warn but don't fail
        return True, f"Model {model_name} not in known list, assuming available"

    @classmethod
    def validate_config_models(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate all models specified in the config.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (all_available, warnings)
        """
        warnings = []
        all_available = True

        # Check main model
        main_model = config.get("model")
        if main_model:
            available, message = cls.check_model(main_model)
            if not available:
                all_available = False
                warnings.append(f"Main model '{main_model}': {message}")
            elif "assuming" in message:
                warnings.append(f"Main model '{main_model}': {message}")

        # Check model_configs
        model_configs = config.get("model_configs", {})
        for role, model_config in model_configs.items():
            if isinstance(model_config, dict):
                model_name = model_config.get("model_name")
                if model_name:
                    available, message = cls.check_model(model_name)
                    if not available:
                        all_available = False
                        warnings.append(f"{role} model '{model_name}': {message}")

        # Check specialist_configs models
        specialist_configs = config.get("specialist_configs", {})
        if isinstance(specialist_configs, dict):
            for specialist_name, specialist_config in specialist_configs.items():
                if isinstance(specialist_config, dict):
                    model_name = specialist_config.get("model")
                    if model_name:
                        available, message = cls.check_model(model_name)
                        if not available:
                            all_available = False
                            warnings.append(f"{specialist_name} model '{model_name}': {message}")

        return all_available, warnings


# =============================================================================
# ENUMS
# =============================================================================

class ProviderType(str, Enum):
    """Valid LLM provider types.

    NOTE: With LiteLLM migration, provider is largely deprecated.
    Model names now include the provider prefix (e.g., "zai-glm-4-5-air").
    The provider field is kept for backward compatibility but is optional.
    """
    LITELLM = "litellm"  # Default - routes through LiteLLM proxy
    ZAI = "zai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LLAMACPP = "llamacpp"  # Local llama.cpp server
    OLLAMA = "ollama"  # Ollama server
    OPENROUTER = "openrouter"  # OpenRouter API
    COHERE = "cohere"  # Cohere API
    GOOGLE = "google"  # Google AI / Gemini


# Valid provider names (for error messages)
VALID_PROVIDERS = [e.value for e in ProviderType]

# LiteLLM model name prefixes (for auto-detection)
LITELLM_PREFIXES = ["zai-", "qwen-", "gpt-", "claude-", "gemini-", "openai-", "anthropic-"]

# Common typos and their corrections
PROVIDER_TYPOS = {
    "lllamacpp": "llamacpp",  # 3 L's instead of 2
    "llamacp": "llamacpp",    # missing 'p'
    "llmacpp": "llamacpp",    # missing 'a'
    "lammacpp": "llamacpp",   # double 'm'
    "zaii": "zai",
    "openia": "openai",
    "open_ai": "openai",
    "antrhopic": "anthropic",
    "hugging_face": "huggingface",
    "hugginface": "huggingface",
    "olama": "ollama",        # missing 'l'
    "ollamma": "ollama",      # double 'm'
    "open_router": "openrouter",
    "openroutr": "openrouter",
    "cohre": "cohere",
    "googel": "google",
    "goggle": "google",
}


class ExecutionMode(str, Enum):
    """Valid execution modes."""
    GENERIC = "generic"
    HYBRID = "hybrid"
    DYNAMIC_GRAPH = "dynamic_graph"


class WorkerSpecializationMode(str, Enum):
    """Valid worker specialization modes."""
    GENERIC = "generic"
    HYBRID = "hybrid"
    DYNAMIC = "dynamic"
    FIXED = "fixed"


# =============================================================================
# MODEL CONFIGS
# =============================================================================

class ModelConfig(BaseModel):
    """Model configuration for a specific role.

    Supports two patterns:
    1. Direct model specification: model_name + provider
    2. Slot reference: model_slot (resolves to models.main or models.tool)
    """
    provider: Optional[ProviderType] = Field(default=ProviderType.ZAI)
    model_name: Optional[str] = Field(None, min_length=1, max_length=200)
    model_slot: Optional[str] = Field(None, min_length=1, max_length=50)
    temperature: float = Field(ge=0.0, le=2.0, default=0.5)
    max_tokens: int = Field(gt=0, le=128000, default=2000)
    enable_thinking: Optional[bool] = Field(None, description="Enable thinking mode for models that support it")

    @field_validator("provider", mode="before")
    @classmethod
    def validate_provider(cls, v) -> Optional[ProviderType]:
        """Validate provider name.

        NOTE: With LiteLLM, provider is largely deprecated. Model names
        now include the provider prefix (e.g., "zai-glm-4-5-air").
        This validation is kept for backward compatibility but is lenient.
        """
        if v is None:
            return None  # Provider is optional with LiteLLM

        # If already a ProviderType enum, return it
        if isinstance(v, ProviderType):
            return v

        # Normalize to lowercase string
        provider_str = str(v).lower().strip()

        # Check if it's a valid provider
        if provider_str in VALID_PROVIDERS:
            return ProviderType(provider_str)

        # Check for common typos
        if provider_str in PROVIDER_TYPOS:
            correct = PROVIDER_TYPOS[provider_str]
            logger.warning(
                f"Provider typo '{v}' -> '{correct}'. "
                f"Consider using LiteLLM model names directly."
            )
            return ProviderType(correct)

        # Unknown provider - log warning but allow (LiteLLM may support it)
        logger.warning(
            f"Unknown provider '{v}'. LiteLLM may support this provider. "
            f"Known providers: {VALID_PROVIDERS}"
        )
        return None  # Return None to let LiteLLM handle routing

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within reasonable bounds."""
        if not (0.0 <= v <= 2.0):
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {v}")
        return v

    @model_validator(mode="after")
    def validate_model_reference(self) -> "ModelConfig":
        """Ensure either model_name or model_slot is provided."""
        if self.model_name is None and self.model_slot is None:
            raise ValueError("Either model_name or model_slot must be specified")
        return self


class ModelConfigs(BaseModel):
    """Container for all model configurations."""
    spawner: Optional[ModelConfig] = None
    researcher: Optional[ModelConfig] = None
    critic: Optional[ModelConfig] = None
    synthesis: Optional[ModelConfig] = None


# =============================================================================
# PROMPTS
# =============================================================================

class PromptConfig(BaseModel):
    """Individual prompt configuration - can be string or dict."""
    root: Optional[str] = None
    worker: Optional[str] = None
    boss: Optional[str] = None
    boss_dependencies: Optional[str] = None
    worker: Optional[str] = None
    critic: Optional[str] = None
    synthesis: Optional[str] = None
    synthesis_role: Optional[str] = None
    worker_role: Optional[str] = None
    output_type: Optional[str] = None
    synthesis_guidance: Optional[str] = None
    citation_instructions: Optional[str] = None
    output_instructions: Optional[str] = None
    evaluation_focus: Optional[str] = None
    critic_role: Optional[str] = None

    class Config:
        extra = "allow"  # Allow additional fields for extensibility


class PromptsConfig(BaseModel):
    """Agent prompt templates - supports both string and structured (dict) prompts."""
    worker: Optional[Union[str, dict]] = None
    boss: Optional[Union[str, dict]] = None
    boss_dependencies: Optional[Union[str, dict]] = None
    worker: Optional[Union[str, dict]] = None
    critic: Optional[Union[str, dict]] = None
    synthesis: Optional[Union[str, dict]] = None
    synthesis_role: Optional[str] = None
    synthesis_guidance: Optional[str] = None
    citation_instructions: Optional[str] = None
    output_instructions: Optional[str] = None
    evaluation_focus: Optional[str] = None
    critic_role: Optional[str] = None
    output_type: Optional[str] = None

    class Config:
        extra = "allow"  # Allow additional fields for extensibility


# =============================================================================
# RECURSION HANDLING CONFIG
# =============================================================================

class RecursionHandlingConfig(BaseModel):
    """Configuration for recursion limit handling in Boss node.

    Controls when the Boss node should start simplifying or forcing completion
    as it approaches the graph recursion limit.
    """
    low_steps_threshold: int = Field(
        ge=1, le=20, default=5,
        description="Warn when remaining_steps <= this value"
    )
    critical_steps_threshold: int = Field(
        ge=1, le=10, default=3,
        description="Force completion when remaining_steps <= this value"
    )
    force_completion_iterations: int = Field(
        ge=1, le=5, default=2,
        description="Min iterations before forcing completion in critical mode"
    )


# =============================================================================
# SYNTHESIS CONFIG
# =============================================================================

class SynthesisConfig(BaseModel):
    """Configuration for synthesis node.

    Controls validation and output requirements for the synthesis phase.
    """
    min_length: int = Field(
        ge=100, le=5000, default=500,
        description="Minimum character length for synthesis output"
    )
    placeholder_patterns: List[str] = Field(
        default_factory=lambda: [
            "i'll help you curate",
            "i'll search for",
            "i need to search for",
            "let me retrieve",
            "i'll help you find",
            "i need to find",
            "let me search",
            "before i can",
            "first before i",
        ],
        description="Patterns that indicate LLM didn't call tools"
    )


# =============================================================================
# CRITIC CONFIG
# =============================================================================

class CriticConfig(BaseModel):
    """Configuration for critic node.

    Controls evaluation criteria and issue detection.
    """
    synthesis_issue_keywords: List[str] = Field(
        default_factory=lambda: [
            "structure", "organization", "flow", "coherent", "unclear", "confusing",
            "repetitive", "contradictory", "inconsistent", "format", "presentation",
            "writing", "style", "tone", "grammar", "spelling", "readability"
        ],
        description="Keywords indicating synthesis issues (not research gaps)"
    )
    research_issue_keywords: List[str] = Field(
        default_factory=lambda: [
            "missing", "incomplete", "gap", "insufficient", "lack of",
            "not enough", "superficial", "shallow", "coverage", "depth"
        ],
        description="Keywords indicating research gaps (not synthesis issues)"
    )


# =============================================================================
# SPAWNER CONFIG
# =============================================================================

class SpawnerConfig(BaseModel):
    """Core spawner configuration."""
    execution_mode: ExecutionMode = Field(default=ExecutionMode.GENERIC)
    worker_specialization_mode: WorkerSpecializationMode = Field(
        default=WorkerSpecializationMode.GENERIC
    )
    max_researchers: int = Field(ge=1, le=10, default=3)
    max_iterations: int = Field(ge=1, le=10, default=1)
    quality_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    completeness_threshold: float = Field(ge=0.0, le=1.0, default=0.8)
    level_of_detail: int = Field(ge=1, le=10, default=5)
    enable_citations: bool = Field(default=True)
    enable_quick_path: bool = Field(default=True)

    # Nested configs for configurable thresholds
    recursion_handling: RecursionHandlingConfig = Field(default_factory=RecursionHandlingConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)

    # Domain configuration
    domain: str = Field(default="general research")
    boss_role: str = Field(default="Project Manager")
    worker_role: str = Field(default="researcher")
    operation_type: str = Field(default="research")
    available_skills: str = Field(default="web search, document analysis, data synthesis")

    # Tool settings container
    tools: Dict[str, Any] = Field(default_factory=dict)

    # Character card mappings for worker types
    character_card_mappings: Dict[str, str] = Field(default_factory=dict)


# =============================================================================
# TOOL SETTINGS
# =============================================================================

class EmbeddingsConfig(BaseModel):
    """Embedding model configuration."""
    provider: ProviderType = Field(default=ProviderType.HUGGINGFACE)
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    model_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"device": "cuda"})
    encode_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"normalize_embeddings": True})


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    enabled: bool = Field(default=False)
    collection: str = Field(default="main_kb")
    persist_directory: str = Field(default="")
    top_k: int = Field(ge=1, le=100, default=5)

    def __init__(self, **data):
        from src.constants import ROOT_DATA_DIR
        if not data.get("persist_directory"):
            data["persist_directory"] = str(ROOT_DATA_DIR / "vectorstores" / "chroma")
        super().__init__(**data)


class ToolSettings(BaseModel):
    """Tool configuration settings."""
    embeddings: Optional[EmbeddingsConfig] = None
    vector_store: Optional[VectorStoreConfig] = None


# =============================================================================
# MIDDLEWARE CONFIG
# =============================================================================

class FlashRankConfig(BaseModel):
    """FlashRank reranking configuration."""
    enabled: bool = Field(default=False)
    top_k: int = Field(ge=1, le=100, default=10)
    threshold_high: int = Field(ge=10, le=200, default=50)


class MiddlewareUseCaseConfig(BaseModel):
    """Middleware configuration for use cases."""
    flashrank: Optional[FlashRankConfig] = None


class MiddlewareConfig(BaseModel):
    """Middleware configuration container."""
    use_case: Optional[MiddlewareUseCaseConfig] = None


# =============================================================================
# METADATA
# =============================================================================

class MetadataConfig(BaseModel):
    """Configuration metadata."""
    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=50)
    description: Optional[str] = None


# =============================================================================
# MAIN CONFIG MODEL
# =============================================================================

class SpawnerConfigSchema(BaseModel):
    """Complete spawner configuration schema.

    This is the main validation model for all YAML configs.
    """

    # Required sections
    _metadata: Optional[MetadataConfig] = None
    model_configs: Optional[ModelConfigs] = None
    prompts: Optional[PromptsConfig] = None
    spawner_config: Optional[SpawnerConfig] = None
    tool_settings: Optional[ToolSettings] = None
    middleware: Optional[MiddlewareConfig] = None
    enabled_tools: Optional[List[str]] = None  # List of enabled tool names

    # Allow extra fields for extensibility
    class Config:
        extra = "allow"

    @model_validator(mode="after")
    def validate_dependencies(self) -> "SpawnerConfigSchema":
        """Validate configuration dependencies.

        Ensures that if certain features are enabled, required
        dependencies are also configured.
        """
        # Check enabled_tools list (NEW PATTERN)
        enabled_tools = self.enabled_tools or []

        # Check vector store dependencies (NEW PATTERN)
        rag_tools = {"vector_search", "add_documents", "list_collections", "pdf_deep_agent", "vector_store", "rag"}
        has_rag = any(tool in rag_tools for tool in enabled_tools)

        if has_rag:
            if not self.tool_settings or not self.tool_settings.vector_store:
                raise ValueError(
                    "vector_store tool_settings required when RAG tools are enabled in enabled_tools. "
                    "Add tool_settings.vector_store configuration or remove RAG tools from enabled_tools."
                )
            if not self.tool_settings.embeddings:
                raise ValueError(
                    "embeddings tool_settings required when RAG tools are enabled in enabled_tools. "
                    "Add tool_settings.embeddings configuration or remove RAG tools from enabled_tools."
                )

        # Check FlashRank dependencies
        if self.middleware and self.middleware.use_case and self.middleware.use_case.flashrank:
            if self.middleware.use_case.flashrank.enabled:
                if not has_rag:
                    raise ValueError(
                        "FlashRank requires vector store to be enabled in enabled_tools. "
                        "Add 'vector_search' or 'rag' to enabled_tools list."
                    )

        # Check character card file existence
        # NOTE: If card_path is a personality ID (uppercase, no path separators),
        # it refers to a personality in the extension's text_injections.yaml, not a file
        if self.spawner_config and self.spawner_config.character_card_mappings:
            from pathlib import Path
            from src.constants import ROOT_DATA_DIR

            for worker_id, card_path in self.spawner_config.character_card_mappings.items():
                # Skip file existence check if this looks like a personality ID
                # Personality IDs are typically uppercase and don't contain path separators
                if isinstance(card_path, str) and card_path.isupper() and '/' not in card_path and '\\' not in card_path:
                    # This is a personality ID reference, not a file path
                    continue

                # Check both relative paths and paths relative to ROOT_DATA_DIR
                paths_to_check = [
                    Path(card_path),
                    ROOT_DATA_DIR / card_path,
                    ROOT_DATA_DIR / "inputs" / "characters" / Path(card_path).name,
                ]

                file_exists = any(p.exists() for p in paths_to_check)

                if not file_exists:
                    raise ValueError(
                        f"Character card file not found for worker '{worker_id}': {card_path}\n"
                        f"Tried: {paths_to_check}\n"
                        f"Please ensure the character card file exists."
                    )

        return self


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config(config: Dict[str, Any]) -> SpawnerConfigSchema:
    """Validate a configuration dictionary against the schema.

    Args:
        config: Configuration dictionary loaded from YAML

    Returns:
        Validated SpawnerConfigSchema instance

    Raises:
        ValidationError: If config is invalid
    """
    # Handle _metadata prefix (YAML doesn't allow keys starting with _)
    metadata = config.pop("_metadata", None) or config.pop("metadata", None)
    if metadata:
        config["_metadata"] = metadata

    # Extract enabled_tools from various possible nested structures
    # The config can have enabled_tools in multiple places:
    # 1. Top-level: enabled_tools: [list]
    # 2. Nested in tools: tools.enabled_tools: [list]
    # 3. Nested in spawner_config: spawner_config.tools.enabled_tools: [list]
    # 4. Nested in worker_types: spawner_config.worker_types.*.tools.enabled_tools: [list]

    if "enabled_tools" not in config or not config["enabled_tools"]:
        enabled_tools_set = set()

        # Check top-level tools.enabled_tools
        tools = config.get("tools", {})
        if isinstance(tools, dict) and "enabled_tools" in tools:
            enabled_tools_set.update(tools["enabled_tools"])

        # Check spawner_config.tools.enabled_tools
        spawner_config = config.get("spawner_config", {})
        if isinstance(spawner_config, dict):
            spawner_tools = spawner_config.get("tools", {})
            if isinstance(spawner_tools, dict) and "enabled_tools" in spawner_tools:
                enabled_tools_set.update(spawner_tools["enabled_tools"])

            # Check spawner_config.worker_types.*.tools.enabled_tools
            worker_types = spawner_config.get("worker_types", {})
            if isinstance(worker_types, dict):
                for worker_name, worker_config in worker_types.items():
                    if isinstance(worker_config, dict):
                        worker_tools = worker_config.get("tools", {})
                        if isinstance(worker_tools, dict) and "enabled_tools" in worker_tools:
                            enabled_tools_set.update(worker_tools["enabled_tools"])

        if enabled_tools_set:
            config["enabled_tools"] = list(enabled_tools_set)

    return SpawnerConfigSchema(**config)


def validate_config_with_details(config: Dict[str, Any], check_models: bool = True) -> tuple[bool, List[str], List[str], SpawnerConfigSchema | None]:
    """Validate config and return detailed results.

    Performs comprehensive validation including:
    1. Schema validation (Pydantic)
    2. Prompt validation (crash-only)
    3. Model availability checking (optional)

    Args:
        config: Configuration dictionary loaded from YAML
        check_models: Whether to check model availability (default: True)

    Returns:
        Tuple of (is_valid, errors, warnings, validated_schema)
    """
    errors = []
    warnings = []

    # First, validate schema
    try:
        validated = validate_config(config.copy())
    except Exception as e:
        # Parse validation errors
        error_str = str(e)

        # Try to extract specific field errors
        if "Field" in error_str or "validation error" in error_str:
            for line in error_str.split("\n"):
                if line.strip() and not line.startswith("For further"):
                    errors.append(line.strip())

        if not errors:
            errors.append(error_str)

        return False, errors, warnings, None

    # Second, validate prompts using prompt_engine validator (crash-only)
    try:
        from src.extensions.content.deep_research.graph.helpers.prompt_registry import get_validation_status
        prompts_valid, prompt_errors, prompt_warnings = get_validation_status(config)
        errors.extend(prompt_errors)
        warnings.extend(prompt_warnings)
    except ImportError:
        # Prompt registry not available - add warning
        warnings.append("Prompt validation skipped: prompt_registry not available")
    except Exception as e:
        # Prompt validation error - add as warning (don't fail)
        warnings.append(f"Prompt validation error: {e}")

    # Third, check model availability
    if check_models:
        try:
            models_available, model_warnings = ModelAvailabilityChecker.validate_config_models(config)
            warnings.extend(model_warnings)
        except Exception as e:
            # Model checking is optional - don't fail on errors
            warnings.append(f"Model availability check failed: {e}")

    return len(errors) == 0, errors, warnings, validated


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SpawnerConfigSchema",
    "ModelConfig",
    "ModelConfigs",
    "PromptsConfig",
    "SpawnerConfig",
    "ToolSettings",
    "ModelAvailabilityChecker",
    "validate_config",
    "validate_config_with_details",
]
