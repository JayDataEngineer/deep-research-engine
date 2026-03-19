"""CRASH-ONLY Configuration Validation Schema.

NO FALLBACKS. NO SILENT ERRORS. CRASH ON ANY ISSUE.

This module provides Pydantic models that validate the ENTIRE config.yaml structure.
If ANY field is missing, unknown, or invalid -> CRASH.

Usage:
    from deep_research.config_schema_strict import validate_config_crash

    # This will CRASH if config is invalid
    config = validate_config_crash(yaml_content)

Prompt Templates:
    ALL prompts are validated for:
    1. Required template names (flash_explorer.planning, etc.)
    2. Required Jinja2 variables in each template

    If a prompt is missing -> CRASH
    If a prompt uses undefined variable -> CRASH (detected at render time)
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError


# =============================================================================
# REQUIRED PROMPT TEMPLATES AND THEIR VARIABLES
# =============================================================================

# Every prompt template that MUST exist in the config
REQUIRED_PROMPTS: Dict[str, Set[str]] = {
    # Worker prompts - variables are optional since there are two worker templates
    "worker": set(),

    # Boss prompts
    "boss": {"original_query", "complexity", "recommended_approach", "ambiguity_detected", "execution_mode", "skills_text", "max_researchers"},
    "boss.structured": {"original_query", "complexity", "research_vectors", "max_researchers"},
    "boss.spawn": {"original_query"},  # For main boss node to decide which workers to spawn
    "boss.guidance": {"original_query", "complexity", "approach", "research_vectors"},  # For preset mode to provide guidance to workers
    "boss.guidance.detailed": {"brand_identity", "brand_tagline", "tone_style", "keywords",
                               "original_query", "complexity", "approach", "research_vectors", "research_team"},
    "boss.adjustment": {"feedback", "worker_list"},

    # Critic prompts
    "critic": {"research_brief", "worker_findings", "history", "quality_threshold"},
    "critic.system": {"tool_info"},

    # Synthesis prompts
    "synthesis": {"findings"},
    "synthesis.default": {"query", "synthesis_role"},

    # Flash Explorer prompts
    "flash_explorer.planning": {"input_data"},  # Legacy combined prompt
    "flash_explorer.planning.system": set(),  # System instructions (no variables)
    "flash_explorer.planning.user": {"input_data", "context_section"},  # User content
    "flash_explorer.synthesis": {"num_explorers", "findings_summary"},  # Legacy
    "flash_explorer.synthesis.system": set(),  # System instructions
    "flash_explorer.synthesis.user": {"num_explorers", "findings_summary"},  # User content

    # =============================================================================
    # EXTRACTION PROMPTS - Used by extract_structured_output() pattern
    # =============================================================================
    # These prompts tell the extraction LLM what fields to extract.
    # The base template (extraction.json) uses {{ agent_output }} and
    # {{ extraction_description }} variables.
    #
    # The extraction_prompt_key values below are plain text descriptions
    # that get injected into extraction.json via the extraction_description var.
    # They must exist in prompts/base.yaml or the config will fail validation.
    #
    # USAGE IN CODE:
    #   extract_structured_output(
    #       ...,
    #       extraction_prompt_key="extraction.boss_decision",  # <-- Must exist in YAML
    #       yaml_config=yaml_config,
    #   )
    # =============================================================================
    "extraction.json": {"agent_output", "extraction_description"},  # Base template
    "extraction.boss_decision": set(),      # Boss decision extraction
    "extraction.critic_response": set(),    # Critic evaluation extraction
    "extraction.research_plan": set(),      # Research plan/avenues extraction
    "extraction.flash_explorer_findings": set(),  # Flash explorer findings extraction

    # =============================================================================
    # SPECIALIST PROMPTS - Domain-specific worker prompts
    # =============================================================================
    # These are used by specialist_prompts.py for specialized research workers.
    # The domain name in code maps to a prompt key:
    #   "security" -> "specialist.cybersecurity"
    #   "financial" -> "specialist.financial"
    #   etc.
    #
    # All specialist prompts require {{ readable_name }} variable.
    # =============================================================================
    "specialist.cybersecurity": {"readable_name"},
    "specialist.financial": {"readable_name"},
    "specialist.technical": {"readable_name"},
    "specialist.legal": {"readable_name"},
    "specialist.scientific": {"readable_name"},
    "specialist.business": {"readable_name"},
    "specialist.creative": {"readable_name"},
    "specialist.medical": {"readable_name"},
    "specialist.general": {"readable_name"},
    "specialist.worker": {"readable_name"},  # Generic worker for initial reconnaissance
}

# Variables that are optional (have defaults in code)
OPTIONAL_TEMPLATE_VARS: Dict[str, Set[str]] = {
    "flash_explorer.planning": {"context_section", "has_rag", "document_count", "documents"},
    "boss": {"worker_types_desc", "refinement_context"},
    # Worker has two templates with different variables
    "worker": {"query", "research_brief", "focus_area", "available_tools", "domain_context", "raw_query", "task", "context", "documents"},
}


class ConfigValidationError(Exception):
    """CRASH-ONLY error for config validation failures."""

    def __init__(self, message: str, field_path: str = ""):
        self.field_path = field_path
        super().__init__(
            f"\n"
            f"╔═══════════════════════════════════════════════════════════════════╗\n"
            f"║         CONFIG VALIDATION ERROR - CRASH IMMEDIATELY                ║\n"
            f"╚═══════════════════════════════════════════════════════════════════╝\n"
            f"\n"
            f"FIELD: {field_path or 'root'}\n"
            f"ERROR: {message}\n"
            f"\n"
            f"ACTION REQUIRED:\n"
            f"  1. Fix the config.yaml file\n"
            f"  2. Ensure all required fields are present\n"
            f"  3. Remove any unknown fields\n"
            f"\n"
            f"NO FALLBACKS. NO SILENT ERRORS. FIX THE CONFIG.\n"
        )


# =============================================================================
# BASE MODEL WITH STRICT EXTRA FORBID
# =============================================================================

class StrictBaseModel(BaseModel):
    """Base model that FORBIDS all extra fields."""

    model_config = {"extra": "forbid"}


# =============================================================================
# PROMPT TEMPLATE VALIDATION
# =============================================================================

def extract_jinja2_variables(template: str) -> Set[str]:
    """Extract all Jinja2 variable names from a template."""
    # Match {{ variable }} and {{ variable.nested }}
    pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)'
    return set(re.findall(pattern, template))


def validate_prompt_template(
    name: str,
    template: str,
    required_vars: Set[str],
    optional_vars: Set[str] = None,
) -> List[str]:
    """Validate a single prompt template.

    Returns list of errors (empty if valid).
    """
    errors = []
    optional_vars = optional_vars or set()

    if not template:
        errors.append(f"Prompt '{name}' is empty")
        return errors

    # Extract variables used in template
    used_vars = extract_jinja2_variables(template)

    # Check that all required variables are present
    all_allowed = required_vars | optional_vars
    for var in required_vars:
        if var not in used_vars:
            errors.append(
                f"Prompt '{name}' is missing required variable '{{{{ {var} }}}}'. "
                f"Template must include this variable."
            )

    # Check for undefined variables (variables used but not declared)
    for var in used_vars:
        if var not in all_allowed:
            errors.append(
                f"Prompt '{name}' uses undefined variable '{{{{ {var} }}}}'. "
                f"Allowed: {sorted(all_allowed)}"
            )

    return errors


def validate_all_prompts(prompts: Dict[str, Any]) -> None:
    """Validate ALL prompt templates.

    Raises ConfigValidationError if any prompt is invalid.
    """
    if not prompts:
        raise ConfigValidationError(
            "prompts section is missing or empty",
            "prompts"
        )

    all_errors = []

    # Check for missing required prompts
    missing_prompts = set(REQUIRED_PROMPTS.keys()) - set(prompts.keys())
    if missing_prompts:
        all_errors.append(
            f"Missing required prompt templates: {sorted(missing_prompts)}. "
            f"These MUST be defined in config.yaml or prompts/base.yaml"
        )

    # Validate each prompt template
    for name, template in prompts.items():
        if not isinstance(template, str):
            # Could be a dict with metadata
            if isinstance(template, dict):
                template = template.get("template", "") or template.get("prompt", "")
            if not isinstance(template, str):
                all_errors.append(f"Prompt '{name}' must be a string, got {type(template).__name__}")
                continue

        required_vars = REQUIRED_PROMPTS.get(name, set())
        optional_vars = OPTIONAL_TEMPLATE_VARS.get(name, set())

        errors = validate_prompt_template(name, template, required_vars, optional_vars)
        all_errors.extend(errors)

    if all_errors:
        raise ConfigValidationError(
            "\n".join(f"  - {e}" for e in all_errors),
            "prompts"
        )


# =============================================================================
# EXTENSION METADATA MODELS
# =============================================================================

class ExtensionMetadata(StrictBaseModel):
    """Extension metadata - REQUIRED."""
    name: str = Field(min_length=1)
    type: str = Field(default="content")
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    version: str = Field(default="1.0.0")
    category: str = Field(default="content")
    enabled: bool = Field(default=True)


class PythonConfig(StrictBaseModel):
    """Python configuration."""
    use_host_env: bool = Field(default=True)


class SystemConfig(StrictBaseModel):
    """System configuration."""
    lazy_load: bool = Field(default=True)


class CapabilitiesConfig(StrictBaseModel):
    """Extension capabilities."""
    input_type: List[str] = Field(min_length=1)
    output_type: List[str] = Field(min_length=1)
    streaming: bool = Field(default=True)


class UIConfig(StrictBaseModel):
    """UI configuration."""
    sidebar_component: Optional[str] = None
    input_component: Optional[str] = None
    message_style: Optional[str] = None


# =============================================================================
# SPAWNER CONFIG MODELS
# =============================================================================

class SubWorkersConfig(StrictBaseModel):
    """Sub-workers configuration."""
    enable: bool = Field(default=False)
    max_concurrent_per_worker: int = Field(ge=1, le=20, default=5)
    max_sub_workers_total: int = Field(ge=1, le=100, default=20)
    default_timeout: int = Field(ge=10, le=3600, default=300)
    default_max_depth: int = Field(ge=1, le=10, default=3)
    default_max_sources: int = Field(ge=1, le=50, default=10)


class SpawnerConfig(StrictBaseModel):
    """Spawner configuration - controls research execution."""
    max_researchers: int = Field(ge=1, le=10, default=3)
    max_iterations: int = Field(ge=1, le=10, default=3)
    enable_prompt_adjustment: bool = Field(default=True)
    max_adjustments_per_worker: int = Field(ge=0, le=10, default=3)
    adjustment_strategy: str = Field(default="replace")
    clear_adjustments_on_pass: bool = Field(default=True)
    flash_recursion_limit: int = Field(ge=10, le=200, default=50)
    sub_workers: Optional[SubWorkersConfig] = None

    # Trivial query detection
    trivial_query_arithmetic_complexity: float = Field(ge=0.0, le=1.0, default=0.05)
    trivial_query_short_query_complexity: float = Field(ge=0.0, le=1.0, default=0.1)
    trivial_query_definition_complexity: float = Field(ge=0.0, le=1.0, default=0.15)
    trivial_query_default_complexity: float = Field(ge=0.0, le=1.0, default=0.2)
    trivial_query_min_length: int = Field(ge=1, default=15)
    trivial_query_max_spaces: int = Field(ge=0, default=2)


# =============================================================================
# AGENT CONFIG MODELS
# =============================================================================

# ScoutConfig removed - worker is now just a default worker
# Complexity thresholds moved to SpawnerConfig


class BossConfig(StrictBaseModel):
    """Boss agent configuration."""
    max_chunks: int = Field(default=-1)
    streaming_timeout: int = Field(ge=10, le=600, default=60)


class LabConfig(StrictBaseModel):
    """Lab agent configuration."""
    artifact_cleanup_enabled: bool = Field(default=True)
    artifact_retention_hours: int = Field(ge=1, le=720, default=24)
    stagger_delay: float = Field(ge=0.0, le=60.0, default=5.0)
    relevance_decay_rate: float = Field(ge=0.0, le=1.0, default=0.02)
    citation_default_confidence: int = Field(ge=0, le=100, default=80)
    documents: List[str] = Field(default_factory=list)


class CriticConfig(StrictBaseModel):
    """Critic agent configuration."""
    quality_threshold: int = Field(ge=0, le=100, default=70)
    direct_confidence: int = Field(ge=0, le=100, default=90)
    default_quality_score: int = Field(ge=0, le=100, default=80)
    default_completeness_score: int = Field(ge=0, le=100, default=90)
    revise_threshold: int = Field(ge=0, le=100, default=50)


class SynthesisConfig(StrictBaseModel):
    """Synthesis configuration."""
    safe_summary_threshold_percent: float = Field(ge=0.0, le=1.0, default=0.05)
    lazy_loading_enabled: bool = Field(default=True)


# =============================================================================
# MIDDLEWARE CONFIG MODELS
# =============================================================================

class ToolRetryConfig(StrictBaseModel):
    """Tool retry middleware configuration."""
    max_retries: int = Field(ge=0, le=10, default=2)
    backoff_factor: float = Field(ge=1.0, le=10.0, default=2.0)
    initial_delay_seconds: float = Field(ge=0.0, le=60.0, default=1.0)
    max_delay_seconds: float = Field(ge=1.0, le=300.0, default=60.0)
    jitter: bool = Field(default=True)
    on_failure: str = Field(default="raise")


class ToolCallLimitConfig(StrictBaseModel):
    """Tool call limit configuration."""
    fact_check_thread_limit: int = Field(ge=1, le=50, default=5)


class MiddlewareConfig(StrictBaseModel):
    """Middleware configuration."""
    tool_retry: ToolRetryConfig = Field(default_factory=ToolRetryConfig)
    tool_call_limit: ToolCallLimitConfig = Field(default_factory=ToolCallLimitConfig)


# =============================================================================
# BEST PRACTICES CONFIG MODEL
# =============================================================================

class BestPracticesConfig(StrictBaseModel):
    """LangGraph best practices configuration."""
    tool_token_limit_before_evict: int = Field(ge=1000, le=100000, default=20000)
    low_steps_threshold: int = Field(ge=1, le=20, default=3)
    streaming_enabled: bool = Field(default=True)
    structured_output_enabled: bool = Field(default=True)


# =============================================================================
# TASKS CONFIG MODEL
# =============================================================================

class TasksConfig(StrictBaseModel):
    """Task configuration."""
    default_max_retries: int = Field(ge=0, le=10, default=3)
    idempotent_enabled: bool = Field(default=True)


# =============================================================================
# DATABASE CONFIG MODEL
# =============================================================================

class DatabaseConfig(StrictBaseModel):
    """Database configuration."""
    max_recent_runs: int = Field(ge=1, le=1000, default=50)
    connection_timeout_seconds: float = Field(ge=1.0, le=300.0, default=30.0)


# =============================================================================
# LOGGING CONFIG MODEL
# =============================================================================

class LoggingComponentsConfig(StrictBaseModel):
    """Component-specific log levels."""
    worker: Optional[str] = None
    boss: Optional[str] = None
    lab: Optional[str] = None
    lab_architect: Optional[str] = None
    critic: Optional[str] = None
    synthesis: Optional[str] = None
    flash_explorer: Optional[str] = None
    web_agent: Optional[str] = None
    run_logger: Optional[str] = None
    context_engine: Optional[str] = None


class LoggingConfig(StrictBaseModel):
    """Logging configuration."""
    level: str = Field(default="normal")
    components: Optional[LoggingComponentsConfig] = None

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = {"debug", "normal", "quiet", "silent"}
        if v.lower() not in valid_levels:
            raise ValueError(f"Invalid log level '{v}'. Must be one of: {valid_levels}")
        return v.lower()


# =============================================================================
# CONTEXT ENGINE CONFIG MODEL
# =============================================================================

class PostgresConfig(StrictBaseModel):
    """Postgres configuration for Context Engine."""
    connection_string: str = Field(min_length=1)
    artifact_table: str = Field(default="artifacts")
    embedding_table: str = Field(default="artifact_embeddings")
    pool_min_size: int = Field(ge=1, default=5)
    pool_max_size: int = Field(ge=1, default=20)
    health_check_timeout: float = Field(ge=0.1, default=1.0)


class Neo4jConfig(StrictBaseModel):
    """Neo4j configuration."""
    enabled: bool = Field(default=True)
    url: str = Field(min_length=1)
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class ContextEngineConfig(StrictBaseModel):
    """Context Engine configuration."""
    enabled: bool = Field(default=True)
    mode: str = Field(default="research")
    isolation: str = Field(default="session")
    postgres: Optional[PostgresConfig] = None
    neo4j: Optional[Neo4jConfig] = None


# =============================================================================
# THRESHOLDS CONFIG MODEL - Centralized threshold configuration
# =============================================================================

class ConfidenceThresholdConfig(StrictBaseModel):
    """Confidence threshold configuration (0-100 scale)."""
    high: int = Field(ge=0, le=100, default=70)
    medium: int = Field(ge=0, le=100, default=50)
    low: int = Field(ge=0, le=100, default=30)
    default: int = Field(ge=0, le=100, default=50)


class QualityThresholdConfig(StrictBaseModel):
    """Quality threshold configuration (0-100 scale)."""
    default: int = Field(ge=0, le=100, default=70)


class TokenLimitsConfig(StrictBaseModel):
    """Token/content limits configuration."""
    max_chars_per_chunk: int = Field(ge=1000, le=100000, default=24000)
    max_output_length: int = Field(ge=100, le=10000, default=500)


class PlanningThresholdConfig(StrictBaseModel):
    """Research planning thresholds."""
    min_angles: int = Field(ge=1, le=10, default=3)
    max_angles: int = Field(ge=1, le=20, default=5)


class ThresholdsConfig(StrictBaseModel):
    """Centralized threshold configuration."""
    confidence: ConfidenceThresholdConfig = Field(default_factory=ConfidenceThresholdConfig)
    quality: QualityThresholdConfig = Field(default_factory=QualityThresholdConfig)
    tokens: TokenLimitsConfig = Field(default_factory=TokenLimitsConfig)
    planning: PlanningThresholdConfig = Field(default_factory=PlanningThresholdConfig)


# =============================================================================
# TOP-LEVEL CONFIG MODEL
# =============================================================================

class DeepResearchConfig(StrictBaseModel):
    """Complete deep_research configuration - STRICT VALIDATION.

    ANY field not defined here will cause a validation error.
    ANY missing required field will cause a validation error.
    """

    # Extension metadata - REQUIRED
    extension: ExtensionMetadata

    # Python/system configuration
    python: Optional[PythonConfig] = None
    system: Optional[SystemConfig] = None
    capabilities: Optional[CapabilitiesConfig] = None
    tags: Optional[List[str]] = None
    ui: Optional[UIConfig] = None

    # LLM concurrency
    llm_concurrency: int = Field(ge=1, le=20, default=3)

    # Tools - MUST be a list of strings
    enabled_tools: List[str] = Field(default_factory=list)

    # Agent configurations
    spawner_config: SpawnerConfig = Field(default_factory=SpawnerConfig)
    boss: BossConfig = Field(default_factory=BossConfig)
    lab: LabConfig = Field(default_factory=LabConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)

    # Thresholds (centralized configuration)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)

    # Middleware
    middleware: MiddlewareConfig = Field(default_factory=MiddlewareConfig)

    # Best practices
    best_practices: BestPracticesConfig = Field(default_factory=BestPracticesConfig)

    # Tasks
    tasks: TasksConfig = Field(default_factory=TasksConfig)

    # Database
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # Logging
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Context Engine
    context_engine: Optional[ContextEngineConfig] = None

    # Prompts - validated separately
    prompts: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_prompts_section(self) -> "DeepResearchConfig":
        """Validate that all required prompts exist and have correct variables."""
        if self.prompts:
            validate_all_prompts(self.prompts)
        return self


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config_crash(config: Dict[str, Any]) -> DeepResearchConfig:
    """Validate config with CRASH-ONLY semantics.

    Args:
        config: Raw config dictionary from YAML

    Returns:
        Validated DeepResearchConfig

    Raises:
        ConfigValidationError: If validation fails (CRASH)
        ValidationError: If Pydantic validation fails (CRASH)
    """
    try:
        # Validate with Pydantic
        validated = DeepResearchConfig(**config)

        # Additional prompt validation
        if validated.prompts:
            validate_all_prompts(validated.prompts)

        return validated

    except ValidationError as e:
        # Convert Pydantic error to our crash format
        errors = []
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  - {loc}: {msg}")

        raise ConfigValidationError(
            f"Pydantic validation failed:\n" + "\n".join(errors),
            "config"
        ) from e


def validate_config_file_crash(config_path: Path | str) -> DeepResearchConfig:
    """Validate a config file with CRASH-ONLY semantics.

    Args:
        config_path: Path to config.yaml

    Returns:
        Validated DeepResearchConfig

    Raises:
        ConfigValidationError: If file not found or validation fails
    """
    import yaml

    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigValidationError(
            f"Config file not found: {config_path}",
            "file"
        )

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(
            f"YAML parse error: {e}",
            "file"
        ) from e

    if not isinstance(config, dict):
        raise ConfigValidationError(
            f"Config must be a dictionary, got {type(config).__name__}",
            "file"
        )

    return validate_config_crash(config)


def validate_prompt_variable_mapping(
    template_name: str,
    code_var_names: Set[str],
    template_required_vars: Set[str],
) -> None:
    """Validate that code passes correct variables to template.

    This catches issues like:
    - Template expects 'input_data' but code passes 'research_avenue'
    - Code passes variables that template doesn't use

    Args:
        template_name: Name of the template
        code_var_names: Variables passed by the code
        template_required_vars: Variables required by the template

    Raises:
        ConfigValidationError: If mapping is incorrect
    """
    # Check for missing required vars
    missing = template_required_vars - code_var_names
    if missing:
        raise ConfigValidationError(
            f"Template '{template_name}' requires variables {missing} "
            f"but code only passes {code_var_names}. "
            f"Either update the template or fix the code.",
            f"prompts.{template_name}"
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DeepResearchConfig",
    "ConfigValidationError",
    "validate_config_crash",
    "validate_config_file_crash",
    "validate_prompt_variable_mapping",
    "validate_all_prompts",
    "REQUIRED_PROMPTS",
    "OPTIONAL_TEMPLATE_VARS",
]
