"""Configuration validation for deep research variants.

This module provides Pydantic-based validation to catch configuration errors
BEFORE the research run starts. This prevents silent failures where specialist
configs are defined but never used.

PHILOSOPHY:
- Configuration issues caught at startup, not during execution
- If a feature requires a config/template/tool, validate it exists
- "Failure is a blessing" - it reveals bugs immediately

This module validates CONFIG FILES ONLY.
It does NOT validate implementation details (see implementation_validation.py).
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class WorkerTypeError(Exception):
    """Raised when worker type configuration is invalid or missing."""
    pass


class SpecialistConfigError(Exception):
    """Raised when specialist configs are defined but cannot be used."""
    pass


class ResearchToolsError(Exception):
    """Raised when specialist worker types won't get research tools."""
    pass


class ConfigSectionError(Exception):
    """Raised when required config sections are missing."""
    pass


class SpawnerConfig(BaseModel):
    """Validation model for spawner_config section."""

    min_complexity: float = 0.3
    max_researchers: int = 3
    max_iterations: int = 3
    quality_threshold: float = 0.8
    completeness_threshold: float = 0.85
    worker_specialization_mode: str = "generic"
    worker_types: Optional[Dict[str, str]] = None
    tools: Optional[Dict[str, bool]] = None
    worker_graph: Optional[Dict[str, Any]] = None

    @field_validator("worker_specialization_mode")
    @classmethod
    def validate_specialization_mode(cls, v):
        valid_modes = ["generic", "hybrid", "fixed", "dynamic"]
        if v not in valid_modes:
            raise ValueError(f"Invalid worker_specialization_mode: {v}. Must be one of {valid_modes}")
        return v


class WorkerGraphConfig(BaseModel):
    """Validation model for worker_graph section."""

    worker_order: List[str] = Field(default_factory=list)
    dependencies: Optional[Dict[str, List[str]]] = None


class VariantConfig(BaseModel):
    """Complete validation model for a research variant configuration."""

    spawner_config: SpawnerConfig
    specialist_configs: Optional[Dict[str, Any]] = None
    worker_graph: Optional[WorkerGraphConfig] = None
    _metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_specialization_consistency(self):
        """Validate that specialization configuration is consistent and usable.

        This catches the silent error where:
        - worker_specialization_mode is "fixed"
        - specialist_configs are defined
        - But worker_types or worker_graph.worker_order don't align

        The system would then run all workers as "generic" without warning.
        """
        spawner = self.spawner_config
        specialist_mode = spawner.worker_specialization_mode

        # Get worker_types from either spawner_config.worker_types or worker_graph.worker_order
        defined_worker_types = spawner.worker_types
        worker_order = []
        if self.worker_graph:
            worker_order = self.worker_graph.worker_order

        # Check for specialist_configs that won't be used
        specialist_config_keys = set(self.specialist_configs.keys()) if self.specialist_configs else set()

        # Case 1: worker_specialization_mode is "fixed" but no worker types defined
        if specialist_mode == "fixed":
            if not defined_worker_types and not worker_order:
                raise WorkerTypeError(
                    "\n"
                    "╔═══════════════════════════════════════════════════════════════════╗\n"
                    "║         CONFIGURATION ERROR: Fixed Mode Without Worker Types         ║\n"
                    "╚═══════════════════════════════════════════════════════════════════╝\n"
                    "\n"
                    f"worker_specialization_mode is set to 'fixed', but no worker types are defined.\n"
                    "\n"
                    "To use fixed specialization mode, you MUST define ONE of:\n"
                    "  1. spawner_config.worker_types: {{philosopher: '...', marketer: '...', ...}}\n"
                    "  2. worker_graph.worker_order: [philosopher, marketer, ...]\n"
                    "\n"
                    f"Found specialist_configs for: {specialist_config_keys if specialist_config_keys else 'none'}\n"
                    "\n"
                    "These specialist configs will NEVER BE USED without proper worker type assignment.\n"
                    "\n"
                    "FIX:\n"
                    "  Add to spawner_config:\n"
                    "    worker_types:\n"
                    "      philosopher: 'Philosopher - timeless wisdom'\n"
                    "      relevancy: 'Relevancy - current zeitgeist'\n"
                    "      marketer: 'Marketer - audience psychology'\n"
                    "\n"
                )

            # Case 2: specialist_configs defined but worker_types don't reference them
            if specialist_config_keys and defined_worker_types:
                defined_type_keys = set(defined_worker_types.keys())
                unused_specialists = specialist_config_keys - defined_type_keys
                if unused_specialists:
                    raise SpecialistConfigError(
                        "\n"
                        "╔═══════════════════════════════════════════════════════════════════╗\n"
                        "║           CONFIGURATION ERROR: Unused Specialist Configs           ║\n"
                        "╚═══════════════════════════════════════════════════════════════════╝\n"
                        "\n"
                        "The following specialist_configs are defined but will NEVER BE USED:\n"
                        f"  {', '.join(unused_specialists)}\n"
                        "\n"
                        "spawner_config.worker_types defines:\n"
                        f"  {', '.join(defined_type_keys)}\n"
                        "\n"
                        "These don't match! The boss node cannot assign workers to specialist configs.\n"
                        "\n"
                        "FIX:\n"
                        "  Either add missing worker_types, or remove unused specialist_configs.\n"
                        "\n"
                    )

            # Case 3: worker_graph.worker_order defined but no worker_types in spawner
            if worker_order and not defined_worker_types:
                raise WorkerTypeError(
                    "\n"
                    "╔═══════════════════════════════════════════════════════════════════╗\n"
                    "║          CONFIGURATION ERROR: Worker Graph Not Connected          ║\n"
                    "╚═══════════════════════════════════════════════════════════════════╝\n"
                    "\n"
                    "worker_graph.worker_order is defined:\n"
                    f"  {', '.join(worker_order)}\n"
                    "\n"
                    "BUT spawner_config.worker_types is NOT defined.\n"
                    "\n"
                    "The boss node reads spawner_config.worker_types, NOT worker_graph.worker_order.\n"
                    "Without worker_types, all workers will run as 'generic' and specialist_configs\n"
                    "will be ignored.\n"
                    "\n"
                    "FIX - Add to spawner_config:\n"
                    "  worker_types:\n"
                    + "\n".join([f"    {w}: '{w.capitalize()} - specialized for {w} tasks'" for w in worker_order])
                    + "\n"
                    "\n"
                )

            # Case 4: Specialist worker types not in needs_research keywords
            # Get needs_research keywords from shared constants
            try:
                from src.extensions.content.deep_research.constants import NEEDS_RESEARCH_KEYWORDS

                if specialist_config_keys:
                    # Check which specialist types aren't in needs_research keywords
                    workers_without_tools = []
                    for worker_type in specialist_config_keys:
                        worker_lower = worker_type.lower()
                        # Check if any part of the worker_type matches a keyword
                        if not any(keyword in worker_lower for keyword in NEEDS_RESEARCH_KEYWORDS):
                            workers_without_tools.append(worker_type)

                    if workers_without_tools:
                        raise ResearchToolsError(
                            "\n"
                            "╔═══════════════════════════════════════════════════════════════════╗\n"
                            "║         CONFIGURATION ERROR: Workers Won't Get Research Tools        ║\n"
                            "╚═══════════════════════════════════════════════════════════════════╝\n"
                            "\n"
                            "The following specialist worker types are NOT in the needs_research keywords:\n"
                            f"  {', '.join(workers_without_tools)}\n"
                            "\n"
                            "This means these workers will NOT get research tools (web_search, web_reader, etc.)\n"
                            "and will produce minimal output. The lab.py needs_research check will return False.\n"
                            "\n"
                            "Current needs_research keywords:\n"
                            "  " + ", ".join(sorted(NEEDS_RESEARCH_KEYWORDS)) + "\n"
                            "\n"
                            "FIX - Add the missing worker types to constants.py NEEDS_RESEARCH_KEYWORDS:\n"
                            "  Or rename your workers to match existing keywords.\n"
                            "\n"
                        )
            except ImportError as e:
                # CRITICAL: Constants module must exist for validation
                raise RuntimeError(
                    f"[CONFIG_VALIDATION] Cannot import NEEDS_RESEARCH_KEYWORDS from constants module: {e}\n"
                    f"The constants module is required for research tools validation."
                ) from e

        return self


def validate_required_config_sections(config: Dict[str, Any]) -> None:
    """Validate that all required config sections exist.

    This prevents silent fallbacks to hardcoded values when config sections
    are missing.

    Args:
        config: The loaded YAML configuration dict

    Raises:
        ConfigSectionError: If required sections are missing
    """
    required_sections = {
        "spawner_config": "Required for worker spawning and orchestration",
    }

    missing_sections = []
    for section, purpose in required_sections.items():
        if section not in config or not config[section]:
            missing_sections.append(f"  - {section}: {purpose}")

    if missing_sections:
        raise ConfigSectionError(
            "\n"
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║            CONFIGURATION ERROR: Missing Required Sections          ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            "The following required config sections are MISSING:\n"
            + "\n".join(missing_sections) + "\n"
            "\n"
            "These sections are REQUIRED. The system does NOT use fallback values.\n"
            "Without these sections, the system will CRASH at runtime.\n"
            "\n"
            "FIX:\n"
            "  Add the missing sections to your variant YAML file.\n"
            "\n"
        )


def validate_variant_config(config: Dict[str, Any]) -> None:
    """Validate a variant configuration before starting the research run.

    Args:
        config: The loaded YAML configuration dict

    Raises:
        WorkerTypeError: If worker type configuration is invalid
        SpecialistConfigError: If specialist configs are defined but unusable
        ValueError: For other configuration errors

    This function should be called early in the graph execution to fail fast
    instead of silently running with incorrect configuration.
    """
    try:
        # Extract relevant sections
        spawner_config = config.get("spawner_config", {})
        specialist_configs = config.get("specialist_configs")
        worker_graph = config.get("worker_graph")
        _metadata = config.get("_metadata")

        # Build validation model
        variant_config = VariantConfig(
            spawner_config=spawner_config,
            specialist_configs=specialist_configs,
            worker_graph=worker_graph,
            _metadata=_metadata,
        )

        # Pydantic validation happens automatically during construction
        # If we get here, validation passed
        return None

    except WorkerTypeError as e:
        # Re-raise worker type errors
        raise e
    except SpecialistConfigError as e:
        # Re-raise specialist config errors
        raise e
    except ResearchToolsError as e:
        # Re-raise research tools errors
        raise e
    except Exception as e:
        # Catch Pydantic validation errors
        raise ValueError(
            f"\n"
            f"╔═══════════════════════════════════════════════════════════════════╗\n"
            f"║              CONFIGURATION VALIDATION ERROR                         ║\n"
            f"╚═══════════════════════════════════════════════════════════════════╝\n"
            f"\n"
            f"Variant configuration failed validation:\n"
            f"  {e}\n"
            f"\n"
            f"Please fix the variant YAML file before running research.\n"
        ) from e
