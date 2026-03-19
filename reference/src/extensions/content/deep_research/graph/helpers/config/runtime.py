"""Runtime config utilities for deep research.

This module provides utilities for converting YAML config to
ResearchRuntimeConfig and extracting runtime config from LangGraph's
RunnableConfig pattern.
"""

from typing import Dict, Any, Optional


# =============================================================================
# RUNTIME CONFIG INTEGRATION (YAML + Runtime API)
# =============================================================================

def yaml_to_runtime_config(
    yaml_config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None
) -> "ResearchRuntimeConfig":
    """Convert YAML config to ResearchRuntimeConfig with optional runtime overrides.

    This function enables the two-layer configuration pattern:
    - Layer 1: YAML config (static defaults from config.yaml files)
    - Layer 2: Runtime overrides (API parameters, user selections, A/B testing)

    Args:
        yaml_config: Loaded YAML config dict from _load_yaml_config()
        overrides: Optional runtime overrides (model_provider, max_parallel_workers, etc.)

    Returns:
        ResearchRuntimeConfig instance ready to pass to graph.invoke()

    Example:
        # Load YAML defaults
        yaml_config = _load_yaml_config(use_case="rag", variant="default")

        # Convert to runtime config (with optional overrides)
        runtime_config = yaml_to_runtime_config(
            yaml_config,
            overrides={"max_parallel_workers": 8, "quality_threshold": 0.9}
        )

        # Invoke graph with runtime config
        result = await graph.ainvoke(
            input={"query": "Research query"},
            config={"configurable": runtime_config}
        )
    """
    from ...state import ResearchRuntimeConfig

    spawner = yaml_config.get("spawner_config", {})

    # Build config dict from YAML defaults
    config_dict = {
        # Execution configuration from spawner_config
        "max_parallel_workers": spawner.get("max_researchers", 3),
        "max_iterations": spawner.get("max_iterations", 3),
        "quality_threshold": spawner.get("quality_threshold", 0.7),
        "enable_reflexion": spawner.get("enable_citations", True),
        "enable_prompt_adjustment": spawner.get("enable_prompt_adjustment", False),
        "enable_quick_path": spawner.get("enable_quick_path", True),
        "enable_citations": spawner.get("enable_citations", True),

        # Feature flags
        "enable_caching": True,
        "cache_ttl": 3600,

        # Debug options (defaults from YAML if present)
        "debug_mode": yaml_config.get("debug_mode", False),
        "save_artifacts": yaml_config.get("save_artifacts", True),
    }

    # Apply runtime overrides if provided (API parameters, user selections)
    if overrides:
        config_dict.update(overrides)

    # Validate and create ResearchRuntimeConfig (Pydantic will validate)
    return ResearchRuntimeConfig(**config_dict)


def get_runtime_config(config: Any) -> "ResearchRuntimeConfig":
    """Extract ResearchRuntimeConfig from LangGraph's RunnableConfig.

    This is the standard way for nodes to access runtime configuration.

    Args:
        config: RunnableConfig from LangGraph (passed to all nodes)

    Returns:
        ResearchRuntimeConfig instance with defaults if not found in config

    Example usage in node:
        async def worker_node(state: GlobalState, config: RunnableConfig):
            runtime_config = get_runtime_config(config)

            # Use runtime config for dynamic behavior
            if runtime_config.debug_mode:
                logger.info("Debug mode enabled")

            if runtime_config.max_parallel_workers > 5:
                logger.info("High parallelism mode")

            # Select LLM based on runtime config
            model = get_llm_for_provider(runtime_config.model_provider)

            return {"research_brief": brief}
    """
    from ...state import ResearchRuntimeConfig

    # Try to get from config["configurable"] (standard Runtime API pattern)
    if config and "configurable" in config:
        configurable = config["configurable"]

        # If it's already a ResearchRuntimeConfig, return it
        if isinstance(configurable, ResearchRuntimeConfig):
            return configurable

        # If it's a dict, try to convert to ResearchRuntimeConfig
        if isinstance(configurable, dict):
            # Extract only fields that ResearchRuntimeConfig understands
            valid_fields = {
                k: v for k, v in configurable.items()
                if k in ResearchRuntimeConfig.model_fields
            }
            return ResearchRuntimeConfig(**valid_fields)

    # Fallback: Return default ResearchRuntimeConfig
    return ResearchRuntimeConfig()
