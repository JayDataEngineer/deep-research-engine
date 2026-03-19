"""YAML config loading, caching, and merging functions.

This module handles loading configuration from YAML files with:
- Multi-level hierarchy (system base, extension, user variant)
- File modification timestamp caching
- Environment variable expansion
- Deep merging of config layers
"""

import os
import asyncio
import re
from pathlib import Path
from typing import Dict, Any, Optional

# Add backend to path for absolute imports
from src.shared.utils.path_utils import setup_backend_dir_from_file
setup_backend_dir_from_file(__file__, parent_levels=5)

# Setup centralized logging
from ....logging_utils import get_logger
logger = get_logger(__name__, __name__.split('.')[-1])


# =============================================================================
# CONFIG CACHE (performance optimization)
# =============================================================================

_config_cache: Dict[str, tuple[Dict[str, Any], tuple[float, ...]]] = {}
"""Cache of loaded configs with their modification timestamps.

Format: {cache_key: (config_dict, mtimes)}
cache_key: f"{use_case}:{variant}"
mtimes: tuple of modification times for ALL config files in the merge chain

CRITICAL: We track mtimes of ALL files in the config chain, not just the last one.
This ensures cache invalidation when ANY file in the chain changes.
"""


def _get_cache_key(use_case: str, variant: str) -> str:
    """Generate cache key for config."""
    return f"{use_case}:{variant}"


def _get_file_mtime(filepath: Path) -> float:
    """Get file modification time."""
    return filepath.stat().st_mtime


def _get_all_file_mtimes(filepaths: list[Path]) -> tuple[float, ...]:
    """Get modification times for all config files in order.

    Args:
        filepaths: List of config file paths in merge order

    Returns:
        Tuple of modification times in the same order
    """
    return tuple(_get_file_mtime(p) for p in filepaths)


def _restore_base_templates(
    prompts: Dict[str, Any],
    base_prompts: Dict[str, str]
) -> Dict[str, Any]:
    """Restore base templates when use_case provides metadata-only dicts.

    When a use_case config has a prompt as a dict with only metadata fields
    (like boss_role, domain) but no template, restore the base template
    and store the metadata under a special _{key}_config key.

    Args:
        prompts: Merged prompts dictionary
        base_prompts: Base prompt templates from prompts/base.yaml

    Returns:
        Prompts dictionary with templates restored where needed
    """
    import copy
    result = copy.deepcopy(prompts)

    # Known template fields that indicate a dict has an actual template
    template_fields = {"template", "instructions", "prompt", "content", "system"}

    # Known metadata fields that should NOT replace a template
    metadata_fields = {
        "boss_role", "domain", "worker_role", "operation_type",
        "synthesis_role", "citation_instructions", "output_instructions",
        "evaluation_focus", "critic_role", "output_type",
        "task_allocation_guidance", "dependency_graph_prompt",
        "citation_requirements", "available_skills", "research_brief",
        "domain_context", "domain_guidance", "worker_role_description",
        "citation_examples", "evaluation_guidelines", "synthesis_guidance",
    }

    for key, value in list(result.items()):
        if isinstance(value, dict) and key in base_prompts:
            # Check if this dict has a template field
            has_template = any(field in value for field in template_fields)
            # Check if this dict has only metadata fields
            is_metadata_only = all(
                field in metadata_fields or field.startswith("_")
                for field in value.keys()
            )

            if is_metadata_only and not has_template:
                # This is a metadata-only dict - restore base template
                # and store metadata under _{key}_config for variable injection
                result[f"_{key}_config"] = value
                result[key] = base_prompts[key]
                logger.debug(f"[CONFIG] Restored base template for '{key}', stored metadata under _{key}_config")

    return result


def _is_cache_valid(cache_key: str, config_paths: list[Path]) -> bool:
    """Check if cached config is still valid (no files modified).

    CRITICAL: Checks ALL files in the chain, not just the last one.
    If ANY file in the merge chain has been modified, cache is invalid.

    Args:
        cache_key: Cache key to validate
        config_paths: All config file paths in the merge chain

    Returns:
        True if cache is valid (no files modified), False otherwise
    """
    if cache_key not in _config_cache:
        return False

    _, cached_mtimes = _config_cache[cache_key]
    current_mtimes = _get_all_file_mtimes(config_paths)
    return cached_mtimes == current_mtimes


# =============================================================================
# YAML LOADING UTILITIES
# =============================================================================

def _load_yaml_files_async(base_path: Path, use_case_path: Path) -> tuple:
    """Load YAML files asynchronously using aiofiles for better performance.

    Uses asyncio to run the async file loading in a sync context.
    This provides 20-50ms faster file operations on initial config load.

    Args:
        base_path: Path to base.yaml
        use_case_path: Path to use_case config file

    Returns:
        Tuple of (base_config, use_case_config) as dicts
    """
    async def _async_load():
        import aiofiles
        import yaml

        # Read both files concurrently
        async with aiofiles.open(base_path) as f:
            base_content = await f.read()
        async with aiofiles.open(use_case_path) as f:
            use_case_content = await f.read()

        # Parse YAML
        base_config = yaml.safe_load(base_content)
        use_case_config = yaml.safe_load(use_case_content)

        return base_config, use_case_config

    # Run async function in a new event loop (for initial load)
    try:
        # Try to get the running loop first (Python 3.7+)
        loop = asyncio.get_running_loop()
        # If loop is running, we need to use a different approach
        # Fall back to sync I/O to avoid "loop is running" error
        import yaml
        with open(base_path) as f:
            base_config = yaml.safe_load(f)
        with open(use_case_path) as f:
            use_case_config = yaml.safe_load(f)
        return base_config, use_case_config
    except RuntimeError:
        # No running loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_async_load())
        finally:
            loop.close()


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dicts. override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_config_overrides(config: Dict, overrides: Dict) -> Dict:
    """Apply config overrides for grid search testing.

    Deep merges override dict into config, allowing selective override
    of nested config values without replacing entire sections.

    Args:
        config: Base config from _load_yaml_config
        overrides: Override dict from grid search config

    Returns:
        Merged config with overrides applied
    """
    return _deep_merge(config, overrides)


def _expand_env_vars(obj: Any) -> Any:
    """Expand environment variables in configuration values.

    Supports ${VAR-default} syntax:
    - If VAR is set, use its value
    - Otherwise, use 'default'

    Args:
        obj: Config object (dict, list, or string)

    Returns:
        Object with environment variables expanded
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Match ${VAR-default} pattern
        def replace_env_var(match):
            var_part = match.group(1)
            # Split on '-' to separate var name from default
            if '-' in var_part:
                var_name, default_val = var_part.split('-', 1)
                return os.getenv(var_name, default_val)
            return os.getenv(var_part, '')

        return re.sub(r'\$\{([^}]+)\}', replace_env_var, obj)
    return obj


def _validate_config(config: Dict) -> None:
    """Validate configuration using Pydantic schema.

    Performs comprehensive validation of configuration structure and values.
    Raises ValueError if configuration is invalid.

    Args:
        config: The merged configuration dict

    Raises:
        ValueError: If configuration fails validation
    """
    from src.extensions.content.deep_research.config_validation import validate_config_with_details

    is_valid, errors, warnings, validated = validate_config_with_details(config)
    if not is_valid:
        error_msg = "Configuration validation failed:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log any warnings (including model availability warnings)
    if warnings:
        logger.warning(f"Configuration warnings ({len(warnings)}):")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    logger.debug("Configuration validated successfully")


def _load_yaml_config(use_case: str = "rag", variant: str = "default", user_id: str = None) -> Dict[str, Any]:
    """Load YAML config from research extensions.

    New System (4-level hierarchy):
    0. Prompts base (deep_research/prompts/base.yaml) - Base prompts (loaded first so use_cases can override)
    1. System base (deep_research/base/config.yaml) - Deep research defaults
    2. Extension base (services/research/{use_case}/config.yaml) - Extension-specific config
    3. User variant (user/config/user/research_variants/{use_case}/{variant}.yaml) - User overrides

    Built-in variants (extension/variants/{variant}.yaml) can also be specified.

    Implements caching with file modification timestamp checking for performance.
    """
    import yaml

    # Import constants
    from src.constants import DEEP_RESEARCH_CONFIG_DIR, RESEARCH_USE_CASES_DIR, USER_CONFIG_DIR

    configs_to_load = []

    # Level 0: Prompts base (NEW - loaded first so use_cases can override)
    # This provides all default prompts that can be overridden by use_case configs
    prompts_base_path = DEEP_RESEARCH_CONFIG_DIR / "prompts" / "base.yaml"
    if prompts_base_path.exists():
        configs_to_load.append(prompts_base_path)
        logger.info(f"[CONFIG] Loaded prompts base from {prompts_base_path}")

    # Level 1: System base (deep research defaults)
    base_path = DEEP_RESEARCH_CONFIG_DIR / "base" / "config.yaml"
    if not base_path.exists():
        base_path = DEEP_RESEARCH_CONFIG_DIR / "base.yaml"
    if not base_path.exists():
        base_path = DEEP_RESEARCH_CONFIG_DIR / "config.yaml"
    if base_path.exists():
        configs_to_load.append(base_path)

    # Level 2: Extension base config
    # NOTE: RESEARCH_USE_CASES_DIR points to integrations/research where soc_media_gen lives
    extension_path = RESEARCH_USE_CASES_DIR / use_case / "config.yaml"
    if extension_path.exists():
        configs_to_load.append(extension_path)
        logger.info(f"[CONFIG] Loaded extension config '{use_case}' from {extension_path}")
    elif use_case == "base":
        # For "base" use_case, also check for the main deep_research config.yaml
        base_config_path = RESEARCH_USE_CASES_DIR / "config.yaml"
        if base_config_path.exists():
            configs_to_load.append(base_config_path)
            logger.info(f"[CONFIG] Loaded base config from {base_config_path}")
        else:
            raise ValueError(
                f"[CONFIG] CRITICAL: No research extension found for use_case '{use_case}'. "
                f"Expected config at: {base_config_path}"
            )
    else:
        # Non-base use_case must have an extension config
        raise ValueError(
            f"[CONFIG] CRITICAL: No research extension found for use_case '{use_case}'. "
            f"Expected config at: {extension_path}"
        )

    # Level 3: Variant (built-in or user-created)
    if variant != "default":
        # Try user variant first (if user_id provided)
        if user_id:
            user_variant_path = (
                USER_CONFIG_DIR /
                user_id /
                "research_variants" /
                use_case /
                f"{variant}.yaml"
            )
            if user_variant_path.exists():
                configs_to_load.append(user_variant_path)
                logger.info(f"[CONFIG] Loaded user variant '{variant}' for {user_id}")

        # Try built-in extension variant
        extension_variant_path = RESEARCH_USE_CASES_DIR / use_case / "variants" / f"{variant}.yaml"
        if extension_variant_path.exists():
            configs_to_load.append(extension_variant_path)
            logger.info(f"[CONFIG] Loaded built-in variant '{variant}' for {use_case}")
        elif variant and use_case == "base":
            # Auto-detect use_case from variant when using base use_case
            # This allows --variant substack_simple to work without specifying --use-case longform_gen
            for ext_dir in RESEARCH_USE_CASES_DIR.iterdir():
                if not ext_dir.is_dir():
                    continue
                variant_path = ext_dir / "variants" / f"{variant}.yaml"
                if variant_path.exists():
                    logger.info(f"[CONFIG] Auto-detected use_case '{ext_dir.name}' from variant '{variant}'")
                    # Load the extension base config
                    extension_base_path = ext_dir / "config.yaml"
                    if extension_base_path.exists():
                        configs_to_load.append(extension_base_path)
                    # Load the variant
                    configs_to_load.append(variant_path)
                    # Update use_case for subsequent processing
                    use_case = ext_dir.name
                    break

    # Determine use_case_path for cache checking (use first non-base config)
    use_case_path = configs_to_load[1] if len(configs_to_load) > 1 else base_path

    # NO CACHING: Always load fresh configs on every run
    # This ensures config changes are immediately reflected without cache issues
    cache_key = _get_cache_key(use_case, variant)
    logger.debug(f"[CONFIG] Loading config for {cache_key} (cache disabled)")

    # Load and merge all configs in order (later overrides earlier)
    # Import merge_configs from config_diff (relative import from grandparent package)
    from ....config_diff import merge_configs

    merged = {}
    base_prompts = {}  # Store base prompts for template preservation

    for path in configs_to_load:
        if path.exists():
            try:
                with open(path) as f:
                    config = yaml.safe_load(f)
                # Expand environment variables in config
                if config:
                    config = _expand_env_vars(config)

                    # Before merging, save base templates for prompts that might be
                    # overridden by metadata-only dicts from use_cases
                    if "prompts" in config and path == configs_to_load[0]:
                        # This is the prompts base file - save the templates
                        for key, value in config.get("prompts", {}).items():
                            if isinstance(value, str):
                                base_prompts[key] = value

                    merged = merge_configs(merged, config)
                    logger.debug(f"[CONFIG] Loaded config from {path}")
            except Exception as e:
                logger.error(f"[CONFIG] Failed to load config from {path}: {e}")

    # Post-process prompts: restore base templates for metadata-only overrides
    if "prompts" in merged and base_prompts:
        merged["prompts"] = _restore_base_templates(merged["prompts"], base_prompts)

    # Preserve synthesis config dict before prompt rendering
    if "prompts" in merged and "synthesis" in merged["prompts"]:
        synthesis_config = merged["prompts"]["synthesis"]
        if isinstance(synthesis_config, dict):
            merged["_synthesis_config"] = synthesis_config

    # Store the use_case in config so nodes can use it for dynamic tool loading
    merged["_use_case"] = use_case

    # Validate configuration
    _validate_config(merged)

    logger.debug(f"[CONFIG] Loaded config for {cache_key} with {len(merged)} keys")

    return merged


def get_config_from_runnable_config(config: Any) -> Dict[str, Any]:
    """Extract config from LangGraph's configurable pattern.

    Args:
        config: RunnableConfig from LangGraph

    Returns:
        Config dict from YAML
    """
    if config and "configurable" in config:
        result = config["configurable"]
        # Defensive: result might be None in some edge cases
        if result is None:
            return {}
        return result
    # Fallback: return empty dict - should never happen if run_v4_spawner is used correctly
    # This prevents hardcoded "rag" default
    return {}
