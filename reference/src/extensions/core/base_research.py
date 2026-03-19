"""Base Research Extension for Research Strategies.

Defines the contract for research strategy extensions (Podcast, Twitter, etc.).
Each extension provides:
1. Profile - Worker behavior, personalities, prefills
2. Output Strategy - How to format the final result
3. User Variants - User-customizable configurations (frontend editable)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ServiceExtension
from ..content.deep_research.config_diff import merge_configs

logger = logging.getLogger(__name__)


class ResearchExtension(ServiceExtension):
    """
    Base class for research strategy extensions.

    Research extensions are "service" extensions that provide:
    - Profile: How workers behave (personalities, prefills, prompt enhancement)
    - Output Strategy: How to format results (thread format, podcast script, etc.)
    - User Variants: User-created configurations (frontend editable)

    Config Hierarchy (3-level merge):
    1. System base (deep_research/base/config.yaml) - Deep research defaults
    2. Extension base (extension/config.yaml) - Extension-specific config
    3. User variant (user_variants/{user}/{extension}/{variant}.yaml) - User overrides
    """

    def get_variant(
        self,
        variant_name: str = "default",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load config with variant override.

        Args:
            variant_name: Name of variant (e.g., "my_custom", "fast", etc.)
            user_id: Optional user ID for user-specific variants

        Returns:
            Merged config (system base → extension base → system variant → user variant)
        """
        from ...constants import DEEP_RESEARCH_CONFIG_DIR, USER_CONFIG_DIR

        configs_to_load = []
        extension_name = self.config.get('extension', {}).get('name', 'unknown')

        # Level 1: System base (deep research defaults)
        system_base = DEEP_RESEARCH_CONFIG_DIR / "base" / "config.yaml"
        if system_base.exists():
            configs_to_load.append(system_base)
        else:
            # Fallback to root config.yaml
            configs_to_load.append(DEEP_RESEARCH_CONFIG_DIR / "config.yaml")

        # Level 2: Extension base
        configs_to_load.append(self.root_path / "config.yaml")

        # Level 3: System variant (built-in variants in extension's variants/ directory)
        if variant_name != "default":
            system_variant_path = self.root_path / "variants" / f"{variant_name}.yaml"
            if system_variant_path.exists():
                configs_to_load.append(system_variant_path)
                logger.info(f"[{extension_name}] Loading system variant '{variant_name}'")
            else:
                logger.warning(f"[{extension_name}] System variant '{variant_name}' not found at {system_variant_path}")

        # Level 4: User variant (frontend editable, overrides everything)
        if user_id and variant_name != "default":
            user_variant_path = (
                USER_CONFIG_DIR /
                user_id /
                "research_variants" /
                extension_name /
                f"{variant_name}.yaml"
            )
            if user_variant_path.exists():
                configs_to_load.append(user_variant_path)
                logger.info(f"[{extension_name}] Loading user variant '{variant_name}' for {user_id} (overrides system variant)")
            else:
                logger.debug(f"[{extension_name}] User variant '{variant_name}' not found for {user_id}")

        # Merge in order (later overrides earlier)
        merged = {}
        for path in configs_to_load:
            if path.exists():
                import yaml
                with open(path) as f:
                    config = yaml.safe_load(f)
                    merged = merge_configs(merged, config)

        return merged

    def list_variants(self) -> list[dict]:
        """List example variants for this extension.

        Returns:
            List of dicts with variant info: [{name, description, user_created}]
        """
        variants_dir = self.root_path / "variants"
        if not variants_dir.exists():
            return []

        variants = []
        for variant_file in variants_dir.glob("*.yaml"):
            variant_name = variant_file.stem  # filename without .yaml
            if variant_name != "default":
                # Read metadata from file
                import yaml
                try:
                    with open(variant_file) as f:
                        config = yaml.safe_load(f)
                    metadata = config.get('_metadata', {})
                    variants.append({
                        'name': variant_name,
                        'description': metadata.get('description', ''),
                        'user_created': metadata.get('user_created', False)
                    })
                except Exception:
                    variants.append({
                        'name': variant_name,
                        'description': '',
                        'user_created': False
                    })

        return sorted(variants, key=lambda x: x['name'])

    def list_user_variants(self, user_id: str) -> list[dict]:
        """List user-created variants for this extension.

        Args:
            user_id: User ID to get variants for

        Returns:
            List of dicts with variant info: [{name, description, user_created}]
        """
        from ...constants import USER_CONFIG_DIR

        extension_name = self.config.get('extension', {}).get('name', 'unknown')
        user_variants_dir = (
            USER_CONFIG_DIR /
            user_id /
            "research_variants" /
            extension_name
        )

        if not user_variants_dir.exists():
            return []

        variants = []
        for variant_file in user_variants_dir.glob("*.yaml"):
            # Read metadata from file
            import yaml
            try:
                with open(variant_file) as f:
                    config = yaml.safe_load(f)
                metadata = config.get('_metadata', {})
                variants.append({
                    'name': variant_file.stem,
                    'description': metadata.get('description', ''),
                    'user_created': metadata.get('user_created', True)  # Default to True for user variants
                })
            except Exception:
                variants.append({
                    'name': variant_file.stem,
                    'description': '',
                    'user_created': True
                })

        return sorted(variants, key=lambda x: x['name'])

    # ==========================================================================
    # SERVICE LIFECYCLE - No-op implementations (research extensions don't run services)
    # ==========================================================================

    async def initialize(self) -> None:
        """Initialize the extension (no-op for research extensions)."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the extension (no-op for research extensions)."""
        pass

    async def start_service(self) -> None:
        """Start the service daemon (no-op for research extensions)."""
        pass

    async def stop_service(self) -> None:
        """Stop the service daemon (no-op for research extensions)."""
        pass

    # ==========================================================================
    # DEFAULT IMPLEMENTATIONS - Override in subclasses if needed
    # ==========================================================================

    def get_profile(self):
        """Return the Profile class for worker behavior.

        The Profile provides:
        - Personality loading (from YAML)
        - Worker behavior (research_directive, prefill_prompt)
        - Prompt enhancement (personality injection)

        Returns:
            ResearchProfile instance

        Default: Uses BaseResearchProfile (generic worker behavior).
        Override this if your extension needs custom profile behavior.
        """
        from src.extensions.content.deep_research.profiles.base import BaseResearchProfile
        return BaseResearchProfile(self.config)

    def get_output_strategy(self, mode: Optional[str] = None):
        """Return the Output Strategy class for formatting results.

        Args:
            mode: Optional output mode override

        Returns:
            OutputStrategy class (has format() method)

        Default: Uses TextFormatter (plain text output).
        Override this if your extension needs custom output formatting.
        """
        from src.extensions.content.deep_research.graph.output import TextFormatter
        return TextFormatter
