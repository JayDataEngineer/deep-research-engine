"""Profiles module for deep research extension.

This module provides profile loading functionality for different research use cases.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Profile:
    """Research profile with configuration."""
    name: str
    use_case: str
    variant: str
    config: Dict[str, Any]

    def __str__(self) -> str:
        return f"Profile({self.name}, use_case={self.use_case}, variant={self.variant})"


def load_profile(
    use_case: str = "general",
    variant: str = "default",
    config: Optional[Dict[str, Any]] = None
) -> Profile:
    """Load a research profile for the given use case and variant.

    Args:
        use_case: Research domain (general, rag, podcast, trading, osint, news, content)
        variant: Config variant (default, or custom)
        config: Optional yaml_config to merge with profile

    Returns:
        Profile object with name and configuration
    """
    # For now, return a simple default profile
    # This can be extended to load from YAML files in the future
    profile_name = f"{use_case}_{variant}"

    return Profile(
        name=profile_name,
        use_case=use_case,
        variant=variant,
        config=config or {}
    )


# Re-export commonly used functions
from .loader import (
    get_profile_loader,
    load_variant_config,
    get_active_profile,
    get_boss_strategy_from_profile,
    get_critic_criteria_from_profile,
    validate_content_against_profile,
)

__all__ = [
    "Profile",
    "load_profile",
    "get_profile_loader",
    "load_variant_config",
    "get_active_profile",
    "get_boss_strategy_from_profile",
    "get_critic_criteria_from_profile",
    "validate_content_against_profile",
]
