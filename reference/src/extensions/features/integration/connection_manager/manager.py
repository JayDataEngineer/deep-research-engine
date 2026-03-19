"""Connection Manager

Manages multi-endpoint profiles and API connection configurations.
Similar to SillyTavern's connection manager feature.
Refactored to use BaseManager and StorageManager.
"""

import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

from src.extensions.libraries.base.manager_base import BaseItem, BaseManager

logger = logging.getLogger(__name__)


@dataclass
class ConnectionProfile(BaseItem):
    """An API endpoint connection profile."""

    id: str
    name: str
    provider: str  # openai, anthropic, google, etc.
    base_url: str
    api_key: str
    model: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    is_active: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "description": self.description,
            "tags": self.tags,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "settings": self.settings,
        }

        if not include_sensitive:
            # Mask API key for display
            if data.get("api_key"):
                data["api_key"] = "***" + data["api_key"][-4:] if len(data["api_key"]) > 4 else "***"

        return {k: v for k, v in data.items() if v or v == [] or v is False}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectionProfile":
        """Create from dictionary."""
        return cls(**data)


class ConnectionManager(BaseManager):
    """Manages API connection profiles."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the connection manager.

        Args:
            base_dir: Base directory for storage (auto-created if None)
        """
        super().__init__(base_dir=base_dir, feature_name="connections", storage_file="profiles.json")

    def _get_item_class(self):
        """Get the ConnectionProfile item class."""
        return ConnectionProfile

    # ========================================================================
    # Connection profile methods
    # ========================================================================

    def create_profile(
        self,
        name: str,
        provider: str,
        base_url: str,
        api_key: str,
        model: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Optional[ConnectionProfile]:
        """Create a new connection profile.

        Args:
            name: Profile name
            provider: Provider name (openai, anthropic, google, etc.)
            base_url: API base URL
            api_key: API key
            model: Default model
            description: Description
            tags: Optional tags
            settings: Additional settings

        Returns:
            The created ConnectionProfile or None
        """
        profile = ConnectionProfile(
            id=str(uuid.uuid4())[:8],
            name=name,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            description=description,
            tags=tags or [],
            settings=settings or {}
        )

        if self.add_item(profile):
            logger.info(f"Created connection profile: {profile.name}")
            return profile
        return None

    def list_profiles(
        self,
        provider: Optional[str] = None,
        active_only: bool = False
    ) -> List[ConnectionProfile]:
        """List profiles.

        Args:
            provider: Filter by provider
            active_only: Only show active profiles

        Returns:
            List of ConnectionProfiles
        """
        profiles = list(self._items.values())

        if provider:
            profiles = [p for p in profiles if p.provider == provider]

        if active_only:
            profiles = [p for p in profiles if p.is_active]

        return profiles

    def get_profile_by_name(self, name: str) -> Optional[ConnectionProfile]:
        """Get a profile by name.

        Args:
            name: The profile name

        Returns:
            The ConnectionProfile or None
        """
        for profile in self._items.values():
            if profile.name == name:
                return profile
        return None

    def set_active(self, profile_id: str) -> Optional[ConnectionProfile]:
        """Set a profile as active (deactivates others).

        Args:
            profile_id: The profile ID

        Returns:
            The activated ConnectionProfile or None
        """
        profile = self.get_item(profile_id)
        if not profile:
            return None

        # Deactivate all profiles of the same provider
        for p in self._items.values():
            if p.provider == profile.provider:
                p.is_active = False

        # Activate the selected profile
        profile.is_active = True
        self._save()
        logger.info(f"Activated connection profile: {profile.name}")
        return profile

    def get_active_profile(self, provider: str) -> Optional[ConnectionProfile]:
        """Get the active profile for a provider.

        Args:
            provider: The provider name

        Returns:
            The active ConnectionProfile or None
        """
        for profile in self._items.values():
            if profile.provider == provider and profile.is_active:
                return profile
        return None

    def test_connection(self, profile_id: str) -> Dict[str, Any]:
        """Test a connection profile.

        Args:
            profile_id: The profile ID

        Returns:
            Test result dict
        """
        profile = self.get_item(profile_id)
        if not profile:
            return {"success": False, "error": "Profile not found"}

        # Basic validation
        result = {
            "success": True,
            "profile": profile.name,
            "provider": profile.provider,
            "base_url": profile.base_url,
            "tests": {
                "api_key_present": bool(profile.api_key),
                "base_url_valid": bool(profile.base_url.startswith("http")),
                "model_configured": bool(profile.model),
            }
        }

        result["all_pass"] = all(result["tests"].values())

        return result

    def export_profile(self, profile_id: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """Export a profile to a file (without sensitive data).

        Args:
            profile_id: The profile ID
            output_path: Optional output path

        Returns:
            Path to exported file or None
        """
        profile = self.get_item(profile_id)
        if not profile:
            return None

        if output_path is None:
            safe_name = profile.name.replace(" ", "_").replace("/", "_")
            output_path = Path.cwd() / f"connection_{safe_name}.json"

        output_path = Path(output_path)

        # Export without sensitive data
        with open(output_path, 'w') as f:
            json.dump(profile.to_dict(include_sensitive=False), f, indent=2)

        logger.info(f"Exported profile to: {output_path}")
        return output_path

    def get_providers(self) -> List[str]:
        """Get list of unique providers."""
        return sorted(set(p.provider for p in self._items.values()))

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        profiles = list(self._items.values())
        providers = self.get_providers()
        active_count = sum(1 for p in profiles if p.is_active)

        return {
            "total_profiles": len(profiles),
            "providers": len(providers),
            "active_profiles": active_count,
            "provider_list": providers,
            "storage_path": str(self.base_dir),
        }
