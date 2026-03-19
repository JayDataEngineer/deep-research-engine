"""
Capability Query

Generates capability information for the frontend.
"""

from typing import Dict, Any, List, Tuple
from ..base import ExtensionStatus


class CapabilityQuery:
    """
    Queries extensions for frontend capabilities.
    """

    def __init__(self, registry):
        """
        Initialize the capability query.

        Args:
            registry: Extension registry to query
        """
        self.registry = registry

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get all capabilities for the frontend.

        Returns:
            Dict with modes, tools, tts, features, and extensions
        """
        modes = []
        tools = {}
        tts = {}
        features = set()
        extension_list = []

        for name, ext in self.registry.get_all().items():
            # Include both LOADED and REGISTERED extensions
            # REGISTERED extensions are subprocess-only (like TTS) that are available but not loaded
            if ext.status not in (ExtensionStatus.LOADED, ExtensionStatus.REGISTERED):
                continue

            category = ext.category
            ui_config = ext.get_ui_config()

            ext_info = {
                "name": name,
                "display_name": ext.config.get('extension', {}).get('display_name', name),
                "category": category,
                "version": ext.config.get('extension', {}).get('version', 'unknown'),
                "provides_features": ui_config.get('provides_features', []),
                "voice_config": ui_config.get('voice_config', {}),
                "settings_schema": ui_config.get('settings_schema', []),
                "sidebar_component": ui_config.get('sidebar_component'),
                "message_style": ui_config.get('message_style'),
                "description": ext.config.get('extension', {}).get('description', ''),
                "status": ext.status.value,  # Include status so frontend knows if extension is loaded
            }
            extension_list.append(ext_info)

            # Categorize
            if category == 'mode' or category == 'text_modes':
                modes.append(name)
            elif category == 'tool' or category == 'job':
                tools[name] = True
            elif category == 'tts':
                tts[name] = True

            # Also check for TTS model extensions (subprocess_only services with 'tts' tag)
            # This filters to only the actual TTS model services, not feature extensions
            ext_tags = ext.config.get('tags', []) or ext.config.get('extension', {}).get('tags', [])
            is_subprocess_only = ext.config.get('extension', {}).get('subprocess_only', False)

            # Only include subprocess-only TTS services (actual TTS models)
            if 'tts' in ext_tags and is_subprocess_only and name not in tts:
                tts[name] = True

            # Collect features (only from loaded extensions)
            if ext.status == ExtensionStatus.LOADED:
                features.update(ui_config.get('provides_features', []))

        return {
            "modes": modes,
            "tools": tools,
            "tts": tts,
            "features": list(features),
            "extensions": extension_list,
        }

    def get_all_router_info(self) -> List[Dict[str, Any]]:
        """
        Get router metadata from all extensions for frontend discovery.

        This uses the new get_router_info() method from extensions
        to provide API information for dynamic frontend routing.

        Returns:
            List of router info dicts with prefix, name, tags, description
        """
        router_info_list = []
        for name, ext in self.registry.get_all().items():
            if ext.status not in (ExtensionStatus.LOADED, ExtensionStatus.REGISTERED):
                continue

            # Get router info from extension
            if hasattr(ext, 'get_router_info'):
                try:
                    info = ext.get_router_info()
                    if info:
                        router_info_list.append({
                            **info,
                            "extension_name": name,
                            "extension_category": ext.category,
                        })
                except Exception as e:
                    print(f"[CapabilityQuery] Error getting router info from {name}: {e}")

        return router_info_list

    def get_extensions_by_feature(self, feature: str) -> List[str]:
        """
        Get extensions that provide a specific feature.

        Args:
            feature: Feature name to look for

        Returns:
            List of extension names providing the feature
        """
        providers = []
        for name, ext in self.registry.get_all().items():
            if ext.status != ExtensionStatus.LOADED:
                continue

            ui_config = ext.get_ui_config()
            if feature in ui_config.get('provides_features', []):
                providers.append(name)

        return providers

    def get_extension_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed info about a specific extension.

        Args:
            name: Extension name

        Returns:
            Extension info dict or empty dict if not found
        """
        ext = self.registry.get_with_fallbacks(name)
        if not ext:
            return {}

        return {
            "name": ext.name,
            "display_name": ext.config.get('extension', {}).get('display_name', ext.name),
            "category": ext.category,
            "status": ext.status.value,
            "version": ext.config.get('extension', {}).get('version', 'unknown'),
            "description": ext.config.get('extension', {}).get('description', ''),
            "author": ext.config.get('extension', {}).get('author', ''),
            "capabilities": ext.get_capabilities(),
            "ui_config": ext.get_ui_config(),
            "root_path": str(ext.root_path),
        }

    def get_categories(self) -> List[str]:
        """
        Get all unique categories.

        Returns:
            List of category names
        """
        categories = set()
        for ext in self.registry.get_all().values():
            categories.add(ext.category)
        return sorted(categories)
