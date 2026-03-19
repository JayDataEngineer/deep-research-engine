"""
Extension Compatibility Layer

Provides backward-compatible imports during extension reorganization.
This file can be deleted once the migration is complete.
"""

import sys
from pathlib import Path

# Import path mappings for backward compatibility
# Format: old_path -> new_path
_IMPORT_MAPPINGS = {
    # Text utilities → new locations
    "src.extensions.features.text_utility.bookmarks": "src.extensions.utilities.bookmarks",
    "src.extensions.features.text_utility.tags": "src.extensions.utilities.tags",
    "src.extensions.features.text_utility.backgrounds": "src.extensions.utilities.backgrounds",
    "src.extensions.features.text_utility.quick_replies": "src.extensions.utilities.quick_replies",

    # Services → integrations
    "src.extensions.services.tts": "src.extensions.integrations.tts",
    "src.extensions.services.comfyui": "src.extensions.integrations.comfyui",
    "src.extensions.services.research": "src.extensions.integrations.research",

    # Research → content
    "src.extensions.features.text_utility.deep_research": "src.extensions.content.deep_research",

    # Audio producer consolidation
    "src.extensions.libraries.audio_producer": "src.extensions.content.audio_producer",

    # Brain modes → modes
    "src.extensions.features.brain_modes.deep_agent": "src.extensions.modes.deep_agent",
    "src.extensions.features.brain_modes.tool_call": "src.extensions.modes.tool_call",
}


def install_compat_hooks():
    """
    Install import hooks for backward compatibility.

    Call this during application startup before importing any extensions.
    """
    class ExtensionCompatFinder:
        """Import hook that redirects old extension paths to new ones."""

        def find_spec(self, fullname, path, target=None):
            if fullname not in _IMPORT_MAPPINGS:
                return None

            new_name = _IMPORT_MAPPINGS[fullname]

            # Try to import from the new location
            try:
                import importlib.util
                spec = importlib.util.find_spec(new_name)
                if spec:
                    return spec
            except (ImportError, ModuleNotFoundError):
                pass

            # Return None to let Python continue searching
            return None

    # Install the import hook
    sys.meta_path.insert(0, ExtensionCompatFinder())


# Auto-install on import (for convenience)
install_compat_hooks()
