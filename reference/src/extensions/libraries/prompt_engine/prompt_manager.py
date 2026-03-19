"""
SillyTavern-compatible Prompt Manager

This module provides prompt management functionality compatible with
SillyTavern's PromptManager for:
- Prompt ordering per character
- Prompt injection at depth (absolute positioning)
- Import/export in ST format
- Prompt collection building with filtering

Architecture based on:
https://github.com/SillyTavern/SillyTavern/blob/develop/public/scripts/PromptManager.js
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Literal
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from .prompt_models import (
    Prompt,
    PromptOrderEntry,
    PromptCollection,
    PromptExport,
    PromptExportData,
    CharacterPromptConfig,
    InjectionPosition,
    PromptRole,
    get_default_prompts,
    get_default_prompt_order,
    SYSTEM_PROMPT_IDS,
    MARKER_PROMPT_IDS,
)


logger = logging.getLogger(__name__)


class PromptManager:
    """
    SillyTavern-compatible Prompt Manager.

    Manages prompt ordering, injection, and export/import for characters.
    Compatible with SillyTavern's prompt export format.
    """

    # Strategy types for prompt ordering
    STRATEGY_GLOBAL = "global"
    STRATEGY_CHARACTER = "character"

    # Dummy character ID for global prompts
    DUMMY_CHARACTER_ID = "global_default"

    def __init__(
        self,
        strategy: Literal["global", "character"] = "global",
        storage_dir: Optional[Path] = None,
    ):
        """
        Initialize the PromptManager.

        Args:
            strategy: "global" = one order for all, "character" = per-character orders
            storage_dir: Optional directory to persist prompt configurations
        """
        self.strategy = strategy
        self.storage_dir = Path(storage_dir) if storage_dir else None

        # All available prompts (indexed by identifier)
        self.prompts: Dict[str, Prompt] = {}

        # Prompt orders per character (character_id -> list of PromptOrderEntry)
        self.prompt_orders: Dict[str, List[PromptOrderEntry]] = {}

        # Character configs (metadata)
        self.character_configs: Dict[str, CharacterPromptConfig] = {}

        # Initialize with default prompts
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Initialize with default SillyTavern prompts."""
        # Add default prompts
        for prompt in get_default_prompts():
            self.prompts[prompt.identifier] = prompt

        # Add default prompt order for global strategy
        if self.strategy == self.STRATEGY_GLOBAL:
            self.prompt_orders[self.DUMMY_CHARACTER_ID] = get_default_prompt_order()

        logger.info(f"[PromptManager] Initialized with {len(self.prompts)} default prompts")

    # ==========================================================================
    # PROMPT CRUD
    # ==========================================================================

    def get_prompt(self, identifier: str) -> Optional[Prompt]:
        """Get a prompt by identifier."""
        return self.prompts.get(identifier)

    def add_prompt(self, prompt: Prompt) -> bool:
        """Add or update a prompt."""
        if not prompt.identifier:
            return False
        self.prompts[prompt.identifier] = prompt
        logger.debug(f"[PromptManager] Added prompt: {prompt.identifier}")
        return True

    def remove_prompt(self, identifier: str) -> bool:
        """Remove a prompt (only if not a system prompt)."""
        prompt = self.get_prompt(identifier)
        if not prompt:
            return False
        if prompt.system_prompt:
            logger.warning(f"[PromptManager] Cannot remove system prompt: {identifier}")
            return False
        del self.prompts[identifier]
        logger.debug(f"[PromptManager] Removed prompt: {identifier}")
        return True

    def list_prompts(
        self,
        include_system: bool = True,
        include_markers: bool = True,
    ) -> List[Prompt]:
        """List all prompts with optional filtering."""
        prompts = list(self.prompts.values())

        if not include_system:
            prompts = [p for p in prompts if not p.system_prompt]

        if not include_markers:
            prompts = [p for p in prompts if not p.marker]

        return prompts

    # ==========================================================================
    # PROMPT ORDER MANAGEMENT
    # ==========================================================================

    def get_prompt_order(
        self,
        character_id: str,
    ) -> List[PromptOrderEntry]:
        """Get prompt order for a character."""
        if self.strategy == self.STRATEGY_GLOBAL:
            # Global strategy - return same order for all
            return self.prompt_orders.get(self.DUMMY_CHARACTER_ID, [])

        # Character strategy - get character-specific order
        return self.prompt_orders.get(character_id, [])

    def set_prompt_order(
        self,
        character_id: str,
        order: List[PromptOrderEntry],
    ) -> bool:
        """Set prompt order for a character."""
        # Validate all identifiers exist
        for entry in order:
            if entry.identifier not in self.prompts:
                logger.warning(f"[PromptManager] Unknown prompt identifier: {entry.identifier}")
                return False

        if self.strategy == self.STRATEGY_GLOBAL:
            # Global strategy - always use dummy ID
            self.prompt_orders[self.DUMMY_CHARACTER_ID] = order
        else:
            self.prompt_orders[character_id] = order

        logger.debug(f"[PromptManager] Set prompt order for {character_id}: {len(order)} entries")
        return True

    def update_prompt_order_entry(
        self,
        character_id: str,
        identifier: str,
        enabled: Optional[bool] = None,
    ) -> bool:
        """Update a single entry in the prompt order."""
        order = self.get_prompt_order(character_id)

        # Find the entry
        for entry in order:
            if entry.identifier == identifier:
                if enabled is not None:
                    entry.enabled = enabled
                return True

        # Entry not found - add it if prompt exists
        if identifier in self.prompts:
            if enabled is None:
                enabled = True
            order.append(PromptOrderEntry(identifier=identifier, enabled=enabled))
            return self.set_prompt_order(character_id, order)

        return False

    def append_prompt_to_order(
        self,
        character_id: str,
        identifier: str,
        enabled: bool = True,
    ) -> bool:
        """Append a prompt to the character's order (if not already present)."""
        if identifier not in self.prompts:
            return False

        order = self.get_prompt_order(character_id)

        # Check if already present
        for entry in order:
            if entry.identifier == identifier:
                return True  # Already present

        # Append to beginning (like ST behavior)
        order.insert(0, PromptOrderEntry(identifier=identifier, enabled=enabled))
        return self.set_prompt_order(character_id, order)

    def detach_prompt_from_order(
        self,
        character_id: str,
        identifier: str,
    ) -> bool:
        """Remove a prompt from the character's order."""
        order = self.get_prompt_order(character_id)

        # Find and remove
        new_order = [e for e in order if e.identifier != identifier]

        if len(new_order) == len(order):
            return False  # Not found

        return self.set_prompt_order(character_id, new_order)

    def reset_prompt_order(
        self,
        character_id: str,
    ) -> bool:
        """Reset prompt order to defaults."""
        default_order = get_default_prompt_order()
        return self.set_prompt_order(character_id, default_order)

    # ==========================================================================
    # PROMPT COLLECTION BUILDING
    # ==========================================================================

    def get_prompt_collection(
        self,
        character_id: str,
        generation_type: str = "normal",
        only_enabled: bool = True,
    ) -> PromptCollection:
        """
        Build a prompt collection for injection.

        This matches SillyTavern's getPromptCollection() method.

        Args:
            character_id: Character to get prompts for
            generation_type: Type of generation (normal, continue, quiet)
            only_enabled: Only include enabled prompts

        Returns:
            PromptCollection with filtered prompts
        """
        order = self.get_prompt_order(character_id)
        collection = PromptCollection()

        for entry in order:
            # Skip disabled prompts
            if only_enabled and not entry.enabled:
                continue

            prompt = self.get_prompt(entry.identifier)
            if not prompt:
                logger.warning(f"[PromptManager] Prompt not found: {entry.identifier}")
                continue

            # Check trigger
            if prompt.injection_trigger:
                if generation_type not in prompt.injection_trigger:
                    continue

            collection.prompts.append(prompt)

        logger.debug(
            f"[PromptManager] Built collection for {character_id}: "
            f"{len(collection.prompts)} prompts (gen_type={generation_type})"
        )

        return collection

    def build_injection_context(
        self,
        character_id: str,
        character_data: Optional[Dict[str, Any]] = None,
        persona_data: Optional[Dict[str, Any]] = None,
        generation_type: str = "normal",
    ) -> List[Dict[str, Any]]:
        """
        Build prompts ready for LLM injection with variable substitution.

        This processes markers, substitutes variables, and orders by
        injection_depth and injection_order.

        Args:
            character_id: Character to build prompts for
            character_data: Character data for markers (description, personality, scenario)
            persona_data: User persona data
            generation_type: Type of generation

        Returns:
            List of prompt dicts ready for LLM (with role, content, depth)
        """
        collection = self.get_prompt_collection(character_id, generation_type)

        # Separate relative and absolute injection prompts
        relative_prompts: List[Prompt] = []
        absolute_prompts: List[Prompt] = []

        for prompt in collection.prompts:
            if prompt.injection_position == InjectionPosition.ABSOLUTE:
                absolute_prompts.append(prompt)
            else:
                relative_prompts.append(prompt)

        # Sort absolute prompts by (depth, order)
        absolute_prompts.sort(key=lambda p: (p.injection_depth, p.injection_order))

        # Build output with marker resolution
        result = []

        # Add relative prompts first (in order)
        for prompt in relative_prompts:
            content = self._resolve_prompt_content(
                prompt,
                character_data,
                persona_data,
            )
            if content:
                result.append({
                    "identifier": prompt.identifier,
                    "role": prompt.role,
                    "content": content,
                    "depth": -1,  # Relative = before chat
                    "marker": prompt.marker,
                })

        # Add absolute prompts at depth
        for prompt in absolute_prompts:
            if prompt.marker:
                # Markers are placeholders - resolve to actual content
                content = self._resolve_marker_content(
                    prompt.identifier,
                    character_data,
                    persona_data,
                )
            else:
                content = prompt.content

            if content:
                result.append({
                    "identifier": prompt.identifier,
                    "role": prompt.role,
                    "content": content,
                    "depth": prompt.injection_depth,
                    "marker": prompt.marker,
                })

        logger.debug(
            f"[PromptManager] Built injection context: "
            f"{len(result)} prompts ({len(relative_prompts)} relative, {len(absolute_prompts)} absolute)"
        )

        return result

    def _resolve_prompt_content(
        self,
        prompt: Prompt,
        character_data: Optional[Dict[str, Any]],
        persona_data: Optional[Dict[str, Any]],
    ) -> str:
        """Resolve prompt content with variable substitution."""
        content = prompt.content

        # Marker prompts get resolved content
        if prompt.marker:
            content = self._resolve_marker_content(
                prompt.identifier,
                character_data,
                persona_data,
            )

        # Substitute common variables
        # {{char}}, {{user}}, etc.
        if character_data:
            char_name = character_data.get("name", "Character")
            content = content.replace("{{char}}", char_name)
            content = content.replace("{{charIfNotGroup}}", char_name)

        if persona_data:
            user_name = persona_data.get("name", "You")
            content = content.replace("{{user}}", user_name)

        return content

    def _resolve_marker_content(
        self,
        identifier: str,
        character_data: Optional[Dict[str, Any]],
        persona_data: Optional[Dict[str, Any]],
    ) -> str:
        """Resolve marker prompt to actual content."""
        if identifier == "charDescription" and character_data:
            return character_data.get("description", "")
        elif identifier == "charPersonality" and character_data:
            return character_data.get("personality", "")
        elif identifier == "scenario" and character_data:
            return character_data.get("scenario", "")
        elif identifier == "personaDescription" and persona_data:
            return persona_data.get("description", "")
        elif identifier == "chatHistory":
            return "<!-- Chat history inserted here -->"
        elif identifier == "dialogueExamples":
            return "<!-- Dialogue examples would go here -->"
        elif identifier == "worldInfoBefore":
            return "<!-- World info (before character) -->"
        elif identifier == "worldInfoAfter":
            return "<!-- World info (after character) -->"
        else:
            return ""

    # ==========================================================================
    # EXPORT/IMPORT (SillyTavern-compatible)
    # ==========================================================================

    def export_full(self) -> PromptExport:
        """Export all prompts (full export)."""
        # Get all user prompts (non-system, non-marker)
        user_prompts = [
            p for p in self.prompts.values()
            if not p.system_prompt and not p.marker
        ]

        # Get prompt order
        if self.strategy == self.STRATEGY_GLOBAL:
            prompt_order = self.get_prompt_order(self.DUMMY_CHARACTER_ID)
        else:
            prompt_order = []

        return PromptExport(
            version=1,
            type="full",
            data=PromptExportData(
                prompts=user_prompts,
                prompt_order=prompt_order,
            ),
        )

    def export_character(self, character_id: str, character_name: str) -> PromptExport:
        """Export prompts and order for a specific character."""
        # Get character-specific prompts
        order = self.get_prompt_order(character_id)

        # Include prompts in the order
        character_prompts = []
        for entry in order:
            prompt = self.get_prompt(entry.identifier)
            if prompt and not prompt.marker:
                character_prompts.append(prompt)

        return PromptExport(
            version=1,
            type="character",
            data=PromptExportData(
                prompts=character_prompts,
                prompt_order=order,
            ),
        )

    def import_export(self, export_data: Dict[str, Any]) -> bool:
        """
        Import prompts from SillyTavern export format.

        Args:
            export_data: Dict from ST export file

        Returns:
            True if import succeeded
        """
        try:
            # Validate structure
            if "version" not in export_data or "data" not in export_data:
                logger.error("[PromptManager] Invalid export format")
                return False

            version = export_data.get("version")
            if version != 1:
                logger.warning(f"[PromptManager] Unknown export version: {version}")

            export_type = export_data.get("type", "full")
            data = export_data["data"]

            # Import prompts
            imported_count = 0
            for prompt_data in data.get("prompts", []):
                prompt = Prompt(**prompt_data)
                # Generate new identifier if conflicts
                identifier = prompt.identifier
                if identifier in self.prompts and not self.prompts[identifier].system_prompt:
                    # Add suffix for user prompts
                    identifier = f"{identifier}_{uuid.uuid4().hex[:6]}"
                    prompt.identifier = identifier

                self.add_prompt(prompt)
                imported_count += 1

            logger.info(f"[PromptManager] Imported {imported_count} prompts")

            # Import prompt order if present
            if "prompt_order" in data:
                order_entries = [
                    PromptOrderEntry(**e) for e in data["prompt_order"]
                ]

                if export_type == "full" and self.strategy == self.STRATEGY_GLOBAL:
                    self.set_prompt_order(self.DUMMY_CHARACTER_ID, order_entries)
                elif export_type == "character":
                    # Store for later character assignment
                    pass  # Would need character_id in export

            return True

        except Exception as e:
            logger.error(f"[PromptManager] Import failed: {e}")
            return False

    # ==========================================================================
    # PERSISTENCE
    # ==========================================================================

    def save_to_file(self, path: Optional[Path] = None) -> bool:
        """Save prompt configuration to file."""
        import json

        if path is None:
            if self.storage_dir is None:
                return False
            path = self.storage_dir / "prompts.json"

        path = Path(path)

        data = {
            "strategy": self.strategy,
            "prompts": [p.model_dump() for p in self.prompts.values()],
            "prompt_orders": {
                k: [e.model_dump() for e in v]
                for k, v in self.prompt_orders.items()
            },
        }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"[PromptManager] Saved prompts to {path}")
            return True
        except Exception as e:
            logger.error(f"[PromptManager] Failed to save: {e}")
            return False

    def load_from_file(self, path: Path) -> bool:
        """Load prompt configuration from file."""
        import json

        path = Path(path)

        if not path.exists():
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Load prompts
            for prompt_data in data.get("prompts", []):
                prompt = Prompt(**prompt_data)
                self.prompts[prompt.identifier] = prompt

            # Load orders
            for char_id, order_data in data.get("prompt_orders", {}).items():
                self.prompt_orders[char_id] = [
                    PromptOrderEntry(**e) for e in order_data
                ]

            logger.info(f"[PromptManager] Loaded prompts from {path}")
            return True

        except Exception as e:
            logger.error(f"[PromptManager] Failed to load: {e}")
            return False


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_prompt_manager_instance: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the global PromptManager instance."""
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        # Get storage dir from config if available
        from src.config import settings
        storage_dir = getattr(settings, "prompt_storage_dir", None)

        _prompt_manager_instance = PromptManager(
            strategy="global",
            storage_dir=storage_dir,
        )
    return _prompt_manager_instance


def create_prompt_manager(
    strategy: Literal["global", "character"] = "global",
    storage_dir: Optional[Path] = None,
) -> PromptManager:
    """Create a new PromptManager instance."""
    return PromptManager(strategy=strategy, storage_dir=storage_dir)
