"""
Presets System

Save and load complete generation presets including:
- Generation parameters
- Chat templates
- Context templates
- Special commands settings
- And more
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import json
import yaml


class PresetCategory(Enum):
    """Categories of presets."""

    GENERATION = "generation"  # Model generation parameters
    FORMAT = "format"  # Chat/context templates
    CHARACTER = "character"  # Character-specific settings
    AUTHOR_NOTE = "author_note"  # Author's note settings
    COMPLETE = "complete"  # Full preset (all categories)


@dataclass
class GenerationPreset:
    """Generation parameters preset."""

    name: str
    description: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_tokens: int = 500
    stop_sequences: List[str] = field(default_factory=list)
    bias: Dict[str, float] = field(default_factory=dict)
    # Mirostat settings
    mirostat_mode: int = 0  # 0=disabled, 1=legacy, 2=v2
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    # Grammar
    grammar: str = ""  # GBNF grammar string
    json_schema: bool = False  # Force JSON output


@dataclass
class FormatPreset:
    """Format preset (chat and context templates)."""

    name: str
    description: str = ""
    chat_template: str = "auto"  # chatml, alpaca, vicuna, etc.
    context_template: str = "plain"  # wpp, stdp, boostyle, etc.
    instruct_template: str = ""  # Optional instruction template
    # Formatting options
    skip_examples: bool = False
    include_names: bool = True
    system_prompt_prefix: str = ""
    system_prompt_suffix: str = ""


@dataclass
class AuthorNotePreset:
    """Author's note preset."""

    name: str
    description: str = ""
    content: str = ""
    position: str = "after_chat"  # before_chat, after_chat, at_depth
    depth: int = 0
    interval: int = 1  # Insert every N turns
    enabled: bool = True


@dataclass
class CharacterPreset:
    """Character-specific preset."""

    name: str
    description: str = ""
    # Talkativeness affects response length
    talkativeness: float = 0.5  # 0.0=brief, 1.0=verbose
    # Bias settings
    biases: Dict[str, float] = field(default_factory=dict)
    # Special behavior
    ooc_handling: str = "pause"  # pause, ignore, respond
    story_mode_handling: str = "narrate"  # narrate, execute, ignore


@dataclass
class CompletePreset:
    """Complete preset combining all categories."""

    name: str
    description: str = ""
    category: str = "complete"
    tags: List[str] = field(default_factory=list)

    generation: Optional[GenerationPreset] = None
    format: Optional[FormatPreset] = None
    author_note: Optional[AuthorNotePreset] = None
    character: Optional[CharacterPreset] = None

    # Additional metadata
    creator: str = ""
    version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        from datetime import datetime
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()


class PresetManager:
    """
    Manage generation presets.

    Presets can be saved, loaded, and applied to configure
    the companion brain behavior.
    """

    def __init__(self, preset_dir: Optional[Path] = None):
        """
        Initialize preset manager.

        Args:
            preset_dir: Directory to save/load presets from
        """
        self.preset_dir = Path(preset_dir) if preset_dir else None
        self.presets: Dict[str, CompletePreset] = {}

        # Add default presets
        self._add_default_presets()

    def _add_default_presets(self) -> None:
        """Add default generation presets."""
        # Creative preset
        creative = CompletePreset(
            name="Creative",
            description="High creativity, longer responses",
            tags=["creative", "roleplay"],
            generation=GenerationPreset(
                name="Creative",
                temperature=1.0,
                top_p=0.95,
                repetition_penalty=1.0,
                max_tokens=800,
            ),
        )
        self.presets["creative"] = creative

        # Balanced preset
        balanced = CompletePreset(
            name="Balanced",
            description="Balanced creativity and coherence",
            tags=["balanced", "default"],
            generation=GenerationPreset(
                name="Balanced",
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                max_tokens=500,
            ),
        )
        self.presets["balanced"] = balanced

        # Precise preset
        precise = CompletePreset(
            name="Precise",
            description="Low temperature, focused responses",
            tags=["precise", "factual"],
            generation=GenerationPreset(
                name="Precise",
                temperature=0.3,
                top_p=0.8,
                top_k=40,
                repetition_penalty=1.15,
                max_tokens=300,
            ),
        )
        self.presets["precise"] = precise

        # Roleplay preset
        roleplay = CompletePreset(
            name="Roleplay",
            description="Optimized for roleplay scenarios",
            tags=["roleplay", "rp"],
            generation=GenerationPreset(
                name="Roleplay",
                temperature=0.85,
                top_p=0.9,
                repetition_penalty=1.05,
                max_tokens=600,
            ),
            format=FormatPreset(
                name="RP Format",
                chat_template="chatml",
                context_template="wpp",
            ),
            character=CharacterPreset(
                name="RP Character",
                talkativeness=0.7,
                ooc_handling="pause",
            ),
        )
        self.presets["roleplay"] = roleplay

        # Assistant preset
        assistant = CompletePreset(
            name="Assistant",
            description="Helpful assistant mode",
            tags=["assistant", "helpful"],
            generation=GenerationPreset(
                name="Assistant",
                temperature=0.5,
                top_p=0.85,
                repetition_penalty=1.0,
                max_tokens=400,
            ),
            format=FormatPreset(
                name="Assistant Format",
                chat_template="chatml",
                context_template="plain",
            ),
        )
        self.presets["assistant"] = assistant

    def register_preset(self, preset: CompletePreset) -> None:
        """Register a preset."""
        self.presets[preset.name.lower()] = preset

    def get_preset(self, name: str) -> Optional[CompletePreset]:
        """Get a preset by name."""
        return self.presets.get(name.lower())

    def list_presets(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[CompletePreset]:
        """
        List presets with optional filtering.

        Args:
            category: Filter by category
            tags: Filter by tags (any match)

        Returns:
            List of matching presets
        """
        presets = list(self.presets.values())

        if category:
            presets = [p for p in presets if p.category == category]

        if tags:
            presets = [
                p for p in presets
                if any(tag in p.tags for tag in tags)
            ]

        return presets

    def apply_preset(
        self,
        name: str,
        config: Any,
    ) -> bool:
        """
        Apply a preset to a configuration object.

        Args:
            name: Preset name
            config: Configuration object to update

        Returns:
            True if applied successfully
        """
        preset = self.get_preset(name)
        if not preset:
            return False

        # Apply generation settings
        if preset.generation and hasattr(config, 'temperature'):
            config.temperature = preset.generation.temperature
            config.top_p = preset.generation.top_p
            config.max_tokens = preset.generation.max_tokens
            # Add more as needed

        # Apply format settings
        if preset.format and hasattr(config, 'chat_template'):
            config.chat_template = preset.format.chat_template
            config.context_template = preset.format.context_template

        return True

    def save_preset(
        self,
        preset: CompletePreset,
        path: Optional[Path] = None,
    ) -> bool:
        """
        Save a preset to file.

        Args:
            preset: Preset to save
            path: Optional path (uses preset_dir if not provided)

        Returns:
            True if saved successfully
        """
        if path is None:
            if self.preset_dir is None:
                return False
            path = self.preset_dir / f"{preset.name.lower()}.yaml"

        path = Path(path)

        # Convert to dict
        data = asdict(preset)

        # Save based on extension
        if path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        elif path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            return False

        return True

    def load_preset(self, path: Path) -> Optional[CompletePreset]:
        """
        Load a preset from file.

        Args:
            path: Path to preset file

        Returns:
            Loaded preset, or None if failed
        """
        path = Path(path)

        if not path.exists():
            return None

        # Load based on extension
        if path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
        else:
            return None

        # Reconstruct nested objects
        generation_data = data.pop("generation", None)
        format_data = data.pop("format", None)
        author_note_data = data.pop("author_note", None)
        character_data = data.pop("character", None)

        generation = GenerationPreset(**generation_data) if generation_data else None
        format_preset = FormatPreset(**format_data) if format_data else None
        author_note = AuthorNotePreset(**author_note_data) if author_note_data else None
        character = CharacterPreset(**character_data) if character_data else None

        preset = CompletePreset(
            generation=generation,
            format=format_preset,
            author_note=author_note,
            character=character,
            **data,
        )

        # Register it
        self.register_preset(preset)

        return preset

    def export_to_dict(self, name: str) -> Optional[Dict[str, Any]]:
        """Export a preset to dictionary."""
        preset = self.get_preset(name)
        if not preset:
            return None
        return asdict(preset)

    def import_from_dict(self, data: Dict[str, Any]) -> Optional[CompletePreset]:
        """Import a preset from dictionary."""
        try:
            generation_data = data.pop("generation", None)
            format_data = data.pop("format", None)
            author_note_data = data.pop("author_note", None)
            character_data = data.pop("character", None)

            generation = GenerationPreset(**generation_data) if generation_data else None
            format_preset = FormatPreset(**format_data) if format_data else None
            author_note = AuthorNotePreset(**author_note_data) if author_note_data else None
            character = CharacterPreset(**character_data) if character_data else None

            preset = CompletePreset(
                generation=generation,
                format=format_preset,
                author_note=author_note,
                character=character,
                **data,
            )

            self.register_preset(preset)
            return preset
        except Exception:
            return None


def create_preset_manager(preset_dir: Optional[Path] = None) -> PresetManager:
    """Factory function to create PresetManager."""
    return PresetManager(preset_dir=preset_dir)


def create_generation_preset(
    name: str,
    temperature: float = 0.7,
    **kwargs
) -> GenerationPreset:
    """Factory function to create GenerationPreset."""
    return GenerationPreset(name=name, temperature=temperature, **kwargs)


def create_format_preset(
    name: str,
    chat_template: str = "auto",
    **kwargs
) -> FormatPreset:
    """Factory function to create FormatPreset."""
    return FormatPreset(name=name, chat_template=chat_template, **kwargs)
