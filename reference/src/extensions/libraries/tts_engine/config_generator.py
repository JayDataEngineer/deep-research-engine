"""
TTS Config Generator - Auto-generate ExtensionConfig from config.yaml.

This module reads extension config.yaml files and auto-generates ExtensionConfig
objects, eliminating the need for manual config.py files in each extension.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from src.extensions.libraries.tts_engine.api.schemas.tts import ExtensionConfig, ExtensionParameter

logger = logging.getLogger(__name__)


# Common parameter definitions that can be reused across extensions
COMMON_PARAMETERS = {
    "speed": ExtensionParameter(
        name="speed",
        type="range",
        label="Speed",
        description="Speech speed multiplier (0.5-2.0, 1.0 = normal)",
        default=1.0,
        min_value=0.5,
        max_value=2.0,
    ),
    "pitch": ExtensionParameter(
        name="pitch",
        type="range",
        label="Pitch",
        description="Pitch adjustment (0.5-2.0, 1.0 = normal)",
        default=1.0,
        min_value=0.5,
        max_value=2.0,
    ),
    "temperature": ExtensionParameter(
        name="temperature",
        type="range",
        label="Temperature",
        description="Sampling temperature (lower = more deterministic)",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
    ),
    "top_p": ExtensionParameter(
        name="top_p",
        type="range",
        label="Top-P",
        description="Nucleus sampling threshold",
        default=0.9,
        min_value=0.0,
        max_value=1.0,
    ),
    "top_k": ExtensionParameter(
        name="top_k",
        type="number",
        label="Top-K",
        description="Top-K sampling parameter",
        default=50,
        min_value=1,
        max_value=100,
    ),
}


class TTSConfigGenerator:
    """
    Generator for creating ExtensionConfig from config.yaml files.

    Eliminates the need for manual config.py files by auto-generating
    configuration from the extension's config.yaml metadata.
    """

    @staticmethod
    def generate_from_yaml(ext_dir: Path) -> ExtensionConfig:
        """
        Generate ExtensionConfig from an extension's config.yaml file.

        Args:
            ext_dir: Path to the extension directory (e.g., /path/to/inference/kokoro)

        Returns:
            ExtensionConfig object with auto-generated parameters
        """
        config_file = ext_dir / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"No config.yaml found in {ext_dir}")

        with open(config_file) as f:
            yaml_config = yaml.safe_load(f)

        ext_config = yaml_config.get("extension", {})
        name = ext_config.get("name")
        if not name:
            raise ValueError(f"No extension name in config.yaml")

        # Build base config
        extension_config = ExtensionConfig(
            extension_name=name,
            display_name=ext_config.get("display_name", name),
            category=ext_config.get("category", "tts"),
            description=ext_config.get("description", f"{name} TTS extension"),
        )

        # Check if extension requires audio sample (voice cloning)
        if ext_config.get("requires_audio_sample") or ext_config.get("supports_voice_cloning"):
            extension_config.requires_audio_sample = True
        if ext_config.get("supports_emotion_tagging"):
            extension_config.supports_emotion_tagging = True

        # Auto-generate parameters from voices section
        voices_config = yaml_config.get("voices", [])
        if voices_config:
            # Generate voice selection parameter from voices config
            voice_options = [v.get("id", v.get("name", "")) for v in voices_config if isinstance(v, dict)]
            if voice_options:
                extension_config.parameters.append(
                    ExtensionParameter(
                        name="voice",
                        type="select",
                        label="Voice",
                        description="Select voice for synthesis",
                        default=voice_options[0],
                        options=voice_options,
                    )
                )

        # Add common parameters if they make sense for this extension
        # (Skip for espeak_ng which has its own complex parameter set)
        if name != "espeak_ng":
            # Most TTS extensions support speed and pitch
            extension_config.parameters.extend([
                COMMON_PARAMETERS["speed"],
                COMMON_PARAMETERS["pitch"],
            ])

            # AI-based extensions typically support temperature
            if name in ["maya1", "gpt_sovits", "vibe_voice", "index_tts"]:
                extension_config.parameters.append(COMMON_PARAMETERS["temperature"])

            # Some support top_p
            if name in ["maya1", "index_tts"]:
                extension_config.parameters.append(COMMON_PARAMETERS["top_p"])

            # Some support top_k
            if name in ["gpt_sovits", "index_tts"]:
                extension_config.parameters.append(COMMON_PARAMETERS["top_k"])

        return extension_config

    @staticmethod
    def get_config_function(ext_dir: Path) -> callable:
        """
        Generate a get_config() function for an extension.

        This can be used to create a module-level function that returns
        the auto-generated ExtensionConfig.

        Args:
            ext_dir: Path to the extension directory

        Returns:
            A callable function that returns ExtensionConfig
        """
        config = TTSConfigGenerator.generate_from_yaml(ext_dir)

        def get_config() -> ExtensionConfig:
            """Get extension configuration (auto-generated from config.yaml)."""
            return config

        return get_config


def create_config_from_yaml(ext_dir: Path) -> ExtensionConfig:
    """
    Convenience function to create ExtensionConfig from config.yaml.

    Usage in extension config.py:
        # config.py (optional - can use this instead of manual config)
        from src.extensions.libraries.tts_engine.config_generator import create_config_from_yaml
        from pathlib import Path

        get_{ext}_config = create_config_from_yaml(Path(__file__).parent.parent)

    Args:
        ext_dir: Path to the extension directory

    Returns:
        ExtensionConfig object
    """
    return TTSConfigGenerator.generate_from_yaml(ext_dir)
