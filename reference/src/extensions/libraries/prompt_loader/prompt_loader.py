"""Prompt Loader Service

Loads and validates prompt configurations from YAML files.
Provides startup validation to catch config errors early.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PromptLoadError:
    """Error details from prompt loading/validation."""
    file: str
    error_type: str  # "missing", "invalid", "parse_error", "schema_error"
    message: str
    path: Optional[str] = None  # Path within YAML


class PromptLoader:
    """Load and validate prompt configurations from YAML files."""

    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialize the prompt loader.

        Args:
            prompts_dir: Directory containing prompt YAML files.
                        Defaults to data/config/prompts/
        """
        if prompts_dir is None:
            # Default to data/config/prompts/
            from src.constants import PROMPTS_CONFIG_DIR
            prompts_dir = PROMPTS_CONFIG_DIR

        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, Any] = {}

    def validate_on_startup(self, raise_on_error: bool = True) -> List[PromptLoadError]:
        """Validate all prompt configurations on startup.

        This should be called when the backend starts up to catch
        configuration errors early.

        Args:
            raise_on_error: If True, raises exception on first error.
                           If False, collects all errors and returns them.

        Returns:
            List of PromptLoadError objects (empty if all valid)

        Raises:
            RuntimeError: If raise_on_error=True and validation fails
        """
        errors = []

        # Check that prompts directory exists
        if not self.prompts_dir.exists():
            error = PromptLoadError(
                file="<directory>",
                error_type="missing",
                message=f"Prompts directory not found: {self.prompts_dir}"
            )
            errors.append(error)
            if raise_on_error:
                raise RuntimeError(f"Prompts directory not found: {self.prompts_dir}")
            return errors

        # Validate each expected file
        expected_files = ["brain.yaml", "personalities.yaml", "examples.yaml"]
        for filename in expected_files:
            file_errors = self._validate_file(filename)
            errors.extend(file_errors)

        # Log results
        if errors:
            logger.error(f"Prompt validation failed with {len(errors)} error(s):")
            for err in errors:
                logger.error(f"  [{err.error_type}] {err.file}: {err.message}")
                if err.path:
                    logger.error(f"    at: {err.path}")
            if raise_on_error:
                raise RuntimeError(f"Prompt validation failed: {errors[0].message}")
        else:
            logger.info("Prompt validation: All configurations valid")

        return errors

    def _validate_file(self, filename: str) -> List[PromptLoadError]:
        """Validate a single YAML file.

        Args:
            filename: Name of the file to validate

        Returns:
            List of errors (empty if valid)
        """
        errors = []
        filepath = self.prompts_dir / filename

        # Check file exists
        if not filepath.exists():
            errors.append(PromptLoadError(
                file=filename,
                error_type="missing",
                message=f"Required file not found: {filepath}"
            ))
            return errors

        # Try to parse YAML
        try:
            with open(filepath) as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            errors.append(PromptLoadError(
                file=filename,
                error_type="parse_error",
                message=f"Failed to parse YAML: {e}"
            ))
            return errors

        # Validate schema based on file type
        if filename == "brain.yaml":
            errors.extend(self._validate_brain_yaml(filename, data))
        elif filename == "personalities.yaml":
            errors.extend(self._validate_personalities_yaml(filename, data))
        elif filename == "examples.yaml":
            errors.extend(self._validate_examples_yaml(filename, data))

        return errors

    def _validate_brain_yaml(self, filename: str, data: Dict) -> List[PromptLoadError]:
        """Validate brain.yaml schema."""
        errors = []

        # Required top-level keys
        required_keys = ["system_prompts", "personality_styles", "markers"]
        for key in required_keys:
            if key not in data:
                errors.append(PromptLoadError(
                    file=filename,
                    error_type="schema_error",
                    message=f"Missing required key: {key}",
                    path=f"$.{key}"
                ))

        # Validate system_prompts
        if "system_prompts" in data:
            required_modes = ["assistant", "companion", "rp"]
            for mode in required_modes:
                if mode not in data["system_prompts"]:
                    errors.append(PromptLoadError(
                        file=filename,
                        error_type="schema_error",
                        message=f"Missing system prompt for mode: {mode}",
                        path=f"$.system_prompts.{mode}"
                    ))

        # Validate personality_styles
        if "personality_styles" in data:
            required_styles = ["brief", "balanced", "elaborate", "witty", "supportive", "professional", "casual"]
            for style in required_styles:
                if style not in data["personality_styles"]:
                    errors.append(PromptLoadError(
                        file=filename,
                        error_type="schema_error",
                        message=f"Missing personality style: {style}",
                        path=f"$.personality_styles.{style}"
                    ))

        # Validate markers
        if "markers" in data:
            required_markers = ["context", "examples", "chat_history", "author_note"]
            for marker in required_markers:
                if marker not in data["markers"]:
                    errors.append(PromptLoadError(
                        file=filename,
                        error_type="schema_error",
                        message=f"Missing marker: {marker}",
                        path=f"$.markers.{marker}"
                    ))

        return errors

    def _validate_personalities_yaml(self, filename: str, data: Dict) -> List[PromptLoadError]:
        """Validate personalities.yaml schema."""
        errors = []

        # Check for personalities key
        if "personalities" not in data:
            errors.append(PromptLoadError(
                file=filename,
                error_type="schema_error",
                message="Missing required key: personalities",
                path="$.personalities"
            ))
            return errors

        # Validate each personality has required fields
        required_fields = ["name", "role", "description", "communication_style",
                          "response_voice", "formality_level", "humor_level", "mood_matching"]

        for persona_name, persona_data in data.get("personalities", {}).items():
            if not isinstance(persona_data, dict):
                errors.append(PromptLoadError(
                    file=filename,
                    error_type="schema_error",
                    message=f"Personality '{persona_name}' must be a dict",
                    path=f"$.personalities.{persona_name}"
                ))
                continue

            for field in required_fields:
                if field not in persona_data:
                    errors.append(PromptLoadError(
                        file=filename,
                        error_type="schema_error",
                        message=f"Personality '{persona_name}' missing field: {field}",
                        path=f"$.personalities.{persona_name}.{field}"
                    ))

            # Validate communication_style is a known style
            valid_styles = ["brief", "balanced", "elaborate", "witty", "supportive", "professional", "casual"]
            if "communication_style" in persona_data:
                style = persona_data["communication_style"]
                if style not in valid_styles:
                    errors.append(PromptLoadError(
                        file=filename,
                        error_type="schema_error",
                        message=f"Invalid communication_style '{style}' in '{persona_name}'",
                        path=f"$.personalities.{persona_name}.communication_style"
                    ))

        # Validate default personality exists
        if "default" in data:
            default_persona = data["default"]
            if default_persona not in data.get("personalities", {}):
                errors.append(PromptLoadError(
                    file=filename,
                    error_type="schema_error",
                    message=f"Default personality '{default_persona}' not defined",
                    path="$.default"
                ))

        return errors

    def _validate_examples_yaml(self, filename: str, data: Dict) -> List[PromptLoadError]:
        """Validate examples.yaml schema."""
        errors = []

        # Check for at least one example section
        if not data:
            errors.append(PromptLoadError(
                file=filename,
                error_type="schema_error",
                message="File is empty or invalid",
                path="$"
            ))
            return errors

        return errors

    def load_brain_prompts(self) -> Dict[str, Any]:
        """Load brain.yaml prompts.

        Returns:
            Dict with brain prompt configuration

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        return self._load_yaml("brain.yaml")

    def load_personalities(self) -> Dict[str, Any]:
        """Load personalities.yaml.

        Returns:
            Dict with personality configurations

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        return self._load_yaml("personalities.yaml")

    def load_examples(self) -> Dict[str, Any]:
        """Load examples.yaml.

        Returns:
            Dict with dialogue examples

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        return self._load_yaml("examples.yaml")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML file with caching.

        Args:
            filename: Name of the file to load

        Returns:
            Parsed YAML data as dict

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        # Check cache first
        if filename in self._cache:
            return self._cache[filename]

        filepath = self.prompts_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Prompt file not found: {filepath}")

        with open(filepath) as f:
            data = yaml.safe_load(f) or {}

        # Cache the result
        self._cache[filename] = data

        return data

    def get_system_prompt(self, mode: str = "companion") -> str:
        """Get system prompt for a specific mode.

        Args:
            mode: "assistant", "companion", or "rp"

        Returns:
            System prompt string
        """
        brain = self.load_brain_prompts()
        prompts = brain.get("system_prompts", {})
        return prompts.get(mode, prompts.get("companion", ""))

    def get_personality_style(self, style: str) -> str:
        """Get personality style prompt.

        Args:
            style: Style name (brief, balanced, witty, etc.)

        Returns:
            Style prompt string
        """
        brain = self.load_brain_prompts()
        styles = brain.get("personality_styles", {})
        return styles.get(style, styles.get("balanced", ""))

    def get_personality_config(self, persona_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific personality configuration.

        Args:
            persona_name: Name of the personality preset

        Returns:
            Personality config dict, or None if not found
        """
        personalities = self.load_personalities()
        return personalities.get("personalities", {}).get(persona_name)

    def get_default_personality(self) -> str:
        """Get the default personality name.

        Returns:
            Default personality name
        """
        personalities = self.load_personalities()
        return personalities.get("default", "friendly")

    def get_examples(self, mode: str = "companion", style: Optional[str] = None) -> List[Dict[str, str]]:
        """Get dialogue examples for a mode/style.

        Args:
            mode: "companion" or "roleplay"
            style: Optional personality style filter

        Returns:
            List of example dicts with "user" and "assistant" keys
        """
        examples = self.load_examples()

        if mode == "companion" and style:
            return examples.get("companion", {}).get(style, [])
        elif mode == "roleplay":
            return examples.get("roleplay", {}).get("default", [])
        else:
            return examples.get("companion", {}).get("friendly", [])

    def get_markers(self) -> Dict[str, Dict[str, Any]]:
        """Get all prompt markers.

        Returns:
            Dict of marker configurations
        """
        brain = self.load_brain_prompts()
        return brain.get("markers", {})

    def clear_cache(self) -> None:
        """Clear the YAML file cache."""
        self._cache.clear()


# Singleton instance
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader(prompts_dir: Optional[Path] = None) -> PromptLoader:
    """Get the singleton prompt loader instance.

    Args:
        prompts_dir: Optional custom prompts directory

    Returns:
        PromptLoader instance
    """
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader(prompts_dir=prompts_dir)
    return _prompt_loader


def validate_prompts_on_startup(prompts_dir: Optional[Path] = None, raise_on_error: bool = True) -> List[PromptLoadError]:
    """Convenience function to validate prompts on startup.

    Args:
        prompts_dir: Optional custom prompts directory
        raise_on_error: If True, raises exception on validation failure

    Returns:
        List of errors (empty if valid)

    Raises:
        RuntimeError: If validation fails and raise_on_error=True
    """
    loader = get_prompt_loader(prompts_dir)
    return loader.validate_on_startup(raise_on_error=raise_on_error)
