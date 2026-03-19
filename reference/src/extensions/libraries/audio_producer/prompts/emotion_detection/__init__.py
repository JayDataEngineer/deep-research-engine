"""
Emotion Detection Prompt Loader

**CRITICAL**: This is the ONLY code that should touch prompt files.
All other code uses this module to access prompts.

Why?
- Prompts are the BRAIN of the operation
- Soft errors only if we mess them up
- Centralized access = easier to audit and test
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class PromptLoader:
    """Loads and formats emotion detection prompts.

    Usage:
        loader = PromptLoader()
        prompt = loader.get_prompt("single_line")
        formatted = prompt.format(speaker="EMMA", text="Hello world")

        # Or with mode preset
        prompts = loader.get_mode_prompts("fast")
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialize prompt loader.

        Args:
            prompts_dir: Path to prompts directory. Defaults to audio_producer/prompts/
        """
        if prompts_dir is None:
            # Default to prompts/emotion_detection/
            prompts_dir = Path(__file__).parent

        self.prompts_dir = prompts_dir
        self.prompts_file = prompts_dir / "prompts.yaml"
        self.metadata_file = prompts_dir / "_metadata.yaml"

        # Load prompts and metadata
        self._prompts = self._load_yaml(self.prompts_file).get("emotion_detection", {})
        self._metadata = self._load_yaml(self.metadata_file)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        with open(path) as f:
            return yaml.safe_load(f)

    def get_prompt(self, prompt_name: str) -> str:
        """Get a prompt template by name.

        Args:
            prompt_name: Name of the prompt (e.g., "single_line", "director")

        Returns:
            Prompt template string (ready for .format())

        Raises:
            KeyError: If prompt not found
        """
        if prompt_name not in self._prompts:
            available = list(self._prompts.keys())
            raise KeyError(
                f"Prompt '{prompt_name}' not found. "
                f"Available prompts: {available}"
            )

        return self._prompts[prompt_name]

    def get_prompt_info(self, prompt_name: str) -> Dict[str, Any]:
        """Get metadata about a prompt.

        Args:
            prompt_name: Name of the prompt

        Returns:
            Dict with prompt metadata (description, use_cases, etc.)
        """
        return self._metadata["prompts"].get(prompt_name, {})

    def list_prompts(self) -> List[str]:
        """List all available prompt names."""
        return list(self._prompts.keys())

    def get_mode_prompts(self, mode: str) -> List[str]:
        """Get the list of prompts for a given mode.

        Args:
            mode: Mode name (e.g., "fast", "overdrive", "sectioned")

        Returns:
            List of prompt names in execution order
        """
        mode_config = self._metadata["modes"].get(mode, {})
        return mode_config.get("prompts", [])

    def get_mode_config(self, mode: str) -> Dict[str, Any]:
        """Get full configuration for a mode.

        Args:
            mode: Mode name

        Returns:
            Dict with mode config (quality_mode, chunking, llm_workers, prompts)
        """
        return self._metadata["modes"].get(mode, {})

    def format_prompt(self, prompt_name: str, **kwargs) -> str:
        """Get and format a prompt in one call.

        Args:
            prompt_name: Name of the prompt
            **kwargs: Format arguments

        Returns:
            Formatted prompt string

        Example:
            loader.format_prompt("single_line", speaker="EMMA", text="Hi!")
        """
        template = self.get_prompt(prompt_name)
        return template.format(**kwargs)

    def validate_prompt_params(self, prompt_name: str, **kwargs) -> bool:
        """Validate that required params are present for formatting.

        Args:
            prompt_name: Name of the prompt
            **kwargs: Params to validate

        Returns:
            True if all required params are present
        """
        template = self.get_prompt(prompt_name)
        # Check if template has placeholders
        import string
        formatter = string.Formatter()
        required_params = {p for _, p, _, _ in formatter.parse(template) if p}

        provided_params = set(kwargs.keys())
        return required_params.issubset(provided_params)


# Singleton instance for easy importing
_loader_instance: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """Get the singleton PromptLoader instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = PromptLoader()
    return _loader_instance


def get_prompt(prompt_name: str) -> str:
    """Quick access to get a prompt template."""
    return get_prompt_loader().get_prompt(prompt_name)


def format_prompt(prompt_name: str, **kwargs) -> str:
    """Quick access to format a prompt."""
    return get_prompt_loader().format_prompt(prompt_name, **kwargs)
