"""
Prompt Configuration Validator

Validates that:
1. All prompts referenced in metadata actually exist
2. All prompt templates are valid Python format strings
3. All required params for prompts are documented
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import string
import yaml


class PromptValidator:
    """Validates prompt configuration files.

    Usage:
        validator = PromptValidator()
        errors = validator.validate_all()
        if errors:
            print("Validation errors:")
            for error in errors:
                print(f"  - {error}")
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialize validator.

        Args:
            prompts_dir: Path to prompts/emotion_detection/
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent

        self.prompts_dir = prompts_dir
        self.prompts_file = prompts_dir / "prompts.yaml"
        self.metadata_file = prompts_dir / "_metadata.yaml"

        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> List[str]:
        """Run all validations.

        Returns:
            List of error messages (empty if all valid)
        """
        self.errors = []
        self.warnings = []

        # Load files
        if not self.prompts_file.exists():
            self.errors.append(f"prompts.yaml not found at {self.prompts_file}")
            return self.errors

        if not self.metadata_file.exists():
            self.errors.append(f"_metadata.yaml not found at {self.metadata_file}")
            return self.errors

        with open(self.prompts_file) as f:
            prompts_data = yaml.safe_load(f)

        with open(self.metadata_file) as f:
            metadata = yaml.safe_load(f)

        # Validate structure
        self._validate_structure(prompts_data, metadata)

        # Validate prompts exist
        self._validate_prompt_existence(prompts_data, metadata)

        # Validate prompt templates
        self._validate_prompt_templates(prompts_data, metadata)

        # Validate mode configurations
        self._validate_mode_configs(prompts_data, metadata)

        return self.errors

    def _validate_structure(self, prompts_data: Dict, metadata: Dict) -> None:
        """Validate basic YAML structure."""
        # Check prompts.yaml has emotion_detection key
        if "emotion_detection" not in prompts_data:
            self.errors.append("prompts.yaml missing 'emotion_detection' key")
            return

        # Check metadata has required keys
        for key in ["prompts", "modes"]:
            if key not in metadata:
                self.errors.append(f"_metadata.yaml missing '{key}' key")

    def _validate_prompt_existence(self, prompts_data: Dict, metadata: Dict) -> None:
        """Validate that all referenced prompts exist."""
        defined_prompts = set(prompts_data.get("emotion_detection", {}).keys())

        # Check all prompts in metadata are defined
        for prompt_name in metadata.get("prompts", {}).keys():
            if prompt_name not in defined_prompts:
                self.errors.append(
                    f"_metadata.yaml references prompt '{prompt_name}' "
                    f"but it's not defined in prompts.yaml"
                )

        # Check all prompts used in modes are defined
        for mode_name, mode_config in metadata.get("modes", {}).items():
            for prompt_name in mode_config.get("prompts", []):
                if prompt_name not in defined_prompts:
                    self.errors.append(
                        f"Mode '{mode_name}' references prompt '{prompt_name}' "
                        f"but it's not defined in prompts.yaml"
                    )

    def _validate_prompt_templates(self, prompts_data: Dict, metadata: Dict) -> None:
        """Validate that prompt templates are valid format strings."""
        formatter = string.Formatter()

        for prompt_name, prompt_template in prompts_data.get("emotion_detection", {}).items():
            # Parse template to extract placeholders
            try:
                placeholders = {
                    p for _, p, _, _ in formatter.parse(prompt_template) if p
                }

                # Check if prompt has metadata documenting required params
                prompt_meta = metadata.get("prompts", {}).get(prompt_name, {})
                required_context = prompt_meta.get("requires_context", False)
                context_fields = prompt_meta.get("context_fields", [])

                # Warn if placeholders aren't documented
                if placeholders and not required_context:
                    self.warnings.append(
                        f"Prompt '{prompt_name}' has placeholders {placeholders} "
                        f"but 'requires_context' is not set in metadata"
                    )

                # Check context fields match placeholders
                if required_context and context_fields:
                    context_fields_set = set(context_fields)
                    missing_fields = context_fields_set - placeholders
                    if missing_fields:
                        self.warnings.append(
                            f"Prompt '{prompt_name}' metadata lists context_fields "
                            f"{context_fields} but template has {placeholders}"
                        )

            except Exception as e:
                self.errors.append(
                    f"Prompt '{prompt_name}' has invalid template syntax: {e}"
                )

    def _validate_mode_configs(self, prompts_data: Dict, metadata: Dict) -> None:
        """Validate mode configurations."""
        defined_prompts = set(prompts_data.get("emotion_detection", {}).keys())

        for mode_name, mode_config in metadata.get("modes", {}).items():
            # Check required fields
            for field in ["quality_mode", "prompts"]:
                if field not in mode_config:
                    self.errors.append(
                        f"Mode '{mode_name}' missing required field '{field}'"
                    )

            # Check prompts list is not empty
            prompts = mode_config.get("prompts", [])
            if not prompts:
                self.errors.append(f"Mode '{mode_name}' has empty prompts list")

            # Check all prompts exist
            for prompt_name in prompts:
                if prompt_name not in defined_prompts:
                    self.errors.append(
                        f"Mode '{mode_name}' references undefined prompt '{prompt_name}'"
                    )

    def print_report(self) -> None:
        """Print validation report."""
        if self.errors:
            print("❌ VALIDATION FAILED")
            print("\nErrors:")
            for error in self.errors:
                print(f"  ✗ {error}")

        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")

        if not self.errors and not self.warnings:
            print("✅ All validations passed!")

        elif not self.errors:
            print("\n✅ No errors (warnings only)")


def validate_prompts(prompts_dir: Optional[Path] = None) -> bool:
    """Convenience function to validate prompts.

    Returns:
        True if validation passed (no errors)
    """
    validator = PromptValidator(prompts_dir)
    errors = validator.validate_all()
    validator.print_report()
    return len(errors) == 0


if __name__ == "__main__":
    import sys
    prompts_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    validate_prompts(prompts_dir)
