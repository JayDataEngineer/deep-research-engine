"""Config loading and validation."""

import os
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field


class WorkflowConfig(BaseModel):
    """Workflow configuration from YAML.

    Attributes:
        workflow: Workflow configuration settings.
        input: Input data configuration.
        phases: Phase configuration (can be dict or list).
        storage: Storage backend configuration.
        llm: LLM configuration.
        tools: Tools configuration.
        execution: Execution settings.
        output: Output configuration.
    """

    workflow: dict[str, Any]
    input: dict[str, Any]
    phases: dict[str, Any] | list[dict[str, Any]]  # Can be dict or list
    storage: dict[str, Any] = Field(default_factory=dict)
    llm: dict[str, Any] = Field(default_factory=dict)
    tools: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    def get_phase_config(self, phase_name: str) -> dict[str, Any]:
        """Get configuration for a specific phase.

        Handles both list and dict formats for self.phases.

        Args:
            phase_name: Name of the phase (e.g., "boss", "workers", "critic").

        Returns:
            Dictionary with phase configuration or empty dict if not found.
        """
        phases = self.phases
        if isinstance(phases, list):
            for phase in phases:
                if isinstance(phase, dict) and phase.get("name") == phase_name:
                    return phase.get("config", {})
        elif isinstance(phases, dict):
            # Try both direct name and phases nested structure
            phase_data = phases.get(phase_name, {})
            if isinstance(phase_data, dict) and "config" in phase_data:
                config_val = phase_data.get("config", {})
                return config_val if isinstance(config_val, dict) else {}
            return phase_data if isinstance(phase_data, dict) else {}

        return {}


def get_phase_config(config: dict[str, Any] | list[dict[str, Any]] | WorkflowConfig, phase_name: str) -> dict[str, Any]:
    """Standalone function to get configuration for a specific phase.

    Supports dict, list, and WorkflowConfig object.
    """
    if isinstance(config, WorkflowConfig):
        return config.get_phase_config(phase_name)
    
    if isinstance(config, list):
        for phase in config:
            if isinstance(phase, dict) and phase.get("name") == phase_name:
                return phase.get("config", {})
    elif isinstance(config, dict):
        # Handle top-level "phases" key if present
        phases = config.get("phases", config)
        if isinstance(phases, list):
            return get_phase_config(phases, phase_name)
        
        phase_data = phases.get(phase_name, {})
        if isinstance(phase_data, dict) and "config" in phase_data:
            return phase_data.get("config", {})
        return phase_data if isinstance(phase_data, dict) else {}

    return {}


def expand_env_vars(value: Any) -> Any:
    """Expand environment variables in config values.

    Supports:
    - ${VAR} - Direct substitution
    - ${VAR-default} - With default value
    """
    if isinstance(value, str):
        # Pattern: ${VAR} or ${VAR-default}
        import re

        def replace(match):
            var_expr = match.group(1)
            if "-" in var_expr:
                var_name, default = var_expr.split("-", 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(var_expr, match.group(0))

        return re.sub(r"\$\{([^}]+)\}", replace, value)

    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]

    return value


def load_config(config_path: str) -> WorkflowConfig:
    """Load workflow configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        WorkflowConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config validation fails
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with path.open() as f:
        config_dict = yaml.safe_load(f)

    # Expand environment variables
    config_dict = expand_env_vars(config_dict)

    # Validate and return
    return WorkflowConfig(**config_dict)


def merge_configs(base_config: dict[str, Any], override_config: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two configs.

    Override values take precedence. Lists are replaced, not merged.
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def render_prompt_template(template_str: str, **variables) -> str:
    """Render a Jinja2 prompt template.

    Args:
        template_str: Jinja2 template string
        **variables: Variables to inject

    Returns:
        Rendered template
    """
    template = Template(template_str)
    return template.render(**variables)
