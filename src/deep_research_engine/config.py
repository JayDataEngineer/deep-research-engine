"""Config loading and validation."""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from jinja2 import Template
import os

from pydantic import BaseModel, Field


class WorkflowConfig(BaseModel):
    """Workflow configuration from YAML."""
    
    workflow: Dict[str, Any]
    input: Dict[str, Any]
    phases: Dict[str, Any]
    storage: Dict[str, Any] = Field(default_factory=dict)
    llm: Dict[str, Any] = Field(default_factory=dict)
    tools: Dict[str, Any] = Field(default_factory=dict)
    execution: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"


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
            if '-' in var_expr:
                var_name, default = var_expr.split('-', 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(var_expr, match.group(0))
        
        return re.sub(r'\$\{([^}]+)\}', replace, value)
    
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
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Expand environment variables
    config_dict = expand_env_vars(config_dict)
    
    # Validate and return
    return WorkflowConfig(**config_dict)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
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
