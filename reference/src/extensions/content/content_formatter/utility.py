"""Utility functions for content formatter extension.

Provides helpers for:
- YAML config loading with merge
- Jinja2 template rendering
- Generic extension prompt loading (factory pattern)
"""

import importlib.util
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from jinja2 import Environment, BaseLoader

# Add backend to path for absolute imports
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(backend_dir) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(backend_dir))


def load_yaml_with_defaults(
    base_config: Dict[str, Any],
    override_path: Path
) -> Dict[str, Any]:
    """Load YAML config and merge with base defaults.

    Args:
        base_config: Base configuration dict
        override_path: Path to format-specific config.yaml

    Returns:
        Merged configuration dict
    """
    if not override_path.exists():
        return base_config.copy()

    with open(override_path, 'r') as f:
        override_config = yaml.safe_load(f)

    # Deep merge override into base
    result = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value

    return result


def render_jinja_template(
    template: str,
    **context: Any
) -> str:
    """Render Jinja2 template with context.

    Args:
        template: Template string with Jinja2 syntax
        **context: Variables to substitute

    Returns:
        Rendered string
    """
    env = Environment(loader=BaseLoader())
    jinja_template = env.from_string(template)
    return jinja_template.render(**context)


def get_extension_prompts(
    format_dir: Path,
    **kwargs: Any
) -> List[str]:
    """Load extension-provided prompts using factory pattern.

    This is a GENERIC factory - it knows NOTHING about specific formats.
    Each format can optionally provide an extend.py that returns extra prompts.
    The format extension owns the logic for what prompts it needs.

    Args:
        format_dir: Path to the format directory (e.g., formats/podcast/)
        **kwargs: Arbitrary arguments passed to the extension's get_prompts()

    Returns:
        List of prompt strings to append to the base prompt.

    Example:
        # Podcast extend.py might accept:
        get_extension_prompts(podcast_dir, character="OLIVIA", session_id="xyz")

        # Tweets extend.py might accept:
        get_extension_prompts(tweets_dir, style="professional")
    """
    extend_path = format_dir / "extend.py"
    if not extend_path.exists():
        return []

    # Dynamic import of the format's extend.py
    spec = importlib.util.spec_from_file_location(
        f"{format_dir.name}.extend",
        extend_path
    )
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Call get_prompts() if it exists - format owns what it returns
    if hasattr(module, 'get_prompts'):
        try:
            return module.get_prompts(**kwargs)
        except Exception:
            # If extension fails, return empty list rather than crash
            return []

    return []


def get_format_directories(formats_dir: Path) -> List[Path]:
    """Get list of available format directories.

    Args:
        formats_dir: Path to formats directory

    Returns:
        List of format directory paths
    """
    if not formats_dir.exists():
        return []

    return [
        d for d in formats_dir.iterdir()
        if d.is_dir() and (d / "config.yaml").exists() and (d / "schema.py").exists()
    ]


def merge_extension_prompts(
    base_prompt: str,
    format_dir: Path,
    **kwargs: Any
) -> str:
    """Merge extension-provided prompts into base prompt.

    Uses the generic get_extension_prompts factory.

    Args:
        base_prompt: Base prompt template
        format_dir: Path to the format directory
        **kwargs: Arbitrary arguments passed to get_extension_prompts

    Returns:
        Merged prompt with all extension prompts loaded
    """
    extension_prompts = get_extension_prompts(format_dir, **kwargs)

    if not extension_prompts:
        return base_prompt

    # Merge extension prompts into base prompt
    prompts_text = "\n\n".join(extension_prompts)
    return f"{base_prompt}\n\n{prompts_text}"
