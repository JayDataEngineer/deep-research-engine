"""Core formatting logic for content formatter extension.

Provides the main formatting function that:
1. Loads format-specific config and schema
2. Builds tools (optional Context Engine access)
3. Creates deep_agent with format-specific prompt
4. Invokes agent and extracts structured output
5. Retries on validation failure with feedback
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from pydantic import BaseModel, ValidationError

# Add backend to path for absolute imports
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(backend_dir) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(backend_dir))

from .utility import load_yaml_with_defaults, render_jinja_template, merge_extension_prompts

logger = logging.getLogger(__name__)


class FormatSpec:
    """Format specification loaded from config and schema."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        schema_class: Type[BaseModel],
        formats_dir: Path
    ):
        self.name = name
        self.config = config
        self.schema_class = schema_class
        self.formats_dir = formats_dir
        self.max_retries = config.get("max_retries", 10)


async def format_content(
    format_name: str,
    content: str,
    base_config: Dict[str, Any],
    formats_dir: Path,
    session_id: Optional[str] = None,
    character: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None,
    max_retries: Optional[int] = None,
    **kwargs: Any
) -> BaseModel:
    """Format content using deep_agent with structured extraction and retry logic.

    Args:
        format_name: Name of format to use (tweets, podcast, etc.)
        content: Input content to format
        base_config: Base configuration from extension config.yaml
        formats_dir: Path to formats directory
        session_id: Optional session ID for Context Engine access
        character: Optional character name for character-based formats
        model_config: Optional model configuration override
        max_retries: Optional override for max retry attempts
        **kwargs: Additional format-specific parameters

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If format not found
        RuntimeError: If formatting fails after all retries
    """
    # Format aliases (common names -> actual format directory names)
    FORMAT_ALIASES = {
        "twitter": "tweets",
        "tweet": "tweets",
        "x": "tweets",
        "thread": "tweets",
    }

    # Resolve alias if applicable
    resolved_format = FORMAT_ALIASES.get(format_name.lower(), format_name)

    # 1. Load format-specific config
    format_config_path = formats_dir / resolved_format / "config.yaml"
    if not format_config_path.exists():
        available = [d.name for d in formats_dir.iterdir() if d.is_dir()]
        raise ValueError(
            f"Format '{format_name}' not found. "
            f"Available formats: {available}"
        )

    format_config = load_yaml_with_defaults(
        base_config.get("base", {}),
        format_config_path
    )

    # 2. Load schema
    schema_class = _load_schema(resolved_format, formats_dir)

    # 3. Get retry count from config
    retries = max_retries or format_config.get(
        "max_retries",
        base_config.get("defaults", {}).get("max_retries", 10)
    )

    # 4. Build prompt with Jinja2 rendering
    prompt_vars = {
        "content": content,
        "session_id": session_id,
        "character": character,
        "base": base_config.get("base", {}),
        **kwargs
    }

    system_prompt = format_config.get("system_prompt", "")
    format_dir = formats_dir / resolved_format

    # Merge extension-provided prompts (generic factory - format owns its logic)
    # Pass through all kwargs so the format's extend.py can use what it needs
    system_prompt = merge_extension_prompts(system_prompt, format_dir, **prompt_vars)

    # Render prompt
    rendered_prompt = render_jinja_template(system_prompt, **prompt_vars)

    # 5. Build tools (optional Context Engine)
    tools = []
    if session_id:
        tools = _get_context_tools(session_id)

    # 6. Create deep_agent
    from deepagents import create_deep_agent

    # Get model config
    if model_config:
        llm = await _create_llm(model_config)
    else:
        llm = await _create_llm(base_config.get("model", {}))

    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=rendered_prompt
    )

    # 7. Invoke agent with retry loop
    from src.extensions.content.deep_research.graph.helpers.deep_agent_helpers import extract_structured_output

    current_prompt = rendered_prompt
    last_error: Optional[ValidationError] = None

    for attempt in range(1, retries + 1):
        try:
            # Invoke agent
            result = await agent.ainvoke({
                "messages": [
                    {"role": "user", "content": current_prompt}
                ]
            })

            # Extract structured output
            agent_output = str(result.get("messages", [])[-1].content) if result else ""
            formatted_result = await extract_structured_output(
                llm=llm,
                agent_output=agent_output,
                pydantic_model=schema_class,
                yaml_config=base_config
            )

            # Success - return validated result
            logger.info(f"[FORMAT_CONTENT] Successfully formatted '{format_name}' on attempt {attempt}")
            return formatted_result

        except ValidationError as e:
            last_error = e

            if attempt == retries:
                # Final attempt failed - crash with detailed error
                raise RuntimeError(
                    f"Format validation failed after {retries} attempts for '{format_name}'. "
                    f"Final validation errors: {_format_validation_errors(e)}"
                ) from e

            # Add validation feedback for retry
            error_details = _format_validation_errors(e)
            current_prompt = _add_retry_feedback(rendered_prompt, error_details, attempt)
            logger.warning(
                f"[FORMAT_CONTENT] Validation failed on attempt {attempt}/{retries} for '{format_name}'. "
                f"Errors: {error_details}. Retrying with feedback..."
            )

            # Simple backoff
            await asyncio.sleep(0.5 * attempt)

        except Exception as e:
            # Non-validation errors should crash immediately (crash-only philosophy)
            raise RuntimeError(
                f"Unexpected error during formatting '{format_name}' on attempt {attempt}: {e}"
            ) from e

    # Should never reach here, but crash if we do
    raise RuntimeError(
        f"Unexpected end of retry loop for '{format_name}'. "
        f"Last validation error: {_format_validation_errors(last_error) if last_error else 'None'}"
    )


def _format_validation_errors(error: ValidationError) -> str:
    """Extract readable error details from Pydantic ValidationError.

    Args:
        error: ValidationError from Pydantic validation

    Returns:
        Semicolon-separated string of error descriptions
    """
    return "; ".join(
        f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
        for e in error.errors()
    )


def _add_retry_feedback(prompt: str, errors: str, attempt: int) -> str:
    """Add validation feedback to prompt for retry.

    Args:
        prompt: Original prompt
        errors: Formatted validation error string
        attempt: Current attempt number

    Returns:
        Prompt with retry feedback appended
    """
    return f"""{prompt}

[RETRY ATTEMPT {attempt}: Previous response had validation errors]
Errors found: {errors}
Please fix these issues and respond with valid JSON that matches the required schema."""


def _load_schema(format_name: str, formats_dir: Path) -> Type[BaseModel]:
    """Dynamically load Pydantic schema from format directory.

    Args:
        format_name: Name of format
        formats_dir: Path to formats directory

    Returns:
        Pydantic model class
    """
    import importlib.util

    schema_path = formats_dir / format_name / "schema.py"
    if not schema_path.exists():
        raise ValueError(f"Schema not found for format '{format_name}'")

    # Dynamic import
    spec = importlib.util.spec_from_file_location(
        f"{format_name}.schema",
        schema_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find first Pydantic BaseModel subclass
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, BaseModel) and attr != BaseModel:
            return attr

    raise ValueError(f"No Pydantic BaseModel found in {schema_path}")


async def _create_llm(model_config: Dict[str, Any]) -> Any:
    """Create LLM from model configuration.

    Args:
        model_config: Model configuration dict

    Returns:
        LLM instance
    """
    from src.shared.chat_llm import get_chat_llm

    model_name = model_config.get("name", "glm-4.7")
    enable_thinking = model_config.get("enable_thinking", False)

    return await get_chat_llm(
        model=model_name,
        enable_thinking=enable_thinking
    )


def _get_context_tools(session_id: str) -> List:
    """Get Context Engine tools for the session.

    Args:
        session_id: Session ID for Context Engine access

    Returns:
        List of Context Engine tools
    """
    # Import Context Engine tools
    try:
        from src.extensions.features.tools.context_engine.tools import (
            query_postgres,
            get_artifact,
            read_vfs_raw,
            query_neo4j,
        )
        return [query_postgres, get_artifact, read_vfs_raw, query_neo4j]
    except ImportError:
        return []
