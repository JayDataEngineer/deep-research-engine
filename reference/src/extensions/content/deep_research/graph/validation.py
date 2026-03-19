"""
Validation module for Deep Research framework.

This module provides validation functions for configuration and implementation.

Architecture:
- Configuration validation: Uses validation/config_validation.py
- Implementation validation: Uses tests/test_deep_research_architecture.py

This module serves as the main entry point for validation, importing from
the appropriate specialized modules.

PHILOSOPHY:
- Fail fast - fail loud - fail at startup
- Configuration issues caught at startup, not during execution
- Implementation issues caught by tests, not at runtime
"""

from typing import Dict, Any

# Import configuration validation
from .validation.config_validation import (
    # Exception classes
    WorkerTypeError,
    SpecialistConfigError,
    ResearchToolsError,
    ConfigSectionError,
    # Pydantic models
    SpawnerConfig,
    WorkerGraphConfig,
    VariantConfig,
    # Validation functions
    validate_variant_config,
    validate_required_config_sections,
)

# Import additional exception classes


class PromptTemplateError(Exception):
    """Raised when prompt templates are missing required placeholders."""
    pass


class DeepAgentError(Exception):
    """Raised when deep_agent is required but not available."""
    pass


def validate_prompt_template_placeholders(
    template_name: str,
    template_content: str,
    required_placeholders: list,
) -> None:
    """Validate that a prompt template contains all required placeholders.

    This prevents silent failures where templates don't have the placeholders
    that the code expects to fill in.

    Args:
        template_name: Name of the template (for error messages)
        template_content: The template content
        required_placeholders: List of required placeholder names (without {})

    Raises:
        PromptTemplateError: If required placeholders are missing
    """
    missing_placeholders = []
    for placeholder in required_placeholders:
        pattern = f"{{{placeholder}}}"
        if pattern not in template_content:
            missing_placeholders.append(f"  - {{{placeholder}}}")

    if missing_placeholders:
        raise PromptTemplateError(
            "\n"
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║          PROMPT TEMPLATE ERROR: Missing Placeholders                ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            f"Template: {template_name}\n"
            "\n"
            "The following required placeholders are MISSING:\n"
            + "\n".join(missing_placeholders) + "\n"
            "\n"
            "The code expects to fill these placeholders, but they're not in the template.\n"
            "This will cause a crash or incorrect behavior at runtime.\n"
            "\n"
            "FIX:\n"
            "  Add the missing placeholders to the template content.\n"
            "\n"
        )


def validate_synthesis_template(config: Dict[str, Any]) -> None:
    """Validate synthesis prompt template has required placeholders.

    synthesis.py expects these placeholders to be available:
    - {findings}: Worker findings content (REQUIRED)
    - {query}: User query (often used)
    - {citations}: Citations section (optional but recommended)

    Args:
        config: The loaded YAML configuration dict

    Raises:
        PromptTemplateError: If synthesis template is missing placeholders
    """
    synthesis_config = config.get("synthesis_config", {})
    template = synthesis_config.get("prompt_template")

    if not template:
        # Template might be in prompts section
        prompts = config.get("prompts", {})
        template = prompts.get("synthesis") or prompts.get("synthesis_structured")

    if not template:
        # No template found - this is a config error
        raise PromptTemplateError(
            "\n"
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║         PROMPT TEMPLATE ERROR: No Synthesis Template Found         ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            "The synthesis prompt template is not defined in:\n"
            "  - synthesis_config.prompt_template\n"
            "  - prompts.synthesis\n"
            "  - prompts.synthesis_structured\n"
            "\n"
            "FIX:\n"
            "  Add a synthesis template to your YAML config.\n"
            "\n"
        )

    # Handle structured prompts (dicts) by extracting the template
    if isinstance(template, dict):
        template = (
            template.get("template") or
            template.get("instructions") or
            template.get("prompt") or
            template.get("content") or
            str(template)
        )

    # Ensure we have a string at this point
    if not isinstance(template, str):
        template = str(template)

    # Check for required placeholders
    required_placeholders = ["findings"]
    validate_prompt_template_placeholders(
        template_name="synthesis",
        template_content=template,
        required_placeholders=required_placeholders,
    )


def validate_boss_template(config: Dict[str, Any]) -> None:
    """Validate boss prompt template exists.

    boss/node.py loads from config or uses hardcoded fallback.
    We want to catch this at startup instead of using the fallback.

    Args:
        config: The loaded YAML configuration dict

    Raises:
        PromptTemplateError: If boss template is missing
    """
    prompts = config.get("prompts", {})
    boss_template = prompts.get("boss") or prompts.get("boss_structured")

    if not boss_template:
        raise PromptTemplateError(
            "\n"
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║           PROMPT TEMPLATE ERROR: Boss Template Missing             ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            "The boss prompt template is not defined in prompts section.\n"
            "\n"
            "Without this template, boss/node.py will fall back to a hardcoded prompt.\n"
            "This HIDDEN FALLBACK should not happen!\n"
            "\n"
            "FIX:\n"
            "  Add to prompts:\n"
            "    boss: |\n"
            "      You are the Boss orchestrator...\n"
            "\n"
        )


def validate_deep_agent_available() -> None:
    """Validate that deep_agent is available for critic/boss nodes.

    The critic and boss nodes use deep_agent for structured LLM output.
    If deep_agent is not available, they fall back to simple LLM calls.

    Args:
        None

    Raises:
        DeepAgentError: If deep_agent is not available
    """
    try:
        from src.shared.tools.ulli_langchain.agent.deep_agent import deep_agent
        return
    except ImportError as e:
        raise DeepAgentError(
            "\n"
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║           CRITICAL ERROR: Deep Agent Not Available                  ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            f"Import error: {e}\n"
            "\n"
            "deep_agent is REQUIRED for:\n"
            "  - Critic node: Structured quality evaluation\n"
            "  - Boss node: Structured worker allocation decisions\n"
            "\n"
            "FIX:\n"
            "  Ensure deep_agent is properly installed and importable.\n"
            "  Check src/shared/tools/ulli_langchain/agent/deep_agent.py exists.\n"
            "\n"
        ) from e


def validate_all(config: Dict[str, Any], skip_deep_agent_check: bool = False) -> None:
    """Run ALL validation checks to catch hidden fallbacks at startup.

    This is the MAIN ENTRY POINT for comprehensive validation.
    Call this early in the research run to fail fast on any configuration issue.

    Args:
        config: The loaded YAML configuration dict
        skip_deep_agent_check: If True, skip deep_agent availability check

    Raises:
        ConfigSectionError: If required config sections are missing
        PromptTemplateError: If prompt templates are invalid
        WorkerTypeError: If worker type configuration is invalid
        SpecialistConfigError: If specialist configs are unusable
        ResearchToolsError: If workers won't get research tools
        DeepAgentError: If deep_agent is required but not available
    """
    validation_errors = []

    # Run all validation checks
    checks = [
        ("Required config sections", lambda: validate_required_config_sections(config)),
        ("Boss template", lambda: validate_boss_template(config)),
        ("Synthesis template", lambda: validate_synthesis_template(config)),
        ("Variant config", lambda: validate_variant_config(config)),
    ]

    if not skip_deep_agent_check:
        checks.append(("DeepAgent availability", lambda: validate_deep_agent_available()))

    # Run all checks
    for check_name, check_func in checks:
        try:
            check_func()
        except Exception as e:
            validation_errors.append((check_name, e))

    # If any validation failed, raise comprehensive error
    if validation_errors:
        error_messages = []
        for check_name, error in validation_errors:
            error_messages.append(f"\n{'='*80}")
            error_messages.append(f"FAILED: {check_name}")
            error_messages.append(f"{'='*80}")
            error_messages.append(str(error))

        raise ValueError(
            "\n"
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║             COMPREHENSIVE VALIDATION FAILED                        ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            "Multiple validation checks failed:\n"
            + "\n".join(error_messages) + "\n"
            "\n"
            "PHILOSOPHY: Fail fast - fail loud - fail at startup!\n"
            "These errors would cause HIDDEN FALLBACKS at runtime.\n"
            "Fix them now before running research.\n"
            "\n"
        )
