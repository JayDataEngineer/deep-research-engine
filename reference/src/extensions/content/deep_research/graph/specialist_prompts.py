"""Specialist prompt templates for dynamically generated workers.

This module loads domain-specific prompt templates from the YAML config
for specialized research workers.

CRASH-ONLY: All prompts are loaded from config, no hardcoded fallbacks.
"""

from typing import Dict, Any, Optional

# =============================================================================
# DOMAIN TO PROMPT KEY MAPPING
# =============================================================================
# Maps domain identifiers used in code to prompt keys in YAML config
DOMAIN_TO_PROMPT_KEY: Dict[str, str] = {
    "security": "specialist.cybersecurity",
    "financial": "specialist.financial",
    "technical": "specialist.technical",
    "legal": "specialist.legal",
    "scientific": "specialist.scientific",
    "business": "specialist.business",
    "creative": "specialist.creative",
    "medical": "specialist.medical",
    "general": "specialist.general",
}


# =============================================================================
# SUB-WORKER INSTRUCTIONS
# =============================================================================

SUB_WORKER_INSTRUCTIONS = """
Web Specialist Sub-Workers

You can delegate web research tasks to autonomous sub-workers that operate
in parallel. This is useful when you need to research multiple different topics
or gather information from multiple sources simultaneously.

How to Use Sub-Workers:

To delegate a research task, use the [SUB_WORKER: ...] syntax:

[SUB_WORKER: What are progressive reactions to Trump's Greenland proposal?]

[SUB_WORKER: How are community organizers responding to ICE raids in Minnesota?]

[SUB_WORKER: What are socialist perspectives on the Canada-China trade deal?]

When to Use Sub-Workers:
- Researching multiple different topics or stories
- Gathering reactions from different perspectives
- Cross-referencing information across multiple sources
- Parallel verification of claims

What Sub-Workers Can Do:
- web_search: Search for relevant information
- web_get: Fetch and read specific URLs
- web_map: Discover site structure and linked pages
- web_crawl: Crawl multiple pages for comprehensive coverage

Sub-Worker Output:
Each sub-worker returns structured findings with:
- Executive summary of key findings
- Detailed findings organized by topic
- Full source URLs
- Confidence assessment

After sub-workers complete, synthesize their findings into your main response.
"""

def get_sub_worker_instructions(enable: bool = False) -> str:
    """Get sub-worker instructions for worker prompts.

    Args:
        enable: Whether sub-workers are enabled for this run

    Returns:
        Sub-worker instruction string (empty if disabled)
    """
    if enable:
        return SUB_WORKER_INSTRUCTIONS
    return ""


# =============================================================================
# SPECIALIST PROMPT TEMPLATES
# =============================================================================

def _create_domain_prompt_template(
    readable_name: str,
    domain: str,
    yaml_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a domain-specific prompt template for a specialist worker.

    CRASH-ONLY: Loads prompts from YAML config. No hardcoded fallbacks.

    Args:
        readable_name: Human-readable worker type name
        domain: Domain identifier (security, financial, technical, etc.)
        yaml_config: Full YAML configuration (REQUIRED)

    Returns:
        Formatted prompt template string

    Raises:
        RuntimeError: If yaml_config is not provided or prompt key is missing
    """
    if not yaml_config:
        raise RuntimeError(
            f"[SPECIALIST_PROMPTS] CRITICAL: yaml_config is REQUIRED for domain '{domain}'. "
            f"No hardcoded fallbacks allowed - pass config from caller."
        )

    # Import here to avoid circular imports
    from src.extensions.content.deep_research.graph.helpers.prompts import get_prompt_from_config

    # Map domain to prompt key
    prompt_key = DOMAIN_TO_PROMPT_KEY.get(domain)
    if not prompt_key:
        # Unknown domain - use general specialist
        prompt_key = "specialist.general"

    # CRASH-ONLY: Get prompt from config (will crash if missing)
    try:
        prompt = get_prompt_from_config(
            yaml_config,
            prompt_key,
            readable_name=readable_name,
        )
        return prompt
    except Exception as e:
        raise RuntimeError(
            f"[SPECIALIST_PROMPTS] CRITICAL: Failed to load prompt '{prompt_key}' "
            f"for domain '{domain}': {e}"
        ) from e


def get_specialist_prompt_template(
    worker_type: str,
    domain: str,
    yaml_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Get the specialist prompt template for a given worker type and domain.

    CRASH-ONLY: Requires yaml_config, no hardcoded fallbacks.

    Args:
        worker_type: The worker type identifier (e.g., "security_analyst", "technical_researcher")
        domain: The domain identifier (e.g., "security", "financial", "technical")
        yaml_config: Full YAML configuration (REQUIRED)

    Returns:
        Formatted prompt template string

    Raises:
        RuntimeError: If yaml_config is not provided
    """
    readable_name = worker_type.replace("_", " ").replace("-", " ").title()
    return _create_domain_prompt_template(readable_name, domain, yaml_config=yaml_config)
