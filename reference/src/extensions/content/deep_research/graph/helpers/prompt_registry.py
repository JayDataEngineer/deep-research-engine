"""Deep research prompt registry using prompt_engine's StrictPromptRegistry.

This module declares all required prompts for the deep_research extension
using the multi-layer StrictPromptRegistry from prompt_engine.

Validation layers:
1. Existence: All required prompts exist in config
2. Syntax: All templates have valid Jinja2 syntax
3. Variables: All template variables are declared in contracts
4. Preflight: All templates render with mock data before runtime

NO HIDDEN ERRORS. If anything is wrong, the system crashes immediately
with a clear error message - NO SILENT FALLBACKS.

"""

from typing import Dict, Any, Set

# Import from prompt_engine library
from src.extensions.libraries.prompt_engine import StrictPromptRegistry


# =============================================================================
# STRICT PROMPT REGISTRY FOR DEEP_RESEARCH
# =============================================================================

# Create registry for deep_research
REGISTRY = StrictPromptRegistry("deep_research")

# =============================================================================
# REQUIRED PROMPTS
# =============================================================================

# Core agent prompts (required for basic operation)
REGISTRY.require(
    "worker",
    "boss",
    "boss.structured",
    "boss.guidance",
    "boss.guidance.detailed",
    "boss.adjustment",
    "boss_dependencies",
    "researcher",
    "critic",
    "critic.system",
    "synthesis",
    "synthesis.default",
)

# Flash explorer prompts (used by common.py and flash_explorer.py)
REGISTRY.require(
    "flash_explorer.planning",
    "flash_explorer.planning.system",
    "flash_explorer.synthesis",
    "flash_explorer.synthesis.system",
)

# Extraction prompt (used by deep_agent_helpers.py)
REGISTRY.require("extraction.json")

# =============================================================================
# VARIABLE CONTRACTS
# =============================================================================

# Boss prompts
REGISTRY.declare_variables(
    "boss",
    required=set(),
    optional={
        "original_query", "complexity", "approach", "previous_strategy", "iteration",
        "research_vectors", "max_researchers", "recommended_approach", "worker_id",
        "boss_role", "domain", "user_query", "ambiguity_detected", "execution_mode",
        "available_skills", "operation_type", "worker_role", "task_allocation_guidance",
        "dependency_graph_prompt", "citation_requirements", "dependencies_output",
        "skills_text", "worker_types_desc", "refinement_context"
    }
)

REGISTRY.declare_variables(
    "boss.guidance",
    required={"original_query"},
    optional={"complexity", "approach", "previous_strategy", "iteration", "research_vectors"}
)

REGISTRY.declare_variables(
    "boss.spawn",
    required={"original_query"},
    optional={"iteration"}
)

REGISTRY.declare_variables(
    "boss.structured",
    required={"original_query"},
    optional={"complexity", "research_vectors", "max_researchers", "recommended_approach"}
)

REGISTRY.declare_variables(
    "boss.guidance.detailed",
    required=set(),
    optional={"original_query", "complexity", "approach", "research_vectors", "research_team",
              "brand_identity", "brand_tagline", "tone_style", "keywords"}
)

REGISTRY.declare_variables(
    "boss.adjustment",
    required=set(),
    optional={"original_query", "worker_id", "feedback", "previous_task", "worker_list"}
)

# Critic prompts
REGISTRY.declare_variables(
    "critic",
    required=set(),
    optional={
        "research_brief", "worker_findings", "quality_threshold", "history", "tool_info",
        "critic_role", "output_type", "evaluation_focus", "evaluation_guidelines"
    }
)

REGISTRY.declare_variables(
    "critic.system",
    required=set(),
    optional={"tool_info", "research_findings", "session_id"}
)

# Synthesis prompts
REGISTRY.declare_variables(
    "synthesis.default",
    required={"query"},
    optional={"session_id", "synthesis_role", "all_findings", "research_findings"}
)

REGISTRY.declare_variables(
    "synthesis",
    required=set(),
    optional={
        "findings", "synthesis_role", "output_type", "worker_role",
        "synthesis_guidance", "citation_instructions", "output_instructions"
    }
)

# Flash explorer prompts
REGISTRY.declare_variables(
    "flash_explorer.planning",
    required=set(),
    optional={"research_avenue", "context_section", "input_data", "has_rag", "document_count", "documents"}
)

REGISTRY.declare_variables(
    "flash_explorer.planning.system",
    required=set(),  # Static system prompt
    optional=set()
)

REGISTRY.declare_variables(
    "flash_explorer.synthesis",
    required={"findings_summary"},
    optional={"num_explorers", "research_avenue", "documents", "deep_agent_results"}
)

REGISTRY.declare_variables(
    "flash_explorer.synthesis.system",
    required=set(),
    optional=set()
)

REGISTRY.declare_variables(
    "flash_explorer.evaluation",
    required=set(),
    optional={"research_avenue", "searches_performed", "successful_agents", "total_agents", "total_sources", "findings_summary"}
)

# Extraction prompts
REGISTRY.declare_variables(
    "extraction.json",
    required=set(),
    optional={"extraction_target", "content_to_extract", "agent_output", "extraction_description"}
)

REGISTRY.declare_variables(
    "extraction.boss_decision",
    required=set(),
    optional={"agent_output"}
)

REGISTRY.declare_variables(
    "extraction.critic_response",
    required=set(),
    optional={"agent_output"}
)

REGISTRY.declare_variables(
    "extraction.research_plan",
    required=set(),
    optional={"agent_output"}
)

REGISTRY.declare_variables(
    "extraction.flash_explorer_findings",
    required=set(),
    optional={"agent_output"}
)

# Worker prompt (default)
REGISTRY.declare_variables(
    "worker",
    required=set(),
    optional={"query", "research_brief", "focus_area", "available_tools", "domain_context",
              "raw_query", "task", "context", "documents", "worker_id", "assigned_task"}
)

# Researcher prompt (from base/config.yaml)
REGISTRY.declare_variables(
    "researcher",
    required=set(),
    optional={
        "worker_id", "worker_role_description", "assigned_task", "available_tools",
        "domain_guidance", "citation_examples"
    }
)

# Specialist prompts (all optional variables)
for specialist in ["cybersecurity", "financial", "technical", "legal", "scientific", "business", "creative", "medical", "general"]:
    REGISTRY.declare_variables(
        f"specialist.{specialist}",
        required=set(),
        optional={"task", "context", "documents", "specialty_context"}
    )

# =============================================================================
# MOCK data for preflight validation
# =============================================================================

REGISTRY.set_mock_data({
    # Boss variables
    "original_query": "[MOCK QUERY]",
    "complexity": 0.5,
    "approach": "standard",
    "previous_strategy": "",
    "iteration": 1,
    "research_vectors": "[MOCK RESEARCH VECTORS]",
    "max_researchers": 3,
    "recommended_approach": "standard",
    "worker_id": "worker_0",
    "worker_list": "worker_0, worker_1",
    "feedback": "[MOCK FEEDBACK]",
    "previous_task": "[MOCK PREVIOUS TASK]",
    # Additional boss variables from base/config.yaml
    "boss_role": "Boss",
    "domain": "research",
    "user_query": "[MOCK USER QUERY]",
    "ambiguity_detected": False,
    "execution_mode": "parallel",
    "available_skills": "web_search, web_get",
    "operation_type": "research",
    "worker_role": "researcher",
    "task_allocation_guidance": "Allocate tasks based on expertise",
    "dependency_graph_prompt": "",
    "citation_requirements": "Cite all sources",
    "dependencies_output": "",
    "skills_text": "web_search, web_get",
    "worker_types_desc": "research workers",
    "refinement_context": "",
    # Brand variables for detailed guidance
    "brand_identity": "Research Brand",
    "brand_tagline": "Your research partner",
    "tone_style": "professional",
    "keywords": "research, analysis",
    "research_team": "worker_0, worker_1",

    # Critic variables
    "tool_info": "[MOCK TOOL INFO]",
    "research_findings": "[MOCK FINDINGS]",
    "research_brief": "[MOCK RESEARCH BRIEF]",
    "worker_findings": "[MOCK WORKER FINDINGS]",
    "quality_threshold": 70,
    "history": "[MOCK HISTORY]",
    "session_id": "mock-session-123",
    "critic_role": "Critic",
    "output_type": "research report",
    "evaluation_focus": "",
    "evaluation_guidelines": "",

    # Synthesis variables
    "query": "[MOCK QUERY]",
    "synthesis_role": "Research Synthesizer",
    "all_findings": "[MOCK ALL FINDINGS]",
    "findings": "[MOCK FINDINGS]",
    "synthesis_guidance": "",
    "citation_instructions": "",
    "output_instructions": "",

    # Flash explorer variables
    "findings_summary": "[MOCK FINDINGS SUMMARY]",
    "num_explorers": 3,
    "research_avenue": "[MOCK AVENUE]",
    "context_section": "[MOCK CONTEXT]",
    "input_data": "[MOCK INPUT DATA]",
    "has_rag": False,
    "document_count": 0,
    "documents": [],
    "deep_agent_results": [],
    "searches_performed": 5,
    "successful_agents": 3,
    "total_agents": 3,
    "total_sources": 10,

    # Extraction variables
    "extraction_target": "json",
    "content_to_extract": "[MOCK CONTENT]",
    "agent_output": "[MOCK AGENT OUTPUT]",
    "extraction_description": "[MOCK DESCRIPTION]",

    # Architect/Boss variables
    "focus_area": "[MOCK FOCUS]",
    "raw_query": "[MOCK RAW QUERY]",
    "available_tools": "web_search, web_get",
    "domain_context": "",

    # Worker/specialist variables
    "task": "[MOCK TASK]",
    "context": "[MOCK CONTEXT]",
    "specialty_context": "[MOCK SPECIALTY CONTEXT]",
    "worker_role_description": "research worker",
    "assigned_task": "[MOCK ASSIGNED TASK]",
    "domain_guidance": "",
    "citation_examples": "[Source: url]",
})

# =============================================================================
# KNOWN KEYS (for typo detection)
# =============================================================================

REGISTRY.known_keys(
    # Core agent prompts
    "worker",
    "boss",
    "boss.structured",
    "boss.spawn",  # Main boss node spawn decision prompt
    "boss.guidance",
    "boss.guidance.detailed",
    "boss.adjustment",
    "boss_dependencies",
    "researcher",
    "critic",
    "critic.system",
    "synthesis",
    "synthesis.default",
    "scout",  # Scout/worker analysis prompt

    # Flash explorer prompts
    "flash_explorer.planning",
    "flash_explorer.planning.system",
    "flash_explorer.planning.user",
    "flash_explorer.synthesis",
    "flash_explorer.synthesis.system",
    "flash_explorer.synthesis.user",
    "flash_explorer.evaluation",

    # Extraction prompts
    "extraction.json",
    "extraction.boss_decision",
    "extraction.critic_response",
    "extraction.research_plan",
    "extraction.flash_explorer_findings",

    # Worker planning prompt (used in configs)
    "worker.planning",

    # Specialist prompts
    "specialist.cybersecurity",
    "specialist.financial",
    "specialist.technical",
    "specialist.legal",
    "specialist.scientific",
    "specialist.business",
    "specialist.creative",
    "specialist.medical",
    "specialist.general",
    "specialist.worker",
    "specialist.scout",

    # Metadata fields (allowed in dict prompts)
    "synthesis_role",
    "synthesis_guidance",
    "citation_instructions",
    "output_instructions",
    "evaluation_focus",
    "critic_role",
    "worker_role",
    "output_type",
    "domain",

    # Internal prompts (for overdrive mode, etc.)
    "overdrive_modifier",
    "parallel_worker_analysis",
    "sub_worker_focused",
    "boss_structured",
)

# STRICT MODE: Crash on unknown prompt keys (catches typos)
# If you need custom prompts, add them to known_keys() above
REGISTRY.allow_unknown(False)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_prompt(config: Dict[str, Any], key: str, **kwargs) -> str:
    """Get and render a prompt. CRASH if missing or contract violated.

    This is the primary function for retrieving prompts in deep_research.
    It uses StrictPromptRegistry for multi-layer validation:
    - Existence check
    - Jinja2 syntax validation
    - Variable contract enforcement
    - Output length constraints

    Args:
        config: Configuration dictionary containing prompts
        key: Type of prompt to retrieve (e.g., 'worker', 'boss.guidance', 'flash_explorer.planning')
        **kwargs: Variables to render into the prompt using Jinja2

    Returns:
        Rendered prompt string

    Raises:
        RuntimeError: If prompt is missing, contract violated, or render fails
    """
    return REGISTRY.get_prompt(config, key, **kwargs)


def validate_prompts(config: Dict[str, Any]) -> None:
    """Validate config with ALL layers. CRASH on ANY issue.

    Runs all validation layers:
    1. Existence: All required prompts exist
    2. Syntax: All templates have valid Jinja2 syntax
    3. Variables: All template variables are in contracts
    4. Preflight: All templates render with mock data before runtime

    Args:
        config: Full configuration dictionary (must contain 'prompts' key)

    Raises:
        RuntimeError: If any validation fails
    """
    REGISTRY.validate(config)


def get_validation_status(config: Dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """Get validation status without crashing.
    Useful for CLI tools that want to show warnings without crashing.
    Args:
        config: Full configuration dictionary

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    try:
        REGISTRY.validate(config)
        return True, [], []
    except RuntimeError as e:
        return False, [str(e)], []


# =============================================================================
# EXPORTed constants (for documentation/error messages)
# =============================================================================

REQUIRED_PROMPTS = sorted(REGISTRY._required)

KNOWN_PROMPT_KEYS = sorted(REGISTRY._known_keys)
