"""Deep Research Module - Multi-agent orchestration for complex tasks.

This module provides the Spawner multi-agent system for deep research,
content generation, and complex task orchestration.
"""

from pathlib import Path
import sys

# Import local constants (self-contained, no src.constants dependency)
from .constants import PROJECT_ROOT, DEEP_RESEARCH_CONFIG_DIR, RESEARCH_EXTENSIONS_DIR

# Define BACKEND_ROOT locally (equivalent to src.constants.BACKEND_ROOT)
BACKEND_ROOT = Path(__file__).parent.parent.parent

# Graph and State (use relative imports from within the module)
from .graph.graph_factory import build_v4_graph
from .graph.graph import run_v4_spawner
from .graph.state import (
    GlobalState,
    ResearchBrief,
    WorkerResult,
    CritiqueResult,
    Citation,
    create_global_state,
)
from .graph.nodes.unified import BossDecision

# Tools - use central registry (src.shared.tools.registry)
from .tools import inject_state_to_tools

# Config
from .config_diff import merge_configs
from .config_validation import validate_config


def load_research_config(use_case: str = "rag", variant: str = "default") -> dict:
    """Load YAML config for deep research from research extensions.

    Config precedence: user variant > extension base > system base

    Args:
        use_case: Research domain (rag, podcast, trading, osint, news, content, etc.)
        variant: Specific variant (default, or custom name)

    Returns:
        Merged configuration dictionary
    """
    import yaml

    configs = []

    # Level 1: System base config
    base_path = DEEP_RESEARCH_CONFIG_DIR / "base" / "config.yaml"
    if base_path.exists():
        configs.append(base_path)

    # Level 2: Extension base config
    ext_path = RESEARCH_EXTENSIONS_DIR / use_case / "config.yaml"
    if ext_path.exists():
        configs.append(ext_path)

    # Level 3: Variant config (from extension variants/)
    if variant != "default":
        variant_path = RESEARCH_EXTENSIONS_DIR / use_case / "variants" / f"{variant}.yaml"
        if variant_path.exists():
            configs.append(variant_path)

    # Merge all configs
    merged = {}
    for config_path in configs:
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                merged = merge_configs(merged, config)

    return merged


__all__ = [
    # Graph
    "build_v4_graph",
    "run_v4_spawner",
    # State
    "GlobalState",
    "ResearchBrief",
    "WorkerResult",
    "CritiqueResult",
    "Citation",
    "BossDecision",
    "create_global_state",
    # Tools
    "inject_state_to_tools",
    # Config
    "load_research_config",
    "merge_configs",
    "validate_config",
]

__version__ = "1.0.0"
