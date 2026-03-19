"""Text Mode Graph Factory - Helper for building chat mode graphs.

This factory provides convenience functions for building chat mode graphs.
It's completely generic with NO hardcoded mode mappings - extensions
pass in their mode name and configuration directly.

Extensions can use this factory OR build their own graphs entirely.
"""

import logging
from typing import Any, Optional, List
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
if str(src_path) not in Path(__file__).resolve().parent.parts:
    import sys
    sys.path.insert(0, str(src_path))

from src.config import settings
from src.shared.checkpointer import get_shared_checkpointer
from src.extensions.libraries.core_chat_engine.brain_graph import create_chat_mode_graph
from src.extensions.libraries.core_chat_engine.config.brain_config import BrainConfig

logger = logging.getLogger(__name__)


async def build_text_mode_graph(
    mode: str,
    brain_config: BrainConfig,
    personality_enabled: bool = True,
    enabled_tools: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Any:
    """Build a chat graph for a text mode.

    This is a helper function that extensions can optionally use.
    Extensions are free to build their own graphs if they need
    custom logic.

    Args:
        mode: Mode identifier (used for internal routing)
        brain_config: BrainConfig instance for this mode
        personality_enabled: Whether to enable personality features
        enabled_tools: Optional list of tool names to enable (from extension config)
        api_key: LLM API key (defaults to settings.llm_api_key)
        base_url: LLM base URL (defaults to settings.llm_base_url)
        model: Model name (defaults to settings.llm_model)

    Returns:
        Compiled StateGraph ready for streaming

    Example:
        from src.extensions.libraries.core_chat_engine.config.brain_config import BrainConfig, PersonalityConfig

        config = BrainConfig(
            mode="my_mode",
            personality=PersonalityConfig(name="My Mode"),
            enable_tool_routing=True,
        )
        graph = build_text_mode_graph(mode="my_mode", brain_config=config)
    """
    # Use defaults from settings if not provided
    api_key = api_key or settings.llm_api_key
    base_url = base_url or settings.llm_base_url
    model = model or settings.llm_model
    checkpointer = await get_shared_checkpointer()

    # Build the graph using the chat_mode_graph factory
    graph = create_chat_mode_graph(
        api_key=api_key,
        base_url=base_url,
        model=model,
        config=brain_config,
        mode=mode,
        personality_enabled=personality_enabled,
        enabled_tools=enabled_tools,
        checkpointer=checkpointer,
    )

    logger.debug(
        f"[FACTORY] Built graph for mode '{mode}' "
        f"(personality_enabled={personality_enabled})"
    )
    return graph
