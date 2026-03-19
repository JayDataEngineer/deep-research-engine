"""
Conversation Management for Companion Brain
"""

from .manager import ConversationManager, create_conversation_manager
from .summarization import (
    ConversationSummarizer,
    SummaryManager,
    create_summarizer,
    create_summary_manager,
)
from .variety import (
    ConversationVarietyManager,
    VarietyConfig,
    ResponsePattern,
    create_variety_manager,
)

__all__ = [
    "ConversationManager",
    "create_conversation_manager",
    "ConversationSummarizer",
    "SummaryManager",
    "create_summarizer",
    "create_summary_manager",
    "ConversationVarietyManager",
    "VarietyConfig",
    "ResponsePattern",
    "create_variety_manager",
]
