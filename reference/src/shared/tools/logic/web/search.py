"""Web search - re-export from ulli_langchain/web.

This module re-exports the LangChain @tool decorated web_search
from ulli_langchain.web to avoid code duplication.

Usage:
    from logic.web import web_search
    result = await web_search(query="AI news", num_results=5)
"""

from src.shared.tools.ulli_langchain.web.web_search import web_search

__all__ = ["web_search"]
