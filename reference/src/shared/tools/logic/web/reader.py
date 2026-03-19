"""Web reader - re-export from ulli_langchain/web.

This module re-exports the LangChain @tool decorated web_get
from ulli_langchain.web to avoid code duplication.

Usage:
    from logic.web import web_read
    content = await web_read(url="https://example.com")
"""

from src.shared.tools.ulli_langchain.web.web_get import web_get as web_read

__all__ = ["web_read"]
