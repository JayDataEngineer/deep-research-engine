"""Document Specialist - Document processing and summarization.

Provides tools and subagent for reading, analyzing, and summarizing
documents stored in the LangGraph Store.

Usage:
    from context_engine.specialists import create_document_specialist

    subagent = create_document_specialist(store, session_id)
"""

from typing import Any

from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from loguru import logger


# =============================================================================
# DOCUMENT TOOLS
# =============================================================================

def create_summarize_document_tool(store: BaseStore, session_id: str) -> Any:
    """Create document summarization tool.

    Args:
        store: LangGraph BaseStore
        session_id: Current session ID

    Returns:
        LangChain tool for document summarization
    """
    @tool
    async def summarize_document(file_key: str, max_length: int = 500) -> str:
        """Summarize a document stored in the context.

        Reads the document from the store and returns a concise summary.

        Args:
            file_key: Key of the document in the store
            max_length: Maximum length of summary in words (default: 500)

        Returns:
            Summary of the document
        """
        # This is a tool definition - actual summarization happens via LLM
        # The tool retrieves the document and returns it for LLM processing
        namespace = (session_id, "input")

        try:
            doc = await store.aget(namespace, file_key)
            if doc is None:
                return f"Document not found: {file_key}"

            content = doc.value.get("content", "")
            if not content:
                return f"Document has no content: {file_key}"

            # Return content for LLM to summarize
            return f"Document: {file_key}\n\nContent:\n{content}"

        except Exception as e:
            logger.error(f"Error reading document {file_key}: {e}")
            return f"Error reading document: {e}"

    return summarize_document


def create_extract_key_points_tool(store: BaseStore, session_id: str) -> Any:
    """Create key points extraction tool.

    Args:
        store: LangGraph BaseStore
        session_id: Current session ID

    Returns:
        LangChain tool for extracting key points
    """
    @tool
    async def extract_key_points(file_key: str, num_points: int = 5) -> str:
        """Extract key points from a document.

        Args:
            file_key: Key of the document in the store
            num_points: Number of key points to extract

        Returns:
            Key points from the document
        """
        namespace = (session_id, "input")

        try:
            doc = await store.aget(namespace, file_key)
            if doc is None:
                return f"Document not found: {file_key}"

            content = doc.value.get("content", "")
            return f"Document: {file_key}\nExtract {num_points} key points:\n\n{content}"

        except Exception as e:
            logger.error(f"Error reading document {file_key}: {e}")
            return f"Error reading document: {e}"

    return extract_key_points


def create_list_documents_tool(store: BaseStore, session_id: str) -> Any:
    """Create document listing tool.

    Args:
        store: LangGraph BaseStore
        session_id: Current session ID

    Returns:
        LangChain tool for listing documents
    """
    @tool
    async def list_documents() -> str:
        """List all documents in the input namespace.

        Returns:
            List of document keys and metadata
        """
        namespace = (session_id, "input")

        try:
            items = await store.alist(namespace)
            if not items:
                return "No documents found in input namespace."

            output = ["## Input Documents\n"]
            for item in items:
                meta = item.value.get("metadata", {})
                output.append(f"- **{item.key}**")
                if meta.get("source"):
                    output.append(f"  - Source: {meta['source']}")
                if meta.get("type"):
                    output.append(f"  - Type: {meta['type']}")

            return "\n".join(output)

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return f"Error listing documents: {e}"

    return list_documents


def create_all_document_tools(store: BaseStore, session_id: str) -> list[Any]:
    """Create all document processing tools.

    Args:
        store: LangGraph BaseStore
        session_id: Current session ID

    Returns:
        List of document tools
    """
    return [
        create_summarize_document_tool(store, session_id),
        create_extract_key_points_tool(store, session_id),
        create_list_documents_tool(store, session_id),
    ]


# =============================================================================
# DOCUMENT SPECIALIST SUBAGENT
# =============================================================================

def create_document_specialist(
    store: BaseStore,
    session_id: str,
) -> dict[str, Any]:
    """Create document specialist subagent definition.

    This specialist can read, analyze, and summarize documents
    stored in the context. Uses FAST model (Haiku).

    Args:
        store: LangGraph BaseStore
        session_id: Current session ID

    Returns:
        Subagent definition dict for document processing
    """
    from context_engine.config import get_model
    from context_engine.store.tools import create_search_vectors_tool

    # Get FAST model (Haiku)
    model = get_model("fast")

    # Tools: context search + document tools
    tools = [
        create_search_vectors_tool(store, session_id),
        *create_all_document_tools(store, session_id),
    ]

    return {
        "name": "document-specialist",
        "description": (
            "Document analysis and summarization specialist. "
            "Can read documents, extract key points, and summarize content. "
            "Give a document key or topic to analyze."
        ),
        "system_prompt": """You are a Document Specialist.

Your job is to analyze and summarize documents stored in the context.

## Available Tools

- **search_vectors(query)**: Search context semantically
- **list_documents()**: List all available documents
- **summarize_document(key)**: Get a document summary
- **extract_key_points(key, n)**: Extract n key points from a document

## Workflow

1. First use list_documents to see what's available
2. Use search_vectors to find relevant documents by topic
3. Use summarize_document or extract_key_points for detailed analysis

## Output Format

```
## Document Analysis: [Document Name]

### Summary
[2-3 sentence summary]

### Key Points
1. Point 1
2. Point 2
3. Point 3

### Relevance
[How this relates to the research query]
```

Be concise and focus on actionable insights.""",
        "model": model,
        "tools": tools,
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_quick_summary_specialist(
    store: BaseStore,
    session_id: str,
) -> dict[str, Any]:
    """Create a quick summary specialist for fast document summaries.

    Simpler than full document specialist - just list and summarize.

    Args:
        store: LangGraph BaseStore
        session_id: Current session ID

    Returns:
        Subagent definition dict for quick summaries
    """
    from context_engine.config import get_model

    model = get_model("fast")

    return {
        "name": "quick-summary",
        "description": (
            "Quick document summarization. "
            "Returns brief summaries of documents. "
            "Use for fast overview, not deep analysis."
        ),
        "system_prompt": """Quick document summarizer.

Use list_documents and summarize_document to provide quick overviews.

Format:
- Document: [name]
- Summary: [1-2 sentences]""",
        "model": model,
        "tools": create_all_document_tools(store, session_id),
    }
