"""RAG (Retrieval-Augmented Generation) Tools - Modern @tool decorator pattern.

Provides semantic search and document ingestion for vector-based knowledge retrieval.
Supports ChromaDB with optional FlashRank reranking.

Both INGESTION and INFERENCE tools are provided:
- vector_search: Search vector store (inference)
- add_documents: Add documents to vector store (ingestion)

RAG DATASOURCES: Extension configs can specify rag_datasources for auto-ingestion:
  tool_settings:
    rag_datasources:
      - type: file
        path: ./data/docs/my_document.txt
        metadata: {source: "my_document.txt"}
      - type: url
        url: https://example.com/article
        metadata: {source: "example.com"}
      - type: text
        content: "Raw text content to ingest"
        metadata: {source: "inline"}
"""

import os
from src.shared.logging_config import get_logger
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator, model_validator

logger = get_logger(__name__)


# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================

# Import PyPDFLoader for PDF ingestion
try:
    from langchain_community.document_loaders import PyPDFLoader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logger.debug("PyPDFLoader not available")

# Import text splitter for chunking
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    TEXT_SPLITTER_AVAILABLE = True
except ImportError:
    TEXT_SPLITTER_AVAILABLE = False
    logger.debug("Text splitter not available")

# Import FlashRank for reranking (REQUIRED - will crash if missing)
from flashrank import Ranker
FLASHRANK_AVAILABLE = True

# Import embedding models
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import OpenAIEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.debug("Embedding models not available")

# Import ChromaDB
try:
    from langchain_chroma import Chroma
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.debug("ChromaDB not available")


# =============================================================================
# GLOBAL VECTOR STORE CLIENT (Singleton)
# =============================================================================

_chroma_client: Optional["Chroma"] = None
_client_config: Optional[Dict[str, Any]] = None
_datasources_ingested: bool = False


def initialize_rag(config: Dict[str, Any]) -> None:
    """Initialize the RAG vector store client.

    Call this once at startup to initialize ChromaDB with embeddings.
    Also auto-ingests any datasources specified in rag_datasources config.

    Args:
        config: Configuration dict with vector_store settings
    """
    global _chroma_client, _client_config, _datasources_ingested

    if not CHROMADB_AVAILABLE:
        logger.warning("ChromaDB not available - RAG tools will be disabled")
        return

    if _chroma_client is not None:
        return  # Already initialized

    _client_config = config

    # Check if vector_store is enabled
    tool_settings = config.get("tool_settings", {})
    vector_settings = tool_settings.get("vector_store", {})

    if not vector_settings.get("enabled", False):
        logger.debug("RAG/Vector store not enabled in config")
        return

    # Collection name and directory
    collection_name = os.getenv("SPAWNER_RAG_COLLECTION", vector_settings.get("collection", "main_kb"))
    persist_directory = os.getenv("SPAWNER_RAG_DIR", vector_settings.get("persist_directory", "./data/vectorstores/chroma"))

    # Create embedding model
    embedding_model = _create_embedding_model(config)
    if embedding_model is None:
        logger.warning("Could not create embedding model - RAG tools will be disabled")
        return

    try:
        settings = Settings(anonymized_telemetry=False)
        _chroma_client = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory,
            client_settings=settings
        )
        logger.info(f"RAG initialized: collection='{collection_name}', dir='{persist_directory}'")

        # Auto-ingest datasources from config
        _ingest_datasources_from_config(config, _chroma_client)

    except Exception as e:
        logger.error(f"Failed to initialize RAG: {e}")


def _ingest_datasources_from_config(config: Dict[str, Any], client: "Chroma") -> None:
    """Ingest datasources specified in extension config.

    Reads rag_datasources from tool_settings and auto-ingests them.
    Supports:
    - file: Local file paths (.txt, .md, .pdf)
    - url: URLs to fetch content from
    - text: Raw text content

    Args:
        config: Full configuration dict
        client: ChromaDB client to add documents to
    """
    global _datasources_ingested

    if _datasources_ingested:
        return  # Already ingested

    tool_settings = config.get("tool_settings", {})
    datasources = tool_settings.get("rag_datasources", [])

    if not datasources:
        logger.debug("[RAG] No datasources configured for auto-ingestion")
        return

    logger.info(f"[RAG] Auto-ingesting {len(datasources)} datasources...")

    for source in datasources:
        try:
            source_type = source.get("type", "text")
            metadata = source.get("metadata", {})

            if source_type == "file":
                _ingest_file(source.get("path"), metadata, client)
            elif source_type == "url":
                _ingest_url(source.get("url"), metadata, client)
            elif source_type == "text":
                _ingest_text(source.get("content"), metadata, client)
            else:
                logger.warning(f"[RAG] Unknown datasource type: {source_type}")
        except Exception as e:
            logger.error(f"[RAG] Failed to ingest datasource {source}: {e}")

    _datasources_ingested = True
    logger.info("[RAG] Datasource auto-ingestion complete")


def _ingest_file(file_path: str, metadata: Dict[str, Any], client: "Chroma") -> None:
    """Ingest a file into the vector store.

    Supports plain text files and PDFs. PDFs are automatically chunked with overlap
    and page numbers are stored in metadata for citation.

    Args:
        file_path: Path to the file
        metadata: Metadata to attach
        client: ChromaDB client
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"[RAG] File not found: {file_path}")
        return

    # Update base metadata with file info
    metadata = metadata.copy()
    metadata["source_type"] = "file"
    metadata["source_path"] = str(path)
    metadata["filename"] = path.name

    # Handle PDF files
    if path.suffix.lower() == '.pdf':
        if not PYPDF_AVAILABLE:
            logger.error(f"[RAG] PyPDFLoader not available, cannot ingest PDF: {file_path}")
            return

        logger.info(f"[RAG] Ingesting PDF: {file_path}")
        _ingest_pdf(path, metadata, client)
        return

    # Handle plain text files
    content = path.read_text(encoding="utf-8", errors="ignore")

    # For text files, chunk if large and splitter is available
    if TEXT_SPLITTER_AVAILABLE and len(content) > 2000:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(content)
        documents = [Document(page_content=text, metadata=metadata) for text in texts]
        logger.info(f"[RAG] Ingested text file (chunked): {file_path} ({len(texts)} chunks)")
    else:
        documents = [Document(page_content=content, metadata=metadata)]
        logger.info(f"[RAG] Ingested text file: {file_path} ({len(content)} chars)")

    client.add_documents(documents)


def _ingest_pdf(path: Path, metadata: Dict[str, Any], client: "Chroma") -> None:
    """Ingest a PDF file into the vector store.

    Extracts text from PDF pages, chunks with overlap, and stores with page metadata.

    Args:
        path: Path to the PDF file
        metadata: Base metadata to attach
        client: ChromaDB client
    """
    try:
        # Load PDF using PyPDFLoader (page mode)
        loader = PyPDFLoader(str(path), mode="page")
        pdf_documents = loader.load()

        logger.info(f"[RAG] PDF loaded: {len(pdf_documents)} pages")

        # Chunk PDF documents with overlap
        if TEXT_SPLITTER_AVAILABLE:
            # Use larger chunks for PDFs to preserve context
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(pdf_documents)
        else:
            # No chunking if splitter not available
            chunks = pdf_documents

        # Update metadata for each chunk
        for chunk in chunks:
            chunk.metadata.update(metadata)
            # Ensure page number is preserved
            if "page" not in chunk.metadata and "source" in chunk.metadata:
                # PyPDFLoader stores page in metadata
                pass

        client.add_documents(chunks)
        logger.info(f"[RAG] PDF ingested: {path.name} ({len(chunks)} chunks from {len(pdf_documents)} pages)")

    except Exception as e:
        logger.error(f"[RAG] Failed to ingest PDF {path}: {e}", exc_info=True)


def _ingest_url(url: str, metadata: Dict[str, Any], client: "Chroma") -> None:
    """Ingest content from a URL into the vector store.

    Args:
        url: URL to fetch content from
        metadata: Metadata to attach
        client: ChromaDB client
    """
    try:
        import requests
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text

        # Update metadata with URL info
        metadata = metadata.copy()
        metadata["source_type"] = "url"
        metadata["source_url"] = url

        # Add to vector store
        client.add_documents([Document(page_content=content, metadata=metadata)])
        logger.info(f"[RAG] Ingested URL: {url} ({len(content)} chars)")
    except Exception as e:
        logger.error(f"[RAG] Failed to fetch URL {url}: {e}")


def _ingest_text(content: str, metadata: Dict[str, Any], client: "Chroma") -> None:
    """Ingest raw text content into the vector store.

    Args:
        content: Text content to ingest
        metadata: Metadata to attach
        client: ChromaDB client
    """
    # Update metadata
    metadata = metadata.copy()
    metadata["source_type"] = "text"

    # Add to vector store
    client.add_documents([Document(page_content=content, metadata=metadata)])
    logger.info(f"[RAG] Ingested text content ({len(content)} chars)")


def _create_embedding_model(config: Dict[str, Any]):
    """Create embedding model using the centralized factory.

    This is a thin wrapper that maintains backward compatibility with
    the existing RAG config format while using the shared embeddings factory.

    Args:
        config: Configuration dict with tool_settings.embeddings

    Returns:
        Embeddings instance from factory
    """
    if not EMBEDDINGS_AVAILABLE:
        return None

    # Use the centralized factory
    from src.shared.embeddings.factory import get_embeddings_from_config
    return get_embeddings_from_config(config)


def get_rag_client() -> Optional["Chroma"]:
    """Get the initialized ChromaDB client."""
    return _chroma_client


# =============================================================================
# TOOL SCHEMAS
# =============================================================================

class VectorSearchInput(BaseModel):
    """Pydantic schema for vector search input validation.

    This provides automatic retry if the LLM generates invalid parameters.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query for semantic similarity search"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return (1-50)"
    )

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty or whitespace only."""
        if not v or v.strip() == "":
            raise ValueError("query cannot be empty or whitespace only")
        return v.strip()


class AddDocumentsInput(BaseModel):
    """Pydantic schema for adding documents input validation.

    This provides automatic retry if the LLM generates invalid parameters.
    """
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of document texts to add (max 100 documents)"
    )
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of metadata dicts (one per text)"
    )

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate texts list is not empty and texts are not empty."""
        if not v:
            raise ValueError("texts cannot be empty")
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"texts[{i}] cannot be empty or whitespace only")
        return v

    @model_validator(mode="after")
    def validate_metadata_length(self) -> "AddDocumentsInput":
        """Validate metadata list length matches texts list."""
        if self.metadatas is not None:
            if len(self.metadatas) != len(self.texts):
                raise ValueError(
                    f"metadatas length ({len(self.metadatas)}) must match "
                    f"texts length ({len(self.texts)})"
                )
        return self


# =============================================================================
# RAG TOOLS (Inference & Ingestion)
# =============================================================================

@tool(args_schema=VectorSearchInput)
async def vector_search(query: str, top_k: int = 5) -> str:
    """Search the vector knowledge base using semantic similarity.

    Retrieves relevant documents from the RAG vector store based on semantic
    similarity to the query. Results are ranked and formatted with citations.

    Use this when you need to:
    - Retrieve information from a knowledge base
    - Find relevant documents for research
    - Access stored facts, documentation, or reference material

    The vector store must be initialized with documents before use.

    Args:
        query: The semantic search query
        top_k: Number of results to return (default: 5, max: 50)

    Returns:
        Formatted search results with citations, or error message if not initialized
    """
    chroma_client = get_rag_client()
    if chroma_client is None:
        return "Error: Vector store not initialized. RAG tools are disabled or not configured."

    config = _client_config or {}
    vector_settings = config.get("tool_settings", {}).get("vector_store", {})
    middleware_config = config.get("middleware", {})

    # FlashRank middleware: fetch more for reranking
    flashrank_conf = middleware_config.get("use_case", {}).get("flashrank", {})
    if flashrank_conf.get("enabled", False) and FLASHRANK_AVAILABLE:
        initial_k = flashrank_conf.get("threshold_high", 50)
    else:
        initial_k = top_k

    try:
        # Search ChromaDB
        results = chroma_client.similarity_search_with_score(query, k=initial_k)

        # Convert to Documents with scores
        docs = []
        for doc, score in results:
            doc.metadata["score"] = float(score)
            docs.append(doc)

        if not docs:
            return "No results found in vector store."

        # Apply FlashRank reranking if enabled
        if flashrank_conf.get("enabled", False) and FLASHRANK_AVAILABLE:
            docs = _apply_flashrank_rerank(docs, query, flashrank_conf)

        # Limit to final top_k
        docs = docs[:top_k]

        # Format results with citations
        output = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', metadata.get('page_number', ''))
            title = metadata.get('title', '')
            chapter = metadata.get('chapter', '')

            # Build citation header
            citation_parts = [f"[{i}]"]
            if title:
                citation_parts.append(f"{title}")
            if chapter:
                citation_parts.append(f"{chapter}")
            if page:
                citation_parts.append(f"page {page}")
            if source and source != 'Unknown':
                citation_parts.append(f"({source})")
            citation_parts.append(f"(score: {metadata.get('score', 0):.3f})")

            output.append(" ".join(citation_parts))
            output.append(doc.page_content[:500])
            output.append("")

        return "\n".join(output) if output else "No results found."

    except Exception as e:
        return f"Error searching vector store: {str(e)}"


@tool(args_schema=AddDocumentsInput)
async def add_documents(texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> str:
    """Add documents to the vector store for later retrieval.

    This is the INGESTION function for RAG - adds new documents to the
    knowledge base so they can be searched later.

    Use this to:
    - Build a knowledge base from documents
    - Add reference material for future queries
    - Index documents for semantic search

    Args:
        texts: List of document texts to add
        metadatas: Optional list of metadata dicts (one per text).
                    Can include: source, title, chapter, page, etc.

    Returns:
        Success message or error
    """
    if not texts:
        return "Error: No texts provided."

    chroma_client = get_rag_client()
    if chroma_client is None:
        return "Error: Vector store not initialized. RAG tools are disabled or not configured."

    try:
        if metadatas is None:
            metadatas = [{"index": i} for i in range(len(texts))]

        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas)
        ]

        chroma_client.add_documents(documents)
        return f"Successfully added {len(texts)} document(s) to vector store."

    except Exception as e:
        return f"Error adding documents: {str(e)}"


@tool
async def list_collections() -> str:
    """List information about the vector store collection.

    Returns:
        Collection info including name, document count, and status
    """
    chroma_client = get_rag_client()
    if chroma_client is None:
        return "Error: Vector store not initialized. RAG tools are disabled or not configured."

    try:
        collection_name = chroma_client._collection.name
        count = chroma_client._collection.count()

        return (
            f"Vector Store Collection Info:\n"
            f"  Collection: {collection_name}\n"
            f"  Document Count: {count}\n"
            f"  Status: {'Ready' if count > 0 else 'Empty - no documents yet'}"
        )
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _apply_flashrank_rerank(docs: List[Document], query: str, flashrank_conf: Dict[str, Any]) -> List[Document]:
    """Apply FlashRank reranking to search results."""
    try:
        top_k = flashrank_conf.get("top_k", 10)
        reranker = Ranker()

        passages = [
            {"id": str(i), "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(docs)
        ]

        reranked = reranker.rerank(query, passages, top_k=top_k)

        reranked_docs = []
        for item in reranked:
            doc = Document(
                page_content=item["text"],
                metadata=item.get("meta", {})
            )
            reranked_docs.append(doc)

        return reranked_docs

    except Exception as e:
        logger.warning(f"FlashRank reranking failed: {e}")
        return docs
