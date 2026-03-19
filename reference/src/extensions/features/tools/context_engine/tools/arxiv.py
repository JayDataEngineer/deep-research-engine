"""
ArXiv Integration Tool

ArXiv integration for academic paper search. Uses the public ArXiv API (no API key required).
"""
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from typing import List, Dict, Any, Optional
from arxiv import Client
import logging

logger = logging.getLogger(__name__)

ARXIV_SORT_OPTIONS = {
    "relevance": "relevance",
    "lastUpdatedDate": "lastUpdatedDate",
    "submittedDate": "submittedDate"
}


def format_arxiv_results(results: List[Dict[str, Any]], query: str) -> str:
    """Format ArXiv search results for LLM consumption."""
    output = f"ArXiv Search Results for: {query}\n"
    output += f"Found {len(results)} papers\n\n"

    for i, paper in enumerate(results, 1):
        output += f"[{i}] {paper['title']}\n"
        output += f"    Authors: {', '.join(paper['authors'][:5])}"
        if len(paper['authors']) > 5:
            output += f" and {len(paper['authors']) - 5} others"
        output += f"\n"
        output += f"    Published: {paper['published']}\n"
        output += f"    ArXiv ID: {paper['arxiv_id']}\n"

        if paper['journal_ref']:
            output += f"    Journal: {paper['journal_ref']}\n"

        output += f"    Category: {paper['primary_category']}\n"
        output += f"    Summary: {paper['summary']}\n"
        output += f"    URL: {paper['arxiv_url']}\n"

        if paper['pdf_url']:
            output += f"    PDF: {paper['pdf_url']}\n"

        output += "\n"

    return output


def format_arxiv_details(paper: Dict[str, Any]) -> str:
    """Format detailed ArXiv paper information."""
    output = f"ArXiv Paper: {paper['title']}\n"
    output += "=" * 80 + "\n\n"

    output += f"ArXiv ID: {paper['arxiv_id']}\n"
    output += f"Published: {paper['published']}\n"

    if paper['journal_ref']:
        output += f"Journal Reference: {paper['journal_ref']}\n"

    if paper.get('doi'):
        output += f"DOI: {paper['doi']}\n"

    output += f"\nCategories: {', '.join(paper['categories'])}\n"
    output += f"Primary: {paper['primary_category']}\n"

    output += f"\nAuthors ({len(paper['authors'])}):\n"
    for author in paper['authors']:
        output += f"  - {author}\n"

    if paper.get('comment'):
        output += f"\nComment: {paper['comment']}\n"

    output += f"\nAbstract:\n{paper['abstract']}\n"

    output += f"\nLinks:\n"
    output += f"  - ArXiv: {paper['arxiv_url']}\n"
    if paper.get('pdf_url'):
        output += f"  - PDF: {paper['pdf_url']}\n"

    return output


@tool
async def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    categories: Optional[List[str]] = None,
    runtime: Optional[ToolRuntime] = None
) -> str:
    """Search academic papers on ArXiv.

    Use this when you need to:
    - Find scholarly sources for research
    - Get citations for academic claims
    - Access peer-reviewed research
    - Find recent papers in specific fields

    No API key required - uses public ArXiv API.

    Args:
        query: Search query (supports AND, OR, quotes, field-specific queries like "au:Hawking")
        max_results: Number of results to return (default: 10, max: 100)
        sort_by: Sort order - "relevance", "lastUpdatedDate", or "submittedDate"
        categories: ArXiv categories to filter (e.g., ["cs.AI", "cs.LG"])
        runtime: ToolRuntime for access to services

    Returns:
        Formatted results with title, authors, summary, publication info, and links

    Examples:
        search_arxiv("large language models efficiency")
        search_arxiv("quantum computing", categories=["quant-ph"])
        search_arxiv('au:"Geoffrey Hinton"')
    """
    try:
        client = Client()

        # Validate sort_by
        if sort_by not in ARXIV_SORT_OPTIONS:
            sort_by = "relevance"

        # Build search query
        search_query = query
        if categories:
            # Add category filters
            cat_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({query}) AND ({cat_filter})"

        # Execute search
        search = client.search(
            query=search_query,
            max_results=min(max_results, 100),  # ArXiv API limit
            sort_by=ARXIV_SORT_OPTIONS[sort_by]
        )

        if not search.results:
            return f"No ArXiv results found for query: {query}"

        # Format results
        results = []
        for result in search.results:
            paper_info = {
                "title": result.title,
                "authors": [a.name for a in result.authors[:10]],  # Limit authors
                "abstract": result.summary[:500] + "..." if len(result.summary) > 500 else result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "arxiv_id": result.entry_id.split("/")[-1],  # Extract ID from URL
                "arxiv_url": result.entry_id,
                "pdf_url": result.pdf_url,
                "primary_category": result.primary_category,
                "categories": result.categories,
                "comment": result.comment,
                "journal_ref": result.journal_ref
            }
            results.append(paper_info)

        return format_arxiv_results(results, query)

    except Exception as e:
        logger.error(f"ArXiv search failed: {e}")
        return f"ArXiv search error: {str(e)}"


@tool
async def get_arxiv_abstract(
    arxiv_id: str,
    runtime: ToolRuntime
) -> str:
    """Get full abstract and details for a specific ArXiv paper.

    Use this when you need to:
    - Get complete abstract for a paper
    - Retrieve detailed metadata
    - Access full author list

    Args:
        arxiv_id: ArXiv paper ID (e.g., "2301.07041" or "cs.AI/2301.07041")
        runtime: ToolRuntime for access to services

    Returns:
        Full paper details with complete abstract, authors, and metadata
    """
    try:
        client = Client()
        search = client.search(id_list=[arxiv_id])

        if not search.results:
            return f"ArXiv paper {arxiv_id} not found"

        result = search.results[0]

        paper_info = {
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "arxiv_id": result.entry_id.split("/")[-1],
            "arxiv_url": result.entry_id,
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "categories": result.categories,
            "comment": result.comment,
            "journal_ref": result.journal_ref,
            "doi": result.doi
        }

        return format_arxiv_details(paper_info)

    except Exception as e:
        logger.error(f"ArXiv abstract retrieval failed: {e}")
        return f"ArXiv abstract error: {str(e)}"


__all__ = ["search_arxiv", "get_arxiv_abstract"]
