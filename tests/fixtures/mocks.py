"""Advanced Mock Fixtures for Testing.

Comprehensive mock fixtures for:
- LLM providers (OpenAI, Anthropic, etc.)
- Web search APIs
- Database connections
- HTTP requests
- External services

Usage:
    from tests.conftest import MockLLMProvider, mock_web_search
"""

import pytest
import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import time


# =============================================================================
# MOCK LLM PROVIDER
# =============================================================================

@dataclass
class LLMResponse:
    """Mock LLM response structure."""
    content: str
    model: str = "mock-model"
    prompt_tokens: int = 100
    completion_tokens: int = 50
    total_tokens: int = 150
    finish_reason: str = "stop"


@dataclass
class LLMCall:
    """Record of an LLM call for verification."""
    messages: List[Dict[str, str]]
    model: str
    temperature: float
    max_tokens: Optional[int]
    response: LLMResponse
    timestamp: float = field(default_factory=time.time)


class MockLLMProvider:
    """Mock LLM provider for testing without API calls.
    
    Features:
    - Pre-programmed responses
    - Call recording for verification
    - Configurable delays
    - Error simulation
    - Token counting
    """
    
    def __init__(self, default_model: str = "mock-model"):
        self.default_model = default_model
        self.responses: Dict[str, LLMResponse] = {}
        self.call_history: List[LLMCall] = []
        self.delay: float = 0.0  # Simulated delay in seconds
        self.error_rate: float = 0.0  # Probability of error
        self.custom_handler: Optional[Callable] = None
        
    def set_response(self, pattern: str, response: str, model: str = "mock-model"):
        """Set a canned response for messages matching pattern."""
        self.responses[pattern] = LLMResponse(content=response, model=model)
        
    def set_default_response(self, response: str, model: str = "mock-model"):
        """Set default response for unmatched messages."""
        self.set_response("__default__", response, model)
        
    def set_delay(self, seconds: float):
        """Set simulated API delay."""
        self.delay = seconds
        
    def set_error_rate(self, rate: float):
        """Set probability of simulated errors (0.0-1.0)."""
        self.error_rate = rate
        
    def set_custom_handler(self, handler: Callable):
        """Set custom handler for dynamic responses."""
        self.custom_handler = handler
        
    def clear_history(self):
        """Clear call history."""
        self.call_history.clear()
        
    def get_call_count(self) -> int:
        """Get number of LLM calls made."""
        return len(self.call_history)
        
    def get_last_call(self) -> Optional[LLMCall]:
        """Get the last LLM call."""
        return self.call_history[-1] if self.call_history else None
        
    def find_call(self, pattern: str) -> Optional[LLMCall]:
        """Find a call containing specific pattern."""
        for call in self.call_history:
            for msg in call.messages:
                if pattern in msg.get("content", ""):
                    return call
        return None
        
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Mock chat completion endpoint."""
        # Simulate delay
        if self.delay > 0:
            await asyncio.sleep(self.delay)
            
        # Simulate random errors
        import random
        if random.random() < self.error_rate:
            raise Exception("Simulated API error")
            
        # Use custom handler if set
        if self.custom_handler:
            response = self.custom_handler(messages, model, temperature)
            if isinstance(response, LLMResponse):
                return response
                
        # Find matching response
        content = " ".join(m.get("content", "") for m in messages)
        response = None
        
        for pattern, resp in self.responses.items():
            if pattern == "__default__" or pattern in content:
                response = resp
                break
                
        if response is None:
            response = LLMResponse(
                content="Default mock response",
                model=model or self.default_model
            )
            
        # Record call
        call = LLMCall(
            messages=messages,
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            response=response
        )
        self.call_history.append(call)
        
        return response
        
    def create_patcher(self) -> patch:
        """Create a patcher for common LLM libraries."""
        # This would patch openai, anthropic, etc.
        return patch.multiple(
            'openai.AsyncOpenAI',
            chat=MagicMock(side_effect=lambda **kwargs: self.chat_completion(**kwargs))
        )


# =============================================================================
# MOCK WEB SEARCH
# =============================================================================

@dataclass
class SearchResult:
    """Mock search result."""
    title: str
    url: str
    snippet: str
    position: int
    source: str = "web"


@dataclass
class SearchCall:
    """Record of a search call."""
    query: str
    num_results: int
    time_range: Optional[str]
    results: List[SearchResult]
    timestamp: float = field(default_factory=time.time)


class MockWebSearch:
    """Mock web search API for testing.
    
    Features:
    - Pre-programmed search results
    - Call recording
    - Configurable result counts
    - Error simulation
    """
    
    def __init__(self):
        self.results: Dict[str, List[SearchResult]] = {}
        self.call_history: List[SearchCall] = []
        self.default_results: List[SearchResult] = []
        self.delay: float = 0.0
        self.error_queries: set = set()
        
    def add_results(self, query_pattern: str, results: List[Dict[str, str]]):
        """Add mock results for queries matching pattern."""
        search_results = [
            SearchResult(
                title=r.get("title", "Result"),
                url=r.get("url", "http://example.com"),
                snippet=r.get("snippet", "Snippet"),
                position=i,
                source=r.get("source", "web")
            )
            for i, r in enumerate(results)
        ]
        self.results[query_pattern] = search_results
        
    def set_default_results(self, results: List[Dict[str, str]]):
        """Set default results for unmatched queries."""
        self.default_results = [
            SearchResult(
                title=r.get("title", "Default Result"),
                url=r.get("url", "http://example.com"),
                snippet=r.get("snippet", "Default snippet"),
                position=i
            )
            for i, r in enumerate(results)
        ]
        
    def set_error_query(self, query: str):
        """Mark a query to always fail."""
        self.error_queries.add(query)
        
    def clear_history(self):
        """Clear call history."""
        self.call_history.clear()
        
    def get_call_count(self) -> int:
        """Get number of search calls made."""
        return len(self.call_history)
        
    async def search(
        self,
        query: str,
        num_results: int = 10,
        time_range: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Mock search endpoint."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
            
        # Check for error query
        if query in self.error_queries:
            raise Exception(f"Simulated search error for: {query}")
            
        # Find matching results
        search_results = None
        for pattern, results in self.results.items():
            if pattern in query:
                search_results = results
                break
                
        if search_results is None:
            search_results = self.default_results or [
                SearchResult(
                    title=f"Result for: {query}",
                    url=f"http://example.com/{i}",
                    snippet=f"Snippet {i} for {query}",
                    position=i
                )
                for i in range(min(5, num_results))
            ]
            
        # Limit results
        search_results = search_results[:num_results]
        
        # Record call
        call = SearchCall(
            query=query,
            num_results=num_results,
            time_range=time_range,
            results=search_results
        )
        self.call_history.append(call)
        
        return search_results
        
    def search_to_findings(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Convert search results to worker findings format."""
        return {
            "query": results[0].url if results else "",
            "findings": [
                {
                    "title": r.title,
                    "url": r.url,
                    "content": r.snippet,
                    "source": r.source
                }
                for r in results
            ],
            "source_count": len(results)
        }


# =============================================================================
# MOCK DATABASE
# =============================================================================

class MockDatabase:
    """Mock database for testing without real connections.

    Supports:
    - PostgreSQL-style operations
    - Query recording
    - Result mocking
    - Transaction simulation
    """

    def __init__(self):
        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.query_history: List[str] = []
        self.in_transaction: bool = False
        self.transaction_queries: List[str] = []
        self.error_queries: set = set()
        self.initialized: bool = False

    def initialize(self):
        """Initialize the mock database."""
        self.initialized = True
        self.tables = {}
        self.query_history = []
        
    def create_table(self, name: str, columns: Dict[str, str]):
        """Create a mock table."""
        self.tables[name] = []
        
    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert data into table."""
        if table not in self.tables:
            self.tables[table] = []
        data["id"] = len(self.tables[table]) + 1
        self.tables[table].append(data)
        return data["id"]
        
    def select(self, table: str, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Select from table with optional where clause."""
        if table not in self.tables:
            return []
            
        results = self.tables[table]
        if where:
            results = [
                row for row in results
                if all(row.get(k) == v for k, v in where.items())
            ]
        return results
        
    def update(self, table: str, data: Dict[str, Any], where: Dict[str, Any]) -> int:
        """Update rows in table."""
        if table not in self.tables:
            return 0
            
        count = 0
        for row in self.tables[table]:
            if all(row.get(k) == v for k, v in where.items()):
                row.update(data)
                count += 1
        return count
        
    def delete(self, table: str, where: Dict[str, Any]) -> int:
        """Delete rows from table."""
        if table not in self.tables:
            return 0
            
        original_len = len(self.tables[table])
        self.tables[table] = [
            row for row in self.tables[table]
            if not all(row.get(k) == v for k, v in where.items())
        ]
        return original_len - len(self.tables[table])
        
    async def execute(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute raw SQL query."""
        self.query_history.append(query)
        
        if query in self.error_queries:
            raise Exception(f"Simulated DB error for: {query}")
            
        # Simple query parsing (very basic)
        query_lower = query.lower().strip()
        
        if query_lower.startswith("select"):
            # Extract table name (very simplified)
            parts = query_lower.split("from")
            if len(parts) > 1:
                table = parts[1].split()[0].strip()
                return self.select(table)
        elif query_lower.startswith("insert"):
            pass  # Would need more parsing
        elif query_lower.startswith("begin"):
            self.in_transaction = True
            self.transaction_queries = []
        elif query_lower.startswith("commit"):
            self.in_transaction = False
            self.transaction_queries = []
        elif query_lower.startswith("rollback"):
            self.in_transaction = False
            self.transaction_queries = []
            
        return []
        
    def set_error_query(self, query: str):
        """Mark a query to always fail."""
        self.error_queries.add(query)
        
    def clear(self):
        """Clear all tables and history."""
        self.tables.clear()
        self.query_history.clear()
        self.in_transaction = False
        self.transaction_queries = []


# =============================================================================
# MOCK HTTP CLIENT
# =============================================================================

class MockHTTPResponse:
    """Mock HTTP response."""
    
    def __init__(
        self,
        status_code: int = 200,
        content: bytes = b"",
        text: str = "",
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.status_code = status_code
        self.content = content or text.encode()
        self._text = text
        self._json_data = json_data
        self.headers = headers or {}
        
    @property
    def text(self) -> str:
        """Return text content."""
        return self._text
        
    def json(self) -> Dict[str, Any]:
        """Return JSON data."""
        return self._json_data or {}
        
    def raise_for_status(self):
        """Raise error for non-2xx status."""
        if 400 <= self.status_code < 600:
            raise Exception(f"HTTP Error: {self.status_code}")


class MockHTTPClient:
    """Mock HTTP client for testing.
    
    Features:
    - URL-based response mocking
    - Request recording
    - Status code simulation
    - Timeout simulation
    """
    
    def __init__(self):
        self.responses: Dict[str, MockHTTPResponse] = {}
        self.request_history: List[Dict[str, Any]] = []
        self.default_response = MockHTTPResponse(status_code=200, text="OK")
        self.timeout_urls: set = set()
        self.error_urls: set = set()
        
    def add_response(
        self,
        url_pattern: str,
        status_code: int = 200,
        json_data: Optional[Dict[str, Any]] = None,
        text: str = "",
        headers: Optional[Dict[str, str]] = None
    ):
        """Add mock response for URL pattern."""
        self.responses[url_pattern] = MockHTTPResponse(
            status_code=status_code,
            json_data=json_data,
            text=text,
            headers=headers
        )
        
    def set_default_response(self, response: MockHTTPResponse):
        """Set default response for unmatched URLs."""
        self.default_response = response
        
    def set_timeout(self, url: str):
        """Mark URL to always timeout."""
        self.timeout_urls.add(url)
        
    def set_error(self, url: str, status_code: int = 500):
        """Mark URL to return error status."""
        self.error_urls.add(url)
        self.add_response(url, status_code=status_code, text="Error")
        
    def clear_history(self):
        """Clear request history."""
        self.request_history.clear()
        
    def get_request_count(self) -> int:
        """Get number of HTTP requests made."""
        return len(self.request_history)
        
    def find_request(self, url_pattern: str) -> Optional[Dict[str, Any]]:
        """Find request matching URL pattern."""
        for req in self.request_history:
            if url_pattern in req["url"]:
                return req
        return None
        
    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        **kwargs
    ) -> MockHTTPResponse:
        """Mock GET request."""
        # Record request
        request = {
            "method": "GET",
            "url": url,
            "headers": headers or {},
            "timeout": timeout
        }
        self.request_history.append(request)
        
        # Simulate timeout
        if url in self.timeout_urls:
            await asyncio.sleep(timeout + 1)
            raise Exception("Request timeout")
            
        # Find matching response
        for pattern, response in self.responses.items():
            if pattern in url:
                return response
                
        return self.default_response
        
    async def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        **kwargs
    ) -> MockHTTPResponse:
        """Mock POST request."""
        request = {
            "method": "POST",
            "url": url,
            "json": json,
            "data": data,
            "headers": headers or {},
            "timeout": timeout
        }
        self.request_history.append(request)
        
        if url in self.timeout_urls:
            await asyncio.sleep(timeout + 1)
            raise Exception("Request timeout")
            
        for pattern, response in self.responses.items():
            if pattern in url:
                return response
                
        return self.default_response


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Create mock LLM provider."""
    llm = MockLLMProvider()
    llm.set_default_response("This is a mock LLM response for testing.")
    return llm


@pytest.fixture
def mock_llm_with_delay() -> MockLLMProvider:
    """Create mock LLM provider with simulated delay."""
    llm = MockLLMProvider()
    llm.set_delay(0.1)  # 100ms delay
    llm.set_default_response("Delayed mock response")
    return llm


@pytest.fixture
def mock_llm_with_errors() -> MockLLMProvider:
    """Create mock LLM provider with error simulation."""
    llm = MockLLMProvider()
    llm.set_error_rate(0.3)  # 30% error rate
    llm.set_default_response("Response (may error)")
    return llm


@pytest.fixture
def mock_web_search() -> MockWebSearch:
    """Create mock web search."""
    search = MockWebSearch()
    search.set_default_results([
        {"title": "Default Result", "url": "http://example.com", "snippet": "Default snippet"}
    ])
    return search


@pytest.fixture
def mock_web_search_populated() -> MockWebSearch:
    """Create mock web search with pre-populated results."""
    search = MockWebSearch()
    search.add_results("AI", [
        {"title": "AI Breakthrough", "url": "http://ai-news.com/breakthrough", "snippet": "Major AI advancement"},
        {"title": "Machine Learning", "url": "http://ml-journal.com/learn", "snippet": "ML fundamentals"},
        {"title": "Neural Networks", "url": "http://deep-learning.org/nn", "snippet": "NN architecture"},
    ])
    search.add_results("climate", [
        {"title": "Climate Report", "url": "http://climate.gov/report", "snippet": "Latest climate data"},
        {"title": "Global Warming", "url": "http://science.org/warming", "snippet": "Temperature trends"},
    ])
    return search


@pytest.fixture
def mock_database() -> MockDatabase:
    """Create mock database."""
    db = MockDatabase()
    db.create_table("users", {"id": "int", "name": "str", "email": "str"})
    db.create_table("documents", {"id": "int", "title": "str", "content": "str"})
    return db


@pytest.fixture
def mock_database_populated(mock_database: MockDatabase) -> MockDatabase:
    """Create mock database with sample data."""
    db = mock_database
    
    # Add sample users
    db.insert("users", {"name": "Alice", "email": "alice@example.com"})
    db.insert("users", {"name": "Bob", "email": "bob@example.com"})
    db.insert("users", {"name": "Charlie", "email": "charlie@example.com"})
    
    # Add sample documents
    db.insert("documents", {"title": "Doc 1", "content": "Content 1"})
    db.insert("documents", {"title": "Doc 2", "content": "Content 2"})
    
    return db


@pytest.fixture
def mock_http_client() -> MockHTTPClient:
    """Create mock HTTP client."""
    client = MockHTTPClient()
    client.set_default_response(MockHTTPResponse(status_code=200, text="OK"))
    return client


@pytest.fixture
def mock_http_client_api() -> MockHTTPClient:
    """Create mock HTTP client with API responses."""
    client = MockHTTPClient()
    
    # Mock API endpoints
    client.add_response(
        "/api/v1/search",
        json_data={"results": [{"id": 1, "title": "Result 1"}]},
        headers={"Content-Type": "application/json"}
    )
    client.add_response(
        "/api/v1/status",
        json_data={"status": "healthy", "version": "1.0.0"}
    )
    
    return client


@pytest.fixture
def mock_all_services(
    mock_llm: MockLLMProvider,
    mock_web_search: MockWebSearch,
    mock_database: MockDatabase,
    mock_http_client: MockHTTPClient
) -> Dict[str, Any]:
    """Create all mock services together."""
    return {
        "llm": mock_llm,
        "web_search": mock_web_search,
        "database": mock_database,
        "http": mock_http_client
    }


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@pytest.fixture
def llm_patcher(mock_llm: MockLLMProvider):
    """Context manager for patching LLM calls."""
    async def mock_chat(**kwargs):
        return await mock_llm.chat_completion(**kwargs)
        
    with patch('openai.AsyncOpenAI') as mock_openai:
        mock_openai.return_value.chat.completions.create = mock_chat
        yield mock_openai


@pytest.fixture
def http_patcher(mock_http_client: MockHTTPClient):
    """Context manager for patching HTTP calls."""
    async def mock_get(url, **kwargs):
        return await mock_http_client.get(url, **kwargs)
        
    async def mock_post(url, **kwargs):
        return await mock_http_client.post(url, **kwargs)
        
    with patch('httpx.AsyncClient.get', mock_get):
        with patch('httpx.AsyncClient.post', mock_post):
            yield mock_http_client
