# Mock Infrastructure Guide

Comprehensive mock infrastructure for testing the Deep Research Engine without external dependencies.

## Overview

The test suite includes sophisticated mocks for:
- **LLM Providers** (OpenAI, Anthropic, etc.)
- **Web Search APIs**
- **Database Connections** (PostgreSQL, MongoDB, etc.)
- **HTTP Clients** (REST APIs, webhooks)
- **External Services**

## Installation

Mocks require these dev dependencies (already in `pyproject.toml`):

```bash
uv sync --group dev
```

This installs:
- `pytest-mock>=3.14.0` - pytest-mock integration
- `vcrpy>=6.0.2` - HTTP recording/playback
- `pytest-recording>=0.13.2` - VCR pytest plugin

## Quick Start

### Using Mock Fixtures

```python
import pytest
from tests.fixtures.mocks import MockLLMProvider, MockWebSearch

@pytest.mark.asyncio
async def test_with_mocks(mock_llm: MockLLMProvider, mock_web_search: MockWebSearch):
    # Configure mock responses
    mock_llm.set_default_response("This is a mock response")
    mock_web_search.add_results("AI", [
        {"title": "AI News", "url": "http://ai.com", "snippet": "AI breakthrough"}
    ])
    
    # Use mocks
    response = await mock_llm.chat_completion(
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    results = await mock_web_search.search("AI developments")
    
    # Verify
    assert response.content == "This is a mock response"
    assert len(results) == 1
```

### Available Fixtures

| Fixture | Type | Description |
|---------|------|-------------|
| `mock_llm` | `MockLLMProvider` | Basic LLM mock |
| `mock_llm_with_delay` | `MockLLMProvider` | LLM with 100ms delay |
| `mock_llm_with_errors` | `MockLLMProvider` | LLM with 30% error rate |
| `mock_web_search` | `MockWebSearch` | Basic search mock |
| `mock_web_search_populated` | `MockWebSearch` | Search with pre-populated results |
| `mock_database` | `MockDatabase` | Basic database mock |
| `mock_database_populated` | `MockDatabase` | Database with sample data |
| `mock_http_client` | `MockHTTPClient` | Basic HTTP mock |
| `mock_http_client_api` | `MockHTTPClient` | HTTP with API responses |
| `mock_all_services` | `Dict` | All mocks together |
| `mocker` | `pytest-mock` | pytest-mock fixture |

## Mock Classes

### MockLLMProvider

Mock LLM provider for testing without API calls.

```python
from tests.fixtures.mocks import MockLLMProvider

llm = MockLLMProvider(default_model="mock-model")

# Set canned responses
llm.set_response("weather", "It's sunny today", model="weather-model")
llm.set_default_response("Default mock response")

# Configure behavior
llm.set_delay(0.1)  # 100ms simulated delay
llm.set_error_rate(0.3)  # 30% error rate

# Use custom handler for dynamic responses
def custom_handler(messages, model, temperature):
    return LLMResponse(content="Dynamic response", model=model)
llm.set_custom_handler(custom_handler)

# Make calls
response = await llm.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4",
    temperature=0.7
)

# Verify calls
assert llm.get_call_count() == 1
last_call = llm.get_last_call()
ai_call = llm.find_call("AI")  # Find call containing "AI"

# Clear history
llm.clear_history()
```

**Features:**
- ✅ Pattern-based response matching
- ✅ Call recording and verification
- ✅ Simulated delays
- ✅ Error rate simulation
- ✅ Custom response handlers
- ✅ Token counting

### MockWebSearch

Mock web search API.

```python
from tests.fixtures.mocks import MockWebSearch

search = MockWebSearch()

# Add results for specific queries
search.add_results("AI", [
    {"title": "AI Breakthrough", "url": "http://ai.com", "snippet": "Major advancement"},
    {"title": "Machine Learning", "url": "http://ml.com", "snippet": "ML basics"}
])

search.set_default_results([
    {"title": "Default Result", "url": "http://example.com", "snippet": "Default"}
])

# Mark queries to fail
search.set_error_query("bad query")

# Search
results = await search.search("AI developments", num_results=10)

# Convert to findings format (for worker integration)
findings = search.search_to_findings(results)
# Returns: {"query": "...", "findings": [...], "source_count": N}

# Verify
assert search.get_call_count() == 1
search.clear_history()
```

**Features:**
- ✅ Pattern-based result matching
- ✅ Configurable result counts
- ✅ Error simulation
- ✅ Call recording
- ✅ Findings format conversion

### MockDatabase

Mock database for testing without real connections.

```python
from tests.fixtures.mocks import MockDatabase

db = MockDatabase()

# Create tables
db.create_table("users", {"id": "int", "name": "str", "email": "str"})

# Insert data
user_id = db.insert("users", {"name": "Alice", "email": "alice@example.com"})

# Select data
all_users = db.select("users")
alice = db.select("users", where={"name": "Alice"})

# Update data
db.update("users", {"email": "new@example.com"}, where={"name": "Alice"})

# Delete data
db.delete("users", where={"name": "Alice"})

# Execute raw SQL
results = await db.execute("SELECT * FROM users")

# Transaction support
await db.execute("BEGIN")
await db.execute("COMMIT")

# Error simulation
db.set_error_query("SELECT * FROM broken")

# Clear all
db.clear()
```

**Features:**
- ✅ Table creation
- ✅ CRUD operations
- ✅ WHERE clause support
- ✅ Raw SQL execution
- ✅ Transaction simulation
- ✅ Error simulation

### MockHTTPClient

Mock HTTP client for testing API calls.

```python
from tests.fixtures.mocks import MockHTTPClient

http = MockHTTPClient()

# Add responses for URL patterns
http.add_response(
    "/api/search",
    status_code=200,
    json_data={"results": [{"id": 1, "title": "Result"}]},
    headers={"Content-Type": "application/json"}
)

http.set_default_response(MockHTTPResponse(status_code=200, text="OK"))

# Simulate timeouts
http.set_timeout("http://slow.com")

# Simulate errors
http.set_error("http://error.com", status_code=500)

# Make requests
response = await http.get("http://api.example.com/api/search")
data = response.json()

response = await http.post(
    "http://api.example.com/api/submit",
    json={"data": "test"}
)

# Verify requests
assert http.get_request_count() == 2
req = http.find_request("/api/search")
assert req["method"] == "GET"

# Clear history
http.clear_history()
```

**Features:**
- ✅ URL pattern matching
- ✅ JSON/text responses
- ✅ Custom headers
- ✅ Timeout simulation
- ✅ Error status simulation
- ✅ Request recording

## pytest-mock Integration

The suite includes pytest-mock for additional mocking capabilities:

```python
def test_with_mocker(mocker):
    # Mock a function
    mock_func = mocker.patch("os.path.exists")
    mock_func.return_value = True
    
    import os
    assert os.path.exists("anything") is True
    mock_func.assert_called_once()

@pytest.mark.asyncio
async def test_async_mock(mocker):
    from unittest.mock import AsyncMock
    
    mock_async = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mock_async.return_value = None
    
    await asyncio.sleep(0.1)
    mock_async.assert_called_once()

def test_spy(mocker):
    import operator
    spy = mocker.spy(operator, "add")
    
    result = operator.add(2, 3)
    assert result == 5
    spy.assert_called_once_with(2, 3)
```

## Integrated Mock Usage

### Testing Complete Workflows

```python
@pytest.mark.asyncio
async def test_workflow_with_mocks(mock_llm, mock_web_search, mock_database):
    """Test a complete research workflow with mocks."""
    
    # Setup mocks
    mock_llm.set_default_response("Based on research, I found...")
    mock_web_search.add_results("topic", [
        {"title": "Source 1", "url": "http://s1.com", "snippet": "Info 1"},
        {"title": "Source 2", "url": "http://s2.com", "snippet": "Info 2"}
    ])
    mock_database.create_table("results", {"id": "int", "content": "str"})
    
    # Simulate workflow
    # 1. Search
    results = await mock_web_search.search("research topic")
    findings = mock_web_search.search_to_findings(results)
    
    # 2. Process with LLM
    llm_response = await mock_llm.chat_completion(
        messages=[{"role": "user", "content": f"Summarize: {findings}"}]
    )
    
    # 3. Store results
    doc_id = mock_database.insert("results", {"content": llm_response.content})
    
    # Verify
    assert mock_llm.get_call_count() == 1
    assert mock_web_search.get_call_count() == 1
    assert doc_id > 0
```

### Using All Services Together

```python
def test_all_services(mock_all_services):
    """Test using all mock services together."""
    llm = mock_all_services["llm"]
    search = mock_all_services["web_search"]
    db = mock_all_services["database"]
    http = mock_all_services["http"]
    
    # Use each service
    llm_response = await llm.chat_completion(messages=[...])
    search_results = await search.search("query")
    db.insert("table", {"data": "value"})
    http_response = await http.get("http://api.com")
```

## Best Practices

### 1. Configure Mocks Before Use

```python
@pytest.fixture
def configured_llm(mock_llm):
    mock_llm.set_default_response("Configured response")
    mock_llm.set_delay(0.05)
    return mock_llm
```

### 2. Verify Mock Calls

```python
async def test_verify_calls(mock_llm):
    await mock_llm.chat_completion(messages=[{"content": "Test"}])
    
    assert mock_llm.get_call_count() == 1
    last_call = mock_llm.get_last_call()
    assert "Test" in str(last_call.messages)
```

### 3. Test Error Scenarios

```python
async def test_error_handling(mock_llm):
    mock_llm.set_error_rate(1.0)  # 100% errors
    
    with pytest.raises(Exception):
        await mock_llm.chat_completion(messages=[...])
```

### 4. Clear State Between Tests

```python
async def test_isolation(mock_llm):
    mock_llm.clear_history()
    # Test with clean state
```

### 5. Use Pattern Matching

```python
mock_llm.set_response("weather", "Sunny")
mock_llm.set_response("joke", "Why did the chicken...")

# Matches any message containing "weather"
response = await mock_llm.chat_completion(
    messages=[{"content": "What's the weather like?"}]
)
```

## Running Mock Tests

```bash
# Run all mock tests
uv run pytest tests/unit/test_mocks.py -v

# Run specific mock test categories
uv run pytest tests/unit/test_mocks.py -k "TestMockLLM" -v
uv run pytest tests/unit/test_mocks.py -k "TestMockWebSearch" -v
uv run pytest tests/unit/test_mocks.py -k "TestMockDatabase" -v
uv run pytest tests/unit/test_mocks.py -k "TestMockHTTP" -v

# Run with coverage
uv run pytest tests/unit/test_mocks.py --cov=tests/fixtures/mocks
```

## Test Markers

Mock tests use these markers:

```python
@pytest.mark.unit      # Unit test
@pytest.mark.mock      # Uses mock infrastructure
@pytest.mark.asyncio   # Async test
```

Run mock tests specifically:

```bash
uv run pytest tests/ -m "mock" -v
```

## Migration from Real Services

### Before (Real API Calls)

```python
async def test_real_api():
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    assert response.choices[0].message.content
```

### After (With Mocks)

```python
async def test_with_mocks(mock_llm):
    mock_llm.set_default_response("Mock response")
    
    response = await mock_llm.chat_completion(
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    assert response.content == "Mock response"
    assert mock_llm.get_call_count() == 1
```

## Troubleshooting

### Mock Not Returning Expected Response

Check pattern matching:

```python
# Pattern must be in the message content
mock_llm.set_response("weather", "Sunny")

# This matches
await mock_llm.chat_completion(messages=[{"content": "What's the weather?"}])

# This doesn't match - uses default
await mock_llm.chat_completion(messages=[{"content": "Hello"}])
```

### Mock Not Recording Calls

Ensure you're awaiting async methods:

```python
# Correct
await mock_llm.chat_completion(messages=[...])

# Incorrect - doesn't await, call not recorded
mock_llm.chat_completion(messages=[...])
```

### pytest-mock Fixture Not Found

Ensure pytest-mock is installed:

```bash
uv pip install pytest-mock
```

## Examples

See `tests/unit/test_mocks.py` for comprehensive examples:
- 40 passing tests
- All mock classes demonstrated
- Integration patterns
- Error scenarios
- pytest-mock usage

## Summary

The mock infrastructure provides:
- ✅ **4 mock classes** (LLM, Search, Database, HTTP)
- ✅ **12+ fixtures** for easy usage
- ✅ **40 passing tests** demonstrating usage
- ✅ **Full pytest-mock integration**
- ✅ **Pattern matching** for flexible responses
- ✅ **Call recording** for verification
- ✅ **Error simulation** for testing edge cases
- ✅ **Zero external dependencies** during tests
