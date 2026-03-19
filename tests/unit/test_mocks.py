"""Unit Tests for Mock Infrastructure.

Tests for:
- MockLLMProvider
- MockWebSearch
- MockDatabase
- MockHTTPClient
- Mock fixtures
"""

import pytest
import asyncio
import time
from typing import Dict, Any

from tests.fixtures.mocks import (
    MockLLMProvider,
    MockWebSearch,
    MockDatabase,
    MockHTTPClient,
    LLMResponse,
    SearchResult,
)


# =============================================================================
# MOCK LLM PROVIDER TESTS
# =============================================================================

class TestMockLLMProvider:
    """Tests for MockLLMProvider."""
    
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_llm_default_response(self, mock_llm: MockLLMProvider):
        """Test LLM returns default response."""
        response = await mock_llm.chat_completion(
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response.content == "This is a mock LLM response for testing."
        assert response.model == "mock-model"
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_llm_pattern_matching(self):
        """Test LLM matches patterns in messages."""
        llm = MockLLMProvider()
        llm.set_response("weather", "It's sunny today", model="weather-model")
        llm.set_response("joke", "Why did the chicken cross the road?", model="joke-model")
        
        # Weather query
        response1 = await llm.chat_completion(
            messages=[{"role": "user", "content": "What's the weather?"}]
        )
        assert "sunny" in response1.content
        
        # Joke query
        response2 = await llm.chat_completion(
            messages=[{"role": "user", "content": "Tell me a joke"}]
        )
        assert "chicken" in response2.content
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_llm_call_recording(self, mock_llm: MockLLMProvider):
        """Test LLM records all calls."""
        await mock_llm.chat_completion(
            messages=[{"role": "user", "content": "First"}],
            model="test-model",
            temperature=0.5
        )
        
        await mock_llm.chat_completion(
            messages=[{"role": "user", "content": "Second"}],
            temperature=0.8
        )
        
        assert mock_llm.get_call_count() == 2
        
        last_call = mock_llm.get_last_call()
        assert last_call is not None
        assert "Second" in str(last_call.messages)
        assert last_call.temperature == 0.8
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_llm_find_call(self, mock_llm: MockLLMProvider):
        """Test finding specific calls in history."""
        await mock_llm.chat_completion(
            messages=[{"role": "user", "content": "Query about AI"}]
        )
        
        await mock_llm.chat_completion(
            messages=[{"role": "user", "content": "Query about climate"}]
        )
        
        # Find AI query
        ai_call = mock_llm.find_call("AI")
        assert ai_call is not None
        assert "AI" in str(ai_call.messages)
        
        # Find climate query
        climate_call = mock_llm.find_call("climate")
        assert climate_call is not None
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_llm_clear_history(self, mock_llm: MockLLMProvider):
        """Test clearing LLM call history."""
        await mock_llm.chat_completion(
            messages=[{"role": "user", "content": "Test"}]
        )
        
        assert mock_llm.get_call_count() == 1
        
        mock_llm.clear_history()
        assert mock_llm.get_call_count() == 0
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_llm_delay(self):
        """Test LLM simulated delay."""
        llm = MockLLMProvider()
        llm.set_delay(0.1)  # 100ms
        
        start = time.time()
        await llm.chat_completion(
            messages=[{"role": "user", "content": "Test"}]
        )
        duration = time.time() - start
        
        assert duration >= 0.1
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_llm_error_simulation(self):
        """Test LLM error rate simulation."""
        llm = MockLLMProvider()
        llm.set_error_rate(1.0)  # 100% error rate
        
        with pytest.raises(Exception) as exc_info:
            await llm.chat_completion(
                messages=[{"role": "user", "content": "Test"}]
            )
        
        assert "Simulated API error" in str(exc_info.value)
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_llm_custom_handler(self):
        """Test LLM custom response handler."""
        llm = MockLLMProvider()
        
        def custom_handler(messages, model, temperature):
            content = " ".join(m.get("content", "") for m in messages)
            if "math" in content.lower():
                return LLMResponse(content="The answer is 42", model=model)
            return LLMResponse(content="I don't understand", model=model)
            
        llm.set_custom_handler(custom_handler)
        
        response1 = await llm.chat_completion(
            messages=[{"role": "user", "content": "Solve this math problem"}]
        )
        assert "42" in response1.content
        
        response2 = await llm.chat_completion(
            messages=[{"role": "user", "content": "Hello there"}]
        )
        assert "don't understand" in response2.content
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_llm_token_counting(self, mock_llm: MockLLMProvider):
        """Test LLM token counting."""
        response = await mock_llm.chat_completion(
            messages=[{"role": "user", "content": "Test message"}]
        )
        
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.total_tokens == response.prompt_tokens + response.completion_tokens


# =============================================================================
# MOCK WEB SEARCH TESTS
# =============================================================================

class TestMockWebSearch:
    """Tests for MockWebSearch."""
    
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_search_default_results(self, mock_web_search: MockWebSearch):
        """Test search returns default results."""
        results = await mock_web_search.search("any query")
        
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].title == "Default Result"
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_search_pattern_matching(self, mock_web_search_populated: MockWebSearch):
        """Test search matches query patterns."""
        # AI query
        ai_results = await mock_web_search_populated.search("AI developments")
        assert len(ai_results) == 3
        assert any("AI" in r.title for r in ai_results)
        
        # Climate query
        climate_results = await mock_web_search_populated.search("climate change")
        assert len(climate_results) == 2
        assert any("Climate" in r.title for r in climate_results)
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_search_result_limit(self, mock_web_search_populated: MockWebSearch):
        """Test search respects result limit."""
        results = await mock_web_search_populated.search("AI", num_results=2)
        assert len(results) == 2
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_search_call_recording(self, mock_web_search: MockWebSearch):
        """Test search records all calls."""
        await mock_web_search.search("query 1", num_results=5)
        await mock_web_search.search("query 2", num_results=10, time_range="year")
        
        assert mock_web_search.get_call_count() == 2
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_search_error_query(self, mock_web_search: MockWebSearch):
        """Test search error simulation."""
        mock_web_search.set_error_query("bad query")
        
        with pytest.raises(Exception) as exc_info:
            await mock_web_search.search("bad query")
        
        assert "Simulated search error" in str(exc_info.value)
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_search_clear_history(self, mock_web_search: MockWebSearch):
        """Test clearing search history."""
        await mock_web_search.search("test")
        assert mock_web_search.get_call_count() == 1
        
        mock_web_search.clear_history()
        assert mock_web_search.get_call_count() == 0
        
    @pytest.mark.unit
    @pytest.mark.mock
    async def test_search_to_findings(self, mock_web_search_populated: MockWebSearch):
        """Test converting search results to findings format."""
        results = await mock_web_search_populated.search("AI")
        findings = mock_web_search_populated.search_to_findings(results)
        
        assert "findings" in findings
        assert "source_count" in findings
        assert findings["source_count"] == 3
        assert len(findings["findings"]) == 3


# =============================================================================
# MOCK DATABASE TESTS
# =============================================================================

class TestMockDatabase:
    """Tests for MockDatabase."""
    
    @pytest.mark.unit
    @pytest.mark.mock
    def test_database_create_table(self, mock_database: MockDatabase):
        """Test creating tables."""
        mock_database.create_table("products", {"id": "int", "name": "str", "price": "float"})
        assert "products" in mock_database.tables
        
    @pytest.mark.unit
    @pytest.mark.mock
    def test_database_insert(self, mock_database: MockDatabase):
        """Test inserting data."""
        user_id = mock_database.insert("users", {"name": "Test", "email": "test@test.com"})
        
        assert user_id > 0
        users = mock_database.select("users")
        assert len(users) > 0
        assert users[-1]["name"] == "Test"
        
    @pytest.mark.unit
    @pytest.mark.mock
    def test_database_select(self, mock_database_populated: MockDatabase):
        """Test selecting data."""
        # Select all
        all_users = mock_database_populated.select("users")
        assert len(all_users) == 3
        
        # Select with where
        alice = mock_database_populated.select("users", where={"name": "Alice"})
        assert len(alice) == 1
        assert alice[0]["email"] == "alice@example.com"
        
    @pytest.mark.unit
    @pytest.mark.mock
    def test_database_update(self, mock_database_populated: MockDatabase):
        """Test updating data."""
        updated = mock_database_populated.update(
            "users",
            {"email": "new@example.com"},
            where={"name": "Bob"}
        )
        
        assert updated == 1
        
        bob = mock_database_populated.select("users", where={"name": "Bob"})
        assert bob[0]["email"] == "new@example.com"
        
    @pytest.mark.unit
    @pytest.mark.mock
    def test_database_delete(self, mock_database_populated: MockDatabase):
        """Test deleting data."""
        deleted = mock_database_populated.delete(
            "users",
            where={"name": "Charlie"}
        )
        
        assert deleted == 1
        
        remaining = mock_database_populated.select("users")
        assert len(remaining) == 2
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_database_execute_select(self, mock_database_populated: MockDatabase):
        """Test executing SELECT query."""
        results = await mock_database_populated.execute("SELECT * FROM users")
        
        assert len(results) == 3
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_database_execute_transaction(self, mock_database: MockDatabase):
        """Test transaction handling."""
        await mock_database.execute("BEGIN")
        assert mock_database.in_transaction is True
        
        await mock_database.execute("COMMIT")
        assert mock_database.in_transaction is False
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_database_error_query(self, mock_database: MockDatabase):
        """Test error query simulation."""
        mock_database.set_error_query("SELECT * FROM broken")
        
        with pytest.raises(Exception):
            await mock_database.execute("SELECT * FROM broken")
            
    @pytest.mark.unit
    @pytest.mark.mock
    def test_database_clear(self, mock_database_populated: MockDatabase):
        """Test clearing database."""
        mock_database_populated.clear()
        
        assert len(mock_database_populated.tables) == 0
        assert len(mock_database_populated.query_history) == 0


# =============================================================================
# MOCK HTTP CLIENT TESTS
# =============================================================================

class TestMockHTTPClient:
    """Tests for MockHTTPClient."""
    
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_http_get_default(self, mock_http_client: MockHTTPClient):
        """Test GET with default response."""
        response = await mock_http_client.get("http://example.com")
        
        assert response.status_code == 200
        assert response.text == "OK"
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_http_get_pattern(self, mock_http_client_api: MockHTTPClient):
        """Test GET with URL pattern matching."""
        response = await mock_http_client_api.get("http://api.example.com/api/v1/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_http_post(self, mock_http_client: MockHTTPClient):
        """Test POST request."""
        mock_http_client.add_response(
            "/submit",
            json_data={"success": True, "id": 123}
        )
        
        response = await mock_http_client.post(
            "http://example.com/submit",
            json={"data": "test"}
        )
        
        data = response.json()
        assert data["success"] is True
        assert data["id"] == 123
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_http_request_recording(self, mock_http_client: MockHTTPClient):
        """Test HTTP request recording."""
        await mock_http_client.get("http://example.com/api")
        await mock_http_client.post("http://example.com/api", json={"test": True})
        
        assert mock_http_client.get_request_count() == 2
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_http_find_request(self, mock_http_client: MockHTTPClient):
        """Test finding recorded requests."""
        await mock_http_client.get("http://api.example.com/users")
        await mock_http_client.get("http://api.example.com/posts")
        
        users_req = mock_http_client.find_request("/users")
        assert users_req is not None
        assert users_req["method"] == "GET"
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_http_timeout(self):
        """Test HTTP timeout simulation."""
        client = MockHTTPClient()
        client.set_timeout("http://slow.com")
        
        with pytest.raises(Exception) as exc_info:
            await client.get("http://slow.com", timeout=0.1)
            
        assert "timeout" in str(exc_info.value).lower()
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_http_error_status(self, mock_http_client: MockHTTPClient):
        """Test HTTP error status simulation."""
        mock_http_client.set_error("http://error.com", status_code=500)
        
        response = await mock_http_client.get("http://error.com")
        assert response.status_code == 500
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_http_clear_history(self, mock_http_client: MockHTTPClient):
        """Test clearing HTTP history."""
        await mock_http_client.get("http://example.com")
        assert mock_http_client.get_request_count() == 1
        
        mock_http_client.clear_history()
        assert mock_http_client.get_request_count() == 0
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_http_response_headers(self, mock_http_client_api: MockHTTPClient):
        """Test HTTP response headers."""
        response = await mock_http_client_api.get("http://api.example.com/api/v1/search")
        
        assert "Content-Type" in response.headers
        assert response.headers["Content-Type"] == "application/json"


# =============================================================================
# INTEGRATED MOCK TESTS
# =============================================================================

class TestIntegratedMocks:
    """Tests for integrated mock usage."""
    
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_all_services_together(self, mock_all_services: Dict[str, Any]):
        """Test using all mock services together."""
        llm = mock_all_services["llm"]
        search = mock_all_services["web_search"]
        db = mock_all_services["database"]
        http = mock_all_services["http"]
        
        # Use LLM
        llm_response = await llm.chat_completion(
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert llm_response is not None
        
        # Use search
        search_results = await search.search("test query")
        assert len(search_results) > 0
        
        # Use database
        doc_id = db.insert("documents", {"title": "Test", "content": "Content"})
        assert doc_id > 0
        
        # Use HTTP
        http_response = await http.get("http://example.com")
        assert http_response.status_code == 200
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_mock_workflow_simulation(self, mock_llm: MockLLMProvider, mock_web_search: MockWebSearch):
        """Test simulating a complete workflow with mocks."""
        # Setup: Configure mock responses - use default response override
        mock_llm.set_default_response("Based on my research, I found the following information.")
        mock_web_search.add_results("topic", [
            {"title": "Source 1", "url": "http://source1.com", "snippet": "Important info"},
            {"title": "Source 2", "url": "http://source2.com", "snippet": "More info"},
        ])
        
        # Simulate workflow
        # Step 1: Search
        results = await mock_web_search.search("research topic")
        findings = mock_web_search.search_to_findings(results)
        
        # Step 2: Process with LLM
        llm_response = await mock_llm.chat_completion(
            messages=[
                {"role": "user", "content": f"Summarize: {findings}"}
            ]
        )
        
        # Verify
        assert len(findings["findings"]) == 2
        assert "Based on" in llm_response.content
        assert mock_llm.get_call_count() == 1
        assert mock_web_search.get_call_count() == 1
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_mock_isolation(self):
        """Test that mocks are properly isolated between tests."""
        llm1 = MockLLMProvider()
        llm2 = MockLLMProvider()
        
        await llm1.chat_completion(messages=[{"role": "user", "content": "Test 1"}])
        await llm2.chat_completion(messages=[{"role": "user", "content": "Test 2"}])
        
        # Each should have independent history
        assert llm1.get_call_count() == 1
        assert llm2.get_call_count() == 1
        assert llm1.find_call("Test 1") is not None
        assert llm2.find_call("Test 2") is not None
        assert llm1.find_call("Test 2") is None


# =============================================================================
# PYTEST-MOCK INTEGRATION TESTS
# =============================================================================

class TestPytestMockIntegration:
    """Tests demonstrating pytest-mock usage."""
    
    @pytest.mark.unit
    @pytest.mark.mock
    def test_mocker_fixture(self, mocker):
        """Test using pytest-mock mocker fixture."""
        # Mock a function
        mock_func = mocker.patch("os.path.exists")
        mock_func.return_value = True
        
        import os
        assert os.path.exists("anything") is True
        mock_func.assert_called_once()
        
    @pytest.mark.unit
    @pytest.mark.mock
    @pytest.mark.asyncio
    async def test_async_mocker(self, mocker):
        """Test mocking async functions."""
        from unittest.mock import AsyncMock
        
        mock_async = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        mock_async.return_value = None
        
        await asyncio.sleep(0.1)
        mock_async.assert_called_once()
        
    @pytest.mark.unit
    @pytest.mark.mock
    def test_mocker_spy(self, mocker):
        """Test using mocker as spy."""
        import operator
        spy = mocker.spy(operator, "add")
        result = operator.add(2, 3)
        
        assert result == 5
        spy.assert_called_once_with(2, 3)
