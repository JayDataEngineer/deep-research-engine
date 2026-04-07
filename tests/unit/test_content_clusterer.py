import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any

from deep_research_engine.nodes.content_clusterer import content_clusterer_node
from deep_research_engine.state import EngineState

@pytest.mark.unit
@pytest.mark.asyncio
async def test_content_clusterer_node_success():
    """Test that content_clusterer_node correctly processes worker results and updates state."""
    
    # Mock state
    state: EngineState = {
        "worker_results": [
            {"worker_id": "w1", "output": "Research about topic A and its impact.", "success": True},
            {"worker_id": "w2", "output": "Topic B is related to technology trends.", "success": True}
        ],
        "context": {
            "config": {
                "phases": [
                    {"name": "content_clusterer", "config": {"enabled": True}}
                ]
            }
        }
    }

    # Mock Neo4jStorage
    with patch("deep_research_engine.nodes.content_clusterer.Neo4jStorage") as mock_neo4j_cls:
        mock_neo4j = MagicMock()
        mock_neo4j.query = AsyncMock(return_value=[{"name": "Existing Topic"}])
        mock_neo4j.initialize = AsyncMock()
        mock_neo4j.close = AsyncMock()
        mock_neo4j_cls.return_value = mock_neo4j

        # Mock get_prompt
        with patch("deep_research_engine.nodes.content_clusterer.get_prompt") as mock_get_prompt:
            mock_wrapper = MagicMock()
            mock_wrapper.compile = MagicMock(return_value="Compiled Prompt")
            mock_get_prompt.return_value = mock_wrapper

            # Mock infra and multimodal helpers
            with patch("deep_research_engine.nodes.content_clusterer._get_infra") as mock_get_infra, \
                 patch("deep_research_engine.nodes.content_clusterer._extract_chat_text") as mock_extract:
                
                mock_infra = MagicMock()
                mock_infra.chat = AsyncMock(return_value="Response Object")
                mock_get_infra.return_value = mock_infra
                
                # LLM response containing JSON block
                mock_extract.return_value = """
                ```json
                [
                  {
                    "name": "Topic A",
                    "summary": "Summary of A",
                    "labels": ["Tag1"],
                    "key_entities": ["Entity1"]
                  }
                ]
                ```
                """

                # Execute node
                command = await content_clusterer_node(state)

                # Verify
                assert command.goto == "neo4j_builder"
                assert "content_clusters" in command.update["context"]
                clusters = command.update["context"]["content_clusters"]
                assert len(clusters) == 1
                assert clusters[0]["name"] == "Topic A"

@pytest.mark.unit
@pytest.mark.asyncio
async def test_content_clusterer_node_disabled():
    """Test that node skips work when disabled in config."""
    state: EngineState = {
        "context": {
            "config": {
                "phases": [
                    {"name": "content_clusterer", "config": {"enabled": False}}
                ]
            }
        }
    }

    command = await content_clusterer_node(state)
    assert command.goto == "neo4j_builder"
    assert "content_clusters" not in command.update.get("context", {})
