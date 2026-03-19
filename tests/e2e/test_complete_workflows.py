"""End-to-End Tests for Complete Workflows.

Tests that run complete workflows with realistic configurations
and verify end-to-end functionality.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

from deep_research_engine.engine import run_workflow
from deep_research_engine.config import WorkflowConfig, load_config
from deep_research_engine.state import create_initial_state


# =============================================================================
# RAG WORKFLOW E2E TESTS
# =============================================================================

class TestRAGWorkflowE2E:
    """End-to-end tests for RAG research workflow."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_rag_workflow_complete(self):
        """Test complete RAG research workflow."""
        # Load actual RAG config
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        if not config_path.exists():
            pytest.skip("RAG config not found")

        config = load_config(str(config_path))

        # Mock all external dependencies
        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.nodes.boss.create_deep_agent') as mock_agent_factory:
                mock_agent = AsyncMock()
                mock_agent.ainvoke = AsyncMock(return_value={
                    "messages": [MagicMock(content='{"tasks": ["Research ML fundamentals", "Research ML applications", "Research ML tools"], "complexity": "medium"}')]
                })
                mock_agent_factory.return_value = mock_agent

                # Mock workers
                class MockWebResearcher:
                    def __init__(self, config):
                        self.config = config

                    async def execute(self, state, config):
                        return {
                            "worker_id": config.worker_id,
                            "success": True,
                            "output": {
                                "findings": [f"Finding for: {config.task}"],
                                "sources": [{"url": "http://example.com", "title": "Example"}]
                            },
                            "confidence": 0.85
                        }

                with patch.object(WorkerRegistry, 'get', return_value=MockWebResearcher):
                    # Mock critic
                    with patch('deep_research_engine.nodes.critic.create_deep_agent') as mock_critic_agent:
                        mock_critic = AsyncMock()
                        mock_critic.ainvoke = AsyncMock(return_value={
                            "messages": [MagicMock(content='{"score": 0.85, "criteria_scores": {"completeness": 0.9, "accuracy": 0.8}}')]
                        })
                        mock_critic_agent.return_value = mock_critic

                        # Mock synthesis
                        class MockSynthesisWorker:
                            def __init__(self, config):
                                self.config = config

                            async def execute(self, state, config):
                                return {
                                    "success": True,
                                    "output": "# Research Report\n\n## Summary\n\nThis report covers ML fundamentals and applications.\n\n## Findings\n\n1. ML is transforming industries\n2. Key tools include TensorFlow and PyTorch\n\n## Sources\n\n- [Example](http://example.com)",
                                    "output_type": "markdown",
                                    "confidence": 0.9
                                }

                        with patch.object(WorkerRegistry, 'get', side_effect=lambda t: MockSynthesisWorker if t == "synthesis" else MockWebResearcher):
                            events = []
                            async for event in run_workflow(
                                input_data="What are the fundamentals of machine learning?",
                                config=config,
                                stream=True
                            ):
                                events.append(event)

                            # Verify workflow completed
                            assert any(e["type"] == "complete" for e in events)
                            complete_event = [e for e in events if e["type"] == "complete"][0]
                            assert complete_event["success"] is True
                            assert "Research Report" in complete_event["output"]

                            # Verify all phases executed
                            phase_starts = [e["phase"] for e in events if e["type"] == "phase_start"]
                            assert "boss" in phase_starts
                            assert "workers" in phase_starts
                            assert "critic" in phase_starts
                            assert "synthesis" in phase_starts

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_rag_workflow_with_citations(self):
        """Test RAG workflow with citations enabled."""
        config_dict = {
            "workflow": {"name": "rag_test", "type": "research"},
            "input": {"type": "query", "source": "test"},
            "phases": [
                {"name": "boss", "node": "BossNode", "config": {"strategy": "simple"}},
                {
                    "name": "workers",
                    "node": "WorkerNode",
                    "config": {"worker_type": "web_researcher"}
                },
                {"name": "critic", "node": "CriticNode"},
                {
                    "name": "synthesis",
                    "node": "SynthesisNode",
                    "config": {
                        "format": "research_report",
                        "citations": {"enabled": True, "format": "APA"}
                    }
                }
            ],
            "storage": {"memory": {"enabled": True}},
            "execution": {"max_steps": 50}
        }
        config = WorkflowConfig(**config_dict)

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.nodes.boss.create_deep_agent') as mock_agent_factory:
                mock_agent = AsyncMock()
                mock_agent.ainvoke = AsyncMock(return_value={
                    "messages": [MagicMock(content='{"tasks": ["Task 1"], "complexity": "simple"}')]
                })
                mock_agent_factory.return_value = mock_agent

                class MockWorker:
                    def __init__(self, config):
                        self.config = config

                    async def execute(self, state, config):
                        return {
                            "worker_id": config.worker_id,
                            "success": True,
                            "output": {"findings": ["Finding 1"]},
                            "confidence": 0.8
                        }

                with patch.object(WorkerRegistry, 'get', return_value=MockWorker):
                    with patch('deep_research_engine.nodes.critic.create_deep_agent') as mock_critic:
                        mock_critic.return_value = AsyncMock()
                        mock_critic.return_value.ainvoke = AsyncMock(return_value={
                            "messages": [MagicMock(content='{"score": 0.85}')]
                        })

                        class MockSynthesis:
                            def __init__(self, config):
                                self.config = config

                            async def execute(self, state, config):
                                return {
                                    "success": True,
                                    "output": "Report with (Author, 2024) citations",
                                    "output_type": "markdown"
                                }

                        with patch.object(WorkerRegistry, 'get', side_effect=lambda t: MockSynthesis if t == "synthesis" else MockWorker):
                            async for event in run_workflow("Test query", config, stream=False):
                                result = event

                            assert result["success"] is True
                            assert "citations" in result["output"].lower() or "Report" in result["output"]


# =============================================================================
# PDF SUMMARIZER WORKFLOW E2E TESTS
# =============================================================================

class TestPDFSummarizerWorkflowE2E:
    """End-to-end tests for PDF summarization workflow."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_pdf_workflow_complete(self):
        """Test complete PDF summarization workflow."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "pdf_summarizer.yaml"
        if not config_path.exists():
            pytest.skip("PDF config not found")

        config = load_config(str(config_path))

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.nodes.boss.create_deep_agent') as mock_agent:
                mock_agent.return_value = AsyncMock()
                mock_agent.return_value.ainvoke = AsyncMock(return_value={
                    "messages": [MagicMock(content='{"tasks": ["Process chunk 1", "Process chunk 2"], "complexity": "medium"}')]
                })

                class MockChunkProcessor:
                    def __init__(self, config):
                        self.config = config

                    async def execute(self, state, config):
                        return {
                            "worker_id": config.worker_id,
                            "success": True,
                            "output": {"summary": f"Summary of {config.task}"},
                            "confidence": 0.9
                        }

                with patch.object(WorkerRegistry, 'get', return_value=MockChunkProcessor):
                    with patch('deep_research_engine.nodes.critic.create_deep_agent') as mock_critic:
                        mock_critic.return_value = AsyncMock()
                        mock_critic.return_value.ainvoke = AsyncMock(return_value={
                            "messages": [MagicMock(content='{"score": 0.9}')]
                        })

                        class MockSynthesis:
                            def __init__(self, config):
                                self.config = config

                            async def execute(self, state, config):
                                return {
                                    "success": True,
                                    "output": "# PDF Summary\n\n## Overview\n\nDocument overview.\n\n## Key Points\n\n1. Point 1\n2. Point 2",
                                    "output_type": "markdown"
                                }

                        with patch.object(WorkerRegistry, 'get', side_effect=lambda t: MockSynthesis if t == "synthesis" else MockChunkProcessor):
                            async for event in run_workflow(
                                {"pdf_path": "/test/doc.pdf"},
                                config,
                                stream=False
                            ):
                                result = event

                            assert result["success"] is True
                            assert "PDF Summary" in result["output"]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_pdf_workflow_handles_large_document(self):
        """Test PDF workflow with many chunks."""
        config_dict = {
            "workflow": {"name": "pdf_test", "type": "pdf_summarization"},
            "input": {"type": "pdf", "source": "/test/large.pdf"},
            "phases": [
                {"name": "boss", "node": "BossNode"},
                {"name": "workers", "node": "WorkerNode", "config": {"worker_type": "chunk_processor"}},
                {"name": "critic", "node": "CriticNode"},
                {"name": "synthesis", "node": "SynthesisNode"}
            ],
            "storage": {"memory": {"enabled": True}},
            "execution": {"max_steps": 100}
        }
        config = WorkflowConfig(**config_dict)

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.nodes.boss.create_deep_agent') as mock_agent:
                # Simulate large document with many chunks
                tasks = [f"Process chunk {i}" for i in range(20)]
                mock_agent.return_value = AsyncMock()
                mock_agent.return_value.ainvoke = AsyncMock(return_value={
                    "messages": [MagicMock(content=f'{{"tasks": {tasks}, "complexity": "high"}}')]
                })

                class MockChunkProcessor:
                    def __init__(self, config):
                        self.config = config

                    async def execute(self, state, config):
                        await asyncio.sleep(0.01)  # Simulate processing
                        return {
                            "worker_id": config.worker_id,
                            "success": True,
                            "output": {"summary": f"Summary of {config.task}"},
                            "confidence": 0.85
                        }

                with patch.object(WorkerRegistry, 'get', return_value=MockChunkProcessor):
                    with patch('deep_research_engine.nodes.critic.create_deep_agent') as mock_critic:
                        mock_critic.return_value = AsyncMock()
                        mock_critic.return_value.ainvoke = AsyncMock(return_value={
                            "messages": [MagicMock(content='{"score": 0.88}')]
                        })

                        class MockSynthesis:
                            def __init__(self, config):
                                self.config = config

                            async def execute(self, state, config):
                                worker_results = state.get("worker_results", [])
                                return {
                                    "success": True,
                                    "output": f"# Large Document Summary\n\nProcessed {len(worker_results)} chunks",
                                    "output_type": "markdown"
                                }

                        with patch.object(WorkerRegistry, 'get', side_effect=lambda t: MockSynthesis if t == "synthesis" else MockChunkProcessor):
                            async for event in run_workflow(
                                {"pdf_path": "/test/large.pdf"},
                                config,
                                stream=False
                            ):
                                result = event

                            assert result["success"] is True
                            assert "20 chunks" in result["output"]


# =============================================================================
# SKILL GENERATOR WORKFLOW E2E TESTS
# =============================================================================

class TestSkillGeneratorWorkflowE2E:
    """End-to-end tests for skill generator workflow."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_skill_generator_complete(self):
        """Test complete skill generator workflow."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "skill_generator.yaml"
        if not config_path.exists():
            pytest.skip("Skill generator config not found")

        config = load_config(str(config_path))

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.nodes.boss.create_deep_agent') as mock_agent:
                mock_agent.return_value = AsyncMock()
                mock_agent.return_value.ainvoke = AsyncMock(return_value={
                    "messages": [MagicMock(content='{"tasks": ["Scrape documentation"], "complexity": "simple"}')]
                })

                class MockScraper:
                    def __init__(self, config):
                        self.config = config

                    async def execute(self, state, config):
                        return {
                            "worker_id": config.worker_id,
                            "success": True,
                            "output": {
                                "content": "# API Documentation\n\n## Methods\n\n- method1()\n- method2()",
                                "metadata": {"url": "https://example.com/docs"}
                            },
                            "confidence": 0.9
                        }

                with patch.object(WorkerRegistry, 'get', return_value=MockScraper):
                    with patch('deep_research_engine.nodes.critic.create_deep_agent') as mock_critic:
                        mock_critic.return_value = AsyncMock()
                        mock_critic.return_value.ainvoke = AsyncMock(return_value={
                            "messages": [MagicMock(content='{"score": 0.9}')]
                        })

                        class MockSynthesis:
                            def __init__(self, config):
                                self.config = config

                            async def execute(self, state, config):
                                return {
                                    "success": True,
                                    "output": "# Skill: API Documentation\n\n## Capabilities\n\n- Understand API structure\n- Extract method signatures\n\n## Knowledge Base\n\n- API Documentation",
                                    "output_type": "markdown"
                                }

                        with patch.object(WorkerRegistry, 'get', side_effect=lambda t: MockSynthesis if t == "synthesis" else MockScraper):
                            async for event in run_workflow(
                                {"url": "https://example.com/docs"},
                                config,
                                stream=False
                            ):
                                result = event

                            assert result["success"] is True
                            assert "Skill:" in result["output"]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_skill_generator_multiple_urls(self):
        """Test skill generator with multiple URLs."""
        config_dict = {
            "workflow": {"name": "skill_test", "type": "skill_generation"},
            "input": {"type": "urls", "source": ["https://docs1.com", "https://docs2.com"]},
            "phases": [
                {"name": "boss", "node": "BossNode"},
                {"name": "workers", "node": "WorkerNode", "config": {"worker_type": "scraper"}},
                {"name": "critic", "node": "CriticNode"},
                {"name": "synthesis", "node": "SynthesisNode"}
            ],
            "storage": {"memory": {"enabled": True}},
            "execution": {"max_steps": 50}
        }
        config = WorkflowConfig(**config_dict)

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.nodes.boss.create_deep_agent') as mock_agent:
                mock_agent.return_value = AsyncMock()
                mock_agent.return_value.ainvoke = AsyncMock(return_value={
                    "messages": [MagicMock(content='{"tasks": ["Scrape URL 1", "Scrape URL 2"], "complexity": "medium"}')]
                })

                class MockScraper:
                    def __init__(self, config):
                        self.config = config

                    async def execute(self, state, config):
                        return {
                            "worker_id": config.worker_id,
                            "success": True,
                            "output": {"content": f"Content from {config.task}", "metadata": {}},
                            "confidence": 0.85
                        }

                with patch.object(WorkerRegistry, 'get', return_value=MockScraper):
                    with patch('deep_research_engine.nodes.critic.create_deep_agent') as mock_critic:
                        mock_critic.return_value = AsyncMock()
                        mock_critic.return_value.ainvoke = AsyncMock(return_value={
                            "messages": [MagicMock(content='{"score": 0.85}')]
                        })

                        class MockSynthesis:
                            def __init__(self, config):
                                self.config = config

                            async def execute(self, state, config):
                                worker_results = state.get("worker_results", [])
                                return {
                                    "success": True,
                                    "output": f"# Generated Skill\n\nSources: {len(worker_results)} URLs processed",
                                    "output_type": "markdown"
                                }

                        with patch.object(WorkerRegistry, 'get', side_effect=lambda t: MockSynthesis if t == "synthesis" else MockScraper):
                            async for event in run_workflow(
                                {"urls": ["https://docs1.com", "https://docs2.com"]},
                                config,
                                stream=False
                            ):
                                result = event

                            assert result["success"] is True
                            assert "2 URLs" in result["output"]


# =============================================================================
# TELEGRAM INGEST WORKFLOW E2E TESTS
# =============================================================================

class TestTelegramIngestWorkflowE2E:
    """End-to-end tests for Telegram ingestion workflow."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_telegram_ingest_complete(self):
        """Test complete Telegram ingestion workflow."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "telegram_ingest.yaml"
        if not config_path.exists():
            pytest.skip("Telegram config not found")

        config = load_config(str(config_path))

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.nodes.boss.create_deep_agent') as mock_agent:
                mock_agent.return_value = AsyncMock()
                mock_agent.return_value.ainvoke = AsyncMock(return_value={
                    "messages": [MagicMock(content='{"tasks": ["Process messages"], "complexity": "simple"}')]
                })

                class MockMultimodalWorker:
                    def __init__(self, config):
                        self.config = config

                    async def execute(self, state, config):
                        return {
                            "worker_id": config.worker_id,
                            "success": True,
                            "output": {
                                "transcripts": ["Transcript 1"],
                                "summaries": ["Summary 1"],
                                "entities": ["Entity 1"]
                            },
                            "confidence": 0.85
                        }

                with patch.object(WorkerRegistry, 'get', return_value=MockMultimodalWorker):
                    with patch('deep_research_engine.nodes.critic.create_deep_agent') as mock_critic:
                        mock_critic.return_value = AsyncMock()
                        mock_critic.return_value.ainvoke = AsyncMock(return_value={
                            "messages": [MagicMock(content='{"score": 0.85}')]
                        })

                        class MockSynthesis:
                            def __init__(self, config):
                                self.config = config

                            async def execute(self, state, config):
                                return {
                                    "success": True,
                                    "output": "# Telegram Export Summary\n\n## Messages Processed: 10\n\n## Key Topics\n\n- Topic 1\n- Topic 2",
                                    "output_type": "markdown"
                                }

                        with patch.object(WorkerRegistry, 'get', side_effect=lambda t: MockSynthesis if t == "synthesis" else MockMultimodalWorker):
                            async for event in run_workflow(
                                {"export_path": "/test/telegram_export"},
                                config,
                                stream=False
                            ):
                                result = event

                            assert result["success"] is True
                            assert "Telegram" in result["output"]


# =============================================================================
# PERFORMANCE E2E TESTS
# =============================================================================

class TestPerformanceE2E:
    """Performance-focused end-to-end tests."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_execution_time(self, sample_config_dict: Dict[str, Any]):
        """Test workflow completes within acceptable time."""
        config = WorkflowConfig(**sample_config_dict)

        import time

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.nodes.boss.create_deep_agent') as mock_agent:
                mock_agent.return_value = AsyncMock()
                mock_agent.return_value.ainvoke = AsyncMock(return_value={
                    "messages": [MagicMock(content='{"tasks": ["Task 1", "Task 2"], "complexity": "simple"}')]
                })

                class FastWorker:
                    def __init__(self, config):
                        self.config = config

                    async def execute(self, state, config):
                        await asyncio.sleep(0.01)
                        return {"worker_id": config.worker_id, "success": True, "output": {}}

                with patch.object(WorkerRegistry, 'get', return_value=FastWorker):
                    with patch('deep_research_engine.nodes.critic.create_deep_agent') as mock_critic:
                        mock_critic.return_value = AsyncMock()
                        mock_critic.return_value.ainvoke = AsyncMock(return_value={
                            "messages": [MagicMock(content='{"score": 0.85}')]
                        })

                        class FastSynthesis:
                            def __init__(self, config):
                                self.config = config

                            async def execute(self, state, config):
                                return {"success": True, "output": "Report"}

                        with patch.object(WorkerRegistry, 'get', side_effect=lambda t: FastSynthesis if t == "synthesis" else FastWorker):
                            start = time.time()

                            async for _ in run_workflow("Test", config, stream=False):
                                pass

                            elapsed = time.time() - start

                            # Should complete in under 2 seconds for simple workflow
                            assert elapsed < 2.0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_memory_usage(self, sample_config_dict: Dict[str, Any]):
        """Test workflow doesn't leak memory."""
        config = WorkflowConfig(**sample_config_dict)

        import tracemalloc
        tracemalloc.start()

        for i in range(5):
            with patch('deep_research_engine.engine.StorageManager'):
                with patch('deep_research_engine.nodes.boss.create_deep_agent') as mock_agent:
                    mock_agent.return_value = AsyncMock()
                    mock_agent.return_value.ainvoke = AsyncMock(return_value={
                        "messages": [MagicMock(content='{"tasks": ["Task"], "complexity": "simple"}')]
                    })

                    class MockWorker:
                        def __init__(self, config):
                            self.config = config

                        async def execute(self, state, config):
                            return {"worker_id": config.worker_id, "success": True, "output": {}}

                    with patch.object(WorkerRegistry, 'get', return_value=MockWorker):
                        with patch('deep_research_engine.nodes.critic.create_deep_agent') as mock_critic:
                            mock_critic.return_value = AsyncMock()
                            mock_critic.return_value.ainvoke = AsyncMock(return_value={
                                "messages": [MagicMock(content='{"score": 0.85}')]
                            })

                            class MockSynthesis:
                                def __init__(self, config):
                                    self.config = config

                                async def execute(self, state, config):
                                    return {"success": True, "output": "Report"}

                            with patch.object(WorkerRegistry, 'get', side_effect=lambda t: MockSynthesis if t == "synthesis" else MockWorker):
                                async for _ in run_workflow(f"Test {i}", config, stream=False):
                                    pass

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory should be reasonable (< 50MB for test)
        assert peak < 50 * 1024 * 1024


# Import WorkerRegistry for tests
from deep_research_engine.workers import WorkerRegistry
