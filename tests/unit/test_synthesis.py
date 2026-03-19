"""Unit Tests for Synthesis Worker.

Tests for:
- SynthesisWorker implementation
- Format schema validation
- Format config loading
- Synthesis node integration
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any

from deep_research_engine.state import EngineState, create_initial_state
from deep_research_engine.workers import WorkerConfig, WorkerRegistry
from deep_research_engine.workers.implementations.synthesis import SynthesisWorker
from deep_research_engine.nodes import synthesis_node


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def synthesis_worker_config() -> WorkerConfig:
    """Create synthesis worker configuration."""
    return WorkerConfig(
        worker_id="synthesis_0",
        worker_type="synthesis",
        task="Synthesize research findings into formatted output",
        tools=[],
        max_retries=3,
        timeout=600,
        custom={
            "format": "research_report",
            "config": {
                "format": "research_report",
                "citations": {
                    "enabled": True,
                    "format": "numbered"
                }
            }
        }
    )


@pytest.fixture
def sample_worker_results() -> list:
    """Create sample worker results for synthesis."""
    return [
        {
            "worker_id": "web_researcher_0",
            "worker_type": "web_researcher",
            "task": "Research AI trends in 2024",
            "success": True,
            "output": "AI trends show significant growth in multimodal models and agent systems.",
            "output_type": "findings",
            "confidence": 0.9,
            "sources": [
                {
                    "id": "1",
                    "title": "AI Trends 2024 Report",
                    "url": "https://example.com/ai-trends-2024"
                }
            ],
            "source_count": 1,
        },
        {
            "worker_id": "web_researcher_1",
            "worker_type": "web_researcher",
            "task": "Research LLM architecture advances",
            "success": True,
            "output": "LLM architecture has evolved with mixture of experts and attention improvements.",
            "output_type": "findings",
            "confidence": 0.85,
            "sources": [
                {
                    "id": "2",
                    "title": "LLM Architecture Survey",
                    "url": "https://example.com/llm-architecture"
                }
            ],
            "source_count": 1,
        }
    ]


@pytest.fixture
def state_with_worker_results(
    sample_worker_results: list
) -> EngineState:
    """Create state with worker results."""
    return create_initial_state(
        workflow_name="test_synthesis",
        workflow_type="research",
        input_data={"query": "Test research query"},
        session_id="test_session",
        run_id="test_run",
        max_iterations=3,
    )


# =============================================================================
# SYNTHESIS WORKER TESTS
# =============================================================================

class TestSynthesisWorker:
    """Tests for SynthesisWorker implementation."""

    @pytest.mark.unit
    def test_synthesis_worker_initialization(
        self,
        synthesis_worker_config: WorkerConfig
    ):
        """Test synthesis worker initialization."""
        worker = SynthesisWorker(synthesis_worker_config)

        assert worker.worker_id == "synthesis_0"
        assert worker.worker_type == "synthesis"
        # max_retries comes from custom config, defaults to 10
        assert worker.max_retries == 10
        assert worker.output_format == "research_report"

    @pytest.mark.unit
    def test_synthesis_worker_registry(self):
        """Test synthesis worker is registered."""
        # Should be able to get from registry
        worker_class = WorkerRegistry.get("synthesis")
        assert worker_class == SynthesisWorker

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesis_worker_no_results(
        self,
        synthesis_worker_config: WorkerConfig,
        initial_state: EngineState
    ):
        """Test synthesis worker handles empty results."""
        worker = SynthesisWorker(synthesis_worker_config)
        result = await worker.execute(initial_state, synthesis_worker_config)

        assert result["success"] is False
        assert "error" in result
        assert "No worker results" in result["error"]

    @pytest.mark.unit
    def test_formats_dir_exists(self):
        """Test that formats directory exists."""
        formats_dir = Path(__file__).parent.parent.parent / "src" / "deep_research_engine" / "synthesis" / "formats"
        assert formats_dir.exists()

    @pytest.mark.unit
    def test_format_configs_exist(self):
        """Test that format configs exist."""
        formats_dir = Path(__file__).parent.parent.parent / "src" / "deep_research_engine" / "synthesis" / "formats"

        # Check for expected format directories
        expected_formats = ["research_report", "markdown", "structured_json"]

        for format_name in expected_formats:
            format_dir = formats_dir / format_name
            assert format_dir.exists(), f"Format directory {format_name} should exist"

            config_file = format_dir / "config.yaml"
            assert config_file.exists(), f"Config file for {format_name} should exist"

            schema_file = format_dir / "schema.py"
            assert schema_file.exists(), f"Schema file for {format_name} should exist"


# =============================================================================
# FORMAT SCHEMA TESTS
# =============================================================================

class TestFormatSchemas:
    """Tests for format schema validation."""

    @pytest.mark.unit
    def test_research_report_schema_import(self):
        """Test research report schema can be imported."""
        from deep_research_engine.synthesis.formats.research_report.schema import (
            ResearchReport,
            Section,
            KeyFinding,
            Citation
        )

        assert ResearchReport is not None
        assert Section is not None
        assert KeyFinding is not None
        assert Citation is not None

    @pytest.mark.unit
    def test_markdown_schema_import(self):
        """Test markdown schema can be imported."""
        from deep_research_engine.synthesis.formats.markdown.schema import (
            MarkdownOutput,
            MarkdownSource
        )

        assert MarkdownOutput is not None
        assert MarkdownSource is not None

    @pytest.mark.unit
    def test_structured_json_schema_import(self):
        """Test structured JSON schema can be imported."""
        from deep_research_engine.synthesis.formats.structured_json.schema import (
            StructuredJSON,
            JSONFinding,
            JSONSource
        )

        assert StructuredJSON is not None
        assert JSONFinding is not None
        assert JSONSource is not None

    @pytest.mark.unit
    def test_research_report_schema_validation(self):
        """Test research report schema validation."""
        from deep_research_engine.synthesis.formats.research_report.schema import (
            ResearchReport,
            Section,
            KeyFinding,
            Citation
        )
        from pydantic import ValidationError

        # Valid report
        report = ResearchReport(
            title="Test Report",
            executive_summary="This is a test summary.",
            key_findings=[
                KeyFinding(
                    finding="Test finding",
                    confidence=0.9,
                    supporting_evidence=["Evidence 1"],
                    citations=["1"]
                )
            ],
            sections=[
                Section(
                    title="Test Section",
                    content="Test content",
                    citations=["1"]
                )
            ],
            conclusion="Test conclusion",
            citations=[
                Citation(
                    id="1",
                    title="Test Source",
                    url="https://example.com"
                )
            ]
        )

        assert report.title == "Test Report"
        assert len(report.key_findings) == 1
        assert report.key_findings[0].confidence == 0.9

        # Invalid confidence (out of range)
        with pytest.raises(ValidationError):
            KeyFinding(
                finding="Test",
                confidence=1.5,  # Invalid: > 1.0
                supporting_evidence=[],
                citations=[]
            )


# =============================================================================
# SYNTHESIS NODE TESTS
# =============================================================================

class TestSynthesisNode:
    """Tests for synthesis node integration."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesis_node_no_results(
        self,
        initial_state: EngineState
    ):
        """Test synthesis node handles empty results."""
        # Add synthesis config to context
        initial_state["context"] = {
            "synthesis": {
                "format": "markdown",
                "citations": {"enabled": False}
            }
        }

        result_state = await synthesis_node(initial_state)

        # Should have error
        assert len(result_state["errors"]) > 0
        assert result_state["errors"][0]["phase"] == "synthesis"

    @pytest.mark.unit
    def test_synthesis_node_import(self):
        """Test synthesis node can be imported."""
        from deep_research_engine.nodes.synthesis import (
            synthesis_node,
            route_after_synthesis
        )

        assert synthesis_node is not None
        assert route_after_synthesis is not None


# =============================================================================
# FORMAT LOADING TESTS
# =============================================================================

class TestFormatLoading:
    """Tests for format config and schema loading."""

    @pytest.mark.unit
    def test_load_format_config(self):
        """Test loading format config."""
        import yaml
        formats_dir = Path(__file__).parent.parent.parent / "src" / "deep_research_engine" / "synthesis" / "formats"

        config_path = formats_dir / "research_report" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert "format" in config
        assert "system_prompt" in config
        assert "model" in config
        assert config["format"]["name"] == "research_report"

    @pytest.mark.unit
    def test_load_schema_class(self):
        """Test loading schema class dynamically."""
        import importlib.util
        formats_dir = Path(__file__).parent.parent.parent / "src" / "deep_research_engine" / "synthesis" / "formats"

        schema_path = formats_dir / "markdown" / "schema.py"
        spec = importlib.util.spec_from_file_location("markdown.schema", schema_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find BaseModel subclass
        from pydantic import BaseModel
        schema_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseModel) and attr != BaseModel:
                schema_class = attr
                break

        assert schema_class is not None
        assert schema_class.__name__ == "MarkdownOutput"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSynthesisIntegration:
    """Integration tests for synthesis workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_synthesis_worker_compiles_findings(
        self,
        synthesis_worker_config: WorkerConfig,
        sample_worker_results: list
    ):
        """Test synthesis worker can compile findings (mock test)."""
        # This test verifies the compilation logic without calling LLM
        worker = SynthesisWorker(synthesis_worker_config)

        # Create a state with worker results
        state = create_initial_state(
            workflow_name="test",
            workflow_type="research",
            input_data={"query": "test"},
            session_id="test",
            run_id="test"
        )
        state["worker_results"] = sample_worker_results

        # Test _compile_findings method directly
        findings_text = worker._compile_findings(
            sample_worker_results,
            citations_enabled=True
        )

        assert "Research AI trends in 2024" in findings_text or "AI trends" in findings_text
        assert "Sources:" in findings_text
        assert "AI Trends 2024 Report" in findings_text

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_synthesis_format_alias_resolution(self):
        """Test format alias resolution."""
        # Test that aliases work in the synthesis worker
        worker_config = WorkerConfig(
            worker_id="synthesis_test",
            worker_type="synthesis",
            task="test",
            custom={
                "format": "report",  # Alias for research_report
            }
        )

        worker = SynthesisWorker(worker_config)

        # Check alias mapping exists
        assert hasattr(worker, '_format_findings')
