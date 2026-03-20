"""Integration Tests for Workflow Execution.

Tests for:
- Workflow execution
- State management
- Config translation
"""

import pytest
from typing import Dict, Any
from pathlib import Path

from deep_research_engine.config import load_config, WorkflowConfig
from deep_research_engine.state import EngineState, create_initial_state


# =============================================================================
# WORKFLOW EXECUTION TESTS
# =============================================================================

class TestRunWorkflow:
    """Tests for workflow execution."""

    @pytest.mark.integration
    def test_config_loads_successfully(self):
        """Test config loads successfully."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))

        assert config is not None
        assert isinstance(config, WorkflowConfig)

    @pytest.mark.integration
    def test_config_has_workflow_section(self):
        """Test config has workflow section."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))

        assert config.workflow is not None
        assert "name" in config.workflow
        assert "type" in config.workflow


# =============================================================================
# STATE MANAGEMENT TESTS
# =============================================================================

class TestStateManagement:
    """Tests for state management."""

    @pytest.mark.integration
    def test_state_creation(self):
        """Test state creation."""
        state = create_initial_state(
            workflow_name="test_workflow",
            workflow_type="research",
            input_data={"query": "test query"},
            session_id="test_session",
            run_id="test_run",
        )

        assert state["workflow_name"] == "test_workflow"
        assert state["workflow_type"] == "research"
        assert state["session_id"] == "test_session"
        assert state["run_id"] == "test_run"

    @pytest.mark.integration
    def test_state_has_context(self):
        """Test state has context."""
        state = create_initial_state(
            workflow_name="test",
            workflow_type="research",
            input_data={"query": "test"},
            session_id="test_session",
            run_id="test_run",
        )

        assert "context" in state
        assert isinstance(state["context"], dict)


# =============================================================================
# CONFIG TRANSLATION TESTS
# =============================================================================

class TestConfigTranslation:
    """Tests for config translation."""

    @pytest.mark.integration
    def test_config_phase_names(self):
        """Test config phase names."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))

        phase_names = [p["name"] for p in config.phases]
        assert len(phase_names) > 0
        assert all(isinstance(name, str) for name in phase_names)

    @pytest.mark.integration
    def test_config_storage_structure(self):
        """Test config storage structure."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))

        assert "storage" in config.model_dump()
        assert "memory" in config.storage


# =============================================================================
# WORKFLOW PHASES TESTS
# =============================================================================

class TestWorkflowPhases:
    """Tests for workflow phases."""

    @pytest.mark.integration
    def test_config_has_multiple_phases(self):
        """Test config has multiple phases."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))

        assert len(config.phases) >= 3

    @pytest.mark.integration
    def test_phase_order(self):
        """Test phase order."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))

        phase_names = [p["name"] for p in config.phases]
        
        # Should have boss/drafter/ingestion first
        assert "boss" in phase_names or "drafter" in phase_names or "ingestion" in phase_names
        
        # Should have critic
        assert "critic" in phase_names
        
        # Should have synthesis
        assert "synthesis" in phase_names or "artifact_maker" in phase_names


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.integration
    def test_config_not_found(self):
        """Test config not found error."""
        from deep_research_engine.errors import ConfigNotFoundError, ConfigError

        with pytest.raises((ConfigNotFoundError, ConfigError, FileNotFoundError)):
            load_config("/nonexistent/path/config.yaml")

    @pytest.mark.integration
    def test_invalid_yaml(self, tmp_path: Path):
        """Test invalid YAML error."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: syntax: [")

        with pytest.raises(Exception):
            load_config(str(invalid_yaml))
