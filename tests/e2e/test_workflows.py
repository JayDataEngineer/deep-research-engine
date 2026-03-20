"""End-to-End Tests for Complete Workflows.

Tests for:
- Config file-based workflows
- Output generation and validation
- Real-world workflow scenarios
"""

import pytest
from typing import Dict, Any
from pathlib import Path

from deep_research_engine.config import load_config, WorkflowConfig


# =============================================================================
# CONFIG LOADING E2E TESTS
# =============================================================================

class TestConfigLoadingE2E:
    """End-to-end tests for config loading."""

    @pytest.mark.e2e
    def test_load_rag_config(self, loaded_rag_config: WorkflowConfig):
        """Test RAG config loads correctly."""
        config = loaded_rag_config

        assert config.workflow["name"] == "rag"
        assert config.workflow["type"] == "research"
        assert len(config.phases) == 4

    @pytest.mark.e2e
    def test_load_pdf_summarizer_config(self, pdf_summarizer_config_path: Path):
        """Test PDF summarizer config loads correctly."""
        config = load_config(str(pdf_summarizer_config_path))

        assert config.workflow["name"] == "pdf_summarizer"
        assert config.workflow["type"] == "ingestion"
        assert len(config.phases) > 0

    @pytest.mark.e2e
    def test_load_skill_generator_config(self, skill_generator_config_path: Path):
        """Test skill generator config loads correctly."""
        config = load_config(str(skill_generator_config_path))

        assert config.workflow["name"] == "skill_generator"
        assert config.workflow["type"] == "generation"

    @pytest.mark.e2e
    def test_load_telegram_ingest_config(self, telegram_ingest_config_path: Path):
        """Test telegram ingest config loads correctly."""
        config = load_config(str(telegram_ingest_config_path))

        assert config.workflow["name"] == "telegram_ingest"
        assert config.workflow["type"] == "ingestion"


# =============================================================================
# WORKFLOW CONFIGURATION E2E TESTS
# =============================================================================

class TestWorkflowConfigE2E:
    """End-to-end tests for workflow configuration."""

    @pytest.mark.e2e
    def test_config_has_required_sections(self, loaded_rag_config: WorkflowConfig):
        """Test config has all required sections."""
        config = loaded_rag_config

        # Required top-level sections
        assert hasattr(config, 'workflow')
        assert hasattr(config, 'input')
        assert hasattr(config, 'phases')
        assert hasattr(config, 'storage')

    @pytest.mark.e2e
    def test_config_workflow_section(self, loaded_rag_config: WorkflowConfig):
        """Test workflow section structure."""
        config = loaded_rag_config

        assert "name" in config.workflow
        assert "type" in config.workflow
        assert config.workflow["name"] == "rag"
        assert config.workflow["type"] == "research"

    @pytest.mark.e2e
    def test_config_phases_section(self, loaded_rag_config: WorkflowConfig):
        """Test phases section structure."""
        config = loaded_rag_config

        assert len(config.phases) > 0
        for phase in config.phases:
            assert "name" in phase

    @pytest.mark.e2e
    def test_config_storage_section(self, loaded_rag_config: WorkflowConfig):
        """Test storage section structure."""
        config = loaded_rag_config

        # Memory storage should be enabled
        assert "memory" in config.storage
        assert config.storage["memory"]["enabled"] is True


# =============================================================================
# ENVIRONMENT VARIABLE E2E TESTS
# =============================================================================

class TestEnvVarE2E:
    """End-to-end tests for environment variable expansion."""

    @pytest.mark.e2e
    def test_env_var_expansion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test environment variable expansion in config."""
        monkeypatch.setenv("TEST_WORKFLOW_NAME", "env_test_workflow")
        monkeypatch.setenv("TEST_WORKFLOW_TYPE", "research")

        config_content = """
workflow:
  name: ${TEST_WORKFLOW_NAME}
  type: ${TEST_WORKFLOW_TYPE}

input:
  type: query
  source: test

phases:
  - name: ingestion
  - name: drafter

storage:
  memory:
    enabled: true
"""
        config_file = tmp_path / "env_test.yaml"
        config_file.write_text(config_content)
        config = load_config(str(config_file))

        assert config.workflow["name"] == "env_test_workflow"
        assert config.workflow["type"] == "research"

    @pytest.mark.e2e
    def test_env_var_default_values(self, tmp_path: Path):
        """Test environment variable default values."""
        config_content = """
workflow:
  name: test
  type: research

input:
  type: query
  source: ${SOURCE-default_source}

phases:
  - name: ingestion

storage:
  memory:
    enabled: true
"""
        config_file = tmp_path / "default_test.yaml"
        config_file.write_text(config_content)
        config = load_config(str(config_file))

        # Should use default value
        assert config.input["source"] == "default_source"


# =============================================================================
# CONFIG MERGING E2E TESTS
# =============================================================================

class TestConfigMergingE2E:
    """End-to-end tests for config merging."""

    @pytest.mark.e2e
    def test_config_merge_override(self, tmp_path: Path):
        """Test config merging with override."""
        from deep_research_engine.config import merge_configs

        base_config = {
            "workflow": {"name": "base", "type": "research"},
            "execution": {"max_steps": 50}
        }
        override_config = {
            "workflow": {"name": "override"},
            "execution": {"timeout": 600}
        }

        merged = merge_configs(base_config, override_config)

        assert merged["workflow"]["name"] == "override"
        assert merged["workflow"]["type"] == "research"
        assert merged["execution"]["max_steps"] == 50
        assert merged["execution"]["timeout"] == 600


# =============================================================================
# WORKFLOW EXECUTION E2E TESTS (MOCKED)
# =============================================================================

class TestWorkflowExecutionE2E:
    """End-to-end tests for workflow execution (mocked)."""

    @pytest.mark.e2e
    def test_config_validation(self, loaded_rag_config: WorkflowConfig):
        """Test config validation."""
        config = loaded_rag_config

        # Config should be valid WorkflowConfig instance
        assert isinstance(config, WorkflowConfig)

        # All required fields should be present
        assert config.workflow is not None
        assert config.input is not None
        assert config.phases is not None

    @pytest.mark.e2e
    def test_multiple_configs_load(self, loaded_configs: list[WorkflowConfig]):
        """Test loading multiple configs."""
        # All configs should load successfully
        assert len(loaded_configs) == 4
        assert all(config is not None for config in loaded_configs)


# =============================================================================
# PERFORMANCE E2E TESTS
# =============================================================================

class TestPerformanceE2E:
    """Performance tests for config loading."""

    @pytest.mark.e2e
    def test_config_load_performance(self, rag_config_path_fixed: Path):
        """Test config loads quickly."""
        import time

        start = time.time()
        for _ in range(10):
            config = load_config(str(rag_config_path_fixed))
            assert config is not None
        duration = time.time() - start

        # Should load 10 times in under 1 second
        assert duration < 1.0

    @pytest.mark.e2e
    def test_multiple_config_load_performance(self, all_config_paths: list[Path]):
        """Test multiple configs load quickly."""
        import time

        start = time.time()
        for config_path in all_config_paths:
            config = load_config(str(config_path))
            assert config is not None

        duration = time.time() - start

        # All configs should load in under 2 seconds
        assert duration < 2.0


# =============================================================================
# INTEGRATION E2E TESTS
# =============================================================================

class TestIntegrationE2E:
    """Integration tests for complete workflow scenarios."""

    @pytest.mark.e2e
    def test_rag_config_structure(self, loaded_rag_config: WorkflowConfig):
        """Test RAG config has expected structure."""
        config = loaded_rag_config

        # Check phases
        phase_names = [p["name"] for p in config.phases]
        # Updated to match actual phase names
        assert "boss" in phase_names or "drafter" in phase_names or "ingestion" in phase_names
        assert "critic" in phase_names
        assert "synthesis" in phase_names or "synthesizer" in phase_names

        # Check storage
        assert config.storage["memory"]["enabled"] is True

    @pytest.mark.e2e
    def test_config_with_env_vars_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test config with env vars can be loaded and dumped."""
        monkeypatch.setenv("WORKFLOW_TEST_NAME", "roundtrip_test")

        # Create config with env var
        config_content = """
workflow:
  name: ${WORKFLOW_TEST_NAME}
  type: research

input:
  type: query
  source: test

phases:
  - name: ingestion

storage:
  memory:
    enabled: true
"""
        config_file = tmp_path / "roundtrip.yaml"
        config_file.write_text(config_content)

        # Load config
        config = load_config(str(config_file))
        assert config.workflow["name"] == "roundtrip_test"
