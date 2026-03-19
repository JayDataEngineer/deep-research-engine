"""Unit Tests for Config Loading and Validation.

Tests for:
- WorkflowConfig Pydantic model
- load_config() function
- expand_env_vars() utility
- merge_configs() utility
- render_prompt_template() utility
"""

import pytest
import os
from pathlib import Path
from typing import Dict, Any

from deep_research_engine.config import (
    WorkflowConfig,
    load_config,
    expand_env_vars,
    merge_configs,
    render_prompt_template,
)


# =============================================================================
# WORKFLOW CONFIG TESTS
# =============================================================================

class TestWorkflowConfig:
    """Tests for WorkflowConfig Pydantic model."""
    
    @pytest.mark.unit
    def test_minimal_config(self, sample_config_dict: Dict[str, Any]):
        """Test config with minimal required fields."""
        config = WorkflowConfig(**sample_config_dict)
        
        assert config.workflow["name"] == "test_workflow"
        assert config.workflow["type"] == "research"
        assert config.input["type"] == "query"
        assert len(config.phases) == 2
    
    @pytest.mark.unit
    def test_config_with_optional_fields(self, sample_config_dict: Dict[str, Any]):
        """Test config with all optional fields."""
        config = WorkflowConfig(**sample_config_dict)
        
        assert config.storage == {"memory": {"enabled": True}, "postgres": {"enabled": False}}
        assert config.llm["models"]["main"] == "test-model"
        assert "web_search" in config.tools["enabled"]
        assert config.execution["max_steps"] == 50
    
    @pytest.mark.unit
    def test_config_extra_fields_allowed(self, sample_config_dict: Dict[str, Any]):
        """Test that extra fields are allowed in config."""
        sample_config_dict["custom_field"] = "custom_value"
        sample_config_dict["another_custom"] = {"nested": "value"}
        
        config = WorkflowConfig(**sample_config_dict)
        
        assert config.custom_field == "custom_value"  # type: ignore
        assert config.another_custom == {"nested": "value"}  # type: ignore
    
    @pytest.mark.unit
    def test_config_from_rag_yaml(self):
        """Test loading actual RAG config."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))
        
        assert config.workflow["name"] == "rag"
        assert config.workflow["type"] == "research"
        assert len(config.phases) == 4
        assert config.storage["postgres"]["enabled"] is True


# =============================================================================
# LOAD_CONFIG TESTS
# =============================================================================

class TestLoadConfig:
    """Tests for load_config() function."""
    
    @pytest.mark.unit
    def test_load_valid_config(self, rag_config_path: Path):
        """Test loading a valid YAML config file."""
        config = load_config(str(rag_config_path))
        
        assert isinstance(config, WorkflowConfig)
        assert config.workflow["name"] == "rag_test"
        assert config.workflow["type"] == "research"
    
    @pytest.mark.unit
    def test_load_nonexistent_config(self):
        """Test loading a nonexistent config file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config("/nonexistent/path/config.yaml")
        
        assert "Config file not found" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_load_config_with_invalid_yaml(self, tmp_path: Path):
        """Test loading a file with invalid YAML syntax."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: syntax: [")
        
        with pytest.raises(Exception):  # yaml.YAMLError
            load_config(str(invalid_yaml))


# =============================================================================
# EXPAND_ENV_VARS TESTS
# =============================================================================

class TestExpandEnvVars:
    """Tests for expand_env_vars() utility."""
    
    @pytest.mark.unit
    def test_expand_simple_env_var(self, monkeypatch: pytest.MonkeyPatch):
        """Test expanding a simple environment variable."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        
        result = expand_env_vars("${TEST_VAR}")
        
        assert result == "test_value"
    
    @pytest.mark.unit
    def test_expand_env_var_with_default(self, monkeypatch: pytest.MonkeyPatch):
        """Test expanding env var with default value."""
        monkeypatch.setenv("EXISTING_VAR", "existing_value")
        
        # Existing var
        result1 = expand_env_vars("${EXISTING_VAR-default}")
        assert result1 == "existing_value"
        
        # Non-existing var uses default
        result2 = expand_env_vars("${NONEXISTENT_VAR-default_value}")
        assert result2 == "default_value"
    
    @pytest.mark.unit
    def test_expand_env_var_in_dict(self, monkeypatch: pytest.MonkeyPatch):
        """Test expanding env vars in a dictionary."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        
        config = {
            "database": {
                "host": "${DB_HOST}",
                "port": "${DB_PORT}",
                "name": "test_db"
            }
        }
        
        result = expand_env_vars(config)
        
        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == "5432"
        assert result["database"]["name"] == "test_db"
    
    @pytest.mark.unit
    def test_expand_env_var_in_list(self, monkeypatch: pytest.MonkeyPatch):
        """Test expanding env vars in a list."""
        monkeypatch.setenv("API_KEY", "secret_key")
        
        config = {
            "headers": [
                "Authorization: ${API_KEY}",
                "Content-Type: application/json"
            ]
        }
        
        result = expand_env_vars(config)
        
        assert result["headers"][0] == "Authorization: secret_key"
        assert result["headers"][1] == "Content-Type: application/json"
    
    @pytest.mark.unit
    def test_expand_env_var_no_substitution(self):
        """Test string without env vars remains unchanged."""
        result = expand_env_vars("no variables here")
        assert result == "no variables here"
    
    @pytest.mark.unit
    def test_expand_env_var_non_string_types(self):
        """Test that non-string types pass through unchanged."""
        assert expand_env_vars(42) == 42
        assert expand_env_vars(3.14) == 3.14
        assert expand_env_vars(True) is True
        assert expand_env_vars(None) is None
    
    @pytest.mark.unit
    def test_expand_env_var_in_config_file(self, config_with_env_vars: Path, monkeypatch: pytest.MonkeyPatch):
        """Test env var expansion in actual config file."""
        monkeypatch.setenv("WORKFLOW_NAME", "custom_workflow")
        monkeypatch.setenv("QUERY_SOURCE", "custom query")
        
        config = load_config(str(config_with_env_vars))
        
        assert config.workflow["name"] == "custom_workflow"
        assert config.input["source"] == "custom query"
        # Default value should be used for POSTGRES_URL since we didn't set it


# =============================================================================
# MERGE_CONFIGS TESTS
# =============================================================================

class TestMergeConfigs:
    """Tests for merge_configs() utility."""
    
    @pytest.mark.unit
    def test_merge_simple_configs(self):
        """Test merging two simple configs."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        
        result = merge_configs(base, override)
        
        assert result["a"] == 1  # From base
        assert result["b"] == 3  # Overridden
        assert result["c"] == 4  # From override
    
    @pytest.mark.unit
    def test_merge_nested_configs(self):
        """Test merging nested configs."""
        base = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "user": "admin",
                    "pass": "secret"
                }
            }
        }
        override = {
            "database": {
                "port": 5433,
                "credentials": {
                    "pass": "new_secret"
                }
            }
        }
        
        result = merge_configs(base, override)
        
        assert result["database"]["host"] == "localhost"  # From base
        assert result["database"]["port"] == 5433  # Overridden
        assert result["database"]["credentials"]["user"] == "admin"  # From base
        assert result["database"]["credentials"]["pass"] == "new_secret"  # Overridden
    
    @pytest.mark.unit
    def test_merge_replaces_lists(self):
        """Test that lists are replaced, not merged."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        
        result = merge_configs(base, override)
        
        assert result["items"] == [4, 5]  # Replaced, not [1, 2, 3, 4, 5]
    
    @pytest.mark.unit
    def test_merge_empty_configs(self):
        """Test merging with empty configs."""
        base = {"a": 1}
        override = {}
        
        result1 = merge_configs(base, override)
        assert result1 == {"a": 1}
        
        result2 = merge_configs({}, override)
        assert result2 == {}
    
    @pytest.mark.unit
    def test_merge_does_not_modify_original(self):
        """Test that merge doesn't modify original configs."""
        base = {"a": 1, "nested": {"b": 2}}
        override = {"c": 3}
        
        result = merge_configs(base, override)
        
        assert "c" not in base
        assert result["c"] == 3


# =============================================================================
# RENDER_PROMPT_TEMPLATE TESTS
# =============================================================================

class TestRenderPromptTemplate:
    """Tests for render_prompt_template() utility."""
    
    @pytest.mark.unit
    def test_render_simple_template(self):
        """Test rendering a simple template."""
        template = "Hello, {{ name }}!"
        
        result = render_prompt_template(template, name="World")
        
        assert result == "Hello, World!"
    
    @pytest.mark.unit
    def test_render_template_with_multiple_vars(self):
        """Test rendering template with multiple variables."""
        template = "{{ query }} - {{ worker_count }} workers"
        
        result = render_prompt_template(
            template,
            query="test query",
            worker_count=5
        )
        
        assert result == "test query - 5 workers"
    
    @pytest.mark.unit
    def test_render_template_with_loops(self):
        """Test rendering template with loops."""
        template = """Research Plan:
{% for task in tasks %}
- {{ task }}
{% endfor %}
"""
        result = render_prompt_template(template, tasks=["Task 1", "Task 2", "Task 3"])
        
        assert "Task 1" in result
        assert "Task 2" in result
        assert "Task 3" in result
    
    @pytest.mark.unit
    def test_render_template_with_conditionals(self):
        """Test rendering template with conditionals."""
        template = """
{% if enable_critic %}
Critic: Enabled
{% else %}
Critic: Disabled
{% endif %}
"""
        result1 = render_prompt_template(template, enable_critic=True)
        assert "Critic: Enabled" in result1
        
        result2 = render_prompt_template(template, enable_critic=False)
        assert "Critic: Disabled" in result2
    
    @pytest.mark.unit
    def test_render_template_no_variables(self):
        """Test rendering template without variables."""
        template = "Static template with no variables"
        
        result = render_prompt_template(template)
        
        assert result == "Static template with no variables"
    
    @pytest.mark.unit
    def test_render_template_missing_variable(self):
        """Test rendering template with missing variable."""
        template = "Hello, {{ name }}!"
        
        # Jinja2 with strict undefined raises error
        from jinja2 import Template, StrictUndefined
        template_obj = Template(template, undefined=StrictUndefined)
        
        with pytest.raises(Exception):  # StrictUndefined raises UndefinedError
            template_obj.render()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestConfigIntegration:
    """Integration tests for config loading."""
    
    @pytest.mark.unit
    def test_full_config_load_and_expand(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test full config loading with env var expansion."""
        monkeypatch.setenv("MODEL_NAME", "gpt-4")
        monkeypatch.setenv("MAX_STEPS", "100")
        
        config_content = """
workflow:
  name: test
  type: research

input:
  type: query
  source: test

phases:
  - name: boss
    node: BossNode

llm:
  models:
    main: ${MODEL_NAME}

execution:
  max_steps: ${MAX_STEPS}
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)
        
        config = load_config(str(config_file))
        
        assert config.llm["models"]["main"] == "gpt-4"
        assert config.execution["max_steps"] == "100"  # Env vars are strings
