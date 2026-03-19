# Deep Research Engine - Test Suite

Comprehensive test suite for the deep research engine, including unit tests, integration tests, and end-to-end tests.

## Test Structure

```
tests/
├── __init__.py              # Package init
├── conftest.py              # Pytest configuration and shared fixtures
├── README.md                # This file
│
├── fixtures/
│   └── __init__.py          # Shared test fixtures
│
├── unit/                    # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── test_config.py       # Config loading tests
│   ├── test_state.py        # State types and factory tests
│   ├── test_workers.py      # Worker ABC and registry tests
│   └── test_storage.py      # Storage backend tests
│
├── integration/             # Integration tests (component interactions)
│   ├── __init__.py
│   └── test_workflow_execution.py  # Workflow execution tests
│
└── e2e/                     # End-to-end tests (complete workflows)
    ├── __init__.py
    └── test_workflows.py    # Full workflow tests
```

## Running Tests

### Basic Commands

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test directory
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/e2e/

# Run specific test file
uv run pytest tests/unit/test_config.py

# Run specific test function
uv run pytest tests/unit/test_config.py::TestLoadConfig::test_load_valid_config
```

### Test Categories

```bash
# Run only unit tests
uv run pytest tests/unit/ -m unit

# Run only integration tests
uv run pytest tests/integration/ -m integration

# Run only e2e tests
uv run pytest tests/e2e/ -m e2e

# Run all except slow tests
uv run pytest tests/ -m "not slow"
```

### Coverage

```bash
# Run with coverage report
uv run pytest tests/ --cov=src/deep_research_engine --cov-report=term-missing

# Generate HTML coverage report
uv run pytest tests/ --cov=src/deep_research_engine --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
uv run pytest tests/ -n auto

# Run with specific number of workers
uv run pytest tests/ -n 4
```

## Test Fixtures

The test suite provides several shared fixtures via `conftest.py`:

### Config Fixtures

- `sample_config_dict` - Sample configuration dictionary
- `rag_config_path` - Temporary RAG config file path
- `config_with_env_vars` - Config with environment variable placeholders

### State Fixtures

- `initial_state` - Fresh EngineState instance
- `state_with_results` - State with worker results populated

### Worker Fixtures

- `worker_config` - Sample WorkerConfig
- `sample_worker_result` - Sample WorkerResult
- `TestWorker` - Test worker implementation
- `test_worker` - TestWorker instance

### Storage Fixtures

- `storage_config` - Sample storage configuration
- `sample_storage_data` - Sample data for storage
- `TestStorageBackend` - Test storage implementation
- `test_storage` - TestStorageBackend instance

### Usage Example

```python
import pytest
from deep_research_engine.config import load_config

def test_with_fixtures(sample_config_dict, initial_state):
    """Test using fixtures."""
    # Use sample_config_dict and initial_state
    assert sample_config_dict["workflow"]["name"] == "test_workflow"
    assert initial_state["current_phase"] == "boss"
```

## Writing Tests

### Unit Test Template

```python
"""Unit Tests for MyComponent."""

import pytest
from deep_research_engine.my_component import MyClass

class TestMyClass:
    """Tests for MyClass."""
    
    @pytest.mark.unit
    def test_something(self):
        """Test description."""
        # Arrange
        obj = MyClass()
        
        # Act
        result = obj.do_something()
        
        # Assert
        assert result is not None
```

### Integration Test Template

```python
"""Integration Tests for MyFeature."""

import pytest
from deep_research_engine.feature import run_feature

class TestMyFeatureIntegration:
    """Integration tests for feature."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_feature_workflow(self, sample_config_dict):
        """Test complete feature workflow."""
        # Test integration between components
        result = await run_feature("input", sample_config_dict)
        assert result.success is True
```

### E2E Test Template

```python
"""End-to-End Tests for Workflow."""

import pytest
from deep_research_engine.engine import run_workflow

class TestWorkflowE2E:
    """End-to-end workflow tests."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete workflow from start to finish."""
        config = load_config("configs/test.yaml")
        
        events = []
        async for event in run_workflow("query", config):
            events.append(event)
        
        assert events[-1]["type"] == "complete"
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.asyncio` - Async tests

## Coverage Goals

| Component | Coverage Goal | Current |
|-----------|--------------|---------|
| Config | 90% | TBD |
| State | 90% | TBD |
| Workers | 85% | TBD |
| Storage | 85% | TBD |
| Engine | 80% | TBD |
| **Overall** | **85%** | **TBD** |

## Continuous Integration

Tests are run automatically on:

- Every pull request
- Every merge to main
- Daily scheduled runs (for slow tests)

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure you're running from project root
cd /home/ubuntu/Documents/programs/deep-research-engine
uv run pytest tests/
```

**Async test failures:**
```bash
# Ensure pytest-asyncio is installed
uv sync --group dev
```

**Fixture not found:**
```bash
# Ensure conftest.py is in tests/ directory
ls tests/conftest.py
```

### Debug Mode

```bash
# Run with print statements visible
uv run pytest tests/ -s

# Run with pdb on failures
uv run pytest tests/ --pdb

# Run specific test with logging
uv run pytest tests/unit/test_config.py -v -s --log-cli-level=DEBUG
```

## Performance

Expected test execution times:

- Unit tests: < 5 seconds total
- Integration tests: < 30 seconds total
- E2E tests: < 2 minutes total
- Full suite: < 3 minutes

## Contributing

When adding new features:

1. Add unit tests for new components
2. Add integration tests for interactions
3. Add e2e tests for workflows
4. Ensure all tests pass
5. Maintain coverage goals

## License

MIT License - See project root for details
