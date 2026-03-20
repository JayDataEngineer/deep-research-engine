# Deep Research Engine - Test Suite

This directory contains the comprehensive test suite for the Deep Research Engine.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared pytest fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_cli.py          # CLI tests
│   ├── test_config.py       # Configuration loading tests
│   ├── test_engine.py       # Engine workflow tests
│   ├── test_mocks.py        # Mock service tests
│   ├── test_nodes.py        # Node tests
│   ├── test_state.py        # State management tests
│   ├── test_storage.py      # Storage backend tests
│   ├── test_synthesis.py    # Synthesis tests
│   └── test_workers.py      # Worker tests
├── integration/             # Integration tests
│   ├── test_graph_execution.py  # Graph execution tests
│   └── test_workflow_execution.py  # Workflow execution tests
├── e2e/                     # End-to-end tests
│   ├── test_complete_workflows.py  # Complete workflow tests
│   └── test_workflows.py    # Config-based workflow tests
└── fixtures/                # Test fixtures and data
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
uv run pytest tests/e2e/
uv run pytest tests/integration/

# Run specific test file
uv run pytest tests/unit/test_config.py

# Run specific test
uv run pytest tests/unit/test_config.py::TestWorkflowConfig::test_minimal_config
```

### Test Markers

```bash
# Run only unit tests
uv run pytest tests/ -m unit

# Run only e2e tests
uv run pytest tests/ -m e2e

# Run only integration tests
uv run pytest tests/ -m integration

# Run only slow tests
uv run pytest tests/ -m slow

# Exclude slow tests
uv run pytest tests/ -m "not slow"
```

### Coverage

```bash
# Run with coverage report
uv run pytest tests/ --cov=src/deep_research_engine --cov-report=html

# Run with coverage report in terminal
uv run pytest tests/ --cov=src/deep_research_engine --cov-report=term-missing
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
uv run pytest tests/ -n auto
```

## Test Categories

### Unit Tests (`tests/unit/`)

Unit tests test individual components in isolation:

- **test_cli.py**: CLI commands and output formatting
- **test_config.py**: Configuration loading, validation, and utilities
- **test_engine.py**: Main workflow engine
- **test_mocks.py**: Mock services (LLM, web search, database, HTTP)
- **test_nodes.py**: Graph nodes (ingestion, drafter, critic, synthesizer)
- **test_state.py**: State management and type definitions
- **test_storage.py**: Storage backends
- **test_synthesis.py**: Synthesis worker and formats
- **test_workers.py**: Worker base class and registry

### Integration Tests (`tests/integration/`)

Integration tests test component interactions:

- **test_graph_execution.py**: Graph compilation and execution
- **test_workflow_execution.py**: Complete workflow execution

### End-to-End Tests (`tests/e2e/`)

E2E tests test complete workflows:

- **test_complete_workflows.py**: Worker results, state management, registry
- **test_workflows.py**: Config loading and validation

## Fixtures

Common fixtures are defined in `conftest.py`:

### State Fixtures
- `mock_state`: Mock engine state
- `initial_state`: Initial engine state for tests
- `state_with_results`: State with worker results

### Worker Fixtures
- `sample_worker_result`: Sample worker result
- `worker_config`: Test worker configuration
- `test_worker`: Test worker instance

### Config Fixtures
- `sample_config_dict`: Sample configuration dictionary
- `tmp_config_file`: Temporary config file
- `rag_config_path`: RAG config file path

### Utility Fixtures
- `sample_text`: Sample text for testing
- `sample_json_data`: Sample JSON data
- `mock_llm_response`: Mock LLM response
- `mock_async_llm_response`: Mock async LLM response

## Writing Tests

### Test Structure

```python
import pytest
from deep_research_engine.config import WorkflowConfig

class TestMyComponent:
    """Tests for my component."""
    
    @pytest.mark.unit
    def test_something(self, sample_config_dict):
        """Test description."""
        config = WorkflowConfig(**sample_config_dict)
        assert config.workflow["name"] == "test_workflow"
```

### Test Markers

Use appropriate markers:

- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (component interaction)
- `@pytest.mark.e2e`: End-to-end tests (complete workflows)
- `@pytest.mark.slow`: Slow tests (can be excluded with `-m "not slow"`)

### Async Tests

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await some_async_function()
    assert result is not None
```

## Current Test Status

As of March 20, 2026:

- **Total Tests**: 277
- **Passing**: 213 (77%)
- **Failed**: 24 (9%)
- **Errors**: 35 (13%)
- **Skipped**: 5 (2%)

### Passing Tests by Category

- **Unit Tests**: 160+ passing
- **E2E Tests**: 35+ passing
- **Integration Tests**: 15+ passing

### Known Issues

1. **CLI Tests**: Some CLI output format tests fail due to output formatting differences
2. **Engine Tests**: Some workflow tests fail due to graph compilation issues
3. **Mock Tests**: Many mock service tests require infrastructure setup
4. **Storage Tests**: Storage backend tests require actual storage services

### Infrastructure Requirements

Some tests require external services:

- PostgreSQL with pgvector
- Neo4j
- MongoDB
- MinIO (S3-compatible storage)

These tests are marked with `@pytest.mark.integration` and can be skipped:

```bash
# Skip integration tests
uv run pytest tests/ -m "not integration"
```

## Debugging Tests

### Verbose Output

```bash
# Show detailed test output
uv run pytest tests/ -v -s

# Show local variables on failure
uv run pytest tests/ -l

# Print output (disable capture)
uv run pytest tests/ -s
```

### Specific Tests

```bash
# Run specific test with debug output
uv run pytest tests/unit/test_config.py::TestWorkflowConfig::test_minimal_config -v -s

# Run tests matching pattern
uv run pytest tests/ -k "config" -v
```

### Coverage

```bash
# Find untested code
uv run pytest tests/ --cov=src/deep_research_engine --cov-report=term-missing
```

## Continuous Integration

Tests are configured to run in CI with:

```bash
# CI test command
uv run pytest tests/ -v --tb=short --strict-markers
```

## Best Practices

1. **Use Fixtures**: Reuse common setup via fixtures
2. **Mark Tests**: Always use appropriate markers (unit, integration, e2e)
3. **Isolate Tests**: Tests should be independent and runnable in any order
4. **Mock External Services**: Use mocks for external dependencies
5. **Fast Tests**: Keep unit tests fast (<100ms each)
6. **Descriptive Names**: Use descriptive test and method names
7. **Test Edge Cases**: Test error conditions and edge cases
8. **Document Tests**: Add docstrings to test methods

## Contributing

When adding new features:

1. Add corresponding unit tests
2. Add integration tests for component interactions
3. Add e2e tests for complete workflows
4. Ensure all tests pass before submitting
5. Maintain or improve code coverage

## Questions?

- See `conftest.py` for available fixtures
- See existing tests for examples
- Check `pyproject.toml` for pytest configuration
