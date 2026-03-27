# Test Suite Documentation

**Location**: `/tests/`  
**Last Updated**: March 27, 2026

---

## Directory Structure

```
tests/
├── __init__.py                 # Test package marker
├── conftest.py                 # Shared fixtures and configuration
├── README.md                   # This file
│
├── unit/                       # Unit tests
│   ├── __init__.py
│   ├── test_core_components.py # Core functionality tests
│   ├── test_artifact_maker.py  # Artifact maker tests
│   ├── test_cli.py             # CLI tests
│   ├── test_config.py          # Configuration tests
│   ├── test_nodes.py           # Node tests
│   ├── test_state.py           # State management tests
│   ├── test_storage.py         # Storage tests
│   ├── test_workers.py         # Worker tests
│   └── test_mocks.py           # Mock provider tests
│
├── integration/                # Integration tests
│   ├── __init__.py
│   ├── test_nodes.py           # Node integration tests
│   ├── test_graph_execution.py # Graph execution tests
│   ├── test_workflow_execution.py # Workflow tests
│   └── test_storage_integration.py # Storage integration
│
├── e2e/                        # End-to-end tests
│   ├── __init__.py
│   ├── test_complete_workflows.py # Full workflow tests
│   └── test_workflows.py       # Workflow config tests
│
├── fixtures/                   # Test fixtures
│   ├── __init__.py
│   ├── sample_configs/         # Sample YAML configs
│   ├── sample_data/            # Sample input data
│   └── expected_outputs/       # Expected output files
│
└── mocks/                      # Mock implementations
    ├── __init__.py
    ├── mock_llm.py             # Mock LLM provider
    ├── mock_storage.py         # Mock storage backends
    └── mock_workers.py         # Mock workers
```

---

## Running Tests

### Quick Start

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src/deep_research_engine --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_core_components.py -v

# Run specific test class
uv run pytest tests/unit/test_core_components.py::TestCriticEvaluation -v

# Run specific test function
uv run pytest tests/unit/test_core_components.py::TestCriticEvaluation::test_critic_evaluation_creation -v
```

### Test Categories

```bash
# Run only unit tests
uv run pytest tests/unit/ -v

# Run only integration tests
uv run pytest tests/integration/ -v

# Run only e2e tests
uv run pytest tests/e2e/ -v

# Run only fast tests (exclude slow)
uv run pytest tests/ -m "not slow" -v

# Run only integration tests
uv run pytest tests/ -m integration -v
```

### Parallel Execution

```bash
# Run tests in parallel (4 workers)
uv run pytest tests/ -n 4

# Run with auto-detected CPU count
uv run pytest tests/ -n auto
```

---

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation.

**Characteristics**:
- Fast execution (<100ms per test)
- No external dependencies
- Mock all I/O operations
- Test single function/method

**Example**:
```python
def test_critic_evaluation_creation():
    """Test creating CriticEvaluation with valid data."""
    from deep_research_engine.state import CriticEvaluation
    
    evaluation = CriticEvaluation(
        passed=True,
        score=0.85,
        completeness=0.9,
        accuracy=0.8,
        citation_quality=0.85,
        coherence=0.85,
        issues=["Minor formatting issue"],
        suggestions=["Add more citations"],
        recommendation="pass",
        feedback_to="artifact_maker"
    )
    
    assert evaluation.passed is True
    assert evaluation.score == 0.85
```

### Integration Tests (`tests/integration/`)

Test component interactions.

**Characteristics**:
- Medium execution (<1s per test)
- May use real dependencies
- Test component interfaces
- Verify data flow

**Example**:
```python
@pytest.mark.asyncio
async def test_critic_node_with_sample_state():
    """Test critic node with realistic state."""
    from deep_research_engine.nodes.critic import critic_node
    from deep_research_engine.state import create_initial_state
    
    state = create_initial_state(...)
    state["worker_results"] = [...]
    
    with patch("deep_research_engine.nodes.critic.create_deep_agent"):
        result = await critic_node(state)
        assert "__goto__" in result
```

### End-to-End Tests (`tests/e2e/`)

Test complete workflows.

**Characteristics**:
- Slow execution (>1s per test)
- Use real dependencies
- Test full user scenarios
- Verify end-to-end functionality

**Example**:
```python
def test_rag_config_structure():
    """Test RAG config has required sections."""
    from deep_research_engine.config import load_config
    
    config = load_config("configs/rag.yaml")
    assert config.workflow["name"] == "rag"
    assert len(config.phases) >= 3
```

---

## Fixtures

### Available Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `sample_worker_results` | function | Sample worker output data |
| `sample_plan` | function | Sample research plan |
| `sample_state` | function | Complete state with results |
| `sample_critic_evaluation` | function | Sample critic evaluation |
| `mock_deep_agent` | function | Mock deep agent |
| `mock_structured_output` | function | Mock structured output factory |
| `tmp_config_file` | function | Temporary config file factory |
| `ffmpeg_metadata_file` | function | FFmpeg metadata file factory |

### Using Fixtures

```python
def test_with_fixture(sample_worker_results):
    """Test using sample_worker_results fixture."""
    assert len(sample_worker_results) == 2
    assert sample_worker_results[0]["worker_type"] == "web_researcher"

@pytest.mark.asyncio
async def test_with_async_fixture(sample_state):
    """Test using sample_state fixture."""
    assert "worker_results" in sample_state
    assert len(sample_state["worker_results"]) == 2
```

### Creating Custom Fixtures

Add to `tests/conftest.py`:

```python
@pytest.fixture
def my_custom_fixture():
    """Custom fixture for specific test needs."""
    # Setup
    data = create_test_data()
    yield data
    # Teardown
    cleanup_test_data()
```

---

## Writing Tests

### Test Naming Conventions

```python
# Test functions
def test_<component>_<action>_<expected>():
    def test_critic_evaluation_creation():
    def test_video_audio_extraction_nonexistent_file():

# Test classes
class Test<Component>:
    class TestCriticEvaluation:
    class TestVideoAudioExtraction:
```

### Test Structure

```python
import pytest

class TestComponent:
    """Tests for component."""
    
    def test_feature(self):
        """Test description."""
        # Arrange
        input_data = ...
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result == expected
```

### Async Tests

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_function()
    assert result is not None
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_doubling(input, expected):
    assert input * 2 == expected
```

---

## Mocking

### Mocking LLM Calls

```python
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_with_mock_llm():
    """Test with mocked LLM."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(content="Mock response")]
    })
    
    with patch("module.create_deep_agent", return_value=mock_llm):
        result = await function_with_llm()
        assert result is not None
```

### Mocking External Services

```python
@pytest.mark.asyncio
async def test_with_mock_storage():
    """Test with mocked storage."""
    mock_storage = AsyncMock()
    mock_storage.store = AsyncMock(return_value={"id": "test"})
    
    with patch("module.get_storage", return_value=mock_storage):
        result = await function_with_storage()
        assert result["id"] == "test"
```

---

## Coverage

### Generate Coverage Report

```bash
# HTML report
uv run pytest tests/ --cov=src/deep_research_engine --cov-report=html
open htmlcov/index.html

# Terminal report
uv run pytest tests/ --cov=src/deep_research_engine --cov-report=term-missing

# XML report (for CI/CD)
uv run pytest tests/ --cov=src/deep_research_engine --cov-report=xml
```

### Coverage Goals

| Component | Target | Current |
|-----------|--------|---------|
| Core Components | 90% | TBD |
| Nodes | 85% | TBD |
| Workers | 85% | TBD |
| Storage | 80% | TBD |
| Overall | 85% | TBD |

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install uv
      - run: uv sync --group dev
      - run: uv run pytest tests/ --cov
```

---

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure you're running from project root
cd /home/ubuntu/Documents/programs/deep-research-engine
uv run pytest tests/
```

**Async Test Issues**:
```python
# Always use @pytest.mark.asyncio decorator
@pytest.mark.asyncio
async def test_async_function():
    ...
```

**Fixture Not Found**:
```python
# Ensure fixture is in conftest.py or imported
# Check fixture name matches exactly
```

### Debug Mode

```bash
# Run with print statements visible
uv run pytest tests/ -s

# Run with Python debugger
uv run pytest tests/ --pdb

# Run specific test with verbose logging
uv run pytest tests/unit/test.py::test_function -vvv
```

---

## Contributing

### Adding New Tests

1. Create test file in appropriate directory
2. Add `__init__.py` if new directory
3. Follow naming conventions
4. Use fixtures where possible
5. Add docstrings to tests
6. Run tests locally before committing

### Test Review Checklist

- [ ] Test has clear purpose
- [ ] Test name describes behavior
- [ ] Test uses fixtures appropriately
- [ ] Test is isolated (no side effects)
- [ ] Test has assertions
- [ ] Test handles edge cases
- [ ] Test documentation is clear

---

**Maintained By**: Deep Research Engine Team  
**Contact**: See AGENTS.md for contribution guidelines
