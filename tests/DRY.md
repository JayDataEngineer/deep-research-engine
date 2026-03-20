# DRY (Don't Repeat Yourself) - Test Suite

## Overview

The test suite follows DRY principles by using fixtures and shared utilities to avoid code duplication.

## Shared Fixtures (`tests/conftest.py`)

### Config Path Fixtures

```python
# Use these instead of hardcoding paths
rag_config_path_fixed          # Path to rag.yaml
pdf_summarizer_config_path     # Path to pdf_summarizer.yaml
skill_generator_config_path    # Path to skill_generator.yaml
telegram_ingest_config_path    # Path to telegram_ingest.yaml
all_config_paths               # List of all config paths
```

**Example:**
```python
# ❌ DON'T: Repeat path construction
def test_something():
    config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
    config = load_config(str(config_path))

# ✅ DO: Use fixture
def test_something(rag_config_path_fixed: Path):
    config = load_config(str(rag_config_path_fixed))
```

### Loaded Config Fixtures

```python
# Pre-loaded configs for common tests
loaded_rag_config       # WorkflowConfig for rag.yaml
loaded_configs          # List of all loaded configs
```

**Example:**
```python
# ❌ DON'T: Load config in every test
def test_rag_config():
    config = load_config("configs/rag.yaml")
    assert config.workflow["name"] == "rag"

# ✅ DO: Use fixture
def test_rag_config(loaded_rag_config: WorkflowConfig):
    assert loaded_rag_config.workflow["name"] == "rag"
```

### State Fixtures

```python
# Reusable state objects
mock_state           # Mock engine state dict
initial_state        # Initial EngineState for tests
state_with_results   # State with worker results
```

### Worker Fixtures

```python
# Reusable worker objects
sample_worker_result  # Sample worker result dict
worker_config         # Test WorkerConfig
test_worker          # Test worker instance
```

### Config Dict Fixtures

```python
# Reusable config dictionaries
sample_config_dict  # Sample config for WorkflowConfig
```

## Test Utilities (`tests/fixtures/`)

### Mock Services

```python
from tests.fixtures.mocks import (
    MockLLMProvider,
    MockWebSearch,
    MockDatabase,
    MockHTTPClient,
)
```

## Common Patterns

### Pattern 1: Config Tests

```python
class TestMyConfig:
    @pytest.mark.unit
    def test_config_something(self, loaded_rag_config: WorkflowConfig):
        # Use pre-loaded config
        assert loaded_rag_config.workflow["name"] == "rag"
```

### Pattern 2: State Tests

```python
class TestMyState:
    @pytest.mark.unit
    def test_state_something(self, initial_state: EngineState):
        # Use pre-created state
        assert "workflow_name" in initial_state
```

### Pattern 3: Worker Tests

```python
class TestMyWorker:
    @pytest.mark.unit
    async def test_worker_something(
        self,
        test_worker: BaseWorker,
        initial_state: EngineState,
        worker_config: WorkerConfig
    ):
        # Use pre-created worker and state
        result = await test_worker.execute(initial_state, worker_config)
        assert result["success"] is True
```

### Pattern 4: Multiple Configs

```python
class TestMultipleConfigs:
    @pytest.mark.e2e
    def test_all_configs(self, loaded_configs: list[WorkflowConfig]):
        # Test all configs at once
        assert len(loaded_configs) == 4
        assert all(config is not None for config in loaded_configs)
```

## DRY Improvements Made

### Before (23 repetitions):
```python
def test_load_rag_config():
    config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
    config = load_config(str(config_path))
    ...

def test_rag_config_structure():
    config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
    config = load_config(str(config_path))
    ...
```

### After (0 repetitions):
```python
def test_load_rag_config(loaded_rag_config: WorkflowConfig):
    config = loaded_rag_config
    ...

def test_rag_config_structure(loaded_rag_config: WorkflowConfig):
    config = loaded_rag_config
    ...
```

## Benefits

1. **Less Code**: Reduced from ~700 lines of repeated path construction to ~50 lines of fixtures
2. **Easier Maintenance**: Change path in one place, not 23 tests
3. **Faster Tests**: Configs loaded once per test session via fixture caching
4. **More Readable**: Tests focus on assertions, not setup
5. **Consistent**: All tests use same config/state objects

## When to Add New Fixtures

Add a new fixture to `conftest.py` when:

1. **3+ Repetitions**: Same setup code appears in 3+ tests
2. **Complex Setup**: Setup requires multiple steps
3. **Shared Resource**: Multiple test files need same object
4. **Performance**: Expensive setup that can be cached

## Fixture Scope

Fixtures use different scopes for efficiency:

- **function** (default): Created per test
- **module**: Created once per test file
- **session**: Created once per test session (best for expensive setup)

```python
@pytest.fixture(scope="session")
def expensive_resource():
    # Created once for all tests
    ...
```

## Questions?

- See `conftest.py` for all available fixtures
- See `tests/fixtures/` for utility classes
- Check pytest docs for fixture best practices
