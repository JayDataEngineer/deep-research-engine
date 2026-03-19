"""Deep Research Engine Test Suite.

This package contains all tests for the deep research engine:

- unit/ - Unit tests for individual components
- integration/ - Integration tests for component interactions
- e2e/ - End-to-end tests for complete workflows
- fixtures/ - Shared test fixtures and utilities

Usage:
    # Run all tests
    pytest tests/
    
    # Run unit tests only
    pytest tests/unit/
    
    # Run integration tests only
    pytest tests/integration/ -m integration
    
    # Run e2e tests only
    pytest tests/e2e/ -m e2e
    
    # Run with coverage
    pytest tests/ --cov=src/deep_research_engine --cov-report=html
    
    # Run specific test file
    pytest tests/unit/test_config.py -v
    
    # Run specific test function
    pytest tests/unit/test_config.py::TestLoadConfig::test_load_valid_config -v
"""

__version__ = "0.1.0"
