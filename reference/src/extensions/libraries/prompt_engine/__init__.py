"""Prompt Engine - YAML-based prompt stack management.

This library provides:
- PromptStack schema with layers and variables
- StackManager for CRUD operations with YAML persistence
- Jinja2 template rendering for advanced prompt templating
- LangChain message conversion for LLM integration
- Dict message conversion for LangGraph state
- Hot-reload on file changes
- Stack registry with categories and tags
- Crash-only prompt validation (PromptValidator, RequiredPromptRegistry)

Example:
    from src.extensions.libraries.prompt_engine import (
        PromptStack,
        StackLayer,
        StackVariable,
        StackManager,
        PromptValidator,
    )

    # Create a stack
    stack = PromptStack(
        name="my_stack",
        layers=[
            StackLayer(id="layer1", content="You are {{char_name}}"),
        ],
    )

    # Convert to LangChain messages for LLM invocation
    messages = stack.to_langchain_messages(char_name="Elena")

    # Or convert to dict for LangGraph state
    dict_messages = stack.to_dict_messages(char_name="Elena")

    # Or use the manager
    manager = StackManager(stacks_dir)
    await manager.initialize()
    stacks = manager.get_all()

    # Or validate required prompts (crash-only)
    validator = PromptValidator("my_extension")
    validator.require("system_prompt", "user_prompt")
    validator.validate_or_crash(config)
"""

# Schemas
from .schema import (
    PromptStack,
    StackLayer,
    StackVariable,
    StackRegistry,
    DEFAULT_STACKS,
)

# Manager
from .manager import StackManager, get_stack_manager, init_stack_manager

# Validation (crash-only)
from .validation import PromptValidator, RequiredPromptRegistry, StrictPromptRegistry

# API Routers (moved from src/api/)
from .prompt_stacks import router as prompt_stacks_router
from .prompts import router as prompts_router

__all__ = [
    # Schemas
    "PromptStack",
    "StackLayer",
    "StackVariable",
    "StackRegistry",
    "DEFAULT_STACKS",
    # Manager
    "StackManager",
    "get_stack_manager",
    "init_stack_manager",
    # Validation (crash-only)
    "PromptValidator",
    "RequiredPromptRegistry",
    "StrictPromptRegistry",
    # API Routers
    "prompt_stacks_router",
    "prompts_router",
]
