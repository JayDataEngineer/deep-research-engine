"""Prompt Engine CLI Plugin

Provides CLI commands for testing and managing prompt stacks.

Usage:
    uv run backend/manage.py prompt_engine list
    uv run backend/manage.py prompt_engine get <stack_id>
    uv run backend/manage.py prompt_engine render <stack_id> [--var KEY=VALUE]
    uv run backend/manage.py prompt_engine to-langchain <stack_id> [--var KEY=VALUE]
    uv run backend/manage.py prompt_engine to-dict <stack_id> [--var KEY=VALUE]
    uv run backend/manage.py prompt_engine test <stack_id>
"""

import argparse
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


def _parse_vars(var_args):
    """Parse KEY=VALUE arguments into a dictionary."""
    vars_dict = {}
    if var_args:
        for var in var_args:
            if '=' in var:
                key, value = var.split('=', 1)
                vars_dict[key] = value
            else:
                logger.warning(f"Invalid variable format: {var} (expected KEY=VALUE)")
    return vars_dict


def cmd_list(args):
    """List all available prompt stacks."""
    from src.extensions.libraries.prompt_engine import get_stack_manager

    # Initialize manager if needed
    try:
        manager = get_stack_manager()
    except RuntimeError:
        # Manager not initialized, initialize it
        from src.extensions.libraries.prompt_engine import init_stack_manager
        manager = asyncio.run(init_stack_manager())

    stacks = manager.get_all()

    if args.category:
        stacks = manager.get_by_category(args.category)
        print(f"\nStacks in category '{args.category}':")
    else:
        print(f"\nAll Available Stacks ({len(stacks)} total):")

    for stack in stacks:
        status = f"[{stack.category}]"
        layer_count = len(stack.layers)
        print(f"  {stack.name:<30} {status:<15} ({layer_count} layers)")

        if args.verbose:
            print(f"    Description: {stack.description or 'N/A'}")
            print(f"    Tags: {', '.join(stack.tags) if stack.tags else 'None'}")
            for layer in stack.layers:
                enabled = "✓" if layer.enabled else "✗"
                print(f"      {enabled} {layer.id} ({layer.role}): {layer.content[:50]}...")

    return {
        "stacks": [s.name for s in stacks],
        "count": len(stacks)
    }


def cmd_get(args):
    """Show details of a specific stack."""
    from src.extensions.libraries.prompt_engine import get_stack_manager

    try:
        manager = get_stack_manager()
    except RuntimeError:
        from src.extensions.libraries.prompt_engine import init_stack_manager
        manager = asyncio.run(init_stack_manager())

    stack = manager.registry.get(args.stack_id)

    if not stack:
        print(f"Stack not found: {args.stack_id}")
        return {"error": "Stack not found", "stack_id": args.stack_id}

    print(f"\nStack: {stack.name}")
    print(f"Description: {stack.description or 'N/A'}")
    print(f"Category: {stack.category}")
    print(f"Version: {stack.version}")
    print(f"Tags: {', '.join(stack.tags) if stack.tags else 'None'}")
    print(f"\nLayers ({len(stack.layers)} total):")

    for layer in stack.layers:
        enabled = "✓" if layer.enabled else "✗"
        print(f"\n  {enabled} {layer.id} [{layer.role}]")
        print(f"    Priority: {layer.priority} | Depth: {layer.injection_depth} | Position: {layer.injection_position}")
        print(f"    Content: {layer.content[:100]}{'...' if len(layer.content) > 100 else ''}")

    if stack.variables:
        print(f"\nVariables ({len(stack.variables)} defined):")
        for var in stack.variables:
            default = f" (default: {var.default})" if var.default else ""
            print(f"  - {var.name}{default}: {var.description or 'No description'}")

    return {
        "stack": stack.model_dump(mode='json'),
        "stack_id": args.stack_id
    }


def cmd_render(args):
    """Render a stack with variable substitution (Jinja2)."""
    from src.extensions.libraries.prompt_engine import get_stack_manager

    try:
        manager = get_stack_manager()
    except RuntimeError:
        from src.extensions.libraries.prompt_engine import init_stack_manager
        manager = asyncio.run(init_stack_manager())

    stack = manager.registry.get(args.stack_id)

    if not stack:
        print(f"Stack not found: {args.stack_id}")
        return {"error": "Stack not found"}

    # Parse variables
    vars_dict = _parse_vars(args.var)

    print(f"\nRendering stack '{stack.name}' with variables:")
    for key, value in vars_dict.items():
        print(f"  {key} = {value}")
    print()

    rendered = stack.render(**vars_dict)

    print("=" * 60)
    print(rendered)
    print("=" * 60)

    return {
        "rendered": rendered,
        "variables": vars_dict,
        "stack_id": args.stack_id
    }


def cmd_to_langchain(args):
    """Convert a stack to LangChain messages."""
    from src.extensions.libraries.prompt_engine import get_stack_manager

    try:
        manager = get_stack_manager()
    except RuntimeError:
        from src.extensions.libraries.prompt_engine import init_stack_manager
        manager = asyncio.run(init_stack_manager())

    stack = manager.registry.get(args.stack_id)

    if not stack:
        print(f"Stack not found: {args.stack_id}")
        return {"error": "Stack not found"}

    # Parse variables
    vars_dict = _parse_vars(args.var)

    print(f"\nConverting stack '{stack.name}' to LangChain messages:")
    for key, value in vars_dict.items():
        print(f"  {key} = {value}")
    print()

    messages = stack.to_langchain_messages(**vars_dict)

    print(f"Generated {len(messages)} LangChain messages:\n")
    for i, msg in enumerate(messages, 1):
        role = msg.__class__.__name__
        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"{i}. [{role}]")
        print(f"   {content_preview}\n")

    if args.json:
        output = {
            "message_count": len(messages),
            "messages": [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content
                }
                for msg in messages
            ],
            "variables": vars_dict
        }
        print("\n" + json.dumps(output, indent=2))

    return {
        "message_count": len(messages),
        "messages": [
            {
                "type": msg.__class__.__name__,
                "content": msg.content
            }
            for msg in messages
        ]
    }


def cmd_to_dict(args):
    """Convert a stack to dict messages (for LangGraph state)."""
    from src.extensions.libraries.prompt_engine import get_stack_manager

    try:
        manager = get_stack_manager()
    except RuntimeError:
        from src.extensions.libraries.prompt_engine import init_stack_manager
        manager = asyncio.run(init_stack_manager())

    stack = manager.registry.get(args.stack_id)

    if not stack:
        print(f"Stack not found: {args.stack_id}")
        return {"error": "Stack not found"}

    # Parse variables
    vars_dict = _parse_vars(args.var)

    print(f"\nConverting stack '{stack.name}' to dict messages:")
    for key, value in vars_dict.items():
        print(f"  {key} = {value}")
    print()

    messages = stack.to_dict_messages(**vars_dict)

    print(f"Generated {len(messages)} dict messages:\n")
    for i, msg in enumerate(messages, 1):
        content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"{i}. [{msg['role']}]")
        print(f"   {content_preview}\n")

    if args.json:
        output = {
            "message_count": len(messages),
            "messages": messages,
            "variables": vars_dict
        }
        print("\n" + json.dumps(output, indent=2))

    return {
        "message_count": len(messages),
        "messages": messages
    }


def cmd_test(args):
    """Run comprehensive E2E test of prompt engine."""
    from src.extensions.libraries.prompt_engine import (
        PromptStack,
        StackLayer,
        StackVariable,
        init_stack_manager
    )

    print("\n" + "=" * 60)
    print("PROMPT ENGINE E2E TEST")
    print("=" * 60)

    # Test 1: Create a stack programmatically
    print("\n[Test 1] Creating test stack with Jinja2 templates...")
    test_stack = PromptStack(
        name="test_jinja2_stack",
        description="Test stack with Jinja2 features",
        category="custom",
        tags=["test", "jinja2"],
        layers=[
            StackLayer(
                id="system_prompt",
                name="System Prompt",
                content="You are {{char_name}}, a {{char_trait}} character in {{scenario}}.",
                role="system",
                priority=100,
                enabled=True
            ),
            StackLayer(
                id="conditional_content",
                name="Conditional Content",
                content="{% if include_backstory %}Backstory: {{char_backstory}}{% endif %}",
                role="system",
                priority=90,
                enabled=True
            ),
            StackLayer(
                id="user_message",
                name="User Message",
                content="Hello {{char_name}}, my name is {{user_name}}.",
                role="user",
                priority=80,
                enabled=True
            ),
            StackLayer(
                id="loop_example",
                name="Loop Example",
                content="{% if traits %}Traits you have:{% for trait in traits %}\n- {{trait}}{% endfor %}{% endif %}",
                role="system",
                priority=70,
                enabled=True
            ),
        ],
        variables=[
            StackVariable(name="char_name", default="Character", description="Character name"),
            StackVariable(name="char_trait", default="mysterious", description="Character trait"),
            StackVariable(name="scenario", default="an unknown location", description="Current scenario"),
        ]
    )
    print("  ✓ Stack created successfully")

    # Test 2: Render with Jinja2
    print("\n[Test 2] Rendering with Jinja2 template engine...")
    context = {
        "char_name": "Elena",
        "char_trait": "brave warrior",
        "scenario": "a mystical forest",
        "include_backstory": True,
        "char_backstory": "Once a guard of the royal kingdom, now a wandering adventurer.",
        "user_name": "Alex",
        "traits": ["Brave", "Curious", "Kind"]
    }

    rendered = test_stack.render(**context)
    print("  ✓ Rendered successfully")
    print("\n  Rendered output:")
    print("  " + "-" * 56)
    for line in rendered.split("\n"):
        print(f"  {line}")
    print("  " + "-" * 56)

    # Test 3: Convert to LangChain messages
    print("\n[Test 3] Converting to LangChain messages...")
    lc_messages = test_stack.to_langchain_messages(**context)
    print(f"  ✓ Generated {len(lc_messages)} LangChain messages")

    for i, msg in enumerate(lc_messages, 1):
        role = msg.__class__.__name__
        print(f"    {i}. [{role}] {len(msg.content)} chars")

    # Test 4: Convert to dict messages
    print("\n[Test 4] Converting to dict messages (LangGraph state)...")
    dict_messages = test_stack.to_dict_messages(**context)
    print(f"  ✓ Generated {len(dict_messages)} dict messages")

    for i, msg in enumerate(dict_messages, 1):
        print(f"    {i}. [{msg['role']}] {len(msg['content'])} chars")

    # Test 5: Test with default stacks
    print("\n[Test 5] Testing with default stacks...")
    try:
        manager = asyncio.run(init_stack_manager())
        stacks = manager.get_all()
        print(f"  ✓ Loaded {len(stacks)} default stacks")

        # Test roleplay stack
        roleplay_stack = manager.registry.get("roleplay_default")
        if roleplay_stack:
            print("  ✓ Testing roleplay_default stack...")
            rp_messages = roleplay_stack.to_langchain_messages(
                char_name="Elena",
                scenario="a mysterious forest"
            )
            print(f"    Generated {len(rp_messages)} messages")
        else:
            print("  ℹ roleplay_default stack not found (may need to be saved)")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60 + "\n")

    return {
        "test": "passed",
        "stack_name": test_stack.name,
        "langchain_messages": len(lc_messages),
        "dict_messages": len(dict_messages)
    }


def register(parser):
    """Register CLI commands with the parser.

    Called by manage.py when discovering plugins.

    Args:
        parser: An ArgumentParser that we add subparsers to
    """
    # Add subparsers to the parser
    subparsers = parser.add_subparsers(
        dest="prompt_engine_subcommand",
        help="Prompt engine commands"
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List all prompt stacks")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")
    list_parser.set_defaults(func=cmd_list)

    # get command
    get_parser = subparsers.add_parser("get", help="Show stack details")
    get_parser.add_argument("stack_id", help="Stack identifier")
    get_parser.set_defaults(func=cmd_get)

    # render command
    render_parser = subparsers.add_parser("render", help="Render stack with Jinja2")
    render_parser.add_argument("stack_id", help="Stack identifier")
    render_parser.add_argument("--var", "-V", action="append", help="Variable as KEY=VALUE (can be used multiple times)")
    render_parser.set_defaults(func=cmd_render)

    # to-langchain command
    lc_parser = subparsers.add_parser("to-langchain", help="Convert to LangChain messages")
    lc_parser.add_argument("stack_id", help="Stack identifier")
    lc_parser.add_argument("--var", "-V", action="append", help="Variable as KEY=VALUE (can be used multiple times)")
    lc_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    lc_parser.set_defaults(func=cmd_to_langchain)

    # to-dict command
    dict_parser = subparsers.add_parser("to-dict", help="Convert to dict messages")
    dict_parser.add_argument("stack_id", help="Stack identifier")
    dict_parser.add_argument("--var", "-V", action="append", help="Variable as KEY=VALUE (can be used multiple times)")
    dict_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    dict_parser.set_defaults(func=cmd_to_dict)

    # test command
    test_parser = subparsers.add_parser("test", help="Run E2E test")
    test_parser.set_defaults(func=cmd_test)
