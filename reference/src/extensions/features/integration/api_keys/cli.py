"""API Keys CLI Plugin

Provides command-line interface for managing encrypted API keys.
"""

import asyncio
import json
from pathlib import Path

from src.services.api_keys import (
    APIKeyService,
    KeyType,
    get_api_key_service,
)


def register(parser):
    """Register CLI commands for API key management."""
    subparsers = parser.add_subparsers(dest="command", help="API key commands")

    # List keys
    list_parser = subparsers.add_parser("list", help="List all stored API keys")
    list_parser.add_argument("--type", help="Filter by key type")

    # Create key
    create_parser = subparsers.add_parser("create", help="Store a new API key")
    create_parser.add_argument("type", help="Key type (openai, anthropic, zai, glm, google, cohere, custom)")
    create_parser.add_argument("name", help="Name for this key")
    create_parser.add_argument("key", help="The API key to store")
    create_parser.add_argument("--base-url", help="Custom base URL for this key")

    # Get key
    get_parser = subparsers.add_parser("get", help="Retrieve and decrypt an API key")
    get_parser.add_argument("type", help="Key type")
    get_parser.add_argument("name", help="Key name")

    # Update key
    update_parser = subparsers.add_parser("update", help="Update an existing API key")
    update_parser.add_argument("type", help="Key type")
    update_parser.add_argument("name", help="Key name")
    update_parser.add_argument("key", help="The new API key")

    # Delete key
    delete_parser = subparsers.add_parser("delete", help="Delete an API key")
    delete_parser.add_argument("type", help="Key type")
    delete_parser.add_argument("name", help="Key name")

    # Deactivate key
    deactivate_parser = subparsers.add_parser("deactivate", help="Deactivate an API key without deleting it")
    deactivate_parser.add_argument("type", help="Key type")
    deactivate_parser.add_argument("name", help="Key name")

    # Test key
    test_parser = subparsers.add_parser("test", help="Test an API key without saving it")
    test_parser.add_argument("type", help="Key type")
    test_parser.add_argument("key", help="The API key to test")
    test_parser.add_argument("--base-url", help="Base URL for the API")

    # Set the handler function
    parser.set_defaults(func=handle_command)


async def _list_keys(args):
    """List all stored API keys."""
    service = get_api_key_service()
    await service.initialize()

    keys = await service.list_keys(key_type=args.type)

    if not keys:
        print("No API keys stored")
        return

    print(f"\n{'Type':<15} {'Name':<25} {'Created':<20} {'Active':<8}")
    print("-" * 70)
    for key in keys:
        created = key.created_at[:19] if key.created_at else "N/A"
        active = "Yes" if key.is_active else "No"
        print(f"{key.key_type:<15} {key.name:<25} {created:<20} {active:<8}")


async def _create_key(args):
    """Store a new API key."""
    from src.services.api_keys import APIKeyCreate

    service = get_api_key_service()
    await service.initialize()

    key_data = APIKeyCreate(
        key_type=args.type,
        name=args.name,
        api_key=args.key,
        base_url=args.base_url
    )

    result = await service.create_key(key_data)
    print(f"✅ API key '{args.name}' of type '{args.type}' stored successfully")
    print(f"   ID: {result.id}")


async def _get_key(args):
    """Retrieve and decrypt an API key."""
    service = get_api_key_service()
    await service.initialize()

    decrypted_key = await service.get_key(args.type, args.name)

    if decrypted_key is None:
        print(f"❌ API key '{args.name}' of type '{args.type}' not found")
        return

    print(f"✅ API key '{args.name}' of type '{args.type}':")
    print(f"   {decrypted_key}")


async def _update_key(args):
    """Update an existing API key."""
    from src.services.api_keys import APIKeyUpdate

    service = get_api_key_service()
    await service.initialize()

    key_data = APIKeyUpdate(api_key=args.key)
    updated = await service.update_key(args.type, args.name, key_data)

    if updated:
        print(f"✅ API key '{args.name}' of type '{args.type}' updated successfully")
    else:
        print(f"❌ API key '{args.name}' of type '{args.type}' not found")


async def _delete_key(args):
    """Delete an API key."""
    service = get_api_key_service()
    await service.initialize()

    deleted = await service.delete_key(args.type, args.name)

    if deleted:
        print(f"✅ API key '{args.name}' of type '{args.type}' deleted successfully")
    else:
        print(f"❌ API key '{args.name}' of type '{args.type}' not found")


async def _deactivate_key(args):
    """Deactivate an API key without deleting it."""
    service = get_api_key_service()
    await service.initialize()

    deactivated = await service.deactivate_key(args.type, args.name)

    if deactivated:
        print(f"✅ API key '{args.name}' of type '{args.type}' deactivated successfully")
    else:
        print(f"❌ API key '{args.name}' of type '{args.type}' not found")


async def _test_key(args):
    """Test an API key without saving it."""
    import httpx

    default_base_urls = {
        KeyType.OPENAI: "https://api.openai.com/v1",
        KeyType.ANTHROPIC: "https://api.anthropic.com",
        KeyType.ZAI: "https://api.z.ai/api/coding/paas/v4",
        KeyType.GLM: "https://open.bigmodel.cn/api/paas/v4",
        KeyType.GOOGLE: "https://generativelanguage.googleapis.com/v1beta",
        KeyType.COHERE: "https://api.cohere.ai/v1",
    }

    base_url = args.base_url or default_base_urls.get(args.type, "")

    if not base_url:
        print(f"❌ Base URL is required for key type: {args.type}")
        return

    print(f"Testing API key for type: {args.type}...")

    try:
        if args.type == KeyType.OPENAI:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {args.key}"}
                )
                if response.status_code == 200:
                    print("✅ API key is valid")
                elif response.status_code == 401:
                    print("❌ Invalid API key")
                else:
                    print(f"⚠️  API returned status {response.status_code}")

        elif args.type == KeyType.ANTHROPIC:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{base_url}/v1/messages",
                    headers={
                        "x-api-key": args.key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={"model": "claude-3-haiku-20240307", "max_tokens": 1, "messages": []}
                )
                if response.status_code == 400:
                    print("✅ API key is valid")
                elif response.status_code == 401:
                    print("❌ Invalid API key")
                else:
                    print(f"⚠️  API returned status {response.status_code}")

        else:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {args.key}"}
                )
                if response.status_code == 200:
                    print("✅ API key is valid")
                elif response.status_code == 401:
                    print("❌ Invalid API key")
                else:
                    print("⚠️  Could not fully validate API key")

    except httpx.TimeoutException:
        print("❌ Request timed out - check your network connection")
    except httpx.ConnectError:
        print("❌ Could not connect to API - check the base URL")
    except Exception as e:
        print(f"❌ Error testing API key: {e}")


async def handle_command(args):
    """Handle the API keys command."""
    if args.command == "list":
        await _list_keys(args)
    elif args.command == "create":
        await _create_key(args)
    elif args.command == "get":
        await _get_key(args)
    elif args.command == "update":
        await _update_key(args)
    elif args.command == "delete":
        await _delete_key(args)
    elif args.command == "deactivate":
        await _deactivate_key(args)
    elif args.command == "test":
        await _test_key(args)
    else:
        print("Unknown command. Available commands: list, create, get, update, delete, deactivate, test")


def main(args):
    """Synchronous entry point for CLI."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(handle_command(args), loop)
        future.result(timeout=60)
    except RuntimeError:
        asyncio.run(handle_command(args))
