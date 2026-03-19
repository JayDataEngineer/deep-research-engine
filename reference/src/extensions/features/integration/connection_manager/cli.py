"""Connection Manager CLI Plugin

Provides commands for managing API connection profiles.

Usage:
    uv run backend/manage.py connections list
    uv run backend/manage.py connections create <name> <provider> <url>
    uv run backend/manage.py connections activate <id>
    uv run backend/manage.py connections test <id>
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def get_manager():
    """Get the ConnectionManager instance."""
    from src.extensions.features.integration.connection_manager.manager import ConnectionManager
    return ConnectionManager()


def cmd_list(args):
    """List all connection profiles."""
    manager = get_manager()

    profiles = manager.list_profiles(
        provider=args.provider,
        active_only=args.active
    )

    if not profiles:
        print("\nNo connection profiles found.")
        return {"profiles": []}

    print(f"\nConnection Profiles ({len(profiles)} found):")
    print("-" * 80)

    # Group by provider
    by_provider = {}
    for profile in profiles:
        if profile.provider not in by_provider:
            by_provider[profile.provider] = []
        by_provider[profile.provider].append(profile)

    for provider, provider_profiles in sorted(by_provider.items()):
        print(f"\n  {provider.upper()}:")
        for profile in provider_profiles:
            active = " [ACTIVE]" if profile.is_active else ""
            print(f"    [{profile.id}] {profile.name}{active}")
            print(f"         URL: {profile.base_url}")
            if profile.model:
                print(f"         Model: {profile.model}")
            if profile.description:
                print(f"         {profile.description}")

    return {"profiles": [p.to_dict(include_sensitive=False) for p in profiles]}


def cmd_create(args):
    """Create a new connection profile."""
    manager = get_manager()

    print(f"\nCreating connection profile: {args.name}")
    print(f"  Provider: {args.provider}")
    print(f"  URL: {args.url}")

    # Get API key (prompt if not provided)
    api_key = args.api_key or input("  Enter API key (optional): ").strip()

    profile = manager.create_profile(
        name=args.name,
        provider=args.provider,
        base_url=args.url,
        api_key=api_key,
        model=args.model or "",
        description=args.description or "",
        tags=args.tags or []
    )

    print(f"\n✓ Connection profile created!")
    print(f"  ID: {profile.id}")
    print(f"  Name: {profile.name}")

    return {"profile": profile.to_dict(include_sensitive=False)}


def cmd_activate(args):
    """Activate a connection profile."""
    manager = get_manager()

    print(f"\nActivating profile: {args.id}")

    profile = manager.set_active(args.id)

    if not profile:
        print(f"\n✗ Profile not found: {args.id}")
        return {"error": "Not found"}

    print(f"\n✓ Profile activated: {profile.name}")
    print(f"  Provider: {profile.provider}")

    return {"profile": profile.to_dict(include_sensitive=False)}


def cmd_test(args):
    """Test a connection profile."""
    manager = get_manager()

    print(f"\nTesting profile: {args.id}")

    result = manager.test_connection(args.id)

    if not result.get("success"):
        print(f"\n✗ {result.get('error', 'Test failed')}")
        return {"error": result.get("error")}

    print(f"\nConnection Test Results: {result['profile']}")
    print("-" * 40)

    for test_name, test_result in result["tests"].items():
        status = "✓" if test_result else "✗"
        print(f"  {status} {test_name}")

    if result["all_pass"]:
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Some tests failed")

    return {"test_result": result}


def cmd_delete(args):
    """Delete a connection profile."""
    manager = get_manager()

    profile = manager.get_profile(args.id)
    if not profile:
        print(f"\n✗ Profile not found: {args.id}")
        return {"error": "Not found"}

    print(f"\nDeleting profile: {profile.name}")
    print(f"  Provider: {profile.provider}")
    print(f"  URL: {profile.base_url}")

    if args.force or input("  Confirm delete? (yes/no): ").lower() == "yes":
        if manager.delete_profile(args.id):
            print("\n✓ Profile deleted")
            return {"deleted": True}
        else:
            print("\n✗ Failed to delete profile")
            return {"error": "Delete failed"}
    else:
        print("\nCancelled")
        return {"cancelled": True}


def cmd_export(args):
    """Export a connection profile."""
    manager = get_manager()

    print(f"\nExporting profile: {args.id}")

    output_path = manager.export_profile(args.id, args.path)

    if not output_path:
        print(f"\n✗ Profile not found: {args.id}")
        return {"error": "Not found"}

    print(f"\n✓ Profile exported to: {output_path}")
    print("  (API key has been masked for security)")

    return {"export_path": str(output_path)}


def cmd_providers(args):
    """List all providers."""
    manager = get_manager()
    providers = manager.get_providers()

    print("\nConfigured Providers:")
    print("-" * 40)
    for provider in providers:
        profiles = manager.list_profiles(provider=provider)
        active = any(p.is_active for p in profiles)
        status = "[ACTIVE]" if active else ""
        print(f"  {provider:20} {len(profiles)} profiles {status}")

    return {"providers": providers}


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(
        dest="connections_subcommand",
        help="Connection commands"
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List connection profiles")
    list_parser.add_argument("--provider", "-p", help="Filter by provider")
    list_parser.add_argument("--active", "-a", action="store_true", help="Show only active")
    list_parser.set_defaults(func=cmd_list)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a connection profile")
    create_parser.add_argument("name", help="Profile name")
    create_parser.add_argument("provider", help="Provider (openai, anthropic, google, etc.)")
    create_parser.add_argument("url", help="API base URL")
    create_parser.add_argument("--api-key", "-k", help="API key (will prompt if not provided)")
    create_parser.add_argument("--model", "-m", help="Default model")
    create_parser.add_argument("--description", "-d", help="Description")
    create_parser.add_argument("--tags", nargs="+", help="Tags")
    create_parser.set_defaults(func=cmd_create)

    # activate command
    activate_parser = subparsers.add_parser("activate", help="Activate a profile")
    activate_parser.add_argument("id", help="Profile ID")
    activate_parser.set_defaults(func=cmd_activate)

    # test command
    test_parser = subparsers.add_parser("test", help="Test a connection")
    test_parser.add_argument("id", help="Profile ID")
    test_parser.set_defaults(func=cmd_test)

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a profile")
    delete_parser.add_argument("id", help="Profile ID")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=cmd_delete)

    # export command
    export_parser = subparsers.add_parser("export", help="Export a profile")
    export_parser.add_argument("id", help="Profile ID")
    export_parser.add_argument("path", nargs="?", help="Output path (optional)")
    export_parser.set_defaults(func=cmd_export)

    # providers command
    providers_parser = subparsers.add_parser("providers", help="List providers")
    providers_parser.set_defaults(func=cmd_providers)
