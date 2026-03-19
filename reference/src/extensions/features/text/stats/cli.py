"""Stats CLI Plugin

Provides commands for viewing and managing usage statistics.

Usage:
    uv run backend/manage.py stats summary
    uv run backend/manage.py stats daily --days 7
    uv run backend/manage.py stats models
    uv run backend/manage.py stats export stats.json
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def get_manager():
    """Get the StatsManager instance."""
    from src.extensions.features.text.stats.manager import StatsManager
    return StatsManager()


def cmd_summary(args):
    """Show statistics summary."""
    manager = get_manager()

    summary = manager.get_summary()

    print("\nUsage Statistics Summary:")
    print("-" * 60)
    print(f"  Total Requests: {summary['total_requests']:,}")
    print(f"  Total Tokens: {summary['total_tokens']:,}")
    print(f"  Total Characters: {summary['total_characters']:,}")
    print(f"  Uptime: {summary['uptime_days']} days")

    if summary['top_models']:
        print("\n  Top Models:")
        for model, count in list(summary['top_models'].items())[:5]:
            print(f"    - {model}: {count:,} requests")

    if summary['top_features']:
        print("\n  Top Features:")
        for feature, count in list(summary['top_features'].items())[:5]:
            print(f"    - {feature}: {count:,} uses")

    if summary['recent_daily']:
        print("\n  Recent Daily Stats:")
        for day in summary['recent_daily'][:7]:
            print(f"    {day['date']}: {day['requests']} requests, {day['tokens']:,} tokens")

    return summary


def cmd_daily(args):
    """Show daily statistics."""
    manager = get_manager()

    daily = manager.get_daily_stats(days=args.days)

    if not daily:
        print(f"\nNo daily stats found for the last {args.days} days.")
        return {"daily": []}

    print(f"\nDaily Statistics (last {args.days} days):")
    print("-" * 60)

    for day in daily:
        print(f"\n  {day['date']}:")
        print(f"    Requests: {day['requests']:,}")
        print(f"    Tokens: {day['tokens']:,}")
        if day.get('models'):
            print(f"    Models: {', '.join(list(day['models'].keys())[:3])}")

    return {"daily": daily}


def cmd_models(args):
    """Show model usage statistics."""
    manager = get_manager()

    models = manager.get_model_stats()

    if not models:
        print("\nNo model usage statistics found.")
        return {"models": {}}

    # Sort by usage
    sorted_models = sorted(models.items(), key=lambda x: x[1], reverse=True)

    print(f"\nModel Usage Statistics ({len(models)} models):")
    print("-" * 60)

    for model, count in sorted_models:
        print(f"  {model}: {count:,} requests")

    return {"models": models}


def cmd_features(args):
    """Show feature usage statistics."""
    manager = get_manager()

    features = manager.get_feature_stats()

    if not features:
        print("\nNo feature usage statistics found.")
        return {"features": {}}

    sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)

    print(f"\nFeature Usage Statistics ({len(features)} features):")
    print("-" * 60)

    for feature, count in sorted_features:
        print(f"  {feature}: {count:,} uses")

    return {"features": features}


def cmd_characters(args):
    """Show character usage statistics."""
    manager = get_manager()

    characters = manager.get_character_stats(limit=args.limit)

    if not characters:
        print("\nNo character usage statistics found.")
        return {"characters": []}

    print(f"\nTop {len(characters)} Characters:")
    print("-" * 60)

    for i, char in enumerate(characters, 1):
        print(f"  {i}. {char['name']}: {char['count']:,} interactions")

    return {"characters": characters}


def cmd_export(args):
    """Export statistics to a file."""
    manager = get_manager()

    manager.export_stats(args.output)

    print(f"\nStatistics exported to: {args.output}")
    return {"exported": args.output}


def cmd_record(args):
    """Record a stat (for testing)."""
    manager = get_manager()

    manager.record_request(
        model=args.model or "test",
        tokens=args.tokens or 0,
        feature=args.feature,
    )

    print(f"\nRecorded request: model={args.model}, tokens={args.tokens}, feature={args.feature}")
    return {"recorded": True}


def cmd_reset(args):
    """Reset all statistics."""
    manager = get_manager()

    if args.force:
        manager.reset_stats()
        print("\nAll statistics have been reset.")
        return {"reset": True}
    else:
        print("\n  Use --force to confirm reset. This cannot be undone!")
        return {"reset": False}


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(dest="stats_subcommand", help="Stats commands")

    # summary command
    summary_parser = subparsers.add_parser("summary", help="Show statistics summary")
    summary_parser.set_defaults(func=cmd_summary)

    # daily command
    daily_parser = subparsers.add_parser("daily", help="Show daily statistics")
    daily_parser.add_argument("--days", "-d", type=int, default=7, help="Number of days")
    daily_parser.set_defaults(func=cmd_daily)

    # models command
    models_parser = subparsers.add_parser("models", help="Show model usage statistics")
    models_parser.set_defaults(func=cmd_models)

    # features command
    features_parser = subparsers.add_parser("features", help="Show feature usage statistics")
    features_parser.set_defaults(func=cmd_features)

    # characters command
    chars_parser = subparsers.add_parser("characters", help="Show character usage statistics")
    chars_parser.add_argument("--limit", "-l", type=int, default=10, help="Max characters")
    chars_parser.set_defaults(func=cmd_characters)

    # export command
    export_parser = subparsers.add_parser("export", help="Export statistics to file")
    export_parser.add_argument("output", help="Output file path")
    export_parser.set_defaults(func=cmd_export)

    # record command (for testing)
    record_parser = subparsers.add_parser("record", help="Record a stat (testing)")
    record_parser.add_argument("--model", "-m", help="Model name")
    record_parser.add_argument("--tokens", "-t", type=int, default=0, help="Token count")
    record_parser.add_argument("--feature", "-f", help="Feature name")
    record_parser.set_defaults(func=cmd_record)

    # reset command
    reset_parser = subparsers.add_parser("reset", help="Reset all statistics")
    reset_parser.add_argument("--force", "-f", action="store_true", help="Confirm reset")
    reset_parser.set_defaults(func=cmd_reset)
