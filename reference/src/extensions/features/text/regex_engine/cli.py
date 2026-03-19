"""Regex Engine CLI Plugin

Provides commands for managing regex patterns.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all regex patterns."""
    from src.extensions.features.text.regex_engine.manager import RegexEngineManager
    
    try:
        manager = RegexEngineManager()
        patterns = manager.list_patterns()
        
        print(f"\nRegex Patterns ({len(patterns)} found):")
        for pattern in patterns:
            print(f"  - {pattern}")
        
        return {"patterns": patterns}
    except Exception as e:
        print(f"Error listing regex patterns: {e}")
        return {"error": str(e)}


def cmd_apply(args):
    """Apply a regex pattern to text."""
    from src.extensions.features.text.regex_engine.manager import RegexEngineManager
    
    try:
        manager = RegexEngineManager()
        
        # Get text from argument or stdin
        text = args.text
        if not text and not args.file:
            print("Please provide text or a file to apply regex to")
            return {"error": "No input provided"}
        
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        
        result = manager.apply_pattern(args.pattern_name, text)
        
        print(f"Applied pattern '{args.pattern_name}' to text:")
        print(f"Result: {result}")
        
        return {"result": result}
    except Exception as e:
        print(f"Error applying regex pattern: {e}")
        return {"error": str(e)}


def cmd_test(args):
    """Test a regex pattern against text."""
    from src.extensions.features.text.regex_engine.manager import RegexEngineManager
    
    try:
        manager = RegexEngineManager()
        
        # Get text from argument or stdin
        text = args.text
        if not text and not args.file:
            print("Please provide text or a file to test regex against")
            return {"error": "No input provided"}
        
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        
        matches = manager.test_pattern(args.pattern_name, text)
        
        print(f"Testing pattern '{args.pattern_name}' against text:")
        print(f"Matches: {matches}")
        
        return {"matches": matches}
    except Exception as e:
        print(f"Error testing regex pattern: {e}")
        return {"error": str(e)}


def register(parser):
    """Register regex engine commands with the parser."""
    subparsers = parser.add_subparsers(dest="regex_subcommand", help="Regex engine commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all regex patterns")
    list_parser.set_defaults(func=cmd_list)
    
    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply a regex pattern to text")
    apply_parser.add_argument("pattern_name", help="Name of the pattern to apply")
    apply_parser.add_argument("text", nargs="?", help="Text to apply regex to")
    apply_parser.add_argument("--file", "-f", help="File to read text from")
    apply_parser.set_defaults(func=cmd_apply)
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test a regex pattern against text")
    test_parser.add_argument("pattern_name", help="Name of the pattern to test")
    test_parser.add_argument("text", nargs="?", help="Text to test regex against")
    test_parser.add_argument("--file", "-f", help="File to read text from")
    test_parser.set_defaults(func=cmd_test)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Regex Engine CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()