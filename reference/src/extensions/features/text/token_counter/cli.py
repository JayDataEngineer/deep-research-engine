"""Token Counter CLI Plugin

Provides commands for counting tokens.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_count(args):
    """Count tokens in text."""
    from src.extensions.features.text.token_counter.manager import TokenCounterManager
    
    try:
        manager = TokenCounterManager()
        
        # Get text from argument or stdin
        text = args.text
        if not text and not args.file:
            print("Please provide text or a file to count tokens for")
            return {"error": "No input provided"}
        
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        
        token_count = manager.count_tokens(text)
        
        print(f"Token count: {token_count}")
        print(f"Character count: {len(text)}")
        print(f"Word count: {len(text.split())}")
        
        return {"token_count": token_count, "char_count": len(text), "word_count": len(text.split())}
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return {"error": str(e)}


def cmd_analyze(args):
    """Analyze token composition."""
    from src.extensions.features.text.token_counter.manager import TokenCounterManager
    
    try:
        manager = TokenCounterManager()
        
        # Get text from argument or stdin
        text = args.text
        if not text and not args.file:
            print("Please provide text or a file to analyze")
            return {"error": "No input provided"}
        
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        
        analysis = manager.analyze_tokens(text)
        
        print(f"Token Analysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        return {"analysis": analysis}
    except Exception as e:
        print(f"Error analyzing tokens: {e}")
        return {"error": str(e)}


def register(parser):
    """Register token counter commands with the parser."""
    subparsers = parser.add_subparsers(dest="token_subcommand", help="Token counter commands")
    
    # Count command
    count_parser = subparsers.add_parser("count", help="Count tokens in text")
    count_parser.add_argument("text", nargs="?", help="Text to count tokens for")
    count_parser.add_argument("--file", "-f", help="File to read text from")
    count_parser.set_defaults(func=cmd_count)
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze token composition")
    analyze_parser.add_argument("text", nargs="?", help="Text to analyze")
    analyze_parser.add_argument("--file", "-f", help="File to read text from")
    analyze_parser.set_defaults(func=cmd_analyze)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Token Counter CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()