"""Probability CLI Plugin

Provides commands for managing probability/random sampling.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_info(args):
    """Show probability information."""
    from src.extensions.features.text.probability.manager import ProbabilityManager
    
    try:
        manager = ProbabilityManager()
        info = manager.get_probability_info()
        
        print(f"\nProbability Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        return {"probability_info": info}
    except Exception as e:
        print(f"Error getting probability info: {e}")
        return {"error": str(e)}


def cmd_sample(args):
    """Generate a random sample."""
    from src.extensions.features.text.probability.manager import ProbabilityManager
    
    try:
        manager = ProbabilityManager()
        
        if args.distribution == "normal":
            sample = manager.sample_normal(args.mean, args.std_dev)
        elif args.distribution == "uniform":
            sample = manager.sample_uniform(args.low, args.high)
        elif args.distribution == "binomial":
            sample = manager.sample_binomial(args.n, args.p)
        else:
            print(f"Unknown distribution: {args.distribution}")
            return {"error": "Unknown distribution"}
        
        print(f"Sample from {args.distribution} distribution: {sample}")
        return {"sample": sample}
    except Exception as e:
        print(f"Error generating sample: {e}")
        return {"error": str(e)}


def register(parser):
    """Register probability commands with the parser."""
    subparsers = parser.add_subparsers(dest="probability_subcommand", help="Probability commands")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show probability information")
    info_parser.set_defaults(func=cmd_info)
    
    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Generate a random sample")
    sample_parser.add_argument("distribution", choices=["normal", "uniform", "binomial"], help="Distribution type")
    sample_parser.add_argument("--mean", type=float, default=0.0, help="Mean for normal distribution")
    sample_parser.add_argument("--std-dev", type=float, default=1.0, help="Standard deviation for normal distribution")
    sample_parser.add_argument("--low", type=float, default=0.0, help="Low bound for uniform distribution")
    sample_parser.add_argument("--high", type=float, default=1.0, help="High bound for uniform distribution")
    sample_parser.add_argument("--n", type=int, default=1, help="Number of trials for binomial distribution")
    sample_parser.add_argument("--p", type=float, default=0.5, help="Probability for binomial distribution")
    sample_parser.set_defaults(func=cmd_sample)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Probability CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()