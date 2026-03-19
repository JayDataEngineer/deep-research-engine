"""Audio Producer Pipeline Mode Tests.

This module runs pipeline tests for all quality modes to verify the
queue-based pipeline architecture works correctly across different processing modes.

Architecture:
- LLM Queue → LLM Workers → TTS Queue → TTS Workers
- All stages run in parallel via queues
- No waiting for all segments to complete before moving to next stage

Modes tested:
1. Fast (Standard) - Rule-based chunking, single-pass LLM
2. Overdrive - Semantic chunking, two-phase refinement
3. Sectioned - Section-based processing, two-phase refinement
"""

import asyncio
import sys
from pathlib import Path

# Add tests directory to path for imports
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

from test_utility import print_summary

# Import mode tests
from test_modes.test_sectioned import test_sectioned_mode
from test_modes.test_overdrive import test_overdrive_mode
from test_modes.test_standard import test_standard_mode


async def run_all_tests():
    """Run all pipeline mode tests sequentially.

    Tests are run sequentially to avoid API rate limiting and resource contention.
    Each test verifies the queue-based pipeline architecture works correctly.
    """
    print("=" * 70)
    print("🧪 AUDIO PRODUCER PIPELINE MODE TESTS")
    print("=" * 70)
    print("\nTesting queue-based pipeline architecture across all quality modes:")
    print("  - LLM Queue → LLM Workers (parallel)")
    print("  - LLM Workers → TTS Queue (push immediately on complete)")
    print("  - TTS Queue → TTS Workers (parallel)")
    print("\n✅ Key verification: TTS starts BEFORE all LLM complete")

    results = []

    # Test 1: Standard (Fast) Mode
    print("\n" + "="*70)
    print("TEST 1/3: Standard (Fast) Mode")
    print("="*70)
    result = await test_standard_mode()
    results.append(result)

    # Brief pause between tests
    await asyncio.sleep(2)

    # Test 2: Sectioned Mode
    print("\n" + "="*70)
    print("TEST 2/3: Sectioned Mode")
    print("="*70)
    result = await test_sectioned_mode()
    results.append(result)

    # Brief pause between tests
    await asyncio.sleep(2)

    # Test 3: Overdrive Mode
    print("\n" + "="*70)
    print("TEST 3/3: Overdrive Mode")
    print("="*70)
    result = await test_overdrive_mode()
    results.append(result)

    # Print summary
    print_summary(results)

    return results


async def run_specific_test(mode: str):
    """Run a specific mode test.

    Args:
        mode: One of 'fast', 'sectioned', 'overdrive', or 'all'
    """
    if mode == "fast":
        result = await test_standard_mode()
        print_summary([result])
        return [result]
    elif mode == "sectioned":
        result = await test_sectioned_mode()
        print_summary([result])
        return [result]
    elif mode == "overdrive":
        result = await test_overdrive_mode()
        print_summary([result])
        return [result]
    elif mode == "all":
        return await run_all_tests()
    else:
        print(f"❌ Unknown mode: {mode}")
        print(f"   Valid modes: fast, sectioned, overdrive, all")
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test audio producer pipeline modes"
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "sectioned", "overdrive", "all"],
        default="all",
        help="Which mode to test (default: all)"
    )

    args = parser.parse_args()

    # Run tests
    results = asyncio.run(run_specific_test(args.mode))

    # Exit with error code if any tests failed
    if results and any(not r['success'] for r in results):
        sys.exit(1)
