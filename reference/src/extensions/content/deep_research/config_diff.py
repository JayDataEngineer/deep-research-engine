"""Configuration Diffing Tool.

Provides utilities to compare different configuration variants side-by-side.
Useful for understanding configuration differences and debugging.

Usage:
    from deep_research.config_diff import compare_configs, print_config_diff

    from .constants import DEEP_RESEARCH_CONFIG_DIR
    diff = compare_configs(
        DEEP_RESEARCH_CONFIG_DIR / "rag" / "config.yaml",
        DEEP_RESEARCH_CONFIG_DIR / "rag" / "variant1.yaml"
    )
    print_config_diff(diff)
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .constants import DEEP_RESEARCH_CONFIG_DIR


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two config dictionaries.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary (takes precedence)

    Returns:
        Merged configuration dictionary
    """
    result = base.copy()

    # Keys that should be REPLACED, not merged
    REPLACE_KEYS = {
        "specialist_configs",  # Completely replace worker definitions
        "spawner_config",  # Replace entire spawner config
        "worker_graph",  # Replace worker graph (variant workers override base)
        "character_card_mappings",  # Replace character mappings
        "brand_context",  # Replace brand context (variant branding overrides base)
    }

    for key, value in override.items():
        if key in REPLACE_KEYS:
            # Replace entirely, don't merge
            result[key] = value
        elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


class DiffType(Enum):
    """Type of configuration difference."""
    ADDED = "added"      # Key exists in config2 but not config1
    REMOVED = "removed"  # Key exists in config1 but not config2
    MODIFIED = "modified"  # Key exists in both but values differ
    SAME = "same"        # Key exists in both with same value


@dataclass
class ConfigDiff:
    """Represents a single configuration difference."""
    key_path: str  # Dot-separated path to the key (e.g., "model_configs.spawner.temperature")
    diff_type: DiffType
    value1: Any  # Value in config1
    value2: Any  # Value in config2
    description: str = ""  # Human-readable description


@dataclass
class ConfigComparison:
    """Result of comparing two configurations."""
    config1_path: str
    config2_path: str
    config1_name: str = ""
    config2_name: str = ""
    differences: List[ConfigDiff] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.config1_name:
            self.config1_name = Path(self.config1_path).stem
        if not self.config2_name:
            self.config2_name = Path(self.config2_path).stem

        # Calculate summary
        self.summary = {
            DiffType.ADDED: sum(1 for d in self.differences if d.diff_type == DiffType.ADDED),
            DiffType.REMOVED: sum(1 for d in self.differences if d.diff_type == DiffType.REMOVED),
            DiffType.MODIFIED: sum(1 for d in self.differences if d.diff_type == DiffType.MODIFIED),
            DiffType.SAME: sum(1 for d in self.differences if d.diff_type == DiffType.SAME),
        }

    @property
    def total_differences(self) -> int:
        """Total number of differences (excluding SAME)."""
        return sum([
            self.summary[DiffType.ADDED],
            self.summary[DiffType.REMOVED],
            self.summary[DiffType.MODIFIED]
        ])

    @property
    def similarity_score(self) -> float:
        """Calculate similarity score between 0 and 1.

        1.0 = identical configs
        0.0 = completely different configs
        """
        total_keys = sum(self.summary.values())
        if total_keys == 0:
            return 1.0
        same_keys = self.summary[DiffType.SAME]
        return same_keys / total_keys


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def compare_dicts(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    path: str = "",
    diffs: List[ConfigDiff] = None,
    include_same: bool = False
) -> List[ConfigDiff]:
    """Compare two dictionaries recursively and find differences.

    Args:
        dict1: First dictionary
        dict2: Second dictionary
        path: Current key path (for recursion)
        diffs: Accumulated differences (for recursion)
        include_same: If True, include keys that are the same

    Returns:
        List of ConfigDiff objects
    """
    if diffs is None:
        diffs = []

    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        key_path = f"{path}.{key}" if path else key

        if key not in dict1:
            diffs.append(ConfigDiff(
                key_path=key_path,
                diff_type=DiffType.ADDED,
                value1=None,
                value2=dict2[key],
                description=f"Added: {key}"
            ))
        elif key not in dict2:
            diffs.append(ConfigDiff(
                key_path=key_path,
                diff_type=DiffType.REMOVED,
                value1=dict1[key],
                value2=None,
                description=f"Removed: {key}"
            ))
        else:
            val1 = dict1[key]
            val2 = dict2[key]

            # Compare based on types
            if isinstance(val1, dict) and isinstance(val2, dict):
                compare_dicts(val1, val2, key_path, diffs, include_same)
            elif isinstance(val1, list) and isinstance(val2, list):
                if val1 != val2:
                    diffs.append(ConfigDiff(
                        key_path=key_path,
                        diff_type=DiffType.MODIFIED,
                        value1=val1,
                        value2=val2,
                        description=f"Modified list: {key}"
                    ))
                elif include_same:
                    diffs.append(ConfigDiff(
                        key_path=key_path,
                        diff_type=DiffType.SAME,
                        value1=val1,
                        value2=val2,
                        description=f"Same: {key}"
                    ))
            elif val1 != val2:
                diffs.append(ConfigDiff(
                    key_path=key_path,
                    diff_type=DiffType.MODIFIED,
                    value1=val1,
                    value2=val2,
                    description=f"Modified: {key}"
                ))
            elif include_same:
                diffs.append(ConfigDiff(
                    key_path=key_path,
                    diff_type=DiffType.SAME,
                    value1=val1,
                    value2=val2,
                    description=f"Same: {key}"
                ))

    return diffs


def compare_configs(
    config1_path: Union[str, Path],
    config2_path: Union[str, Path],
    include_same: bool = False
) -> ConfigComparison:
    """Compare two configuration files.

    Args:
        config1_path: Path to first config file
        config2_path: Path to second config file
        include_same: If True, include keys that are the same

    Returns:
        ConfigComparison object with differences
    """
    config1 = load_config(config1_path)
    config2 = load_config(config2_path)

    differences = compare_dicts(config1, config2, include_same=include_same)

    return ConfigComparison(
        config1_path=str(config1_path),
        config2_path=str(config2_path),
        differences=differences
    )


def print_config_diff(comparison: ConfigComparison, show_same: bool = False) -> None:
    """Print a human-readable configuration diff.

    Args:
        comparison: ConfigComparison object
        show_same: If True, also show keys that are the same
    """
    print(f"\n{'=' * 70}")
    print(f"Configuration Comparison")
    print(f"{'=' * 70}")
    print(f"\nComparing:")
    print(f"  {comparison.config1_name} <- {comparison.config1_path}")
    print(f"  {comparison.config2_name} <- {comparison.config2_path}")
    print(f"\nSimilarity Score: {comparison.similarity_score:.1%}")
    print(f"Total Differences: {comparison.total_differences}")

    # Summary by type
    print(f"\nSummary:")
    for diff_type, count in comparison.summary.items():
        if count > 0 and (diff_type != DiffType.SAME or show_same):
            print(f"  {diff_type.value.capitalize()}: {count}")

    # Group differences by type
    by_type: Dict[DiffType, List[ConfigDiff]] = {}
    for diff in comparison.differences:
        if diff.diff_type != DiffType.SAME or show_same:
            if diff.diff_type not in by_type:
                by_type[diff.diff_type] = []
            by_type[diff.diff_type].append(diff)

    # Print differences by type
    for diff_type in [DiffType.ADDED, DiffType.REMOVED, DiffType.MODIFIED, DiffType.SAME]:
        if diff_type in by_type and (diff_type != DiffType.SAME or show_same):
            print(f"\n{diff_type.value.upper()} ({len(by_type[diff_type])} items):")
            for diff in sorted(by_type[diff_type], key=lambda d: d.key_path):
                print(f"\n  {diff.key_path}")
                if diff.diff_type == DiffType.ADDED:
                    print(f"    New value: {_format_value(diff.value2)}")
                elif diff.diff_type == DiffType.REMOVED:
                    print(f"    Old value: {_format_value(diff.value1)}")
                elif diff.diff_type == DiffType.MODIFIED:
                    print(f"    - {_format_value(diff.value1)}")
                    print(f"    + {_format_value(diff.value2)}")
                elif diff.diff_type == DiffType.SAME:
                    print(f"    Value: {_format_value(diff.value1)}")

    print(f"\n{'=' * 70}\n")


def _format_value(value: Any, max_length: int = 100) -> str:
    """Format a value for display.

    Args:
        value: Value to format
        max_length: Maximum length for string representation

    Returns:
        Formatted string representation
    """
    if value is None:
        return "None"
    elif isinstance(value, str):
        if len(value) > max_length:
            return f'"{value[:max_length]}..."'
        return f'"{value}"'
    elif isinstance(value, (dict, list)):
        s = str(value)
        if len(s) > max_length:
            return f"{s[:max_length]}..."
        return s
    else:
        return str(value)


def find_config_diffs(
    config_path: Union[str, Path],
    base_config_path: Union[str, Path] = None
) -> List[ConfigDiff]:
    """Find all differences between a config and the base config.

    Args:
        config_path: Path to config file to compare
        base_config_path: Path to base config file (defaults to DEEP_RESEARCH_CONFIG_DIR/base.yaml)

    Returns:
        List of ConfigDiff objects (excludes SAME)
    """
    if base_config_path is None:
        base_config_path = DEEP_RESEARCH_CONFIG_DIR / "base.yaml"
    return compare_configs(base_config_path, config_path, include_same=False).differences


def generate_config_diff_markdown(comparison: ConfigComparison) -> str:
    """Generate a markdown representation of the config diff.

    Args:
        comparison: ConfigComparison object

    Returns:
        Markdown string
    """
    lines = [
        f"# Configuration Comparison: {comparison.config1_name} vs {comparison.config2_name}",
        "",
        f"**Similarity Score:** {comparison.similarity_score:.1%}",
        f"**Total Differences:** {comparison.total_differences}",
        "",
        "## Summary",
        ""
    ]

    for diff_type, count in comparison.summary.items():
        if count > 0:
            lines.append(f"- **{diff_type.value.capitalize()}:** {count}")

    lines.append("")

    # Group differences by type
    by_type: Dict[DiffType, List[ConfigDiff]] = {}
    for diff in comparison.differences:
        if diff.diff_type != DiffType.SAME:
            if diff.diff_type not in by_type:
                by_type[diff.diff_type] = []
            by_type[diff.diff_type].append(diff)

    for diff_type in [DiffType.ADDED, DiffType.REMOVED, DiffType.MODIFIED]:
        if diff_type in by_type:
            lines.append(f"## {diff_type.value.capitalize()} ({len(by_type[diff_type])} items)")
            lines.append("")
            for diff in sorted(by_type[diff_type], key=lambda d: d.key_path):
                lines.append(f"### `{diff.key_path}`")
                lines.append("")

                if diff_type == DiffType.ADDED:
                    lines.append(f"**New value:** `{_format_value(diff.value2, 200)}`")
                elif diff_type == DiffType.REMOVED:
                    lines.append(f"**Old value:** `{_format_value(diff.value1, 200)}`")
                elif diff_type == DiffType.MODIFIED:
                    lines.append(f"**Old value:** `{_format_value(diff.value1, 200)}`")
                    lines.append(f"**New value:** `{_format_value(diff.value2, 200)}`")

                lines.append("")

    return "\n".join(lines)


# =============================================================================
# CLI UTILITY
# =============================================================================

def main():
    """CLI entry point for config diffing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two configuration files"
    )
    parser.add_argument(
        "config1",
        help="Path to first config file"
    )
    parser.add_argument(
        "config2",
        help="Path to second config file"
    )
    parser.add_argument(
        "--show-same",
        action="store_true",
        help="Show keys that are the same"
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output as markdown instead of human-readable"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file (default: stdout)"
    )

    args = parser.parse_args()

    comparison = compare_configs(args.config1, args.config2, include_same=args.show_same)

    if args.markdown:
        output = generate_config_diff_markdown(comparison)
    else:
        # Capture print output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        print_config_diff(comparison, show_same=args.show_same)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Diff written to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
