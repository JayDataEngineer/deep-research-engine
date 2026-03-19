"""Rating System - Centralized 0-100 scale rating for deep research decisions.

This module provides:
- Constants for the 0-100 rating scale
- Threshold comparisons for decision-making
- Quality/Confidence evaluation helpers
- LLM output normalization

CRITICAL: ALL ratings in deep research use the 0-100 scale.
This is the SINGLE SOURCE OF TRUTH for rating decisions.

Why 0-100 scale?
- LLMs understand percentages better (clearer prompts)
- Integer math avoids floating-point comparison issues
- More intuitive for configuration (quality_threshold: 70)
- Consistent with config.yaml and state_models.py

Usage:
    from .rating import Rating, RatingDecision, normalize_rating

    # Check if quality passes threshold
    if RatingDecision.passes(quality_score, threshold=70):
        ...

    # Normalize LLM output to 0-100
    confidence = normalize_rating(llm_output)  # Handles 0-1, 1-10, 0-100
"""

from typing import Any, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from ..logging_utils import get_logger

logger = get_logger(__name__, __name__.split('.')[-1])


# =============================================================================
# RATING SCALE CONSTANTS (0-100)
# =============================================================================

class Rating:
    """Rating scale constants for 0-100 scale.

    ALL thresholds and ratings in deep research use this scale.
    Config values in YAML should use 0-100 for consistency.

    Examples:
        quality_threshold: 70  # In config.yaml
        confidence_threshold: 80  # In spawner_config
    """

    # Scale bounds
    MIN = 0
    MAX = 100

    # Quality thresholds (used by critic)
    EXCELLENT = 90      # Outstanding quality
    GOOD = 70           # Acceptable quality (default pass threshold)
    FAIR = 50           # Needs revision
    POOR = 30           # Significant issues
    FAIL = 0            # Unacceptable

    # Confidence thresholds (used by explorer/boss)
    HIGH_CONFIDENCE = 80    # Very confident in results
    MEDIUM_CONFIDENCE = 50  # Moderately confident
    LOW_CONFIDENCE = 30     # Low confidence, needs more research

    # Default values
    DEFAULT_QUALITY = 80
    DEFAULT_CONFIDENCE = 50
    DEFAULT_COMPLETENESS = 90

    @classmethod
    def clamp(cls, value: int) -> int:
        """Clamp a rating to 0-100 range.

        Args:
            value: Rating value

        Returns:
            Value clamped to 0-100
        """
        return max(cls.MIN, min(cls.MAX, value))

    @classmethod
    def is_valid(cls, value: int) -> bool:
        """Check if a rating is in valid range.

        Args:
            value: Rating value

        Returns:
            True if 0 <= value <= 100
        """
        return cls.MIN <= value <= cls.MAX

    @classmethod
    def to_decimal(cls, value: int) -> float:
        """Convert 0-100 rating to 0-1 decimal.

        Args:
            value: Rating in 0-100 scale

        Returns:
            Float in 0-1 range
        """
        return cls.clamp(value) / 100.0

    @classmethod
    def from_decimal(cls, value: float) -> int:
        """Convert 0-1 decimal to 0-100 rating.

        Args:
            value: Float in 0-1 range

        Returns:
            Rating in 0-100 scale
        """
        return cls.clamp(int(round(value * 100)))

    @classmethod
    def get_quality_label(cls, score: int) -> str:
        """Get human-readable quality label.

        Args:
            score: Quality score (0-100)

        Returns:
            Label string
        """
        score = cls.clamp(score)
        if score >= cls.EXCELLENT:
            return "excellent"
        elif score >= cls.GOOD:
            return "good"
        elif score >= cls.FAIR:
            return "fair"
        elif score >= cls.POOR:
            return "poor"
        else:
            return "fail"

    @classmethod
    def get_confidence_label(cls, score: int) -> str:
        """Get human-readable confidence label.

        Args:
            score: Confidence score (0-100)

        Returns:
            Label string
        """
        score = cls.clamp(score)
        if score >= cls.HIGH_CONFIDENCE:
            return "high"
        elif score >= cls.MEDIUM_CONFIDENCE:
            return "medium"
        elif score >= cls.LOW_CONFIDENCE:
            return "low"
        else:
            return "very_low"


# =============================================================================
# RATING NORMALIZATION (handles LLM quirks)
# =============================================================================

def normalize_rating(value: Any, default: int = Rating.DEFAULT_CONFIDENCE) -> int:
    """Normalize any rating value to 0-100 scale.

    LLMs sometimes return weird values like:
    - 0.85 (correct 0-1 scale, convert to 85)
    - 1 or 1.0 (could be 0-1 scale -> convert to 100)
    - 5 or 8 (1-10 scale, convert to 50 or 80)
    - 85 (correct 0-100 scale, keep as 85)

    FAILSAFE RULES:
    - If value <= 1, multiply by 100 (0-1 scale -> 0-100)
    - If value is 2-10, multiply by 10 (1-10 scale -> 0-100)
    - If value is > 10, assume 0-100 scale (clamp if needed)

    Args:
        value: The rating value from LLM
        default: Default value if conversion fails

    Returns:
        Integer in 0-100 range
    """
    if value is None:
        return default

    try:
        val = float(value)

        # Handle negative values
        if val < 0:
            return Rating.MIN

        # FAILSAFE RULE 1: If <= 1, treat as 0-1 scale (multiply by 100)
        # This catches: 0.85 -> 85, 1.0 -> 100, 0.5 -> 50
        if val <= 1:
            return Rating.clamp(int(round(val * 100)))

        # FAILSAFE RULE 2: If 2-10, treat as 1-10 scale (multiply by 10)
        # This catches: 5 -> 50, 8 -> 80, 10 -> 100
        if 2 <= val <= 10:
            return Rating.clamp(int(round(val * 10)))

        # FAILSAFE RULE 3: If > 10, assume 0-100 scale
        return Rating.clamp(int(round(val)))

    except (TypeError, ValueError):
        return default


# Alias for backward compatibility
normalize_confidence = normalize_rating


# =============================================================================
# RATING DECISIONS (threshold-based decisions)
# =============================================================================

class RatingDecision:
    """Rating-based decision helpers.

    Provides consistent threshold-based decision making across
    the deep research system.

    Usage:
        # Check if quality passes
        if RatingDecision.passes(quality_score, threshold=70):
            proceed_to_synthesis()

        # Get decision with reason
        decision = RatingDecision.evaluate(quality_score, threshold=70)
        if decision.passed:
            print(f"Passed with {decision.score}")
    """

    @staticmethod
    def passes(score: int, threshold: int = Rating.GOOD) -> bool:
        """Check if a score passes a threshold.

        Args:
            score: The score to check (0-100)
            threshold: The threshold to compare against (0-100)

        Returns:
            True if score >= threshold
        """
        return Rating.clamp(score) >= Rating.clamp(threshold)

    @staticmethod
    def fails(score: int, threshold: int = Rating.FAIR) -> bool:
        """Check if a score fails below a threshold.

        Args:
            score: The score to check (0-100)
            threshold: The threshold to compare against (0-100)

        Returns:
            True if score < threshold
        """
        return Rating.clamp(score) < Rating.clamp(threshold)

    @staticmethod
    def needs_revision(score: int, threshold: int = Rating.FAIR) -> bool:
        """Check if a score needs revision (between fail and pass).

        Args:
            score: The score to check (0-100)
            threshold: The minimum acceptable threshold (0-100)

        Returns:
            True if score is in revision range
        """
        score = Rating.clamp(score)
        return Rating.FAIR <= score < threshold

    @staticmethod
    def evaluate(
        score: int,
        pass_threshold: int = Rating.GOOD,
        fail_threshold: int = Rating.FAIR,
    ) -> 'RatingResult':
        """Evaluate a score against thresholds.

        Args:
            score: The score to evaluate (0-100)
            pass_threshold: Threshold for passing (default: 70)
            fail_threshold: Threshold for failing (default: 50)

        Returns:
            RatingResult with decision and details
        """
        score = Rating.clamp(score)

        if score >= pass_threshold:
            return RatingResult(
                passed=True,
                score=score,
                decision="pass",
                reason=f"Score {score} >= threshold {pass_threshold}",
            )
        elif score >= fail_threshold:
            return RatingResult(
                passed=False,
                score=score,
                decision="revise",
                reason=f"Score {score} in revision range ({fail_threshold}-{pass_threshold})",
            )
        else:
            return RatingResult(
                passed=False,
                score=score,
                decision="fail",
                reason=f"Score {score} < fail threshold {fail_threshold}",
            )

    @staticmethod
    def should_continue_research(
        confidence: int,
        gaps_count: int,
        min_confidence: int = Rating.GOOD,
        max_gaps: int = 3,
    ) -> bool:
        """Decide if research should continue based on confidence and gaps.

        This is used by the explorer/flash_explorer to determine if
        more research is needed.

        Args:
            confidence: Confidence score (0-100)
            gaps_count: Number of identified gaps
            min_confidence: Minimum confidence to stop (default: 70)
            max_gaps: Maximum acceptable gaps (default: 3)

        Returns:
            True if research should continue (not confident enough)
        """
        confidence = Rating.clamp(confidence)

        # High confidence with few gaps = done
        if confidence >= min_confidence and gaps_count <= max_gaps:
            return False

        # Low confidence or many gaps = continue
        return True

    @staticmethod
    def accept_research(
        confidence: int,
        has_no_gaps: bool,
        threshold: int = Rating.GOOD,
    ) -> bool:
        """Decide if research output should be accepted.

        Accept if ANY of these conditions are met:
        1. Confidence >= 80 (high confidence)
        2. No gaps identified
        3. Confidence >= threshold

        Args:
            confidence: Confidence score (0-100)
            has_no_gaps: Whether no gaps were identified
            threshold: Minimum acceptable threshold

        Returns:
            True if research should be accepted
        """
        confidence = Rating.clamp(confidence)

        # Accept if confidence >= 80 (high confidence)
        if confidence >= Rating.HIGH_CONFIDENCE:
            return True

        # Accept if no gaps identified
        if has_no_gaps:
            return True

        # Accept if confidence >= threshold
        if confidence >= threshold:
            return True

        return False


@dataclass
class RatingResult:
    """Result of a rating evaluation."""

    passed: bool
    score: int
    decision: str  # "pass", "revise", "fail"
    reason: str

    @property
    def label(self) -> str:
        """Get quality label for the score."""
        return Rating.get_quality_label(self.score)


# =============================================================================
# THRESHOLD HELPERS
# =============================================================================

def get_thresholds_from_config(yaml_config: Dict[str, Any]) -> Dict[str, int]:
    """Get all thresholds from config with fallbacks.

    Ensures all thresholds are normalized to 0-100 scale.

    Args:
        yaml_config: The full YAML configuration

    Returns:
        Dict with all threshold values (0-100 scale)
    """
    thresholds = yaml_config.get("thresholds", {})
    confidence = thresholds.get("confidence", {})
    quality = thresholds.get("quality", {})

    return {
        # Confidence thresholds (0-100 scale)
        "confidence_high": normalize_rating(confidence.get("high", Rating.HIGH_CONFIDENCE)),
        "confidence_medium": normalize_rating(confidence.get("medium", Rating.MEDIUM_CONFIDENCE)),
        "confidence_low": normalize_rating(confidence.get("low", Rating.LOW_CONFIDENCE)),

        # Quality thresholds (0-100 scale)
        "quality_excellent": normalize_rating(quality.get("excellent", Rating.EXCELLENT)),
        "quality_good": normalize_rating(quality.get("good", Rating.GOOD)),
        "quality_fair": normalize_rating(quality.get("fair", Rating.FAIR)),
        "quality_poor": normalize_rating(quality.get("poor", Rating.POOR)),

        # Critic thresholds (from critic config section)
        "quality_threshold": normalize_rating(
            yaml_config.get("critic", {}).get("quality_threshold", Rating.GOOD)
        ),
        "revise_threshold": normalize_rating(
            yaml_config.get("critic", {}).get("revise_threshold", Rating.FAIR)
        ),
        "direct_confidence": normalize_rating(
            yaml_config.get("critic", {}).get("direct_confidence", Rating.HIGH_CONFIDENCE)
        ),

        # Spawner thresholds
        "spawner_quality_threshold": normalize_rating(
            yaml_config.get("spawner_config", {}).get("quality_threshold", Rating.GOOD)
        ),
    }


def validate_threshold_config(value: Any, name: str) -> int:
    """Validate and normalize a threshold configuration value.

    Args:
        value: Config value (may be 0-1 or 0-100)
        name: Name of the threshold for error messages

    Returns:
        Validated threshold in 0-100 scale

    Raises:
        ValueError: If value cannot be normalized
    """
    normalized = normalize_rating(value)

    if not Rating.is_valid(normalized):
        raise ValueError(
            f"Invalid threshold '{name}': {value}. "
            f"Must be in range 0-100 (or 0-1 decimal)."
        )

    return normalized


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def confidence_to_decimal(value: int) -> float:
    """Convert 0-100 confidence to 0-1 decimal for storage/API.

    Args:
        value: Confidence in 0-100 scale

    Returns:
        Float in 0-1 range
    """
    return Rating.to_decimal(value)


def decimal_to_confidence(value: float) -> int:
    """Convert 0-1 decimal to 0-100 confidence.

    Args:
        value: Float in 0-1 range

    Returns:
        Confidence in 0-100 scale
    """
    return Rating.from_decimal(value)
