"""Error Utilities - Centralized error formatting and crash-only helpers.

This module provides:
- Standardized critical error box formatting
- Consistent error message templates
- Crash-only error helpers (no silent failures)

CRASH-ONLY PHILOSOPHY:
All critical errors should crash with clear, actionable error messages.
No fallbacks that hide problems - fail fast and loud.
"""

from typing import Optional, Any


# =============================================================================
# CRITICAL ERROR BOX TEMPLATES
# =============================================================================

class ErrorBoxTemplates:
    """Standardized error message templates with ASCII art boxes.

    All critical errors use this format for consistency and visibility.
    """

    @staticmethod
    def critical_failure(component: str, reason: str, solutions: list[str]) -> str:
        """Format a critical failure error box.

        Args:
            component: The component that failed (e.g., "CONTEXT ENGINE")
            reason: Why the failure occurred
            solutions: List of solution steps

        Returns:
            Formatted error box as string
        """
        border = "╔═══════════════════════════════════════════════════════════════════╗"
        thin_border = "║" + " " * 63 + "║"

        lines = [
            border,
            thin_border,
            f"║     {component} - CRITICAL FAILURE         ".ljust(64) + "║",
            thin_border,
            border,
            "",
        ]

        if reason:
            lines.append(reason)
            lines.append("")

        if solutions:
            lines.append("Solutions:")
            for i, solution in enumerate(solutions, 1):
                lines.append(f"  {i}. {solution}")

        return "\n".join(lines)

    @staticmethod
    def not_initialized(component: str, usage: str) -> str:
        """Error for component not initialized.

        Args:
            component: Component name (e.g., "Context Engine")
            usage: Where/how it should be used

        Returns:
            Formatted error message
        """
        return ErrorBoxTemplates.critical_failure(
            component=f"{component} NOT INITIALIZED",
            reason=f"{component} was not initialized properly.",
            solutions=[
                "Ensure configuration is correct",
                f"{usage}",
                "Check logs for initialization errors",
            ]
        )

    @staticmethod
    def missing_required(field: str, component: str) -> str:
        """Error for missing required field.

        Args:
            field: The missing field name
            component: Component requiring the field

        Returns:
            Formatted error message
        """
        return ErrorBoxTemplates.critical_failure(
            component=f"MISSING REQUIRED FIELD",
            reason=f"{field} is required for {component} but was not provided.",
            solutions=[
                f"Ensure {field} is set before calling this function",
                "Check your configuration",
                "Verify state initialization",
            ]
        )

    @staticmethod
    def service_unavailable(
        service: str,
        reason: str,
        startup_command: Optional[str] = None
    ) -> str:
        """Error for external service unavailable.

        Args:
            service: Service name (e.g., "Neo4j")
            reason: Why it's unavailable
            startup_command: Optional command to start the service

        Returns:
            Formatted error message
        """
        solutions = [
            f"Ensure {service} is running",
            "Check connection configuration",
        ]
        if startup_command:
            solutions.append(f"Start with: {startup_command}")

        return ErrorBoxTemplates.critical_failure(
            component=f"{service} UNAVAILABLE",
            reason=reason,
            solutions=solutions
        )


# =============================================================================
# CRASH-ONLY ERROR RAISERS
# =============================================================================

class CrashErrors:
    """Crash-only error raising functions.

    These functions raise exceptions instead of returning None or empty values.
    Use these to enforce crash-only architecture.
    """

    @staticmethod
    def crash_if_none(value: Any, error_message: str) -> Any:
        """Crash if value is None.

        Args:
            value: Value to check
            error_message: Error message if None

        Returns:
            The value if not None

        Raises:
            RuntimeError: If value is None
        """
        if value is None:
            raise RuntimeError(error_message)
        return value

    @staticmethod
    def crash_if_false(condition: bool, error_message: str) -> None:
        """Crash if condition is False.

        Args:
            condition: Condition to check
            error_message: Error message if False

        Raises:
            RuntimeError: If condition is False
        """
        if not condition:
            raise RuntimeError(error_message)

    @staticmethod
    def crash_if_empty(collection: Any, error_message: str) -> Any:
        """Crash if collection is empty.

        Args:
            collection: Collection to check (list, dict, str, etc.)
            error_message: Error message if empty

        Returns:
            The collection if not empty

        Raises:
            RuntimeError: If collection is empty
        """
        if not collection:
            raise RuntimeError(error_message)
        return collection


# =============================================================================
# CONTEXT ENGINE SPECIFIC ERRORS
# =============================================================================

class ContextEngineErrors:
    """Specialized errors for Context Engine integration.

    These provide consistent error messages for Context Engine failures.
    """

    NOT_ENABLED = ErrorBoxTemplates.critical_failure(
        component="CONTEXT ENGINE NOT ENABLED",
        reason="Context Engine is REQUIRED for deep research to function.",
        solutions=[
            "Enable Context Engine in your config:",
            "  context_engine:",
            "    enabled: true",
            "    postgres:",
            "      connection_string: \"postgresql://...\"",
            "    neo4j:",
            "      enabled: true",
            "      url: \"bolt://localhost:7687\"",
        ]
    )

    INITIALIZATION_FAILED = ErrorBoxTemplates.critical_failure(
        component="CONTEXT ENGINE INITIALIZATION FAILED",
        reason="Direct initialization error occurred.",
        solutions=[
            "Ensure services are running: docker ps | grep lang-tools",
            "Check database connections are available",
            "Verify context_engine config in data/config/deep_research/base/config.yaml",
        ]
    )

    SEARCH_FAILED_NOT_INITIALIZED = ErrorBoxTemplates.critical_failure(
        component="CONTEXT ENGINE SEARCH FAILED",
        reason="search_research_context() called with no context_engine.",
        solutions=[
            "Context Engine must be initialized before calling this function.",
            "Check initialization sequence.",
        ]
    )

    NO_POSTGRES_STORE = ErrorBoxTemplates.critical_failure(
        component="CONTEXT ENGINE SEARCH FAILED",
        reason="postgres_store is not available.",
        solutions=[
            "Check database connection configuration.",
            "Ensure Postgres service is running.",
        ]
    )

    @staticmethod
    def get_context_failed(function_name: str) -> str:
        """Error for get_context functions when context engine missing."""
        return ErrorBoxTemplates.critical_failure(
            component=f"{function_name} FAILED - NO CONTEXT ENGINE",
            reason="Context Engine is not initialized in context variable.",
            solutions=[
                "Context Engine must be initialized before this operation.",
                "Check initialization sequence in graph startup.",
            ]
        )

    @staticmethod
    def no_session_id(function_name: str) -> str:
        """Error for missing session_id in state."""
        return ErrorBoxTemplates.critical_failure(
            component=f"{function_name} FAILED - NO SESSION_ID",
            reason="session_id not found in state.",
            solutions=[
                "Every research session MUST have a session_id for Context Engine isolation.",
                "Check state initialization.",
                "Verify session creation in graph startup.",
            ]
        )


# =============================================================================
# VALIDATION ERROR HELPERS
# =============================================================================

class ValidationErrors:
    """Specialized errors for validation failures."""

    @staticmethod
    def missing_session_id(function_name: str) -> RuntimeError:
        """Create error for missing session_id."""
        return RuntimeError(ContextEngineErrors.no_session_id(function_name))

    @staticmethod
    def missing_context_engine(function_name: str) -> RuntimeError:
        """Create error for missing context engine."""
        return RuntimeError(ContextEngineErrors.get_context_failed(function_name))

    @staticmethod
    def missing_postgres_store() -> RuntimeError:
        """Create error for missing postgres store."""
        return RuntimeError(ContextEngineErrors.NO_POSTGRES_STORE)
