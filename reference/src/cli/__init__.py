"""Unified LLM Interface CLI - Plugin Architecture

A CLI system where extensions define their own commands via cli.py plugins.

Usage:
    uv run backend/manage.py <extension> <command> [args...]

Example:
    uv run backend/manage.py groups list
    uv run backend/manage.py instruct list
    uv run backend/manage.py tools list
"""

__version__ = "2.0.0"
