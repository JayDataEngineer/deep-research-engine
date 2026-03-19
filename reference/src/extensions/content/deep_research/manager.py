"""Deep Research Manager

Manages deep research functionality.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from .constants import DATA_DIR, ROOT_DATA_DIR


class DeepResearchManager:
    """Manages deep research functionality."""

    def __init__(self):
        """Initialize the deep research manager."""
        self.research_dir = ROOT_DATA_DIR / "research"  # Default location
        self.research_dir.mkdir(exist_ok=True)

    @staticmethod
    def _sanitize_filename(topic: str, max_length: int = 60) -> str:
        """Sanitize topic for use in filename.

        Args:
            topic: The research topic string
            max_length: Maximum length for filename (default 60)

        Returns:
            Sanitized filename-safe string
        """
        # Remove characters that are unsafe for filenames
        # Keep: letters, numbers, spaces, hyphens, underscores
        sanitized = re.sub(r'[^\w\s-]', '', topic)
        # Replace multiple spaces/hyphens with single hyphen
        sanitized = re.sub(r'[\s-]+', '-', sanitized)
        # Remove leading/trailing hyphens
        sanitized = sanitized.strip('-')
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip('-')
        return sanitized or "research"

    def perform_research(self, topic: str, use_case: str = "base", variant: str = "default", overdrive: bool = False, pdf_file: str = None, pdf_only: bool = False, checkpointer=None, store=None) -> str:
        """
        Perform deep research on a topic.

        Args:
            topic: Topic to research
            use_case: Use case name (base, rag, podcast, news, etc.)
            variant: Config variant
            overdrive: Enable overdrive mode (max quality, max context, deeper research)
            pdf_file: Optional path to PDF file for RAG
            pdf_only: If True, skip web research and use only PDF content
            checkpointer: Optional checkpointer for persistence (default: auto-create SQLiteSaver)
            store: REQUIRED - BaseStore for InformationFunnel and cross-thread persistence

        Returns:
            Research results
        """
        from src.extensions.content.deep_research.graph import run_v4_spawner
        from src.extensions.content.deep_research.observability import RunLogger
        from src.extensions.content.deep_research.utils import ensure_mcp_accessible

        # Capture checkpointer and store for use in nested function
        _checkpointer = checkpointer
        _store = store

        async def _run_research():
            nonlocal _checkpointer, _store

            # Ensure lang-tools (Docker services) are running
            from src.extensions.content.deep_research.graph.helpers.config import (
                get_run_logger_from_config,
                _load_yaml_config
            )
            from .logging_utils import get_logger

            docker_logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level

            docker_result = await ensure_mcp_accessible()
            if docker_result["success"]:
                if docker_result["started"]:
                    docker_logger.info("[DOCKER_HELPER] lang-tools started successfully")
                elif docker_result["already_running"]:
                    docker_logger.info("[DOCKER_HELPER] lang-tools already running")
            else:
                docker_logger.warning(f"[DOCKER_HELPER] lang-tools issue: {docker_result.get('error', 'Unknown error')}")
                docker_logger.warning("Research will proceed but web_search may not work properly.")

            # Load config to detect actual use_case from variant
            # Some variants (like substack_single) map to different use_cases (like longform_gen)
            final_use_case = use_case  # Default to original
            try:
                config = _load_yaml_config(use_case, variant)
                # Check if variant has a use_case override
                # The detected use_case is stored at "_use_case" key by _load_yaml_config
                detected_use_case = config.get("_use_case", use_case)
                if detected_use_case != use_case:
                    docker_logger.info(f"[CONFIG] Auto-detected use_case '{detected_use_case}' from variant '{variant}'")
                    final_use_case = detected_use_case
            except Exception as e:
                docker_logger.warning(f"[CONFIG] Could not detect use_case from variant '{variant}': {e}")

            # Create RunLogger to capture output
            # Sanitize topic for filename (max 60 chars, remove unsafe chars)
            sanitized_topic = self._sanitize_filename(topic)

            # Build config dict for RunLogger
            run_logger_config = {
                "use_case_config": {
                    "use_case": final_use_case,
                    "variant": variant,
                },
                "overdrive": overdrive,
            }
            run_logger = RunLogger(
                run_name=f"research_{sanitized_topic}",
                config=run_logger_config,
                base_dir=Path("data/outputs/logs/deep_research"),
            )

            # Create checkpointer using factory (unless provided)
            if _checkpointer is None:
                from src.shared.checkpointer.factory import get_checkpointer

                _checkpointer = await get_checkpointer()
                checkpointer_type = type(_checkpointer).__name__
                docker_logger.info(f"[CHECKPOINTER] Using {checkpointer_type} from factory")

            # Create Store using LangGraph PostgresStore with pgvector (auto-vectorization)
            from src.extensions.features.tools.context_engine.storage.langgraph_store import get_langgraph_store

            store = await get_langgraph_store()
            store_type = type(store).__name__
            docker_logger.info(f"[STORE] Using {store_type} with pgvector auto-vectorization")

            # Build config overrides for PDF support
            config_overrides = {}
            if pdf_file:
                config_overrides["pdf_file"] = str(pdf_file)
            if pdf_only:
                config_overrides["pdf_only"] = True

            result = await run_v4_spawner(
                query=topic,
                use_case=final_use_case,
                variant=variant,
                run_logger=run_logger,
                overdrive=overdrive,
                config_overrides=config_overrides if config_overrides else None,
                checkpointer=_checkpointer,
                store=store  # Pass Store - REQUIRED for InformationFunnel
            )

            return result.get("final_response", "No response generated")

        # Run async function
        try:
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, create a task
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(_run_research(), loop)
                result = future.result(timeout=600)  # 10 minute timeout
                return result
            except RuntimeError:
                # No running loop, create a new one
                result = asyncio.run(_run_research())
                return result
        except Exception as e:
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            return f"Error performing research: {e}"

    def refine_research(self, topic: str, refinement_query: str) -> str:
        """
        Refine existing research with a query.

        Args:
            topic: Original research topic
            refinement_query: Query to refine the research

        Returns:
            Refined research results
        """
        # For now, just run a new research with the combined query
        combined_query = f"{topic}. Refinement: {refinement_query}"
        return self.perform_research(combined_query)