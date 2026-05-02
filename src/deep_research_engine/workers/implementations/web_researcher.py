"""Web Researcher Worker - Autonomous web research using orchestrator SDK.

Routes web research through the orchestrator's /api/tools/exec endpoint,
giving access to MCP tools (search, scrape, research, crawl, extract)
without direct MCP client management.

The orchestrator handles tool routing, timeout management, and provides
access to 37+ tools through a single SDK client.

Usage:
    worker = WebResearcherWorker(config)
    result = await worker.execute(state, config)
"""

import asyncio
from datetime import datetime
from typing import Any

from langgraph.types import Command

from ...shared.orch_client import (
    is_available as orch_is_available,
    orch_deep_research,
    orch_extract,
    orch_research,
    orch_scrape,
    orch_search,
)
from ...state import EngineState, WorkerResult
from ...utils.logging import get_loguru_logger
from ..base import BaseWorker, WorkerConfig, WorkerRegistry

logger = get_loguru_logger("worker.web_researcher")


@WorkerRegistry.register("web_researcher")
class WebResearcherWorker(BaseWorker):
    """Autonomous web research agent using Flash Explorer pattern.

    This worker:
    1. Receives a research avenue (broad topic)
    2. Plans multiple research angles
    3. Spawns deep agents for each angle in parallel
    4. Collects and synthesizes findings
    5. Returns structured research results

    Deep Agents Used:
    - web_deep_agent: Web research (search, scrape, extract)
    - pdf_deep_agent: Document RAG (vector search in PDFs)
    - community_deep_agent: Community sentiment (Reddit, HN, etc.)
    """

    worker_type = "web_researcher"
    description = "Autonomous web research agent with parallel deep agents"
    required_tools = ["orch_search", "orch_scrape", "orch_research"]

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.max_angles = config.custom.get("max_angles", 3)
        self.max_steps = config.custom.get("max_steps", 3)
        self.enable_rag = config.custom.get("enable_rag", False)
        self.documents = config.custom.get("documents", [])
        self.enable_community = config.custom.get("enable_community", True)

    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Execute web research using Flash Explorer pattern.

        Args:
            state: Current engine state with config and context
            config: Worker configuration

        Returns:
            WorkerResult with research findings
        """
        self._start_timing()

        try:
            # Get research query from task or state
            query = self.task or state.get("input_data", {}).get("query", "")
            if not query:
                return self._create_result(
                    success=False,
                    output=None,
                    error="No research query provided",
                )

            # Get YAML config for LLM/tools
            yaml_config = state.get("context", {}).get("config", {})

            # Get session_id for Context Engine
            session_id = state.get("session_id", f"session_{datetime.now().isoformat()}")

            # Run Flash Explorer
            findings = await self._run_flash_explorer(
                state=state,
                query=query,
                yaml_config=yaml_config,
                session_id=session_id,
                documents=self.documents if self.enable_rag else [],
            )

            # Extract sources from findings
            sources = self._extract_sources(findings)

            self._log(f"Web research complete: {len(sources)} sources found")

            return self._create_result(
                success=True,
                output=findings,
                output_type="research_findings",
                confidence=0.85,
                sources=sources,
            )

        except Exception as e:
            self._log(f"Web research failed: {e}", level="error")
            return self._create_result(
                success=False,
                output=None,
                error=str(e),
            )

    async def _run_flash_explorer(
        self,
        state: EngineState,
        query: str,
        yaml_config: dict[str, Any],
        session_id: str,
        documents: list[str] | None = None,
    ) -> str:
        """Run Flash Explorer for autonomous research.

        This is a simplified wrapper that:
        1. Creates a research plan
        2. Spawns deep agents for each angle in parallel
        3. Collects and combines findings

        Args:
            state: Engine state
            query: Research query/avenue
            yaml_config: LLM and tool configuration
            session_id: Session ID for tracing
            documents: Optional list of document paths for RAG

        Returns:
            Combined research findings
        """
        # Determine which deep agents to spawn
        has_pdf = len(documents or []) > 0
        has_community = self._needs_community_sentiment(query)

        self._log(
            f"Running Flash Explorer with: web_deep_agent"
            f"{', pdf_deep_agent' if has_pdf else ''}"
            f"{', community_deep_agent' if has_community else ''}"
        )

        # Prepare tasks for parallel execution
        tasks = []

        # 1. Web deep agent task
        tasks.append(
            self._run_web_agent(
                state=state,
                query=query,
                yaml_config=yaml_config,
                session_id=session_id,
            )
        )

        # 2. PDF deep agent task if documents provided
        if has_pdf and documents:
            tasks.append(
                self._run_pdf_agent(
                    state=state,
                    query=query,
                    documents=documents,
                    session_id=session_id,
                )
            )
        else:
            tasks.append(asyncio.sleep(0, result=""))

        # 3. Community deep agent task if sentiment query
        if has_community:
            tasks.append(
                self._run_community_agent(
                    state=state,
                    query=query,
                    session_id=session_id,
                )
            )
        else:
            tasks.append(asyncio.sleep(0, result=""))

        # Run all deep agents in parallel
        results = await asyncio.gather(*tasks)
        web_findings, pdf_findings, community_findings = results

        # Combine findings
        combined = []
        if web_findings:
            combined.append(f"## Web Research\n\n{web_findings}")
        if pdf_findings:
            combined.append(f"## Document Research\n\n{pdf_findings}")
        if community_findings:
            combined.append(f"## Community Sentiment\n\n{community_findings}")

        return "\n\n".join(combined) if combined else "No findings returned."

    async def _run_web_agent(
        self,
        state: EngineState,
        query: str,
        yaml_config: dict[str, Any],
        session_id: str,
    ) -> str:
        """Run web research via orchestrator SDK.

        Uses the orchestrator's MCP tools (research, search, scrape)
        to gather web findings. Falls back gracefully if orchestrator
        is unavailable.

        Args:
            state: Engine state
            query: Research query
            yaml_config: LLM configuration
            session_id: Session ID

        Returns:
            Web research findings
        """
        if not orch_is_available():
            self._log("Orchestrator not available, skipping web research", level="warning")
            return "Web research skipped: orchestrator not reachable."

        try:
            # Use deep research (search + scrape in one call) for maximum coverage
            findings = await orch_deep_research(query, max_results=self.max_steps)
            self._log(f"Orchestrator web research returned {len(findings)} chars")
            return findings
        except Exception as e:
            self._log(f"Web research via orchestrator failed: {e}", level="warning")
            # Fallback to basic search
            try:
                return await orch_search(query, max_results=5)
            except Exception as e2:
                return f"Web research failed: {e2}"

    async def _run_pdf_agent(
        self,
        state: EngineState,
        query: str,
        documents: list[str],
        session_id: str,
    ) -> str:
        """Run document research via orchestrator sandbox.

        Reads documents from sandbox filesystem and extracts key information.

        Args:
            query: Research query
            documents: List of document paths
            session_id: Session ID

        Returns:
            Document research findings
        """
        if not documents:
            return ""

        if not orch_is_available():
            self._log("Orchestrator not available, skipping PDF research", level="warning")
            return ""

        try:
            # Use sandbox file_read to access documents
            from ...shared.orch_client import get_orch

            orch = get_orch()
            sandboxes = await asyncio.to_thread(orch.list_sandboxes)
            if not sandboxes:
                self._log("No sandboxes available for PDF research", level="warning")
                return ""

            sandbox_id = sandboxes[0]["id"]
            doc_contents = []
            for doc_path in documents[:5]:  # Limit to 5 documents
                result = await asyncio.to_thread(
                    orch.file_read, sandbox_id, doc_path
                )
                if result.success:
                    doc_contents.append(f"### {doc_path}\n\n{result.text[:5000]}")

            if doc_contents:
                return "\n\n".join(doc_contents)
            return ""

        except Exception as e:
            self._log(f"PDF research via orchestrator failed: {e}", level="warning")
            return ""

    async def _run_community_agent(
        self,
        state: EngineState,
        query: str,
        session_id: str,
    ) -> str:
        """Run community sentiment research via orchestrator MCP.

        Uses the orchestrator's research tool with community-focused
        query modifications to find forum/reddit/social discussions.

        Args:
            query: Research query
            session_id: Session ID

        Returns:
            Community sentiment findings
        """
        if not orch_is_available():
            self._log("Orchestrator not available, skipping community research", level="warning")
            return ""

        try:
            # Search for community discussions with targeted query
            community_query = f"{query} site:reddit.com OR site:news.ycombinator.com OR opinion OR discussion"
            findings = await orch_research(community_query, max_results=3)
            self._log(f"Community research returned {len(findings)} chars")
            return findings
        except Exception as e:
            self._log(f"Community research via orchestrator failed: {e}", level="warning")
            return ""


    def _needs_community_sentiment(self, query: str) -> bool:
        """Detect if query needs community sentiment research.

        Args:
            query: Research query

        Returns:
            True if community_deep_agent should be spawned
        """
        query_lower = query.lower()

        sentiment_keywords = [
            "what do people think",
            "what does reddit think",
            "community sentiment",
            "public opinion",
            "user sentiment",
            "opinion on",
            "thoughts on",
            "reaction",
            "reactions",
        ]

        return any(keyword in query_lower for keyword in sentiment_keywords)

    def _extract_sources(self, findings: str) -> list[dict[str, Any]]:
        """Extract sources from research findings.

        Args:
            findings: Research findings text

        Returns:
            List of source dicts
        """
        import re

        # Extract URLs from findings
        url_pattern = r"https?://[^\s<>\"]+"
        urls = re.findall(url_pattern, findings)

        return [
            {
                "url": url,
                "title": f"Source from {url.split('/')[2] if '/' in url else url}",
                "type": "web",
            }
            for url in urls
        ]
