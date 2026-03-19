"""Web Researcher Worker - Autonomous web research using Flash Explorer pattern.

This worker wraps the Flash Explorer agent for autonomous web research.
It spawns deep agents (web_deep_agent, pdf_deep_agent, community_deep_agent)
to research from multiple angles in parallel.

Key Features:
- Spawns web_deep_agent for parallel research angles
- Supports RAG with local documents (pdf_deep_agent)
- Detects sentiment queries and spawns community_deep_agent
- Parallel execution of all research angles
- Structured findings with citations

Usage:
    worker = WebResearcherWorker(config)
    result = await worker.execute(state, config)
"""

from datetime import datetime
from typing import Any

from langgraph.types import Command

from ...state import EngineState, WorkerResult
from ..base import BaseWorker, WorkerConfig, WorkerRegistry


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
    required_tools = ["web_search", "web_get", "web_research_agent"]

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
                source_count=len(sources),
            )

        except Exception as e:
            self._log(f"Web research failed: {e}", level="error")
            return self._create_result(
                success=False,
                output=None,
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _run_flash_explorer(
        self,
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

        # Run web deep agent
        web_findings = await self._run_web_agent(
            query=query,
            yaml_config=yaml_config,
            session_id=session_id,
        )

        # Run PDF deep agent if documents provided
        pdf_findings = ""
        if has_pdf and documents:
            pdf_findings = await self._run_pdf_agent(
                query=query,
                documents=documents,
                session_id=session_id,
            )

        # Run community deep agent if sentiment query
        community_findings = ""
        if has_community:
            community_findings = await self._run_community_agent(
                query=query,
                session_id=session_id,
            )

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
        query: str,
        yaml_config: dict[str, Any],
        session_id: str,
    ) -> str:
        """Run web deep agent for research.

        Uses the web_research_agent tool which runs the full graph:
        plan → search → select → scrape → extract → critic → write_results

        Args:
            query: Research query
            yaml_config: LLM configuration
            session_id: Session ID

        Returns:
            Web research findings
        """
        try:
            from src.shared.tools.ulli_langchain.web.web_agent import web_research_agent

            # Get tool model from config
            tool_model = self._get_tool_model(yaml_config)

            # Run web research agent
            result = await web_research_agent.ainvoke(
                {
                    "query": query,
                    "num_results": 10,
                    "max_urls": 5,
                    "max_steps": self.max_steps,
                    "max_refinements": 1,
                    "tool_model": tool_model,
                }
            )

            # Extract content from result
            if isinstance(result, Command) and result.update:
                messages = result.update.get("messages", [])
                if messages:
                    return str(messages[0].content)
                return "Web research completed."
            else:
                return str(result)

        except Exception as e:
            self._log(f"Web deep agent failed: {e}", level="warning")
            return f"Web research failed: {e}"

    async def _run_pdf_agent(
        self,
        query: str,
        documents: list[str],
        session_id: str,
    ) -> str:
        """Run PDF deep agent for document RAG.

        Args:
            query: Research query
            documents: List of document paths
            session_id: Session ID

        Returns:
            Document research findings
        """
        try:
            from src.shared.tools.ulli_langchain.memory.pdf_agent import pdf_deep_agent

            result = await pdf_deep_agent.ainvoke(
                {
                    "query": query,
                    "documents": documents,
                    "top_k": 10,
                    "max_attempts": 3,
                }
            )

            return str(result) if result else ""

        except Exception as e:
            self._log(f"PDF deep agent failed: {e}", level="warning")
            return ""

    async def _run_community_agent(
        self,
        query: str,
        session_id: str,
    ) -> str:
        """Run community deep agent for sentiment research.

        Args:
            query: Research query
            session_id: Session ID

        Returns:
            Community sentiment findings
        """
        try:
            from src.shared.tools.ulli_langchain.web.community_deep_agent import (
                simple_community_research,
            )

            findings, _ = await simple_community_research(
                query=query,
                platform="auto",
                limit=20,
            )

            return findings if findings else ""

        except Exception as e:
            self._log(f"Community deep agent failed: {e}", level="warning")
            return ""

    def _get_tool_model(self, yaml_config: dict[str, Any]) -> str | None:
        """Get tool model from YAML config.

        Args:
            yaml_config: YAML configuration dict

        Returns:
            Model name string or None
        """
        try:
            from src.extensions.content.deep_research.graph.helpers.llm import (
                _resolve_model_slot,
            )

            tool_model_info = _resolve_model_slot(yaml_config, role="worker")
            return tool_model_info.get("model")  # type: ignore[union-attr]
        except Exception:
            return None

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
