# ruff: noqa: D101
"""Natural Language to Cypher Query Chain.

Converts natural language questions to Cypher queries, executes against Neo4j,
and returns natural language answers.

Uses LLM with structured output for query generation.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ...utils.logging import get_loguru_logger

logger = get_loguru_logger("shared.neo4j.qa_chain")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GeneratedCypher(BaseModel):
    query: str = Field(description="Cypher query to execute")
    explanation: str = Field(description="Explanation of what the query does")


class GraphQueryResult(BaseModel):
    question: str
    cypher: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    answer: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


class NaturalLanguageGraphQuery:
    """NL -> Cypher -> Answer chain with enhanced schema.

    Usage::

        chain = NaturalLanguageGraphQuery(llm=llm, neo4j_driver=driver, database="neo4j")
        result = await chain.query("Who works for SpaceX?")
    """

    def __init__(
        self,
        llm: Any,
        neo4j_driver: Any,
        database: str = "neo4j",
        enhanced_schema: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.driver = neo4j_driver
        self.database = database
        self.enhanced_schema = enhanced_schema

    def _build_schema_prompt(self) -> str:
        if not self.enhanced_schema:
            return "Use the available node and relationship types in the graph."

        lines = ["Graph Schema:", ""]
        for nt in self.enhanced_schema.get("node_types", []):
            label = nt.get("label", "Unknown")
            count = nt.get("count", 0)
            lines.append(f"Node: {label} ({count} nodes)")
            for prop in nt.get("properties", []):
                examples = prop.get("example_values", [])
                ex = f" (e.g., {examples[0]})" if examples else ""
                lines.append(f"  - {prop['name']}: {prop['type']}{ex}")

        for rt in self.enhanced_schema.get("relationship_types", []):
            label = rt.get("label", "Unknown")
            count = rt.get("count", 0)
            lines.append(f"Rel: :{label} ({count} relationships)")

        return "\n".join(lines)

    async def _generate_cypher(self, question: str) -> GeneratedCypher:
        schema_prompt = self._build_schema_prompt()

        system = (
            "You are a Neo4j Cypher query expert. "
            "Convert natural language questions into Cypher queries.\n\n"
            f"{schema_prompt}\n\n"
            "Guidelines:\n"
            "- Use MATCH to find nodes and relationships\n"
            "- Use RETURN to specify what to return\n"
            "- Keep queries simple and efficient\n"
            "- Use LIMIT when appropriate\n"
        )

        try:
            structured_llm = self.llm.with_structured_output(GeneratedCypher)
            return await structured_llm.ainvoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=f"Question: {question}"),
                ]
            )
        except Exception as e:
            logger.warning(f"Structured output failed: {e}, using fallback")
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=f"Question: {question}"),
                ]
            )
            content = str(response.content)
            match = re.search(r"```(?:cypher)?\s*(.*?)```", content, re.DOTALL)
            query = match.group(1).strip() if match else ""
            if not query:
                for line in content.split("\n"):
                    line = line.strip()
                    if line.upper().startswith(("MATCH", "CREATE", "MERGE", "RETURN")):
                        query = line
                        break
            return GeneratedCypher(query=query, explanation="Fallback generation")

    async def _execute_cypher(self, cypher: str) -> list[dict[str, Any]]:
        async with self.driver.session(database=self.database) as session:
            result = await session.run(cypher)
            return await result.data()

    async def _generate_answer(
        self, question: str, cypher: str, results: list[dict[str, Any]]
    ) -> str:
        if not results:
            return "No results found for this query."

        results_str = "\n".join(str(r) for r in results[:10])
        prompt = (
            f"Based on the following query results, answer the question.\n\n"
            f"Question: {question}\n\n"
            f"Cypher: {cypher}\n\n"
            f"Results:\n{results_str}\n\n"
            "Provide a concise, direct answer."
        )
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return str(response.content).strip()
        except Exception:
            return f"Found {len(results)} results."

    async def query(self, question: str) -> GraphQueryResult:
        """Execute natural language query against the graph."""
        try:
            generated = await self._generate_cypher(question)
            results = await self._execute_cypher(generated.query)
            answer = await self._generate_answer(question, generated.query, results)
            return GraphQueryResult(
                question=question,
                cypher=generated.query,
                results=results,
                answer=answer,
            )
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return GraphQueryResult(
                question=question, cypher="", answer=f"Query failed: {e}", error=str(e)
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_enhanced_schema_for_llm(schema: dict[str, Any]) -> str:
    """Format enhanced schema dict for LLM consumption."""
    lines = ["Graph Schema:", ""]
    for nt in schema.get("node_types", []):
        label = nt.get("label", "Unknown")
        lines.append(f"Node: {label}")
        for prop in nt.get("properties", []):
            examples = prop.get("example_values", [])
            ex = f" (e.g., {examples[0]})" if examples else ""
            lines.append(f"  - {prop['name']}: {prop['type']}{ex}")
    return "\n".join(lines)
