"""
Natural Language to Cypher Query Chain.

This module implements natural language querying for Neo4j knowledge graphs:
- Converts natural language questions to Cypher queries
- Executes queries against Neo4j
- Returns natural language answers with citations

Uses LLM with structured output for query generation (no external GraphCypherQAChain dependency).
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Cypher Generation
# ============================================================================

class GeneratedCypher(BaseModel):
    """Generated Cypher query with explanation."""
    query: str = Field(description="Cypher query to execute")
    explanation: str = Field(description="Explanation of what the query does")


class GraphQueryResult(BaseModel):
    """Result of a natural language graph query."""
    question: str = Field(description="Original question")
    cypher: str = Field(description="Generated Cypher query")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Query results")
    answer: str = Field(description="Natural language answer")
    error: Optional[str] = Field(default=None, description="Error message if query failed")


# ============================================================================
# Natural Language Query Chain
# ============================================================================

class NaturalLanguageGraphQuery:
    """
    Natural language to Cypher query chain with enhanced schema.

    Usage:
        query_chain = NaturalLanguageGraphQuery(
            llm=llm,
            neo4j_store=neo4j_store,
            enhanced_schema=enhanced_schema_dict
        )

        result = await query_chain.query("Who works for SpaceX?")
        # Returns: GraphQueryResult with answer, cypher, and results
    """

    def __init__(
        self,
        llm,
        neo4j_store,
        enhanced_schema: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ):
        """Initialize natural language query chain.

        Args:
            llm: LangChain LLM instance for query generation
            neo4j_store: Neo4jStore instance
            enhanced_schema: Pre-computed enhanced schema with example values
            verbose: Enable verbose logging
        """
        self.llm = llm
        self.neo4j_store = neo4j_store
        self.enhanced_schema = enhanced_schema
        self.verbose = verbose

    def _build_schema_prompt(self) -> str:
        """Build schema description for the LLM."""
        if self.enhanced_schema:
            return self._format_enhanced_schema(self.enhanced_schema)
        else:
            # Fall back to basic schema query
            return "Use the available node and relationship types in the graph."

    def _format_enhanced_schema(self, schema: Dict[str, Any]) -> str:
        """Format enhanced schema with example values."""
        lines = []
        lines.append("Graph Schema:")
        lines.append("")

        # Node types
        node_types = schema.get("node_types", [])
        if node_types:
            lines.append("Node Types:")
            for nt in node_types:
                label = nt.get("label", "Unknown")
                count = nt.get("count", 0)
                lines.append(f"  - {label} ({count} nodes)")

                properties = nt.get("properties", [])
                if properties:
                    lines.append(f"    Properties:")
                    for prop in properties:
                        name = prop.get("name", "")
                        prop_type = prop.get("type", "STRING")
                        examples = prop.get("example_values", [])
                        if examples:
                            example_str = ", ".join(str(e) for e in examples[:3])
                            lines.append(f"      - {name}: {prop_type} (examples: {example_str})")
                        else:
                            lines.append(f"      - {name}: {prop_type}")

        # Relationship types
        rel_types = schema.get("relationship_types", [])
        if rel_types:
            lines.append("")
            lines.append("Relationship Types:")
            for rt in rel_types:
                label = rt.get("label", "Unknown")
                count = rt.get("count", 0)
                lines.append(f"  - :{label} ({count} relationships)")

                properties = rt.get("properties", [])
                if properties:
                    lines.append(f"    Properties:")
                    for prop in properties:
                        name = prop.get("name", "")
                        prop_type = prop.get("type", "STRING")
                        examples = prop.get("example_values", [])
                        if examples:
                            example_str = ", ".join(str(e) for e in examples[:3])
                            lines.append(f"      - {name}: {prop_type} (examples: {example_str})")
                        else:
                            lines.append(f"      - {name}: {prop_type}")

        return "\n".join(lines)

    async def _generate_cypher(self, question: str) -> GeneratedCypher:
        """Generate Cypher query from natural language question."""
        schema_prompt = self._build_schema_prompt()

        system_prompt = f"""You are a Neo4j Cypher query expert. Convert natural language questions into Cypher queries.

{schema_prompt}

Guidelines:
- Use MATCH to find nodes and relationships
- Use RETURN to specify what to return
- Keep queries simple and efficient
- Use LIMIT to limit results when appropriate
- Assume property names exist based on the schema above

Return ONLY the Cypher query and a brief explanation."""

        try:
            # Try structured output first
            structured_llm = self.llm.with_structured_output(GeneratedCypher)
            result = await structured_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Question: {question}")
            ])
            return result
        except Exception as e:
            logger.warning(f"[GRAPH_QA] Structured output failed: {e}, using fallback")
            # Fallback to text generation
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Question: {question}\n\nProvide your answer as:")
            ])
            import re
            content = str(response.content)
            # Extract Cypher from code block
            cypher_match = re.search(r'```(?:cypher)?\s*(.*?)```', content, re.DOTALL)
            if cypher_match:
                query = cypher_match.group(1).strip()
            else:
                # Try to find a line that looks like Cypher
                lines = content.split('\n')
                query = ""
                for line in lines:
                    line = line.strip()
                    if line.upper().startswith(('MATCH', 'CREATE', 'MERGE', 'RETURN')):
                        query = line
                        break

            return GeneratedCypher(
                query=query,
                explanation="Generated via fallback method"
            )

    async def _execute_cypher(self, cypher: str) -> List[Dict[str, Any]]:
        """Execute Cypher query against Neo4j."""
        try:
            driver = self.neo4j_store.driver
            async with driver.session(database=self.neo4j_store.database) as session:
                result = await session.run(cypher)
                records = await result.data()
            return records
        except Exception as e:
            logger.error(f"[GRAPH_QA] Cypher execution failed: {e}")
            raise

    async def _generate_answer(
        self,
        question: str,
        cypher: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Generate natural language answer from query results."""
        if not results:
            return "No results found for this query."

        # Format results for LLM
        results_str = "\n".join(str(r) for r in results[:10])  # Limit to 10 results

        prompt = f"""Based on the following query results, provide a natural language answer to the question.

Question: {question}

Cypher Query: {cypher}

Results:
{results_str}

Provide a concise, direct answer."""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return str(response.content).strip()
        except Exception as e:
            logger.warning(f"[GRAPH_QA] Answer generation failed: {e}")
            # Fallback to simple answer
            if len(results) == 1:
                return f"Found: {results[0]}"
            return f"Found {len(results)} results."

    async def query(self, question: str) -> GraphQueryResult:
        """
        Execute natural language query against the graph.

        Args:
            question: Natural language question

        Returns:
            GraphQueryResult with answer, cypher, and context
        """
        try:
            if self.verbose:
                logger.info(f"[GRAPH_QA] Question: {question}")

            # Step 1: Generate Cypher query
            generated = await self._generate_cypher(question)

            if self.verbose:
                logger.info(f"[GRAPH_QA] Generated Cypher: {generated.query}")
                logger.info(f"[GRAPH_QA] Explanation: {generated.explanation}")

            # Step 2: Execute query
            results = await self._execute_cypher(generated.query)

            if self.verbose:
                logger.info(f"[GRAPH_QA] Results: {len(results)} records")

            # Step 3: Generate natural language answer
            answer = await self._generate_answer(question, generated.query, results)

            return GraphQueryResult(
                question=question,
                cypher=generated.query,
                results=results,
                answer=answer,
                error=None
            )

        except Exception as e:
            logger.error(f"[GRAPH_QA] Query failed: {e}", exc_info=True)
            return GraphQueryResult(
                question=question,
                cypher="",
                results=[],
                answer=f"Query failed: {e}",
                error=str(e)
            )

    async def batch_query(self, questions: List[str]) -> List[GraphQueryResult]:
        """
        Execute multiple natural language queries.

        Args:
            questions: List of natural language questions

        Returns:
            List of GraphQueryResult
        """
        results = []
        for question in questions:
            result = await self.query(question)
            results.append(result)
        return results


# ============================================================================
# Convenience Functions
# ============================================================================

async def query_graph_natural_language(
    question: str,
    llm,
    neo4j_store,
    enhanced_schema: Optional[Dict[str, Any]] = None
) -> GraphQueryResult:
    """
    Convenience function for natural language graph query.

    Args:
        question: Natural language question
        llm: LangChain LLM instance
        neo4j_store: Neo4jStore instance
        enhanced_schema: Optional enhanced schema

    Returns:
        GraphQueryResult with answer, cypher, and context
    """
    query_chain = NaturalLanguageGraphQuery(
        llm=llm,
        neo4j_store=neo4j_store,
        enhanced_schema=enhanced_schema
    )

    return await query_chain.query(question)


def format_enhanced_schema_for_llm(schema: Dict[str, Any]) -> str:
    """
    Format enhanced schema for LLM consumption.

    Args:
        schema: Enhanced schema dict from ResearchGraphTransformer

    Returns:
        Formatted schema string
    """
    lines = ["Graph Schema:", ""]
    node_types = schema.get("node_types", [])
    for nt in node_types:
        label = nt.get("label", "Unknown")
        lines.append(f"Node: {label}")
        for prop in nt.get("properties", []):
            examples = prop.get("example_values", [])
            example_str = f" (e.g., {examples[0]})" if examples else ""
            lines.append(f"  - {prop['name']}: {prop['type']}{example_str}")
    return "\n".join(lines)


# ============================================================================
# Schema Conversion
# ============================================================================

def schema_to_dict(graph_schema) -> Dict[str, Any]:
    """
    Convert DynamicGraphSchema to dict for serialization.

    Args:
        graph_schema: DynamicGraphSchema from ResearchGraphTransformer

    Returns:
        Dict representation
    """
    return {
        "node_types": [
            {
                "label": nt.label,
                "properties": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "example_values": p.example_values
                    }
                    for p in nt.properties
                ],
                "count": nt.count,
                "description": nt.description
            }
            for nt in graph_schema.node_types
        ],
        "relationship_types": [
            {
                "label": rt.label,
                "properties": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "example_values": p.example_values
                    }
                    for p in rt.properties
                ],
                "count": rt.count,
                "description": rt.description
            }
            for rt in graph_schema.relationship_types
        ],
        "patterns": [
            {
                "source": p.source,
                "relationship": p.relationship,
                "target": p.target
            }
            for p in graph_schema.patterns
        ]
    }
