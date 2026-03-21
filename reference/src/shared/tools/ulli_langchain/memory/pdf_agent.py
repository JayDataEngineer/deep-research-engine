"""PDF Deep Agent Tool - Agentic RAG for PDF analysis.

Provides three modes of PDF analysis with increasing sophistication:
- basic: Fast vector search (~2s)
- enhanced: Graded retrieval (~4-5s)
- agentic: Full CRAG with self-correction (~8-12s, default)

The agentic mode implements Corrective Retrieval Augmented Generation (CRAG):
1. Query expansion (3 variations)
2. Adaptive retrieval with parallel search
3. Three-way confidence grading (correct/incorrect/ambiguous)
4. Knowledge refinement (decompose-then-recompose)
5. Query transformation on low confidence
6. Dual generation grading (hallucination + answer)

Reference: https://arxiv.org/abs/2401.15884
"""

import asyncio
from .logging_config import get_logger
from typing import List, Dict, Any, Optional, Literal, TypedDict
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator, model_validator

logger = get_logger(__name__)

# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================

# Import LangGraph for agentic mode
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.debug("LangGraph not available - agentic mode will be disabled")

# Import vector store
try:
    from .rag import get_rag_client, _client_config
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.debug("RAG not available")

# Import LLM
try:
    from src.shared import get_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.debug("LLM not available")


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class PDFAgentState(TypedDict):
    """State for PDF agent workflow."""
    query: str
    original_query: str
    documents: List[Document]
    refined_docs: Optional[List[Document]]
    confidence: Optional[str]  # "correct" | "incorrect" | "ambiguous"
    attempts: int
    answer: Optional[str]
    max_attempts: int
    tool_model_provider: Optional[str]  # Provider for LLM (bound by research framework)


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class DocumentGrade(BaseModel):
    """Grade document relevance with three-way confidence."""
    confidence: Literal["correct", "incorrect", "ambiguous"] = Field(
        description="Retrieval quality: 'correct' (high relevance), 'incorrect' (no relevance), 'ambiguous' (mixed)"
    )
    reasoning: str = Field(description="Brief explanation of the grading decision")


class GenerationGrade(BaseModel):
    """Grade generation quality."""
    is_grounded: bool = Field(description="Is the answer grounded in the retrieved documents?")
    answers_question: bool = Field(description="Does the answer address the user's question?")
    reasoning: str = Field(description="Brief explanation of the grading decision")


class PDFDeepAgentInput(BaseModel):
    """Pydantic schema for PDF Deep Agent tool input validation.

    This provides automatic retry if the LLM generates invalid parameters.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question or analysis task for the PDF"
    )
    mode: Literal["basic", "enhanced", "agentic"] = Field(
        default="agentic",
        description="Analysis mode - 'basic' (fast), 'enhanced' (graded), or 'agentic' (full CRAG)"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of documents to retrieve (1-50)"
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum query transformation retries in agentic mode (1-10)"
    )

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty or whitespace only."""
        if not v or v.strip() == "":
            raise ValueError("query cannot be empty or whitespace only")
        return v.strip()

    @model_validator(mode="after")
    def validate_mode_consistency(self) -> "PDFDeepAgentInput":
        """Cross-field validation: max_attempts only relevant for agentic mode."""
        if self.mode == "basic" and self.max_attempts != 3:
            # Warn but don't fail - max_attempts is ignored in basic mode
            pass
        return self


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

QUERY_EXPANSION_PROMPT = """You are a query expansion expert. Given a user's question about a PDF document, generate 3 optimized search query variations.

Each variation should:
- Use different terminology and phrasing
- Target different aspects of the question
- Be optimized for semantic vector search

Return ONLY 3 queries, one per line. No numbering, no explanation.

Original question: {query}

Optimized queries:"""

DOCUMENT_GRADE_PROMPT = """You are a retrieval quality evaluator. Assess whether the retrieved documents contain relevant information to answer the user's question.

User question: {query}

Retrieved documents:
{documents}

Evaluate the overall quality of these documents:
- CORRECT: At least one document directly addresses the question with specific, relevant information
- INCORRECT: None of the documents contain relevant information to the question
- AMBIGUOUS: Some documents may be partially relevant, but you're uncertain about the overall quality

Respond with the confidence level and a brief explanation."""


GENERATION_GRADE_PROMPT = """You are a response quality evaluator. Assess whether the generated answer is accurate and complete.

User question: {query}

Retrieved documents:
{documents}

Generated answer:
{answer}

Evaluate:
1. Is the answer grounded in/supported by the retrieved documents? (Not making claims beyond what the documents say)
2. Does the answer directly address the user's question?

Respond with your assessment and brief explanation."""


QUERY_TRANSFORM_PROMPT = """You are a query refinement expert. The previous search did not yield relevant results. Rewrite the question to find different information.

Original question: {original_query}

Previous search attempts failed to find relevant information.

Rewrite the question to:
- Use different terminology or synonyms
- Focus on a different aspect of the topic
- Be more general or more specific as needed

Return ONLY the rewritten question. No explanation."""


KNOWLEDGE_REFINE_PROMPT = """You are a knowledge extraction specialist. Extract only the most relevant information from the retrieved documents for answering the question.

User question: {query}

Documents:
{documents}

Extract and organize ONLY the information directly relevant to answering the question.
- Remove irrelevant context and filler
- Group related information together
- Preserve important details and citations

Return the refined knowledge:"""


# =============================================================================
# BASIC MODE: Fast Vector Search
# =============================================================================

async def _basic_mode(query: str, top_k: int = 5, tool_model_provider: Optional[str] = None) -> str:
    """Basic mode: Simple vector search → generate.

    Fastest mode (~2s) but may hallucinate on poor retrieval.

    Args:
        query: Search query
        top_k: Number of documents to retrieve
        tool_model_provider: Provider for LLM (bound by research framework)
    """
    if not RAG_AVAILABLE:
        return "Error: Vector store not available"

    chroma_client = get_rag_client()
    if chroma_client is None:
        return "Error: RAG not initialized"

    try:
        # Simple vector search
        results = chroma_client.similarity_search_with_score(query, k=top_k)

        if not results:
            return f"No relevant information found in the PDF for: {query}"

        # Format results
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', metadata.get('page_number', ''))
            title = metadata.get('title', '')

            citation = f"[{i}]"
            if title:
                citation += f" {title}"
            if page:
                citation += f" (page {page})"
            if source != 'Unknown':
                citation += f" - {source}"
            citation += f" [relevance: {score:.3f}]"

            context_parts.append(f"{citation}\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # Generate answer
        if LLM_AVAILABLE:
            if tool_model_provider:
                llm = await get_llm(provider=tool_model_provider)
            else:
                llm = await get_llm()
            prompt = f"""Answer the question using ONLY the provided context from PDF documents. If the context doesn't contain the answer, say "The PDF does not contain information about this."

Question: {query}

Context:
{context}

Answer:"""

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        else:
            # Return raw context if no LLM
            return f"Context from PDF:\n\n{context}"

    except Exception as e:
        logger.error(f"[PDF_AGENT] Basic mode error: {e}")
        return f"Error searching PDF: {str(e)}"


# =============================================================================
# ENHANCED MODE: Graded Retrieval
# =============================================================================

async def _enhanced_mode(query: str, top_k: int = 10, tool_model_provider: Optional[str] = None) -> str:
    """Enhanced mode: Query expansion → graded retrieval → generate.

    Medium speed (~4-5s) with quality filtering.

    Args:
        query: Search query
        top_k: Number of documents to retrieve
        tool_model_provider: Provider for LLM (bound by research framework)
    """
    if not RAG_AVAILABLE or not LLM_AVAILABLE:
        return "Error: Vector store or LLM not available"

    chroma_client = get_rag_client()
    if chroma_client is None:
        return "Error: RAG not initialized"

    try:
        if tool_model_provider:
            llm = await get_llm(provider=tool_model_provider)
        else:
            llm = await get_llm()

        # Step 1: Query expansion
        logger.info(f"[PDF_AGENT] Enhanced mode: Expanding query...")
        expansion_prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        response = await llm.ainvoke([HumanMessage(content=expansion_prompt)])
        queries_text = response.content.strip().split('\n')

        # Clean up queries
        queries = [q.strip() for q in queries_text if q.strip()]
        queries = [query] + queries[:2]  # Original + 2 expansions

        logger.info(f"[PDF_AGENT] Expanded to {len(queries)} queries: {queries}")

        # Step 2: Parallel retrieval
        logger.info(f"[PDF_AGENT] Enhanced mode: Parallel retrieval...")
        search_tasks = [chroma_client.similarity_search_with_score(q, k=top_k) for q in queries]
        all_results = await asyncio.gather(*search_tasks)

        # Flatten and deduplicate results
        all_docs = {}
        for results in all_results:
            for doc, score in results:
                # Use doc content as key to deduplicate
                key = doc.page_content[:200]
                if key not in all_docs or all_docs[key][1] > score:
                    all_docs[key] = (doc, score)

        sorted_docs = sorted(all_docs.values(), key=lambda x: x[1])[:top_k]

        if not sorted_docs:
            return f"No relevant information found in the PDF for: {query}"

        # Step 3: Grade documents
        logger.info(f"[PDF_AGENT] Enhanced mode: Grading {len(sorted_docs)} documents...")
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content[:500]}"
            for i, (doc, _) in enumerate(sorted_docs[:5])  # Grade top 5 for speed
        ])

        grade_prompt = DOCUMENT_GRADE_PROMPT.format(query=query, documents=docs_text)

        try:
            graded = await llm.with_structured_output(DocumentGrade).ainvoke(
                [HumanMessage(content=grade_prompt)]
            )
            logger.info(f"[PDF_AGENT] Document grade: {graded.confidence} - {graded.reasoning}")
        except Exception as e:
            logger.warning(f"[PDF_AGENT] Grading failed, using all docs: {e}")
            graded = DocumentGrade(confidence="correct", reasoning="Grading failed, proceeding")

        # Step 4: Filter or use based on grade
        if graded.confidence == "incorrect":
            return f"The PDF does not contain relevant information for: {query}\n\nReasoning: {graded.reasoning}"

        # Use top documents for generation
        context_parts = []
        for i, (doc, score) in enumerate(sorted_docs[:5], 1):
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', metadata.get('page_number', ''))

            citation = f"[{i}]"
            if page:
                citation += f" (page {page})"
            if source != 'Unknown':
                citation += f" - {source}"

            context_parts.append(f"{citation}\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # Step 5: Generate answer
        logger.info(f"[PDF_AGENT] Enhanced mode: Generating answer...")
        gen_prompt = f"""Answer the question using ONLY the provided context from PDF documents.

Question: {query}

Context:
{context}

Instructions:
- Use only information from the provided context
- If the context doesn't contain the answer, say so clearly
- Include source citations in your answer

Answer:"""

        response = await llm.ainvoke([HumanMessage(content=gen_prompt)])
        return response.content

    except Exception as e:
        logger.error(f"[PDF_AGENT] Enhanced mode error: {e}", exc_info=True)
        return f"Error in enhanced PDF search: {str(e)}"


# =============================================================================
# AGENTIC MODE: Full CRAG with LangGraph
# =============================================================================

def _create_agentic_graph(max_attempts: int = 3, tool_model_provider: Optional[str] = None):
    """Create the LangGraph for agentic RAG workflow.

    Args:
        max_attempts: Maximum query transformation retries
        tool_model_provider: Provider for LLM (bound by research framework)
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph is not available")

    # Node functions
    async def analyze_query(state: PDFAgentState) -> PDFAgentState:
        """Expand query into 3 variations for parallel retrieval."""
        logger.info("[PDF_AGENT] Agentic: Analyzing and expanding query...")

        provider = state.get("tool_model_provider") or tool_model_provider
        if provider:
            llm = await get_llm(provider=provider)
        else:
            llm = await get_llm()
        prompt = QUERY_EXPANSION_PROMPT.format(query=state["query"])

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            expanded = response.content.strip().split('\n')
            expanded = [q.strip() for q in expanded if q.strip()][:2]
            logger.info(f"[PDF_AGENT] Generated {len(expanded)} query variations")
        except Exception as e:
            logger.warning(f"[PDF_AGENT] Query expansion failed: {e}")
            expanded = []

        state["expanded_queries"] = [state["query"]] + expanded
        return state

    async def retrieve(state: PDFAgentState) -> PDFAgentState:
        """Parallel retrieval with expanded queries."""
        logger.info(f"[PDF_AGENT] Agentic: Retrieving with {len(state.get('expanded_queries', [state['query']]))} queries...")

        chroma_client = get_rag_client()
        queries = state.get("expanded_queries", [state["query"]])

        search_tasks = [chroma_client.similarity_search_with_score(q, k=10) for q in queries]
        all_results = await asyncio.gather(*search_tasks)

        # Deduplicate by content
        all_docs = {}
        for results in all_results:
            for doc, score in results:
                key = doc.page_content[:200]
                if key not in all_docs or all_docs[key][1] > score:
                    all_docs[key] = (doc, score)

        sorted_docs = sorted(all_docs.values(), key=lambda x: x[1])[:10]
        state["documents"] = [doc for doc, _ in sorted_docs]
        state["raw_scores"] = [score for _, score in sorted_docs]

        logger.info(f"[PDF_AGENT] Retrieved {len(state['documents'])} documents")
        return state

    async def grade_documents(state: PDFAgentState) -> Dict[str, str]:
        """Grade documents and route to next step."""
        logger.info("[PDF_AGENT] Agentic: Grading documents...")

        provider = state.get("tool_model_provider") or tool_model_provider
        if provider:
            llm = await get_llm(provider=provider)
        else:
            llm = await get_llm()

        # Grade top 5 documents for speed
        docs_to_grade = state["documents"][:5]
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content[:500]}"
            for i, doc in enumerate(docs_to_grade)
        ])

        prompt = DOCUMENT_GRADE_PROMPT.format(
            query=state["query"],
            documents=docs_text
        )

        try:
            grade = await llm.with_structured_output(DocumentGrade).ainvoke(
                [HumanMessage(content=prompt)]
            )
            state["confidence"] = grade.confidence
            logger.info(f"[PDF_AGENT] Grade: {grade.confidence} - {grade.reasoning}")
        except Exception as e:
            logger.warning(f"[PDF_AGENT] Grading failed: {e}")
            grade = DocumentGrade(confidence="ambiguous", reasoning="Grading failed")

        # Route based on confidence
        if grade.confidence == "correct":
            return "next_refine"
        elif grade.confidence == "ambiguous":
            return "next_refine"
        else:  # incorrect
            if state["attempts"] < state["max_attempts"]:
                return "next_transform"
            else:
                return "next_honest_failure"

    async def refine_documents(state: PDFAgentState) -> PDFAgentState:
        """Decompose-then-recompose: extract relevant strips from documents."""
        logger.info("[PDF_AGENT] Agentic: Refining documents...")

        provider = state.get("tool_model_provider") or tool_model_provider
        if provider:
            llm = await get_llm(provider=provider)
        else:
            llm = await get_llm()

        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(state["documents"][:5])
        ])

        prompt = KNOWLEDGE_REFINE_PROMPT.format(
            query=state["query"],
            documents=docs_text
        )

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            refined_content = response.content.strip()

            # Create refined document
            refined_doc = Document(
                page_content=refined_content,
                metadata={"source": "refined", "type": "synthesized"}
            )
            state["refined_docs"] = [refined_doc]
            logger.info(f"[PDF_AGENT] Refined to {len(refined_content)} chars")
        except Exception as e:
            logger.warning(f"[PDF_AGENT] Refinement failed, using original docs: {e}")
            state["refined_docs"] = state["documents"][:5]

        return state

    async def transform_query(state: PDFAgentState) -> PDFAgentState:
        """Transform query when retrieval fails."""
        logger.info(f"[PDF_AGENT] Agentic: Transforming query (attempt {state['attempts'] + 1})...")

        provider = state.get("tool_model_provider") or tool_model_provider
        if provider:
            llm = await get_llm(provider=provider)
        else:
            llm = await get_llm()
        prompt = QUERY_TRANSFORM_PROMPT.format(
            original_query=state["original_query"]
        )

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            new_query = response.content.strip()
            state["query"] = new_query
            state["attempts"] += 1
            logger.info(f"[PDF_AGENT] Transformed query: {new_query[:80]}...")
        except Exception as e:
            logger.warning(f"[PDF_AGENT] Query transformation failed: {e}")
            state["attempts"] += 1

        return state

    async def generate_answer(state: PDFAgentState) -> PDFAgentState:
        """Generate answer from refined documents."""
        logger.info("[PDF_AGENT] Agentic: Generating answer...")

        provider = state.get("tool_model_provider") or tool_model_provider
        if provider:
            llm = await get_llm(provider=provider)
        else:
            llm = await get_llm()

        docs_to_use = state.get("refined_docs", state["documents"][:5])
        context = "\n\n".join([
            f"[{i+1}] {doc.page_content}"
            for i, doc in enumerate(docs_to_use)
        ])

        prompt = f"""Answer the question using ONLY the provided context from PDF documents.

Question: {state["original_query"]}

Context:
{context}

Instructions:
- Use ONLY information from the provided context
- Include source citations [1], [2], etc.
- If the context doesn't contain the answer, say so clearly
- Do not make claims beyond what the documents state

Answer:"""

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        state["answer"] = response.content
        logger.info(f"[PDF_AGENT] Generated answer: {len(response.content)} chars")
        return state

    async def check_generation(state: PDFAgentState) -> Dict[str, str]:
        """Check generation quality (hallucination + answer grading)."""
        logger.info("[PDF_AGENT] Agentic: Checking generation quality...")

        provider = state.get("tool_model_provider") or tool_model_provider
        if provider:
            llm = await get_llm(provider=provider)
        else:
            llm = await get_llm()

        docs_to_use = state.get("refined_docs", state["documents"][:5])
        docs_text = "\n\n".join([doc.page_content for doc in docs_to_use])

        prompt = GENERATION_GRADE_PROMPT.format(
            query=state["original_query"],
            documents=docs_text,
            answer=state["answer"]
        )

        try:
            grade = await llm.with_structured_output(GenerationGrade).ainvoke(
                [HumanMessage(content=prompt)]
            )
            logger.info(f"[PDF_AGENT] Generation grade: grounded={grade.is_grounded}, answers={grade.answers_question}")

            if grade.is_grounded and grade.answers_question:
                return "end"
            elif not grade.is_grounded:
                # Hallucination - retry
                if state["attempts"] < state["max_attempts"]:
                    return "retry_refine"
                else:
                    return "end"
            else:
                # Doesn't answer - maybe transform query
                if state["attempts"] < state["max_attempts"]:
                    return "retry_transform"
                else:
                    return "end"
        except Exception as e:
            logger.warning(f"[PDF_AGENT] Generation grading failed: {e}")
            return "end"

    async def honest_failure(state: PDFAgentState) -> PDFAgentState:
        """Admit when PDF doesn't contain the information."""
        state["answer"] = f"""Based on my search through the PDF documents, I could not find relevant information to answer: {state["original_query"]}

The documents either:
- Do not contain information on this topic
- Discuss it in a way that doesn't address your specific question
- Use different terminology that didn't match my search queries

You may want to:
- Rephrase your question using different terms
- Check if the topic is covered in these documents
- Consult additional sources"""

        logger.info("[PDF_AGENT] Agentic: Honest failure - no relevant information found")
        return state

    # Build the graph
    workflow = StateGraph(PDFAgentState)

    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("refine", refine_documents)
    workflow.add_node("transform", transform_query)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("honest_failure", honest_failure)

    # Add edges
    workflow.add_edge(START, "retrieve")

    # Conditional routing from retrieval
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "next_refine": "refine",
            "next_transform": "transform",
            "next_honest_failure": "honest_failure"
        }
    )

    # After refinement, generate
    workflow.add_edge("refine", "generate")

    # After transformation, retry retrieval
    workflow.add_edge("transform", "retrieve")

    # After generation, check quality
    workflow.add_conditional_edges(
        "generate",
        check_generation,
        {
            "end": END,
            "retry_refine": "refine",
            "retry_transform": "transform"
        }
    )

    # Honest failure goes to end
    workflow.add_edge("honest_failure", END)

    return workflow.compile()


async def _agentic_mode(query: str, top_k: int = 10, max_attempts: int = 3, tool_model_provider: Optional[str] = None) -> str:
    """Agentic mode: Full CRAG with self-correction.

    Slowest mode (~8-12s) but highest quality with self-correction loops.

    Args:
        query: Search query
        top_k: Number of documents to retrieve
        max_attempts: Maximum query transformation retries
        tool_model_provider: Provider for LLM (bound by research framework)
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("[PDF_AGENT] LangGraph not available, falling back to enhanced mode")
        return await _enhanced_mode(query, top_k, tool_model_provider=tool_model_provider)

    if not RAG_AVAILABLE or not LLM_AVAILABLE:
        return "Error: Vector store or LLM not available"

    try:
        logger.info(f"[PDF_AGENT] Agentic mode: Starting with max_attempts={max_attempts}")

        # Create and run the graph
        graph = _create_agentic_graph(max_attempts, tool_model_provider=tool_model_provider)

        initial_state: PDFAgentState = {
            "query": query,
            "original_query": query,
            "documents": [],
            "refined_docs": None,
            "confidence": None,
            "attempts": 0,
            "answer": None,
            "max_attempts": max_attempts,
            "tool_model_provider": tool_model_provider,  # Bind tool model provider
        }

        result = await graph.ainvoke(initial_state)

        if result.get("answer"):
            logger.info(f"[PDF_AGENT] Agentic mode completed successfully")
            return result["answer"]
        else:
            return "Error: Failed to generate answer"

    except Exception as e:
        logger.error(f"[PDF_AGENT] Agentic mode error: {e}", exc_info=True)
        # Fallback to enhanced mode
        logger.info("[PDF_AGENT] Falling back to enhanced mode")
        return await _enhanced_mode(query, top_k, tool_model_provider=tool_model_provider)


# =============================================================================
# MAIN TOOL FUNCTION
# =============================================================================

@tool(args_schema=PDFDeepAgentInput)
async def pdf_deep_agent(
    query: str,
    mode: Literal["basic", "enhanced", "agentic"] = "agentic",
    top_k: int = 10,
    max_attempts: int = 3,
    runtime: Optional[Any] = None,  # ToolRuntime for accessing state
) -> str:
    """Agentic PDF analysis tool for deep document understanding.

    Three modes with different quality/speed tradeoffs:
    - basic: Fast vector search (~2s). Simple retrieval → generation.
    - enhanced: Graded retrieval (~4-5s). Query expansion + quality filtering.
    - agentic: Full CRAG with self-correction (~8-12s, default).

    The agentic mode implements Corrective Retrieval Augmented Generation:
    1. Expands query into 3 variations for parallel search
    2. Retrieves and grades documents (correct/incorrect/ambiguous)
    3. Refines relevant documents (decompose-then-recompose)
    4. Transforms query and retries if retrieval fails
    5. Grades generation for hallucination and completeness
    6. Admits failure honestly if information is not in PDF

    Use this tool when you need to:
    - Extract specific information from PDF documents
    - Analyze books, reports, or academic papers
    - Find evidence or quotes for claims
    - Synthesize information across multiple PDF pages

    Args:
        query: The question or analysis task for the PDF. Be specific about what
               information you need. Examples: "What are the main arguments against
               central banking?", "Find statistics about climate change impacts",
               "Summarize the methodology used in this research".
        mode: Analysis mode - "basic" (fast), "enhanced" (graded), or "agentic" (full CRAG, default).
        top_k: Number of documents to retrieve for grading (default: 10). More documents
              may find better results but take longer to process.
        max_attempts: Maximum query transformation retries in agentic mode (default: 3).
                      The agent will try different query formulations before giving up.
        runtime: ToolRuntime context for accessing state (used for tool_model_provider binding).

    Returns:
        Synthesized answer with citations from the PDF, or honest failure message
        if the document doesn't contain relevant information.
    """
    if not RAG_AVAILABLE:
        return "Error: RAG system not available. Ensure vector store is initialized."

    # Read tool_model_provider from runtime.state (bound by research framework)
    tool_model_provider = None
    if runtime is not None:
        tool_model_provider = runtime.state.get("tool_model_provider")
        if tool_model_provider:
            logger.info(f"[PDF_AGENT] Using tool model provider: {tool_model_provider}")

    logger.info(f"[PDF_AGENT] Starting PDF deep agent: mode={mode}, query='{query[:80]}'...")

    try:
        if mode == "basic":
            logger.info("[PDF_AGENT] Using basic mode (fast)")
            return await _basic_mode(query, top_k, tool_model_provider=tool_model_provider)

        elif mode == "enhanced":
            logger.info("[PDF_AGENT] Using enhanced mode (graded)")
            return await _enhanced_mode(query, top_k, tool_model_provider=tool_model_provider)

        elif mode == "agentic":
            logger.info("[PDF_AGENT] Using agentic mode (full CRAG)")
            return await _agentic_mode(query, top_k, max_attempts, tool_model_provider=tool_model_provider)

        else:
            return f"Error: Unknown mode '{mode}'. Use 'basic', 'enhanced', or 'agentic'."

    except Exception as e:
        logger.error(f"[PDF_AGENT] Error: {e}", exc_info=True)
        return f"Error in PDF deep agent: {str(e)}"


# =============================================================================
# TOOL REGISTRATION HELPERS
# =============================================================================

def get_pdf_deep_agent_tool(mode: str = "agentic"):
    """Get a pdf_deep_agent tool with a specific mode pre-configured.

    This allows configs to create mode-specific tools.

    Args:
        mode: The analysis mode ("basic", "enhanced", or "agentic")

    Returns:
        A tool function with the mode pre-configured
    """
    async def bound_tool(query: str, top_k: int = 10, max_attempts: int = 3) -> str:
        return await pdf_deep_agent(query, mode=mode, top_k=top_k, max_attempts=max_attempts)

    bound_tool.name = f"pdf_deep_agent_{mode}"
    bound_tool.description = f"""PDF analysis tool in {mode} mode.

    {pdf_deep_agent.description}

    Mode-specific behavior:
    - basic: Fast vector search (~2s)
    - enhanced: Graded retrieval (~4-5s)
    - agentic: Full CRAG with self-correction (~8-12s)
    """

    return bound_tool
