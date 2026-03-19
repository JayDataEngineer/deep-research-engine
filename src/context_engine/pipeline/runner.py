"""Main pipeline orchestration.

Orchestrates: Main Agent → Critic → (loop if fail) → Complete
"""

import uuid
from pathlib import Path
from typing import Any

from langgraph.store.base import BaseStore
from loguru import logger

from context_engine.agents.critic import create_critic_agent, parse_critic_response
from context_engine.agents.main import create_main_agent
from context_engine.logging.rich_logging import log_critic_review, log_stage
from context_engine.prompts.langfuse import get_prompt_manager
from context_engine.store.setup import get_store_and_checkpointer


async def run_pipeline(
    query: str,
    input_files: dict[str, str] | None = None,
    session_id: str | None = None,
    max_iterations: int = 3,
    pass_threshold: float = 0.7,
    use_langfuse_prompts: bool = True,
) -> dict[str, Any]:
    """Run the full context engine pipeline.

    Pipeline:
    1. Setup: Initialize store, checkpointer, prompts
    2. Ingest: Load input files into store (if provided)
    3. Build: Main agent gathers context
    4. Critic: Evaluate quality
    5. Loop: If failed, feed back and retry
    6. Complete: Return results

    Args:
        query: Research query to build context for
        input_files: Optional dict of filename -> content to ingest
        session_id: Session ID (auto-generated if None)
        max_iterations: Maximum critic loop iterations
        pass_threshold: Minimum score to pass (0.0-1.0)
        use_langfuse_prompts: Whether to load prompts from LangFuse

    Returns:
        Dict with:
        - session_id: str
        - passed: bool
        - score: float
        - iterations: int
        - issues: list[str]
        - suggestions: list[str]
    """
    # Generate session ID if not provided
    session_id = session_id or str(uuid.uuid4())[:8]

    logger.info(f"Starting pipeline for session {session_id}")
    log_stage("INIT", {"session_id": session_id, "query": query[:50] + "..."})

    # Setup: Store + Checkpointer
    store, checkpointer = await get_store_and_checkpointer()

    # Setup: Prompts
    prompt_manager = get_prompt_manager(use_langfuse=use_langfuse_prompts)
    prompts = {
        "main_agent": prompt_manager.get_main_agent_prompt(session_id),
        "critic_agent": prompt_manager.get_critic_agent_prompt(),
    }

    # Phase: Ingest input files
    if input_files:
        log_stage("INGEST", {"files": len(input_files)})
        await _ingest_files(store, session_id, input_files)

    # Create agents
    log_stage("CREATE_AGENTS")
    main_agent = create_main_agent(store, session_id, prompts)
    critic_agent = create_critic_agent(store, session_id, prompts)

    # Pipeline state
    iteration = 0
    passed = False
    score = 0.0
    issues: list[str] = []
    suggestions: list[str] = []
    current_query = query

    # Main pipeline loop
    while iteration < max_iterations:
        iteration += 1
        log_stage("BUILD", {"iteration": iteration})

        # Build context
        build_result = await main_agent.ainvoke(
            {"messages": [{"role": "user", "content": current_query}]},
            config={
                "configurable": {"thread_id": session_id},
                "context": {"session_id": session_id},
            },
        )

        # Evaluate with critic
        log_stage("CRITIC", {"iteration": iteration})
        critic_result = await critic_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Evaluate the context quality for this query:

QUERY: {query}

Review all stored memories and assess completeness, accuracy, and relevance.
Return your evaluation as JSON.""",
                    }
                ]
            },
            config={"configurable": {"thread_id": f"{session_id}-critic"}},
        )

        # Parse critic response
        evaluation = parse_critic_response(critic_result)
        score = evaluation["score"]
        issues = evaluation["issues"]
        suggestions = evaluation["suggestions"]
        passed = evaluation["passed"] or score >= pass_threshold

        # Log critic review
        log_critic_review(score, passed, issues, suggestions)

        if passed:
            log_stage("COMPLETE", {"score": score, "iterations": iteration})
            break

        # Failed: prepare feedback for next iteration
        log_stage("REFINE", {"reason": "critic_feedback", "score": score})

        current_query = f"""The critic reviewed your context and found issues that need to be addressed.

CRITIC SCORE: {score:.2f}
PASS THRESHOLD: {pass_threshold}

ISSUES FOUND:
{chr(10).join(f"- {issue}" for issue in issues)}

SUGGESTIONS FOR IMPROVEMENT:
{chr(10).join(f"- {sug}" for sug in suggestions)}

Please improve the context to address these issues.
The original query was: {query}
"""

    else:
        # Max iterations reached
        log_stage(
            "COMPLETE",
            {"note": f"Max iterations ({max_iterations}) reached", "score": score},
        )
        logger.warning(f"Max iterations reached with score {score:.2f}")

    return {
        "session_id": session_id,
        "passed": passed,
        "score": score,
        "iterations": iteration,
        "issues": issues,
        "suggestions": suggestions,
    }


async def _ingest_files(
    store: BaseStore,
    session_id: str,
    files: dict[str, str],
) -> None:
    """Ingest input files into store.

    Args:
        store: LangGraph store
        session_id: Session ID for namespacing
        files: Dict of filename -> content
    """
    namespace = (session_id, "input")

    for filename, content in files.items():
        # Store with auto-vectorization
        await store.aput(
            namespace=namespace,
            key=filename,
            value={
                "content": content,
                "source": "user_input",
            },
        )
        logger.debug(f"Ingested {filename} into {namespace}")

    logger.info(f"Ingested {len(files)} files for session {session_id}")


async def build_context(
    query: str,
    session_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Simplified context building entry point.

    Args:
        query: Research query
        session_id: Optional session ID
        **kwargs: Additional arguments passed to run_pipeline

    Returns:
        Pipeline result dict
    """
    return await run_pipeline(
        query=query,
        session_id=session_id,
        **kwargs,
    )


# CLI entry point
async def main() -> None:
    """CLI entry point for context engine."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run python -m context_engine 'your query here'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = await run_pipeline(query)

    print(f"\n{'='*50}")
    print(f"Session: {result['session_id']}")
    print(f"Passed: {result['passed']}")
    print(f"Score: {result['score']:.2f}")
    print(f"Iterations: {result['iterations']}")
    if result["issues"]:
        print(f"Issues: {result['issues']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
