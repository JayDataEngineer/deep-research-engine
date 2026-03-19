"""
Output markdown generation for Run Logger V3.

Handles the generation of output.md with all metrics and formatting.
"""

import json
from typing import Dict, Any, List, Optional

from .models import RunMetrics, WorkerMetrics


def save_output_markdown(
    run_dir,
    run_name: str,
    run_id: str,
    metrics: RunMetrics,
    use_case: str,
    variant: str,
    config: Dict[str, Any],
    response: str,
    sources: List[str],
    citations: List[Any],
    include_images: bool = False
) -> None:
    """Generate and save the output.md file with all metrics.

    Args:
        run_dir: The run directory path
        run_name: The run name
        run_id: The run ID
        metrics: The run metrics
        use_case: The use case
        variant: The config variant
        config: The configuration dict
        response: The final response text
        sources: List of source names
        citations: List of full citation objects
        include_images: Ignored (charts removed for simplicity)
    """
    output_file = run_dir / "output.md"
    with open(output_file, 'w') as f:
        f.write(f"# {run_name}\n\n")
        f.write(f"**Run ID:** `{run_id}`\n")
        f.write(f"**Date:** {metrics.timestamp}\n")
        f.write(f"**Use Case:** {use_case}\n")
        f.write(f"**Variant:** {variant}\n")
        f.write(f"**Overdrive:** {'✅ Enabled' if metrics.overdrive else '❌ Disabled'}\n\n")

        # Config section
        _write_config_section(f, metrics, config)

        # Query
        f.write(f"## Query\n\n")
        f.write(f"{metrics.query}\n\n")

        # Errors and Issues
        _write_errors_section(f, metrics)

        # Phase Metrics Matrix
        _write_phase_metrics_matrix(f, metrics, config)

        # Summary metrics table
        _write_summary_metrics(f, metrics)

        # Critic Rejections / Iterations
        if metrics.critique_history:
            _write_critic_evaluations(f, metrics)

        # Tool Call Metrics
        if metrics.tool_stats:
            _write_tool_metrics(f, metrics)

        # Researcher-Tool Matrix
        if metrics.researcher_tool_stats:
            _write_researcher_tool_matrix(f, metrics)

        # Dynamic Prompt Breakdown
        if metrics.prompt_breakdown:
            _write_prompt_breakdown(f, metrics)

        # Response
        f.write(f"## Response\n\n")
        f.write(f"{response}\n")

        # Sources and Citations
        _write_sources_and_citations(f, response, sources, citations)


def _write_config_section(f, metrics: RunMetrics, config: Dict[str, Any]) -> None:
    """Write the configuration section to output.md."""
    f.write(f"## Configuration\n\n")

    # Show useful config details that are always available
    f.write(f"### Execution Details\n\n")
    f.write(f"- **Workers Spawned:** {metrics.workers_spawned}\n")
    f.write(f"- **Research Iterations:** {metrics.iterations}\n")
    if metrics.quality_score > 0:
        f.write(f"- **Quality Score:** {metrics.quality_score:.2f}/1.00\n")
    f.write(f"\n")

    # Show profile information if available
    if config:
        profile = config.get("_profile")
        if profile and hasattr(profile, 'name'):
            f.write(f"### Profile\n\n")
            f.write(f"- **Name:** {profile.name}\n")
            f.write(f"- **Description:** {profile.description if hasattr(profile, 'description') else 'N/A'}\n")
            f.write(f"\n")

    # Show spawner_config details if available
    if config:
        spawner_config = config.get("spawner_config", {})
        if spawner_config:
            f.write(f"### Spawner Settings\n\n")
            worker_execution_mode = spawner_config.get("worker_execution_mode", "parallel")
            max_workers = spawner_config.get("max_workers")
            if max_workers is None or max_workers == "N/A":
                max_workers = metrics.workers_spawned if metrics.workers_spawned > 0 else "N/A"
            f.write(f"- **Execution Mode:** {worker_execution_mode}\n")
            f.write(f"- **Max Workers:** {max_workers}\n")
            f.write(f"\n")

            # Show tool settings if available
            tools_config = spawner_config.get("tools", {})
            enabled_tools = [k for k, v in tools_config.items() if v is True and not k.startswith("_")]
            if enabled_tools:
                f.write(f"### Enabled Tools\n\n")
                for tool in enabled_tools[:10]:  # Limit to first 10
                    f.write(f"- {tool}\n")
                if len(enabled_tools) > 10:
                    f.write(f"- ... and {len(enabled_tools) - 10} more\n")
                f.write(f"\n")


def _write_errors_section(f, metrics: RunMetrics) -> None:
    """Write the errors and issues section to output.md."""
    errors_found = False
    errors_output = []

    # Check for failed workers
    failed_workers = [w for w in metrics.worker_metrics if w.status != "success"]
    if failed_workers:
        errors_found = True
        errors_output.append("### Failed Workers\n\n")
        for worker in failed_workers:
            errors_output.append(f"**{worker.worker_id}** ({worker.worker_type})\n\n")
            errors_output.append(f"- **Status:** {worker.status}\n")
            if worker.error_message:
                errors_output.append(f"- **Error:** {worker.error_message}\n")
            errors_output.append(f"- **Task:** {worker.task_description[:200]}...\n\n")

    # Check for worker error messages (including refusals)
    workers_with_errors = [w for w in metrics.worker_metrics if w.error_message and w.status == "error"]
    if workers_with_errors:
        errors_found = True
        if "### Worker Errors\n\n" not in "".join(errors_output):
            errors_output.append("### Worker Errors\n\n")
        for worker in workers_with_errors:
            errors_output.append(f"**{worker.worker_id}**: {worker.error_message}\n\n")

    # Check for workers with no tool calls (potential silent failures)
    workers_no_tools = [w for w in metrics.worker_metrics if w.tool_calls == 0 and w.worker_type != "scriptwriter"]
    if workers_no_tools:
        errors_found = True
        errors_output.append("### Workers With No Tool Calls\n\n")
        for worker in workers_no_tools:
            errors_output.append(f"- **{worker.worker_id}** ({worker.worker_type}): {worker.output_length} chars output, {worker.llm_calls} LLM calls\n\n")

    # Check for slow/failed tool calls
    slow_tool_calls = {}
    failed_tool_calls = []
    tool_errors = {}

    error_patterns = [
        "408 Request Timeout",
        "scrape error",
        "Access Denied",
        "Access Restricted",
        "LocalPRO.*error",
        "Client error",
    ]

    for call in metrics.tool_calls:
        tool_name = call.get("tool", "unknown")
        researcher = call.get("researcher", "unknown")
        duration = call.get("duration", 0)
        output = str(call.get("output", call.get("output_content", "")))

        has_explicit_error = "error" in call

        has_text_error = False
        error_type = None
        for pattern in error_patterns:
            if pattern.lower() in output.lower():
                has_text_error = True
                error_type = pattern
                break

        if has_text_error or has_explicit_error:
            error_key = error_type if error_type else "unknown_error"
            if error_key not in tool_errors:
                tool_errors[error_key] = {"count": 0, "tools": {}}
            tool_errors[error_key]["count"] += 1
            if tool_name not in tool_errors[error_key]["tools"]:
                tool_errors[error_key]["tools"][tool_name] = 0
            tool_errors[error_key]["tools"][tool_name] += 1

        if (has_explicit_error or has_text_error) and len(failed_tool_calls) < 50:
            failed_tool_calls.append(f"- **{tool_name}** (by {researcher}): {duration:.1f}s - {error_type if error_type else 'FAILED'}\n")

        elif duration > 5:
            if tool_name not in slow_tool_calls:
                slow_tool_calls[tool_name] = {"count": 0, "total_time": 0, "max_time": 0}
            slow_tool_calls[tool_name]["count"] += 1
            slow_tool_calls[tool_name]["total_time"] += duration
            slow_tool_calls[tool_name]["max_time"] = max(slow_tool_calls[tool_name]["max_time"], duration)

    # Write tool error summary
    if tool_errors:
        errors_found = True
        errors_output.append("### Tool Errors & Retries\n\n")
        errors_output.append("*Errors detected in tool outputs. ToolRetryMiddleware automatically retries failed calls.*\n\n")
        for error_type, stats in sorted(tool_errors.items(), key=lambda x: x[1]["count"], reverse=True):
            errors_output.append(f"**{error_type}**:\n")
            errors_output.append(f"  - Total occurrences: {stats['count']}\n")
            errors_output.append(f"  - Affected tools: {', '.join(stats['tools'].keys())}\n\n")

    # Write detailed failed call list
    if failed_tool_calls:
        errors_found = True
        if not tool_errors:
            errors_output.append("### Failed Tool Calls\n\n")
        errors_output.append(f"*Showing first {len(failed_tool_calls)} failed calls*\n\n")
        errors_output.extend(failed_tool_calls)
        errors_output.append("\n")

    # Write slow tool calls
    if slow_tool_calls:
        errors_found = True
        errors_output.append("### Slow Tool Calls (>5s)\n\n")
        errors_output.append("*Tool calls taking >5 seconds may indicate scrape service overload.*\n\n")
        for tool_name, stats in sorted(slow_tool_calls.items(), key=lambda x: x[1]["total_time"], reverse=True):
            avg_time = stats["total_time"] / stats["count"]
            errors_output.append(f"- **{tool_name}**:\n")
            errors_output.append(f"  - Count: {stats['count']} calls\n")
            errors_output.append(f"  - Total time: {stats['total_time']:.1f}s\n")
            errors_output.append(f"  - Average: {avg_time:.1f}s/call\n")
            errors_output.append(f"  - Max: {stats['max_time']:.1f}s\n\n")

    # Determine if run failed completely
    run_failed_completely = any(w.status == "error" for w in metrics.worker_metrics if w.status)

    # Write errors section if any found
    if errors_found:
        f.write(f"## ⚠️ Errors and Issues\n\n")

        if run_failed_completely:
            f.write(f"## ❌ FAILED TO COMPLETE\n\n")
            f.write(f"**Status:** Research run failed due to critical errors.\n\n")
            f.write(f"The research system encountered unrecoverable errors and could not complete the requested task.\n\n")
            f.write(f"---\n\n")

        for line in errors_output:
            f.write(line)
        f.write("\n")


def _write_phase_metrics_matrix(f, metrics: RunMetrics, config: Dict[str, Any]) -> None:
    """Write the phase metrics matrix table to output.md."""
    f.write(f"## Phase Metrics Matrix\n\n")
    f.write(f"| Phase | Worker Type | LLM Calls | Tool Calls | Phase Time | Prompt Tokens | Completion Tokens | Total Tokens |\n")
    f.write(f"|-------|-------------|-----------|------------|------------|---------------|-------------------|--------------|\n")

    def fmt_tokens(t: int) -> str:
        return f"{t:,}" if t > 0 else "0"

    phases = [
        ("Worker", metrics.worker_calls, metrics.worker_time,
         metrics.worker_prompt_tokens, metrics.worker_completion_tokens),
        ("Boss", metrics.boss_calls, metrics.boss_time,
         metrics.boss_prompt_tokens, metrics.boss_completion_tokens),
        ("Critic", metrics.critic_calls, metrics.critic_time,
         metrics.critic_prompt_tokens, metrics.critic_completion_tokens),
        ("Synthesis", metrics.synthesis_calls, metrics.synthesis_time,
         metrics.synthesis_prompt_tokens, metrics.synthesis_completion_tokens),
    ]

    for phase_name, calls, time_val, prompt_toks, completion_toks in phases:
        total_toks = prompt_toks + completion_toks
        f.write(f"| {phase_name} | - | {calls} | - | {time_val:.2f}s | {fmt_tokens(prompt_toks)} | {fmt_tokens(completion_toks)} | {fmt_tokens(total_toks)} |\n")

    # Worker breakdown by individual sessions
    worker_session_stats = {}
    for call in metrics.calls:
        if call.get("phase") == "worker" and "worker_id" in call:
            worker_id = call["worker_id"]
            if worker_id not in worker_session_stats:
                worker_session_stats[worker_id] = {
                    "time": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0
                }
            worker_session_stats[worker_id]["time"] += call.get("time", 0.0)
            worker_session_stats[worker_id]["prompt_tokens"] += call.get("prompt_tokens", 0)
            worker_session_stats[worker_id]["completion_tokens"] += call.get("completion_tokens", 0)

    # Build mapping of worker_id to worker_type, llm_calls, tool_calls
    worker_type_map = {}
    worker_llm_calls_map = {}
    worker_tool_calls_map = {}
    for wm in metrics.worker_metrics:
        if wm.worker_id:
            worker_type_map[wm.worker_id] = (wm.worker_type.upper() if wm.worker_type else "GENERIC")
            worker_llm_calls_map[wm.worker_id] = wm.llm_calls
            worker_tool_calls_map[wm.worker_id] = wm.tool_calls

    # Write individual worker sessions
    for worker_id, stats in sorted(worker_session_stats.items()):
        worker_type = worker_type_map.get(worker_id, "-")
        llm_call_count = worker_llm_calls_map.get(worker_id, 0)
        tool_call_count = worker_tool_calls_map.get(worker_id, 0)
        total_toks = stats["prompt_tokens"] + stats["completion_tokens"]
        f.write(f"| {worker_id} | {worker_type} | {llm_call_count} | {tool_call_count} | {stats['time']:.2f}s | {fmt_tokens(stats['prompt_tokens'])} | {fmt_tokens(stats['completion_tokens'])} | {fmt_tokens(total_toks)} |\n")

    # Worker aggregate row
    worker_total_toks = metrics.worker_prompt_tokens + metrics.worker_completion_tokens
    total_worker_llm_calls = sum(wm.llm_calls for wm in metrics.worker_metrics)
    total_worker_tool_calls = sum(wm.tool_calls for wm in metrics.worker_metrics)

    lab_display_time = metrics.lab_wall_time if metrics.lab_wall_time > 0 else metrics.lab_time

    if metrics.workers_spawned > 0 or total_worker_llm_calls > 0:
        worker_execution_mode = "parallel"
        if config:
            spawner_config = config.get("spawner_config", {})
            worker_execution_mode = spawner_config.get("worker_execution_mode", "parallel")
        parallel_indicator = "✓ PARALLEL" if worker_execution_mode == "parallel" else "SEQUENTIAL"
        f.write(f"| Lab (Workers) | - | {total_worker_llm_calls} | {total_worker_tool_calls} | {lab_display_time:.2f}s ({parallel_indicator}) | {fmt_tokens(metrics.worker_prompt_tokens)} | {fmt_tokens(metrics.worker_completion_tokens)} | {fmt_tokens(worker_total_toks)} |\n")

    # Calculate totals
    all_phase_totals = list(phases)
    if metrics.workers_spawned > 0 or total_worker_llm_calls > 0:
        all_phase_totals.append(
            ("Lab (Workers)", total_worker_llm_calls, lab_display_time,
             metrics.worker_prompt_tokens, metrics.worker_completion_tokens)
        )
    total_llm_calls = sum(c for _, c, _, _, _ in all_phase_totals)
    total_tool_calls = total_worker_tool_calls
    total_time = sum(t for _, _, t, _, _ in all_phase_totals)
    total_prompt_tokens = sum(p for _, _, _, p, _ in all_phase_totals)
    total_completion_tokens = sum(c for _, _, _, _, c in all_phase_totals)
    total_tokens = total_prompt_tokens + total_completion_tokens

    f.write(f"| **TOTAL** | - | **{total_llm_calls}** | **{total_tool_calls}** | **{total_time:.2f}s** | **{fmt_tokens(total_prompt_tokens)}** | **{fmt_tokens(total_completion_tokens)}** | **{fmt_tokens(total_tokens)}** |\n\n")


def _write_summary_metrics(f, metrics: RunMetrics) -> None:
    """Write the summary metrics table to output.md."""
    f.write(f"## Summary Metrics\n\n")
    f.write(f"| Metric | Value |\n")
    f.write(f"|--------|-------|\n")
    f.write(f"| **Total Time** | {metrics.total_time:.2f}s |\n")
    f.write(f"| **Tokens/sec** | {metrics.tokens_per_sec:.2f} |\n")
    f.write(f"| **Avg Call Time** | {metrics.avg_call_time:.2f}s |\n")
    f.write(f"| **Response Length** | {metrics.response_length} chars |\n")
    f.write(f"| **Workers Spawned** | {metrics.workers_spawned} |\n")
    f.write(f"| **Iterations** | {metrics.iterations} |\n")

    if metrics.workers_spawned == 0 and not metrics.critique_history:
        f.write(f"| **Quality Score** | N/A (quick path) |\n")
        f.write(f"| **Passed Quality Gate** | N/A (quick path) |\n\n")
    else:
        f.write(f"| **Quality Score** | {metrics.quality_score:.2f} |\n")
        f.write(f"| **Passed Quality Gate** | {metrics.passed_quality_gate} |\n\n")


def _write_critic_evaluations(f, metrics: RunMetrics) -> None:
    """Write the critic evaluations section to output.md."""
    f.write(f"## Critic Evaluations\n\n")

    passes = sum(1 for c in metrics.critique_history if c.get("passed", False))
    revisions = sum(1 for c in metrics.critique_history if c.get("recommendation") == "revise")
    rejections = len(metrics.critique_history) - passes - revisions

    f.write(f"**Total Evaluations:** {len(metrics.critique_history)}\n")
    f.write(f"**Passed:** {passes} | **Revisions:** {revisions} | **Rejected:** {rejections}\n\n")

    for i, critique in enumerate(metrics.critique_history, 1):
        recommendation = critique.get("recommendation")
        if recommendation == "pass":
            status = "✅ PASSED"
        elif recommendation == "revise":
            status = "🔄 REVISE"
        elif critique.get("passed", False):
            status = "✅ PASSED"
        else:
            status = "❌ REJECTED"

        f.write(f"### Iteration {critique.get('iteration', i)} - {status}\n\n")
        f.write(f"**Quality Score:** {critique.get('quality_score', 0):.2f}/1.0\n")
        f.write(f"**Timestamp:** {critique.get('timestamp', 'Unknown')}\n\n")

        feedback = critique.get("feedback", "")
        if feedback:
            f.write(f"**Feedback:**\n")
            f.write(f"<details><summary>Click to expand feedback</summary>\n\n")
            f.write(f"{feedback}\n\n")
            f.write(f"</details>\n\n")

        gaps = critique.get("gaps_identified", [])
        if gaps:
            f.write(f"**Gaps Identified ({len(gaps)}):**\n")
            for gap in gaps:
                f.write(f"- {gap}\n")
            f.write("\n")

    f.write("---\n\n")


def _write_tool_metrics(f, metrics: RunMetrics) -> None:
    """Write the tool call metrics section to output.md."""
    f.write(f"## Tool Call Metrics\n\n")
    f.write(f"| Tool | Calls | Total Time | Avg Time | Total Output |\n")
    f.write(f"|------|-------|------------|----------|-------------|\n")

    sorted_tools = sorted(
        metrics.tool_stats.items(),
        key=lambda x: x[1]["total_time"],
        reverse=True
    )

    for tool_name, stats in sorted_tools:
        f.write(f"| {tool_name} | {stats['count']} | {stats['total_time']:.3f}s | {stats['avg_time']:.3f}s | {stats['total_output']:,} chars |\n")

    total_tool_time = sum(s["total_time"] for s in metrics.tool_stats.values())
    total_tool_calls = sum(s["count"] for s in metrics.tool_stats.values())
    f.write(f"| **TOTAL** | **{total_tool_calls}** | **{total_tool_time:.3f}s** | **-** | **-** |\n\n")


def _write_researcher_tool_matrix(f, metrics: RunMetrics) -> None:
    """Write the researcher-tool matrix section to output.md."""
    f.write(f"## Researcher Tool Matrix\n\n")

    researchers = sorted(metrics.researcher_tool_stats.keys())
    researchers = [r for r in researchers if r != "unknown"]

    all_tools = set()
    for tool_dict in metrics.researcher_tool_stats.values():
        all_tools.update(tool_dict.keys())
    tools = sorted(all_tools)

    if researchers and tools:
        # Table 1: Researchers x Tools
        f.write(f"### Researchers → Tools\n\n")
        f.write(f"| Researcher | " + " | ".join(tools) + " | **Total** |\n")
        f.write(f"|------------|" + "|".join(["-----------"] * len(tools)) + "|----------|\n")

        for researcher in researchers:
            tool_counts = metrics.researcher_tool_stats.get(researcher, {})
            row_data = [str(tool_counts.get(tool, 0)) for tool in tools]
            row_total = sum(int(c) for c in row_data)
            f.write(f"| {researcher} | " + " | ".join(row_data) + f" | **{row_total}** |\n")

        f.write(f"| **TOTAL** |")
        for tool in tools:
            total = sum(metrics.researcher_tool_stats.get(r, {}).get(tool, 0) for r in researchers)
            f.write(f" **{total}** |")
        grand_total = sum(sum(metrics.researcher_tool_stats.get(r, {}).values()) for r in researchers)
        f.write(f" **{grand_total}** |\n\n")


def _write_prompt_breakdown(f, metrics: RunMetrics) -> None:
    """Write the prompt breakdown section to output.md."""
    f.write(f"## Dynamic Prompt Breakdown\n\n")

    worker_to_boss = metrics.prompt_breakdown.get("worker_to_boss", [])
    if worker_to_boss:
        f.write(f"### Architect -> Boss\n\n")
        for i, info in enumerate(worker_to_boss, 1):
            f.write(f"**{i}. Research Brief**\n")
            f.write(f"- Complexity: `{info.get('complexity', 'N/A')}`\n")
            f.write(f"- Brief Length: {info.get('brief_length', 0)} chars\n")
            recommendations = info.get('recommendations', [])
            if recommendations:
                f.write(f"- Recommendations: {', '.join(recommendations[:5])}\n")
            brief_preview = info.get('brief_preview', '')
            if brief_preview:
                f.write(f"- Preview: {brief_preview[:200]}...\n\n")

    boss_to_workers = metrics.prompt_breakdown.get("boss_to_workers", [])
    has_actual_tasks = False
    for task_group in boss_to_workers:
        if isinstance(task_group, dict):
            for worker_id, task_info in task_group.items():
                if isinstance(task_info, dict) and task_info.get('task_length', 0) > 0:
                    has_actual_tasks = True
                    break
        if has_actual_tasks:
            break

    if has_actual_tasks:
        f.write(f"### Boss -> Workers\n\n")
        for i, task_group in enumerate(boss_to_workers, 1):
            if isinstance(task_group, dict):
                f.write(f"**Task Assignment {i}:**\n\n")
                for worker_id, task_info in task_group.items():
                    f.write(f"- **{worker_id}**\n")
                    if isinstance(task_info, dict):
                        task_preview = task_info.get('task_preview', '')
                        if task_preview:
                            preview = task_preview[:5000] + "..." if len(task_preview) > 5000 else task_preview
                            f.write(f"  <details>\n")
                            f.write(f"  <summary>Click to expand task preview</summary>\n\n")
                            f.write(f"  {preview}\n\n")
                            f.write(f"  </details>\n")
                f.write(f"\n")
        f.write(f"\n---\n\n")

    worker_breakdowns = metrics.prompt_breakdown.get("worker_prompt_breakdowns", {})
    if worker_breakdowns:
        f.write(f"### Worker Prompt Breakdowns\n\n")
        f.write(f"Shows the three key components sent to each worker:\n")
        f.write(f"1. **Hardcoded Prompt** (from prompt_engine schema)\n")
        f.write(f"2. **Character/Personality** (from YAML config)\n")
        f.write(f"3. **Boss Guidance** (if Boss injected extra context)\n\n")

        for worker_id, breakdown in worker_breakdowns.items():
            f.write(f"#### {worker_id}\n\n")

            personality_id = breakdown.get("personality_id", "generic")
            f.write(f"**Personality:** {personality_id}\n\n")

            personality_preview = breakdown.get("personality_preview", "")
            if personality_preview:
                f.write(f"**Character/Personality (from YAML):**\n")
                f.write(f"<details><summary>Click to expand</summary>\n\n")
                preview = personality_preview[:10000] + "..." if len(personality_preview) > 10000 else personality_preview
                f.write(f"{preview}\n\n")
                f.write(f"</details>\n\n")

            boss_guidance = breakdown.get("boss_guidance_preview", "")
            if boss_guidance:
                f.write(f"**Boss Guidance (injected by Boss):**\n")
                f.write(f"<details><summary>Click to expand</summary>\n\n")
                preview = boss_guidance[:5000] + "..." if len(boss_guidance) > 5000 else boss_guidance
                f.write(f"{preview}\n\n")
                f.write(f"</details>\n\n")
            else:
                f.write(f"**Boss Guidance:** (none)\n\n")

            f.write(f"---\n\n")


def _write_sources_and_citations(f, response: str, sources: List[str], citations: List[Any]) -> None:
    """Write the sources and citations section to output.md."""
    response_contains_sources = (
        "## Sources" in response or "## Citations" in response
    )

    if not response_contains_sources and sources:
        f.write(f"\n## Sources\n\n")
        for i, source in enumerate(sources, 1):
            f.write(f"{i}. {source}\n")

    if not response_contains_sources and citations:
        f.write(f"\n## Citations\n\n")
        f.write(f"Total citations: {len(citations)}\n\n")
        for i, cit in enumerate(citations[:20], 1):
            source_name = cit.get('source_name', 'Unknown') if isinstance(cit, dict) else getattr(cit, 'source_name', 'Unknown')
            excerpt = cit.get('excerpt', '')[:200] if isinstance(cit, dict) else getattr(cit, 'excerpt', '')[:200]
            url = cit.get('url', '') if isinstance(cit, dict) else getattr(cit, 'url', '')

            f.write(f"### [{i}] {source_name}\n")
            if url:
                f.write(f"**URL:** {url}\n")
            f.write(f"**Excerpt:** {excerpt}...\n\n")

        if len(citations) > 20:
            f.write(f"*... and {len(citations) - 20} more citations*\n\n")
