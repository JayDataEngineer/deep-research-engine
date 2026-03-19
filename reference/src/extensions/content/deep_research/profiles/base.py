"""Base Research Profile - Default implementation.

This module provides the default profile implementation that preserves
the original deep research behavior without any customizations.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
sys.path.insert(0, str(Path(__file__).parent.parent))
from ..logging_utils import get_logger
from langgraph.graph import StateGraph

from ..core.interface import ResearchProfile
from ..graph.state import ResearchBrief
from ..graph.helpers.llm import create_llm_from_config
from ..graph.helpers.prompts import get_prompt_from_config


logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level


class BaseResearchProfile(ResearchProfile):
    """Default profile with no customizations.

    This profile implements all hooks as no-ops, ensuring the core
    orchestrator works with default behavior when no specific profile
    is loaded.

    IMPORTANT: This profile also provides GENERIC preset graph support
    for any use case that defines specialist_configs + worker_graph in YAML.
    This allows fitness_content, substack_research, and other use cases to
    use their custom worker graphs without needing a dedicated profile class.
    """

    @property
    def name(self) -> str:
        return "base"

    def get_custom_topology(self, builder: StateGraph) -> Optional[StateGraph]:
        """No topology changes - use default graph structure."""
        return None

    def enhance_system_prompt(
        self,
        base_prompt: str,
        worker_type: str,
        worker_id: str,
        state: Dict[str, Any]
    ) -> str:
        """No prompt enhancement - use base prompt as-is."""
        return base_prompt

    def process_critic_feedback(self, feedback: str, state: Dict[str, Any]) -> str:
        """No feedback processing - use critic feedback as-is."""
        return feedback

    def get_worker_specialist_prompt(
        self,
        worker_type: str,
        worker_id: str,
        state: Dict[str, Any]
    ) -> Optional[str]:
        """No specialist prompt customization."""
        return None

    # ==========================================================================
    # GENERIC PRESET GRAPH - Works for ALL use cases with specialist_configs
    # ==========================================================================

    async def get_preset_boss_decision(
        self,
        brief: ResearchBrief,
        run_logger=None
    ) -> Optional[Dict[str, Any]]:
        """Generate preset boss decision from YAML config (GENERIC).

        This method provides GENERIC preset graph support for ANY use case
        that defines specialist_configs + worker_graph in their YAML config.

        This allows use cases like fitness_content, substack_research, etc.
        to use custom worker graphs without needing a dedicated profile class.

        Args:
            brief: Research brief from worker phase
            run_logger: Optional run logger for metrics

        Returns:
            Boss decision dict or None if no specialist_configs defined
        """
        logger.info(f"[{self.name.upper()}] get_preset_boss_decision called for profile {self.name}")

        spawner_config = self.spawner_config
        specialist_configs = self.config.get("specialist_configs", {})

        # DEBUG: Log what we found
        if run_logger:
            run_logger._write_to_file(f"[{self.name.upper()}] get_preset_boss_decision called")
            run_logger._write_to_file(f"[{self.name.upper()}] config keys: {list(self.config.keys())[:10]}")
            run_logger._write_to_file(f"[{self.name.upper()}] specialist_configs: {list(specialist_configs.keys()) if specialist_configs else 'None'}")
            run_logger._write_to_file(f"[{self.name.upper()}] spawner_config keys: {list(spawner_config.keys()) if spawner_config else 'None'}")

        # Check if this use case has specialist_configs defined
        if not specialist_configs:
            logger.debug(f"[{self.name.upper()}] No specialist_configs, using LLM boss")
            return None

        # NOTE: Single-worker configs are now supported (e.g., twitter_1worker variant)
        # Previously required minimum 2 workers, but now allow 1 worker for quick testing

        logger.info(f"[{self.name.upper()}] Found {len(specialist_configs)} specialist_configs, spawning workers")

        start_time = time.time()

        # READ FROM CONFIG - Generic for all use cases
        # worker_graph can be at root level OR under spawner_config
        worker_graph = self.config.get("worker_graph") or spawner_config.get("worker_graph", {})
        worker_order = worker_graph.get("worker_order", [])
        dependencies_config = worker_graph.get("dependencies", {})

        # Build worker_id mapping: specialist_name → researcher_0, etc.
        worker_type_to_id = {}
        worker_dependencies = {}
        worker_types = {}
        worker_roles = {}
        worker_tasks = {}

        # Track if any worker needs boss guidance (requires LLM call)
        needs_boss_guidance = False

        for idx, specialist_type in enumerate(worker_order):
            worker_id = f"researcher_{idx}"
            worker_type_to_id[specialist_type] = worker_id

            # Map dependencies from specialist names to worker IDs
            dep_names = dependencies_config.get(specialist_type, [])
            worker_dependencies[worker_id] = [
                worker_type_to_id.get(dep_name) for dep_name in dep_names if dep_name in worker_type_to_id
            ]

            # Store worker type
            worker_types[worker_id] = specialist_type

            # Build task from specialist_config
            specialist_config = specialist_configs.get(specialist_type, {})
            specialist_prompt = specialist_config.get("specialist_prompt", "")
            boss_guided = specialist_config.get("boss_guided", False)

            # Store worker role (default to "output" if not specified)
            # Valid roles: "output" (passed to Critic/Synthesis) or "context_only" (provides context to other workers only)
            worker_role = specialist_config.get("worker_role", "output")
            worker_roles[worker_id] = worker_role

            # Check if this worker needs boss LLM guidance
            if boss_guided:
                needs_boss_guidance = True
                # Placeholder - will be filled by boss LLM
                worker_tasks[worker_id] = {
                    "base_prompt": specialist_prompt,
                    "needs_guidance": True,
                    "worker_type": specialist_type,
                    "dependencies": worker_dependencies[worker_id]
                }
            else:
                # Standard preset task
                # For first worker, include query
                if idx == 0:
                    worker_tasks[worker_id] = f"{specialist_prompt}\n\nQuery: {brief.original_query}"
                else:
                    worker_tasks[worker_id] = specialist_prompt

        # ============================================================================
        # BOSS GUIDED MODE: Generate contextual tasks for boss_guided workers
        # ============================================================================
        if needs_boss_guidance:
            import asyncio

            llm = await create_llm_from_config(self.config, role="spawner")

            # Build prompt for boss to generate contextual tasks
            guided_workers_info = []
            for worker_id, task_data in worker_tasks.items():
                if isinstance(task_data, dict) and task_data.get("needs_guidance"):
                    worker_type = task_data["worker_type"]
                    base_prompt = task_data["base_prompt"]
                    dependencies = task_data.get("dependencies", [])

                    # Find which workers this depends on
                    dep_worker_types = []
                    for dep_id in dependencies:
                        dep_type = worker_types.get(dep_id)
                        if dep_type:
                            dep_worker_types.append(dep_type)

                    guided_workers_info.append({
                        "worker_id": worker_id,
                        "worker_type": worker_type,
                        "base_prompt": base_prompt,
                        "depends_on": dep_worker_types
                    })

            if guided_workers_info and run_logger:
                run_logger._write_to_file(f"[BOSS] Generating contextual guidance for {len(guided_workers_info)} boss_guided workers...")

            # Build workers info for prompt
            workers_info_text = ""
            for i, worker_info in enumerate(guided_workers_info):
                workers_info_text += f"""
{i+1}. Worker: {worker_info['worker_type']}
   Base Role: {worker_info['base_prompt'][:200]}...
   Depends On: {', '.join(worker_info['depends_on']) if worker_info['depends_on'] else 'None'}
"""

            # CRASH-ONLY: brief must exist, no silent fallbacks
            if not brief:
                raise RuntimeError(
                    "[BOSS_GUIDANCE] CRITICAL: brief is None! "
                    "Cannot generate guidance without research brief."
                )

            # Get boss guidance prompt from config (CRASH-ONLY - no fallbacks)
            guidance_prompt = get_prompt_from_config(
                self.config,
                "boss.guidance",
                original_query=brief.original_query,
                complexity=brief.complexity_estimate,
                approach=brief.recommended_approach,
                research_vectors=brief.research_vectors,
            )

            # Add worker-specific guidance requirements
            guidance_prompt += f"""

WORKERS NEEDING CONTEXTUAL GUIDANCE:
{workers_info_text}

Your job: Generate contextual task guidance for EACH worker above.

For each worker, provide:
1. CONTEXTUAL_TASK: A specific research task tailored to the query and this worker's role
2. RESEARCH_FOCUS: What specific angles/aspects they should investigate
3. OUTPUT_FORMAT: What format their findings should take

Respond in JSON format:
{{
  "tasks": [
    {{
      "worker_id": "researcher_0",
      "contextual_task": "...",
      "research_focus": "...",
      "output_format": "..."
    }}
  ]
}}

IMPORTANT:
- The contextual_task should STACK with the worker's base_prompt
- Make tasks SPECIFIC to the query, not generic
- Consider dependencies - workers can build on previous workers' outputs
"""

            try:
                if run_logger:
                    run_logger._write_to_file(f"[BOSS] Calling LLM for boss_guided task generation...")

                # CRASH-ONLY: Use native structured output - NO JSON PARSING
                if not hasattr(llm, 'llm') or not hasattr(llm.llm, 'with_structured_output'):
                    raise RuntimeError(
                        f"LLM {type(llm).__name__} does not support native structured output. "
                        f"Boss guidance requires with_structured_output() for Pydantic validation. "
                        f"JSON parsing fallback has been EXTERMINATED."
                    )

                from pydantic import BaseModel, Field

                class BossGuidance(BaseModel):
                    """Boss guidance for worker tasks."""
                    tasks: List[Dict[str, Any]] = Field(..., description="Worker task guidance")

                # Use structured output
                structured_llm = llm.llm.with_structured_output(BossGuidance)
                guidance_data = await structured_llm.ainvoke([{"role": "system", "content": guidance_prompt}])

                if guidance_data and guidance_data.tasks:
                    # Apply guidance to each worker
                    for task_info in guidance_data.tasks:
                        worker_id = task_info.get("worker_id")
                        if worker_id in worker_tasks:
                            task_data = worker_tasks[worker_id]
                            if isinstance(task_data, dict) and task_data.get("needs_guidance"):
                                base_prompt = task_data["base_prompt"]
                                contextual_task = task_info.get("contextual_task", "")
                                research_focus = task_info.get("research_focus", "")
                                output_format = task_info.get("output_format", "")

                                # Stack: base_prompt + boss_guidance
                                stacked_prompt = f"""{base_prompt}

BOSS GUIDANCE (Contextual Task):
{contextual_task}

Research Focus: {research_focus}
Output Format: {output_format}
"""
                                # Replace the dict with the actual stacked prompt
                                worker_tasks[worker_id] = stacked_prompt

                                if run_logger:
                                    run_logger._write_to_file(f"[BOSS] Generated contextual task for {worker_id} ({task_data['worker_type']})")
                else:
                    if run_logger:
                        run_logger._write_to_file(f"[BOSS] Failed to extract JSON from boss guidance response, using base prompts")

            except Exception as e:
                if run_logger:
                    run_logger._write_to_file(f"[BOSS] Error generating boss guidance: {e}, using base prompts")
                # Fall back to base prompts
                for worker_id, task_data in list(worker_tasks.items()):
                    if isinstance(task_data, dict) and task_data.get("needs_guidance"):
                        worker_tasks[worker_id] = task_data["base_prompt"]

        # Log the preset graph usage
        if run_logger:
            run_logger._write_to_file(f"[BOSS] Using PRESET GRAPH ({self.name}): {len(worker_order)} workers from YAML config")
            run_logger._write_to_file(f"[BOSS] Worker dependencies: {worker_dependencies}")
            run_logger._write_to_file(f"[BOSS] Worker types: {worker_types}")
            run_logger._write_to_file(f"[BOSS] Worker roles: {worker_roles}")

            elapsed = time.time() - start_time
            # Note: log_phase_call is handled by boss/node.py for the guidance LLM
            # We skip logging 0 tokens here to avoid double-logging
            run_logger.log_phase_end("boss", elapsed, details={
                "decision": "preset_from_yaml",
                "workers": len(worker_order),
                "worker_tasks": list(worker_tasks.keys()),
                "graph_type": "preset_from_yaml"
            })
            run_logger.log_boss_to_workers(worker_tasks)
            run_logger.log_worker_spawn(list(worker_tasks.keys()))  # CRITICAL: Log spawned workers!
            run_logger.log_boss_rationale(
                decision="preset_from_yaml",
                num_workers=len(worker_order),
                reasoning=f"Using YAML-defined {self.name} dependency graph with {len(worker_order)} workers: {', '.join(worker_order)}",
                complexity=brief.complexity_estimate if brief else 0.5,
                recommendations=[f"Preset {self.name} graph from YAML", "Config-driven dependencies", f"{len(worker_order)} specialist workers"]
            )

        return {
            "routing_decision": "preset_from_yaml",
            "workers_to_spawn": list(worker_tasks.keys()),
            "worker_tasks": worker_tasks,
            "worker_dependencies": worker_dependencies,
            "worker_types": worker_types,
            "worker_roles": worker_roles,
            "current_phase": "lab",
        }
