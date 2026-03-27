"""Flash Agent - Autonomous research subagent with iterative refinement.

Implements the Flash Explorer pattern for autonomous research with:
- Task planning and decomposition
- Parallel specialist deployment
- Gap detection and filling
- Iterative refinement loops
"""

import asyncio
from typing import Any, List, Dict, Optional
from loguru import logger
from deepagents import create_deep_agent, SubAgent


class FlashAgent:
    """Autonomous research agent with Flash Explorer pattern."""
    
    def __init__(
        self,
        name: str,
        task: str,
        model: str = "openai:gpt-4o",
        max_iterations: int = 3,
        specialists: Optional[List[SubAgent]] = None
    ):
        """Initialize Flash Agent.
        
        Args:
            name: Agent identifier
            task: Research task description
            model: LLM model to use
            max_iterations: Max refinement loops
            specialists: Optional list of specialist subagents
        """
        self.name = name
        self.task = task
        self.model = model
        self.max_iterations = max_iterations
        self.specialists = specialists or []
        
    def _build_system_prompt(self) -> str:
        """Build system prompt for the flash agent."""
        return (
            f"You are {self.name}, an autonomous research agent.\n\n"
            f"Your goal: {self.task}\n\n"
            "CAPABILITIES:\n"
            "1. PLAN: Break down complex tasks into subtasks\n"
            "2. DELEGATE: Use specialists for specific tasks\n"
            "3. SYNTHESIZE: Combine findings into coherent output\n"
            "4. ITERATE: Identify gaps and fill them\n\n"
            "WORKFLOW:\n"
            "1. Analyze the task and create a research plan\n"
            "2. Deploy specialists for each subtask\n"
            "3. Review results and identify gaps\n"
            "4. Fill gaps with targeted research\n"
            "5. Synthesize final comprehensive output\n\n"
            "Be thorough, cite sources, and ensure factual accuracy."
        )
    
    async def _identify_gaps(self, result: str) -> List[str]:
        """Identify gaps in the research result.
        
        Args:
            result: Current research output
            
        Returns:
            List of gap descriptions
        """
        gap_agent = create_deep_agent(
            model=self.model,
            system_prompt=(
                "You are a research quality analyst. Identify gaps, missing information, "
                "and areas that need more depth in the provided research output."
            )
        )
        
        gap_prompt = (
            "Analyze this research output and identify specific gaps:\n\n"
            f"{result[:4000]}...\n\n"  # Truncate for context
            "List 3-5 specific gaps that need to be filled. Be concise."
        )
        
        gap_result = await gap_agent.ainvoke({
            "messages": [("user", gap_prompt)]
        })
        
        gaps_text = gap_result["messages"][-1].content
        
        # Parse gaps into list
        gaps = []
        for line in gaps_text.split("\n"):
            line = line.strip()
            if line and len(line) > 10:
                # Remove numbering/bullets
                line = line.lstrip("1234567890.-*•")
                gaps.append(line.strip())
        
        return gaps[:5]  # Limit to 5 gaps
    
    async def _fill_gaps(
        self,
        gaps: List[str],
        existing_result: str
    ) -> str:
        """Fill identified gaps with targeted research.
        
        Args:
            gaps: List of gap descriptions
            existing_result: Current research output
            
        Returns:
            New research addressing gaps
        """
        gap_agent = create_deep_agent(
            model=self.model,
            system_prompt=(
                "You are a research specialist. Fill the identified gaps with "
                "targeted, high-quality research. Provide specific facts and citations."
            ),
            subagents=self.specialists
        )
        
        gap_prompt = (
            "Fill these research gaps with targeted information:\n\n"
            f"Gaps:\n" + "\n".join([f"- {g}" for g in gaps]) + "\n\n"
            f"Existing research:\n{existing_result[:3000]}...\n\n"
            "Provide detailed information to address each gap."
        )
        
        result = await gap_agent.ainvoke({
            "messages": [("user", gap_prompt)]
        })

        return result["messages"][-1].content

    async def _merge_results(self, original: str, new: str) -> str:
        """Merge original and new research results.

        Args:
            original: Original research output
            new: New research filling gaps

        Returns:
            Merged comprehensive output
        """
        merge_agent = create_deep_agent(
            model=self.model,
            system_prompt=(
                "You are an editor. Merge these two research outputs into a single, "
                "coherent, comprehensive report. Remove redundancies and ensure logical flow."
            )
        )

        merge_prompt = (
            "Merge these research outputs into one comprehensive report:\n\n"
            f"Original:\n{original[:3000]}...\n\n"
            f"New research:\n{new[:2000]}...\n\n"
            "Create a unified, well-organized final report."
        )

        result = await merge_agent.ainvoke({
            "messages": [("user", merge_prompt)]
        })

        return result["messages"][-1].content

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research task with autonomous refinement.
        
        Args:
            state: Current state dict
            
        Returns:
            WorkerResult-style dict
        """
        logger.info(f"FlashAgent '{self.name}' starting execution")
        
        # Iteration 1: Initial research
        agent = create_deep_agent(
            model=self.model,
            system_prompt=self._build_system_prompt(),
            subagents=self.specialists,
        )
        
        result = await agent.ainvoke({
            "messages": [("user", self.task)]
        })
        
        current_result = result["messages"][-1].content
        iteration = 0
        
        # Iteration 2+: Gap detection and filling
        while iteration < self.max_iterations - 1:
            iteration += 1
            logger.info(f"FlashAgent '{self.name}' iteration {iteration + 1}/{self.max_iterations}")
            
            # Identify gaps
            gaps = await self._identify_gaps(current_result)
            
            if not gaps:
                logger.info(f"No gaps identified - research complete")
                break
            
            logger.info(f"Identified {len(gaps)} gaps to fill")
            
            # Fill gaps
            gap_result = await self._fill_gaps(gaps, current_result)
            
            # Merge results
            current_result = self._merge_results(current_result, gap_result)
        
        logger.info(f"FlashAgent '{self.name}' completed after {iteration + 1} iterations")
        
        return {
            "success": True,
            "output": current_result,
            "output_type": "markdown",
            "worker_id": f"flash_{self.name}",
            "worker_type": "flash_agent",
            "task": self.task,
            "iterations": iteration + 1
        }


async def create_flash_subagents(
    worker_configs: List[Dict[str, Any]]
) -> List[SubAgent]:
    """Create specialist sub-sub agents from worker configs.
    
    Args:
        worker_configs: List of worker configurations
        
    Returns:
        List of SubAgent specs for deepagents
    """
    specialists = []
    
    for config in worker_configs:
        worker_type = config.get("type")
        if not config.get("enabled", True):
            continue
        
        task = config.get("task", "General research task")
        model = config.get("model", "openai:gpt-4o-mini")
        
        # Create specialist system prompts based on worker type
        specialist_prompts = {
            "web_researcher": (
                "You are a web search specialist. Find credible sources, "
                "verify facts, and extract key information from web content."
            ),
            "chunk_processor": (
                "You are a document processing specialist. Analyze PDF chunks, "
                "extract key information, and summarize content accurately."
            ),
            "multimodal": (
                "You are a multimodal analysis specialist. Analyze images, "
                "audio, and video content to extract meaningful information."
            ),
            "scraper": (
                "You are a web scraping specialist. Extract structured data "
                "from websites while respecting robots.txt and rate limits."
            ),
            "asr_worker": (
                "You are a speech recognition specialist. Transcribe audio "
                "content accurately and identify speakers."
            ),
            "vlm_worker": (
                "You are a vision-language specialist. Analyze images and "
                "generate detailed descriptions."
            ),
            "face_worker": (
                "You are a face recognition specialist. Identify and cluster "
                "facial features across images and videos."
            ),
        }
        
        system_prompt = specialist_prompts.get(
            worker_type,
            f"You are a specialist in {worker_type}. {task}"
        )
        
        specialists.append({
            "name": f"{worker_type}_specialist",
            "description": f"Expert at {worker_type.replace('_', ' ')}. Task: {task}",
            "system_prompt": system_prompt,
            "model": model
        })
    
    return specialists
