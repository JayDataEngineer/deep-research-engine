"""Social Archive Worker - Specialist for digesting and learning from social data dumps.

This worker processes the 'big text dump' from ingestion to:
1. Map relationships and the 'social graph'
2. Extract key memories and shared history
3. Create personality profiles for each friend
4. Summarize the 'vibe' and inside jokes
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Any

from ...state import EngineState, WorkerResult
from ...utils.logging import get_loguru_logger
from ..base import BaseWorker, WorkerConfig, WorkerRegistry

logger = get_loguru_logger("worker.social_archive")

@WorkerRegistry.register("social_archive")
class SocialArchiveWorker(BaseWorker):
    """Specialist worker for personal social archives."""

    worker_type = "social_archive"
    description = "Digests social data to learn about friends and history"
    required_tools = ["text_analyzer", "entity_extractor"]

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.extract_profiles = config.custom.get("extract_profiles", True)
        self.map_relationships = config.custom.get("map_relationships", True)
        self.find_memories = config.custom.get("find_memories", True)

    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Execute social archive digestion.

        Args:
            state: Current engine state
            config: Worker configuration

        Returns:
            WorkerResult with social insights
        """
        self._start_timing()

        try:
            # Get the big text dump from context
            big_dump_path = state.get("context", {}).get("big_text_dump")
            if not big_dump_path:
                return self._create_result(
                    success=False,
                    output=None,
                    error="No big text dump found in context. Ingestion may have failed.",
                )

            # Read the dump
            with open(big_dump_path, encoding="utf-8") as f:
                content = f.read()

            # Truncate content for now to avoid hitting context limits, or use a large context model
            # Ideally we'd chunk this if it's truly massive.
            max_chars = 100000  # Reasonable for gpt-4o or similar
            if len(content) > max_chars:
                self._log(f"Truncating social archive from {len(content)} to {max_chars} chars")
                content = content[:max_chars]

            self._log(f"Digesting social archive: {len(content)} characters")

            # 1. Flash Subagent: Overview & Vibe
            vibe_task = self._extract_vibe(state, content)

            # 2. Specialist: Relationship Mapper
            relationships_task = (
                self._map_social_graph(state, content)
                if self.map_relationships
                else asyncio.sleep(0, result=[])
            )

            # 3. Specialist: Personality Profiler
            profiles_task = (
                self._extract_friend_profiles(state, content)
                if self.extract_profiles
                else asyncio.sleep(0, result=[])
            )

            # 4. Specialist: Memory Finder
            memories_task = (
                self._find_key_memories(state, content)
                if self.find_memories
                else asyncio.sleep(0, result=[])
            )

            # Run all in parallel
            vibe, relationships, profiles, memories = await asyncio.gather(
                vibe_task, relationships_task, profiles_task, memories_task
            )

            output = {
                "vibe_summary": vibe,
                "social_graph": relationships,
                "friend_profiles": profiles,
                "key_memories": memories,
                "processed_at": datetime.now().isoformat(),
            }

            self._log(
                f"Digestion complete: Learned about {len(profiles)} friends and {len(memories)} memories."
            )

            return self._create_result(
                success=True,
                output=output,
                output_type="social_archive_insights",
                confidence=0.9,
            )

        except Exception as e:
            self._log(f"Social archive digestion failed: {e}", level="error")
            return self._create_result(
                success=False,
                output=None,
                error=str(e),
            )

    async def _extract_vibe(self, state: EngineState, content: str) -> str:
        """Flash subagent to extract the overall vibe."""
        llm = await self._get_llm(state)
        prompt = (
            "Analyze the following social archive data and describe the overall 'vibe', "
            "common themes, and shared history of the group.\n\n"
            f"Content:\n{content}"
        )
        response = await llm.ainvoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    async def _map_social_graph(self, state: EngineState, content: str) -> list[dict[str, Any]]:
        """Specialist sub-sub agent to map relationships."""
        llm = await self._get_llm(state)
        prompt = (
            "Identify the key people in this social archive and their relationships to each other. "
            "Respond with a JSON list of objects, each with 'person_a', 'person_b', and 'relationship'.\n\n"
            f"Content:\n{content}"
        )
        response = await llm.ainvoke(prompt)
        content_str = response.content if hasattr(response, "content") else str(response)
        return self._parse_json_list(content_str)

    async def _extract_friend_profiles(self, state: EngineState, content: str) -> list[dict[str, Any]]:
        """Specialist sub-sub agent to create personality profiles."""
        llm = await self._get_llm(state)
        prompt = (
            "Based on the interactions in this social archive, create personality profiles for "
            "the main participants. Respond with a JSON list of objects, each with 'name', "
            "'traits' (list of strings), and 'summary'.\n\n"
            f"Content:\n{content}"
        )
        response = await llm.ainvoke(prompt)
        content_str = response.content if hasattr(response, "content") else str(response)
        return self._parse_json_list(content_str)

    async def _find_key_memories(self, state: EngineState, content: str) -> list[dict[str, Any]]:
        """Specialist sub-sub agent to find key moments."""
        llm = await self._get_llm(state)
        prompt = (
            "Extract significant shared memories, events, or inside jokes from this social archive. "
            "Respond with a JSON list of objects, each with 'event' and 'description'.\n\n"
            f"Content:\n{content}"
        )
        response = await llm.ainvoke(prompt)
        content_str = response.content if hasattr(response, "content") else str(response)
        return self._parse_json_list(content_str)

    def _parse_json_list(self, text: str) -> list[dict[str, Any]]:
        """Extract and parse a JSON list from text."""
        try:
            # Try to find JSON list in the text
            match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
            if match:
                return json.loads(match.group())

            # Try to parse the whole thing
            return json.loads(text)
        except Exception as e:
            self._log(f"Failed to parse JSON list from response: {e}", level="warning")
            return []
