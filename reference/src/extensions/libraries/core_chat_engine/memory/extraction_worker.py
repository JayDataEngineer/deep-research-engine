"""Background Memory Extraction Worker - Async non-blocking memory extraction.

This worker runs in the background to extract and store memories from conversations
without blocking the main chat flow. It uses a separate model that can run in
parallel while the user continues chatting.

Key Features:
- Async execution - doesn't block main conversation
- Parallel model usage - can use smaller local model (e.g., QWEN) for extraction
- Smart memory extraction - LLM-powered extraction of important information
- Automatic deduplication - avoids storing duplicate memories
- Background processing - user unaware, unbothered, unblocked
- Trait evolution integration - memories can trigger trait changes
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ..memory.memory_manager import get_memory_manager
from ..memory.memory_schema import MemoryType, MemoryImportance
from src.shared import get_llm

logger = logging.getLogger(__name__)


# =============================================================================
# Memory Extraction Prompts
# =============================================================================

MEMORY_EXTRACTION_SYSTEM_PROMPT = """You are a MEMORY EXTRACTION SPECIALIST. Your job is to analyze a conversation summary and extract important information that should be remembered long-term.

Extract memories in these categories:
1. **PREFERENCES**: User likes/dislikes (e.g., "User loves coffee, especially lattes")
2. **FACTS**: Important facts about the user (e.g., "User is a software engineer living in Seattle")
3. **GOALS**: User's goals and aspirations (e.g., "User wants to learn Spanish before their Mexico trip")
4. **RELATIONSHIPS**: Important people in user's life (e.g., "User's spouse is named Alex")
5. **EVENTS**: Important life events (e.g., "User got promoted last week")
6. **EPISODES**: Notable conversation moments (e.g., "User was really happy when discussing their hobby")

Rules:
- Extract ONLY information that seems important or recurring
- Ignore trivial/temporary information (e.g., "user said hello")
- Be concise but specific
- Extract 3-5 memories maximum
- If nothing important is found, return empty list

Output JSON format:
{
  "memories": [
    {
      "type": "preference|fact|goal|relationship|event|episode",
      "content": "The memory content",
      "importance": "low|medium|high|critical",
      "keywords": ["keyword1", "keyword2"]
    }
  ],
  "reasoning": "Brief explanation of what was extracted and why"
}
"""


# =============================================================================
# Memory Extraction Worker
# =============================================================================

class MemoryExtractionWorker:
    """Background worker for extracting and storing memories from conversations.

    This worker runs asynchronously to avoid blocking the main chat flow.
    It uses a separate LLM instance for memory extraction, allowing parallel
    processing with smaller local models.

    Example:
        ```python
        worker = MemoryExtractionWorker(user_id="alice")

        # Extract from conversation summary (runs in background)
        await worker.extract_from_conversation(
            conversation_summary="User mentioned they love hiking...",
            thread_id="thread_123"
        )

        # Worker runs in background, doesn't block!
        ```
    """

    def __init__(
        self,
        user_id: str,
        extraction_model: Optional[str] = None,
        extraction_api_key: Optional[str] = None,
        extraction_base_url: Optional[str] = None,
    ):
        """Initialize the memory extraction worker.

        Args:
            user_id: User ID for memory namespacing
            extraction_model: Optional separate model for extraction (defaults to main LLM)
            extraction_api_key: Optional separate API key for extraction model
            extraction_base_url: Optional separate base URL for extraction model
        """
        self.user_id = user_id
        self.extraction_model = extraction_model
        self.extraction_api_key = extraction_api_key
        self.extraction_base_url = extraction_base_url

        # Get memory manager for this user
        self.memory_manager = get_memory_manager(user_id=user_id)

        # Track extraction tasks
        self._active_tasks: List[asyncio.Task] = []
        self._extraction_queue: asyncio.Queue = asyncio.Queue()

        logger.info(f"[MEMORY WORKER] Initialized for user={user_id}")

    async def extract_from_conversation(
        self,
        conversation_summary: str,
        thread_id: Optional[str] = None,
        priority: str = "normal",
    ) -> Dict[str, Any]:
        """Extract memories from a conversation summary.

        This method is async and non-blocking. It runs extraction in the background
        and returns immediately with a task ID for tracking.

        Args:
            conversation_summary: Summary of the conversation to extract from
            thread_id: Optional thread ID for memory tracking
            priority: Task priority ("low", "normal", "high")

        Returns:
            Dict with task_id and status for tracking
        """
        # Create background task
        task = asyncio.create_task(
            self._extract_and_store(conversation_summary, thread_id)
        )

        self._active_tasks.append(task)

        # Return immediately (non-blocking)
        return {
            "task_id": id(task),
            "status": "scheduled",
            "message": "Memory extraction scheduled in background"
        }

    async def _extract_and_store(
        self,
        conversation_summary: str,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Internal method: Extract memories and store them.

        This runs in the background after extract_from_conversation returns.
        """
        start_time = time.time()
        logger.info(f"[MEMORY WORKER] Starting extraction for user={self.user_id}")

        try:
            # Step 1: Extract memories using LLM
            extracted_memories = await self._llm_extract(conversation_summary)

            if not extracted_memories:
                logger.info(f"[MEMORY WORKER] No memories extracted from conversation")
                return {
                    "status": "completed",
                    "memories_stored": 0,
                    "duration": time.time() - start_time,
                }

            # Step 2: Check for duplicates and store new memories
            stored_count = 0
            for memory_data in extracted_memories:
                # Check for similar existing memories
                if not self._is_duplicate(memory_data):
                    # Store the memory
                    memory = self.memory_manager.store_memory(
                        memory_type=MemoryType(memory_data["type"]),
                        content=memory_data["content"],
                        importance=MemoryImportance[memory_data["importance"].upper()],
                        keywords=memory_data.get("keywords", []),
                        thread_id=thread_id,
                    )
                    stored_count += 1
                    logger.info(f"[MEMORY WORKER] Stored memory: {memory.memory_id} - {memory_data['content'][:50]}...")
                else:
                    logger.info(f"[MEMORY WORKER] Skipping duplicate memory: {memory_data['content'][:50]}...")

            logger.info(f"[MEMORY WORKER] Extraction complete: {stored_count} memories stored in {time.time() - start_time:.2f}s")

            return {
                "status": "completed",
                "memories_stored": stored_count,
                "duration": time.time() - start_time,
            }

        except Exception as e:
            logger.error(f"[MEMORY WORKER] Extraction failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "duration": time.time() - start_time,
            }

    async def _llm_extract(self, conversation_summary: str) -> List[Dict[str, Any]]:
        """Use LLM to extract memories from conversation summary.

        This uses a separate LLM call that doesn't block the main conversation.
        """
        try:
            # Get LLM for extraction (can use smaller/faster model)
            llm = await self._get_extraction_llm()

            # Build extraction messages
            messages = [
                {"role": "system", "content": MEMORY_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract important memories from this conversation:\n\n{conversation_summary}"}
            ]

            # Call LLM (non-streaming for extraction)
            response = await llm.ainvoke(messages)
            response_text = response.content.strip()

            logger.debug(f"[MEMORY WORKER] LLM extraction response: {response_text[:200]}...")

            # Parse JSON response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                response_text = json_match.group(0)

            result = json.loads(response_text)
            memories = result.get("memories", [])

            logger.info(f"[MEMORY WORKER] Extracted {len(memories)} candidate memories")
            return memories

        except json.JSONDecodeError as e:
            logger.error(f"[MEMORY WORKER] Failed to parse extraction JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"[MEMORY WORKER] LLM extraction failed: {e}")
            return []

    async def _get_extraction_llm(self):
        """Get LLM instance for memory extraction.

        Can use a separate, smaller/faster model for extraction.
        """
        from src.config import settings

        # Use separate extraction model if configured
        model = self.extraction_model or settings.llm_model

        # Use lower temperature for consistent extraction
        return await get_llm(
            model=model,
            temperature=0.3,
            enable_thinking=False,  # No thinking needed for extraction
        )

    def _is_duplicate(self, memory_data: Dict[str, Any]) -> bool:
        """Check if a memory already exists (avoid duplicates).

        Simple check: look for similar content in recent memories.
        """
        try:
            # Search for similar memories
            content = memory_data["content"]
            results = self.memory_manager.search_memories(
                query=content,
                limit=5,
            )

            # Check if any result is very similar (>80% similarity)
            for result in results:
                if result.relevance_score > 0.8:
                    # Check if content is substantially similar
                    existing_content = result.memory.content.lower()
                    new_content = content.lower()
                    # Simple word overlap check
                    words = set(new_content.split())
                    if words and any(word in existing_content for word in words):
                        return True

            return False

        except Exception as e:
            logger.warning(f"[MEMORY WORKER] Duplicate check failed: {e}")
            return False

    async def cleanup_tasks(self):
        """Wait for all active extraction tasks to complete."""
        if self._active_tasks:
            logger.info(f"[MEMORY WORKER] Waiting for {len(self._active_tasks)} active tasks...")
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()


# =============================================================================
# Global Worker Factory
# =============================================================================

_workers: Dict[str, MemoryExtractionWorker] = {}


def get_memory_worker(user_id: str = "default") -> MemoryExtractionWorker:
    """Get or create a memory extraction worker for a user.

    Args:
        user_id: User identifier

    Returns:
        MemoryExtractionWorker instance
    """
    if user_id not in _workers:
        _workers[user_id] = MemoryExtractionWorker(user_id=user_id)
    return _workers[user_id]


async def extract_memories_background(
    conversation_summary: str,
    user_id: str = "default",
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to extract memories in background.

    This is the main entry point for background memory extraction.
    Can be called from the companion graph after a conversation.

    Args:
        conversation_summary: Summary of the conversation
        user_id: User ID
        thread_id: Optional thread ID

    Returns:
        Dict with task status (non-blocking)
    """
    worker = get_memory_worker(user_id=user_id)
    return await worker.extract_from_conversation(
        conversation_summary=conversation_summary,
        thread_id=thread_id,
    )


async def extract_memories_with_trait_evolution(
    conversation_summary: str,
    user_id: str,
    assistant_id: str,
    interaction_signals: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Extract memories AND evolve companion traits based on conversation.

    This is the enhanced version that:
    1. Extracts and stores memories (background, non-blocking)
    2. Updates companion interaction count
    3. Evolves traits based on interaction signals

    Args:
        conversation_summary: Summary of the conversation
        user_id: User ID
        assistant_id: Companion ID for trait evolution
        interaction_signals: Optional dict of signals like
            {"user_laughed": True, "user_thanked": True, ...}

    Returns:
        Dict with memory extraction status and trait evolution results
    """
    from ..memory.assistant_profile import get_assistant_manager

    # Step 1: Extract memories (background)
    memory_result = await extract_memories_background(
        conversation_summary=conversation_summary,
        user_id=user_id,
        thread_id=assistant_id,  # Use assistant_id for thread isolation
    )

    # Step 2: Update companion interaction count
    manager = get_assistant_manager()
    profile = manager.get(user_id, assistant_id)

    trait_changes = []
    new_stage = None

    if profile:
        # Record interaction (may advance relationship stage)
        new_stage = profile.record_interaction()

        # Step 3: Evolve traits based on interaction signals
        if interaction_signals:
            for signal, occurred in interaction_signals.items():
                if not occurred:
                    continue

                # Map signals to trait evolution patterns
                from ..memory.traits import TraitEvolution
                adjusted = TraitEvolution.evolve_from_interaction(
                    profile, signal, strength=0.05
                )

                for trait_name in adjusted:
                    trait_changes.append({
                        "trait": trait_name,
                        "weight": profile.traits[trait_name].weight,
                    })

        # Save profile
        manager.update(profile)

    return {
        "memory_extraction": memory_result,
        "trait_changes": trait_changes,
        "relationship_stage": new_stage.value if new_stage else None,
    }
