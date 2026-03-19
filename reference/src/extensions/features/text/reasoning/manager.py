"""Reasoning Display Manager

Stores and retrieves chain-of-thought reasoning data from LLM responses.
Also handles logprob (token probability) storage.

Similar to SillyTavern's reasoning display and logprobs features.
Refactored to use BaseManager infrastructure where applicable.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from src.extensions.libraries.base.manager_base import BaseItem, BaseManager
from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class TokenLogprob(BaseItem):
    """Token probability information."""
    token: str
    logprob: float  # Log probability
    top_logprobs: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "logprob": self.logprob,
            "top_logprobs": self.top_logprobs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenLogprob":
        return cls(**data)


@dataclass
class ReasoningData(BaseItem):
    """Chain-of-thought reasoning data from an LLM response."""
    id: str
    timestamp: str
    model: str
    prompt: str
    response: str
    reasoning: str = ""
    reasoning_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0

    # Token probabilities (optional, can be large)
    logprobs: List[TokenLogprob] = field(default_factory=list)

    # Metadata
    temperature: float = 0.7
    max_tokens: int = 4096
    character_id: Optional[str] = None
    chat_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "model": self.model,
            "prompt": self.prompt,
            "response": self.response,
            "reasoning": self.reasoning,
            "reasoning_tokens": self.reasoning_tokens,
            "response_tokens": self.response_tokens,
            "total_tokens": self.total_tokens,
            "logprobs": [lp.to_dict() for lp in self.logprobs],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "character_id": self.character_id,
            "chat_id": self.chat_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningData":
        logprobs = [
            TokenLogprob.from_dict(lp)
            for lp in data.pop("logprobs", [])
        ]
        return cls(logprobs=logprobs, **data)


class ReasoningManager(BaseManager):
    """Manages reasoning and logprob data storage using BaseManager infrastructure.

    Uses a custom storage structure:
    - reasonings/ - separate JSON file per reasoning entry
    - logprobs/ - separate JSON file per logprob entry
    - index.json - quick lookup index
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the reasoning manager.

        Args:
            base_dir: Base directory for storage
        """
        super().__init__(base_dir=base_dir, feature_name="reasoning", storage_file="index.json")

        # Create subdirectories for the custom storage structure
        self.reasoning_dir = self.base_dir / "reasonings"
        self.logprobs_dir = self.base_dir / "logprobs"
        self.reasoning_dir.mkdir(exist_ok=True)
        self.logprobs_dir.mkdir(exist_ok=True)

        # The index is now loaded into self._items by BaseManager
        # Each item in the index is a dict with metadata

    def _get_item_class(self):
        """ReasoningManager doesn't use item_class for the index."""
        return None

    def _save(self):
        """Save the index (not individual reasoning files)."""
        # Save only the index, not the full reasoning data
        # Individual reasoning files are saved as they're created
        StorageManager.save_json_file(self.storage_file, list(self._items.values()))

    # ========================================================================
    # Storing Reasoning Data
    # ========================================================================

    def store_reasoning(
        self,
        id: str,
        model: str,
        prompt: str,
        response: str,
        reasoning: str = "",
        reasoning_tokens: int = 0,
        response_tokens: int = 0,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        character_id: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> ReasoningData:
        """Store reasoning data.

        Args:
            id: Unique identifier for this response
            model: Model name
            prompt: The prompt sent
            response: The response received
            reasoning: Chain-of-thought reasoning (if available)
            reasoning_tokens: Number of tokens in reasoning
            response_tokens: Number of tokens in response
            temperature: Temperature used
            max_tokens: Max tokens setting
            character_id: Associated character (optional)
            chat_id: Associated chat (optional)

        Returns:
            Created ReasoningData
        """
        data = ReasoningData(
            id=id,
            timestamp=datetime.now().isoformat(),
            model=model,
            prompt=prompt[:10000],  # Truncate for storage
            response=response[:50000],  # Truncate for storage
            reasoning=reasoning[:50000],
            reasoning_tokens=reasoning_tokens,
            response_tokens=response_tokens,
            total_tokens=reasoning_tokens + response_tokens,
            temperature=temperature,
            max_tokens=max_tokens,
            character_id=character_id,
            chat_id=chat_id,
        )

        # Store reasoning data in separate file
        reasoning_file = self.reasoning_dir / f"{id}.json"
        StorageManager.save_json_file(reasoning_file, data.to_dict())

        # Update index
        self._items[id] = {
            "timestamp": data.timestamp,
            "model": model,
            "has_reasoning": bool(reasoning),
            "tokens": data.total_tokens,
            "character_id": character_id,
            "chat_id": chat_id,
        }

        self._save()
        logger.info(f"Stored reasoning data: {id}")
        return data

    def store_logprobs(
        self,
        id: str,
        logprobs: List[TokenLogprob],
    ) -> bool:
        """Store token logprobs for a response.

        Args:
            id: Response ID (should match a reasoning entry)
            logprobs: List of token logprobs

        Returns:
            True if stored
        """
        logprobs_file = self.logprobs_dir / f"{id}.json"
        data = [lp.to_dict() for lp in logprobs]
        StorageManager.save_json_file(logprobs_file, data)

        logger.info(f"Stored logprobs: {id} ({len(logprobs)} tokens)")
        return True

    # ========================================================================
    # Retrieving Reasoning Data
    # ========================================================================

    def get_reasoning(self, id: str) -> Optional[ReasoningData]:
        """Get reasoning data by ID.

        Args:
            id: Response ID

        Returns:
            ReasoningData or None
        """
        reasoning_file = self.reasoning_dir / f"{id}.json"
        if not reasoning_file.exists():
            return None

        data = StorageManager.load_json_file(reasoning_file)
        return ReasoningData.from_dict(data)

    def get_logprobs(self, id: str) -> Optional[List[TokenLogprob]]:
        """Get logprobs for a response.

        Args:
            id: Response ID

        Returns:
            List of TokenLogprob or None
        """
        logprobs_file = self.logprobs_dir / f"{id}.json"
        if not logprobs_file.exists():
            return None

        data = StorageManager.load_json_file(logprobs_file)
        return [TokenLogprob.from_dict(lp) for lp in data]

    def get_full_response(self, id: str) -> Optional[Dict[str, Any]]:
        """Get both reasoning and logprobs for a response.

        Args:
            id: Response ID

        Returns:
            Combined data dict or None
        """
        reasoning = self.get_reasoning(id)
        if not reasoning:
            return None

        logprobs = self.get_logprobs(id)

        return {
            "reasoning": reasoning.to_dict(),
            "logprobs": [lp.to_dict() for lp in logprobs] if logprobs else [],
        }

    # ========================================================================
    # Searching and Listing
    # ========================================================================

    def list_reasonings(
        self,
        limit: int = 50,
        model: Optional[str] = None,
        character_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        has_reasoning: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """List stored reasonings with filters.

        Args:
            limit: Max results
            model: Filter by model
            character_id: Filter by character
            chat_id: Filter by chat
            has_reasoning: Filter by has reasoning content

        Returns:
            List of reasoning metadata
        """
        results = []

        for id, meta in self._items.items():
            # Apply filters
            if model and meta.get("model") != model:
                continue
            if character_id and meta.get("character_id") != character_id:
                continue
            if chat_id and meta.get("chat_id") != chat_id:
                continue
            if has_reasoning is not None and meta.get("has_reasoning") != has_reasoning:
                continue

            results.append({"id": id, **meta})

        # Sort by timestamp (newest first) and limit
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return results[:limit]

    def search_reasonings(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search reasoning content.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching entries
        """
        query_lower = query.lower()
        results = []

        for id in self._items.keys():
            reasoning = self.get_reasoning(id)
            if not reasoning:
                continue

            # Search in reasoning, response, and prompt
            searchable = f"{reasoning.reasoning} {reasoning.response} {reasoning.prompt}".lower()

            if query_lower in searchable:
                results.append({
                    "id": id,
                    "timestamp": reasoning.timestamp,
                    "model": reasoning.model,
                    "preview": reasoning.reasoning[:200] if reasoning.reasoning else reasoning.response[:200],
                })

        return results[:limit]

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get reasoning storage statistics.

        Returns:
            Stats dict
        """
        total_reasonings = len(self._items)
        with_reasoning = sum(1 for m in self._items.values() if m.get("has_reasoning"))
        total_tokens = sum(m.get("tokens", 0) for m in self._items.values())

        # Model breakdown
        models = {}
        for meta in self._items.values():
            model = meta.get("model", "unknown")
            models[model] = models.get(model, 0) + 1

        # Count logprobs files
        logprobs_count = len(list(self.logprobs_dir.glob("*.json")))

        return {
            "total_reasonings": total_reasonings,
            "with_reasoning": with_reasoning,
            "total_tokens": total_tokens,
            "models": models,
            "logprobs_stored": logprobs_count,
        }

    # ========================================================================
    # Management
    # ========================================================================

    def delete_reasoning(self, id: str) -> bool:
        """Delete reasoning data.

        Args:
            id: Response ID

        Returns:
            True if deleted
        """
        # Delete reasoning file
        reasoning_file = self.reasoning_dir / f"{id}.json"
        if reasoning_file.exists():
            reasoning_file.unlink()

        # Delete logprobs file
        logprobs_file = self.logprobs_dir / f"{id}.json"
        if logprobs_file.exists():
            logprobs_file.unlink()

        # Remove from index
        if id in self._items:
            del self._items[id]
            self._save()

        return True

    def clear_old(self, days: int = 30) -> int:
        """Clear reasonings older than specified days.

        Args:
            days: Days threshold

        Returns:
            Number of entries deleted
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)
        cutoff_iso = cutoff.isoformat()

        to_delete = [
            id for id, meta in self._items.items()
            if meta.get("timestamp", "") < cutoff_iso
        ]

        for id in to_delete:
            self.delete_reasoning(id)

        return len(to_delete)
