"""Lorebook Vector Engine - Semantic activation using embeddings.

Activates lorebook entries based on semantic similarity rather than
just keyword matching.

This requires an embedding model to be available.
"""

import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class VectorActivator:
    """Semantic activation engine using vector embeddings.

    Uses embedding models to find semantically similar entries
    based on conversation context.
    """

    def __init__(
        self,
        embedding_model=None,
        threshold: float = 0.7,
        max_results: int = 10
    ):
        """Initialize the vector activator.

        Args:
            embedding_model: Embedding model (e.g., from sentence_transformers)
            threshold: Similarity threshold for activation
            max_results: Maximum entries to activate
        """
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.max_results = max_results
        self._entry_embeddings = {}

    def index_entries(self, entries: List) -> None:
        """Build embedding index for entries.

        Args:
            entries: List of lorebook entries to index
        """
        if not self.embedding_model:
            logger.warning("No embedding model available, skipping vector indexing")
            return

        for entry in entries:
            entry_id = getattr(entry, 'id', id(entry))
            content = getattr(entry, 'content', '')

            # Create embedding
            try:
                embedding = self.embedding_model.encode(content)
                self._entry_embeddings[entry_id] = embedding
            except Exception as e:
                logger.error(f"Failed to embed entry {entry_id}: {e}")

        logger.info(f"Indexed {len(self._entry_embeddings)} entries for vector search")

    def activate(
        self,
        entries: List,
        query_text: str,
        top_k: int = None
    ) -> 'ActivationResult':
        """Activate entries based on semantic similarity.

        Args:
            entries: List of lorebook entries
            query_text: Conversation text to find similar entries
            top_k: Maximum entries to activate

        Returns:
            ActivationResult with semantically similar entries
        """
        from .result import ActivationResult
        from .trigger import KeywordMatcher

        result = ActivationResult()

        if not self.embedding_model:
            return result

        top_k = top_k or self.max_results

        # Encode query
        try:
            query_embedding = self.embedding_model.encode(query_text)
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            return result

        # Calculate similarities
        similarities = []
        for entry in entries:
            entry_id = getattr(entry, 'id', id(entry))

            if entry_id not in self._entry_embeddings:
                continue

            entry_embedding = self._entry_embeddings[entry_id]

            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, entry_embedding)

            if similarity >= self.threshold:
                similarities.append((similarity, entry))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Activate top-k
        for similarity, entry in similarities[:top_k]:
            content = getattr(entry, 'content', '')
            position = getattr(entry, 'position', 'after_char')
            result.add_entry(entry, content, position)

        return result

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between -1 and 1
        """
        import numpy as np

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
