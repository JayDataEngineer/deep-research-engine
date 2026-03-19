"""
Tokenizer Integration

Accurate token counting for context management.
Supports multiple tokenizers (tiktoken, transformers, etc.).
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class Tokenizer:
    """
    Token wrapper for accurate token counting.

    Supports multiple tokenizer backends:
    - tiktoken (OpenAI)
    - transformers (Hugging Face)
    - Approximate (character-based fallback)
    """

    def __init__(
        self,
        backend: str = "tiktoken",
        model: str = "gpt-4",
        encoding_name: str = None,
    ):
        """
        Initialize tokenizer.

        Args:
            backend: Tokenizer backend ("tiktoken", "transformers", "approximate")
            model: Model name (for tiktoken: "gpt-4", "gpt-3.5-turbo", etc.)
            encoding_name: Specific encoding name (optional)
        """
        self.backend = backend
        self.model = model
        self.encoding_name = encoding_name

        # Initialize tokenizer
        self.tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        """Initialize the tokenizer based on backend."""
        if self.backend == "tiktoken":
            try:
                import tiktoken
                if self.encoding_name:
                    self.tokenizer = tiktoken.get_encoding(self.encoding_name)
                else:
                    self.tokenizer = tiktoken.encoding_for_model(self.model)
            except ImportError:
                self.backend = "approximate"

        elif self.backend == "transformers":
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            except ImportError:
                self.backend = "approximate"

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        if self.backend == "tiktoken" and self.tokenizer:
            return len(self.tokenizer.encode(text))

        elif self.backend == "transformers" and self.tokenizer:
            tokens = self.tokenizer.tokenize(text)
            return len(tokens)

        else:
            # Approximate: ~4 characters per token for English
            return len(text) // 4

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of messages.

        Args:
            messages: List of message dicts with "role" and "content"

        Returns:
            Total token count
        """
        total = 0
        for message in messages:
            content = message.get("content", "")
            # Add role overhead (rough estimate)
            total += self.count_tokens(content) + 4
        return total

    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """
        Truncate text to fit within max_tokens.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens

        Returns:
            Truncated text
        """
        if self.backend == "tiktoken" and self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                return self.tokenizer.decode(tokens)
            return text

        elif self.backend == "transformers" and self.tokenizer:
            # Truncate at character level (approximate)
            chars_per_token = 4
            max_chars = max_tokens * chars_per_token
            if len(text) > max_chars:
                return text[:max_chars] + "..."
            return text

        else:
            # Approximate
            chars_per_token = 4
            max_chars = max_tokens * chars_per_token
            if len(text) > max_chars:
                return text[:max_chars] + "..."
            return text

    def get_token_estimate(self, text: str) -> Dict[str, int]:
        """
        Get detailed token estimate.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with token statistics
        """
        return {
            "tokens": self.count_tokens(text),
            "characters": len(text),
            "chars_per_token": len(text) / max(self.count_tokens(text), 1),
            "estimated": self.backend == "approximate",
        }


class TokenBudget:
    """
    Manage token budget for context.

    Useful for:
    - Lorebook token limits
    - Context window management
    - Prompt budget allocation
    """

    def __init__(
        self,
        total_budget: int = 4096,
        reserved: int = 512,
    ):
        """
        Initialize token budget.

        Args:
            total_budget: Total token budget (context window size)
            reserved: Reserved tokens for system/response
        """
        self.total_budget = total_budget
        self.reserved = reserved
        self.available = total_budget - reserved

        # Budget allocations
        self.allocations: Dict[str, int] = {}

    def allocate(self, category: str, tokens: int) -> bool:
        """
        Allocate tokens to a category.

        Args:
            category: Category name (e.g., "lorebook", "chat_history")
            tokens: Number of tokens to allocate

        Returns:
            True if allocation succeeded, False otherwise
        """
        if self.available >= tokens:
            self.allocations[category] = self.allocations.get(category, 0) + tokens
            self.available -= tokens
            return True
        return False

    def deallocate(self, category: str, tokens: int = None) -> None:
        """
        Deallocate tokens from a category.

        Args:
            category: Category name
            tokens: Number of tokens to deallocate (None = all)
        """
        if tokens is None:
            tokens = self.allocations.get(category, 0)

        if category in self.allocations:
            self.allocations[category] -= tokens
            self.available += tokens

            if self.allocations[category] <= 0:
                del self.allocations[category]

    def get_allocation(self, category: str) -> int:
        """Get token allocation for a category."""
        return self.allocations.get(category, 0)

    def get_remaining(self) -> int:
        """Get remaining available tokens."""
        return self.available

    def reset(self) -> None:
        """Reset allocations."""
        self.allocations = {}
        self.available = self.total_budget - self.reserved


def create_tokenizer(
    backend: str = "tiktoken",
    model: str = "gpt-4",
    encoding_name: str = None,
) -> Tokenizer:
    """Factory function to create Tokenizer."""
    return Tokenizer(
        backend=backend,
        model=model,
        encoding_name=encoding_name,
    )


def create_token_budget(
    total_budget: int = 4096,
    reserved: int = 512,
) -> TokenBudget:
    """Factory function to create TokenBudget."""
    return TokenBudget(
        total_budget=total_budget,
        reserved=reserved,
    )
