"""Token Counter Manager

Manages token counting functionality.
"""

import tiktoken
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class TokenCounterManager:
    """Manages token counting functionality."""

    def __init__(self):
        """Initialize the token counter manager."""
        # Try to use a common tokenizer
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            # Fallback to cl100k_base if model-specific encoder fails
            logger.warning(f"Could not load gpt-3.5-turbo tokenizer: {e}, trying cl100k_base")
            try:
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as e2:
                # If all else fails, use a simple character-based approach
                logger.warning(f"Could not load cl100k_base tokenizer: {e2}, using character-based fallback")
                self.encoder = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                # Fallback to character count if encoding fails
                logger.debug(f"Token encoding failed: {e}, using character fallback")
                return len(text) // 4  # Rough estimate: 1 token ~ 4 characters
        else:
            # Simple fallback: estimate 1 token per 4 characters
            return max(1, len(text) // 4)
    
    def analyze_tokens(self, text: str) -> Dict[str, Any]:
        """
        Analyze token composition.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        token_count = self.count_tokens(text)
        char_count = len(text)
        word_count = len(text.split())
        
        return {
            "token_count": token_count,
            "character_count": char_count,
            "word_count": word_count,
            "avg_chars_per_token": round(char_count / max(token_count, 1), 2),
            "avg_words_per_token": round(word_count / max(token_count, 1), 2)
        }