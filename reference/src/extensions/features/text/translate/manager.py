"""Translation Services Manager

Provides translation services using the LLM.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any

from src.shared.factory import get_llm

logger = logging.getLogger(__name__)


class TranslateManager:
    """Manages translation services."""

    def __init__(self):
        self.supported_languages = [
            "english", "spanish", "french", "german", "italian", "portuguese",
            "russian", "chinese", "japanese", "korean", "arabic", "hindi",
            "dutch", "swedish", "norwegian", "danish", "finnish", "polish",
            "turkish", "vietnamese", "thai", "indonesian", "filipino",
        ]

    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: str = "auto"
    ) -> str:
        """Translate text to target language.

        Args:
            text: Text to translate
            target_language: Target language
            source_language: Source language (auto-detect if empty)

        Returns:
            Translated text
        """
        llm = await get_llm()

        prompt = f"""Translate the following text to {target_language}.

Text: {text}

Provide only the translation, no explanation."""

        try:
            result = await llm.generate(prompt, system="You are a professional translator.")
            return result.strip()
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return f"Error: {str(e)}"

    async def translate_batch(
        self,
        texts: List[str],
        target_language: str,
        source_language: str = "auto"
    ) -> List[str]:
        """Translate multiple texts.

        Args:
            texts: List of texts to translate
            target_language: Target language
            source_language: Source language

        Returns:
            List of translated texts
        """
        results = []
        for text in texts:
            result = await self.translate(text, target_language, source_language)
            results.append(result)
        return results

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages

    def detect_language(self, text: str) -> str:
        """Detect the language of text (basic implementation).

        Args:
            text: Text to analyze

        Returns:
            Detected language name
        """
        # Basic detection based on character patterns
        # In production, this would use a proper language detection library
        text_sample = text[:100]

        if any(ord(c) > 0x4E00 and ord(c) < 0x9FFF for c in text_sample):
            return "chinese"
        elif any(ord(c) > 0x0400 and ord(c) < 0x052F for c in text_sample):
            return "russian"
        elif any(ord(c) > 0x0600 and ord(c) < 0x06FF for c in text_sample):
            return "arabic"
        elif any(ord(c) > 0x0900 and ord(c) < 0x097F for c in text_sample):
            return "hindi"
        elif any(c in "じゃじょ" for c in text_sample):
            return "japanese"
        elif any(c in "가힣" for c in text_sample):
            return "korean"
        elif any(c in "áéíóúñü" for c in text_sample.lower()):
            return "spanish"
        else:
            return "english"  # Default fallback
