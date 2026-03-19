"""OSINT Profile - Open Source Intelligence Research.

This profile implements OSINT-specific research with focus on:
- Digital footprint analysis
- Social media investigation
- Domain and infrastructure analysis
- Public records research
"""

import logging
from typing import Any, Dict, Optional

from ....content.deep_research.profiles.base import BaseResearchProfile


logger = logging.getLogger(__name__)


class OsintProfile(BaseResearchProfile):
    """Profile for OSINT (Open Source Intelligence) research.

    This profile provides OSINT-specific behavior on top of the generic
    preset graph provided by BaseResearchProfile.

    Key Features:
    - Focus on verifiable, factual findings with proper source attribution
    - Ethical guidelines for OSINT investigations
    - Multi-source intelligence gathering
    """

    @property
    def name(self) -> str:
        return "osint"

    @property
    def description(self) -> str:
        return "Open Source Intelligence research with focus on digital footprint and public records"

    def enhance_system_prompt(
        self,
        base_prompt: str,
        worker_type: str,
        worker_id: str,
        state: Dict[str, Any]
    ) -> str:
        """Enhance worker prompts with OSINT-specific guidelines.

        Args:
            base_prompt: The generic system prompt
            worker_type: Type of worker (e.g., 'investigator', 'analyst')
            worker_id: Unique worker ID
            state: Current global state

        Returns:
            Enhanced prompt with OSINT guidelines
        """
        osint_guidance = """

OSINT GUIDELINES:
- Only use publicly available information
- Respect privacy and legal boundaries
- Do not attempt unauthorized access
- Document all findings transparently
- Verify information from multiple sources when possible

SOURCE CITATION (CRITICAL):
- web_search/web_reader: Include URL and date accessed
- osint tools: Reference tool name and search parameters
- news_query: Include article source, date, and URL
- vector_store: Reference document title and section

DO NOT cite other investigators without source attribution.
Focus on verifiable, factual findings with clear attribution.
"""

        return base_prompt + osint_guidance

    def process_critic_feedback(self, feedback: str, state: Dict[str, Any]) -> str:
        """Add OSINT-specific quality checks to critic feedback.

        Args:
            feedback: Raw critic feedback
            state: Current global state

        Returns:
            Enhanced feedback with OSINT quality requirements
        """
        osint_quality_check = """

OSINT QUALITY REQUIREMENTS:
- All findings MUST cite actual sources with timestamps
- Findings MUST NOT only cite other investigators
- Acceptable citations: URLs with dates, tool names with parameters, document references
- If NO external sources cited: quality_score MUST be < 0.3
- Verify information freshness and source reliability
"""

        return feedback + osint_quality_check
