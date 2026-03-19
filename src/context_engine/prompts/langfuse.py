"""LangFuse prompt management.

Loads prompts from LangFuse with local fallbacks for development.
"""

import os
from typing import Optional

from loguru import logger

# LangFuse configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


# =============================================================================
# LOCAL FALLBACK PROMPTS (for development without LangFuse)
# =============================================================================

MAIN_AGENT_PROMPT = """You are the Context Engine Main Agent, a sophisticated research coordinator.

Your job is to build comprehensive context for research queries by:
1. Planning your approach using write_todos
2. Gathering information using available tools
3. Storing findings as memories using write_file
4. Delegating specialized tasks to subagents when needed

## Available Tools

- **write_todos**: Plan and track your research tasks
- **read_file**: Read from the context store
- **write_file**: Write findings to the context store (auto-vectorized!)
- **ls**: List files in the store
- **search_vectors**: Semantic search across all stored context
- **task**: Delegate to specialized subagents

## Memory Structure

All context is stored in namespaces:
- `(session_id, "input")` - Original input documents
- `(session_id, "memories")` - Your gathered context and findings
- `(session_id, "insights")` - Synthesized insights

## Guidelines

1. **Plan first**: Break down the query into focused research tasks
2. **Search before asking**: Use search_vectors to find existing relevant context
3. **Store incrementally**: Write findings as you go, not all at once
4. **Be thorough**: Address all aspects of the query
5. **Cite sources**: When using search results, note where info came from

Current session: {session_id}
"""

CRITIC_AGENT_PROMPT = """You are the Context Engine Critic, a quality evaluator.

Your job is to review context gathered by the Main Agent and provide
structured feedback on completeness and quality.

## Your Role

You have READ-ONLY access:
- **read_file**: Read stored context
- **ls**: List available files
- **search_vectors**: Search context semantically

You CANNOT write or modify any files.

## Evaluation Criteria

1. **Completeness** (0.0-1.0): Does the context address all aspects of the query?
2. **Accuracy**: Are claims supported by sources?
3. **Organization**: Is the context well-structured and navigable?
4. **Depth**: Is there sufficient detail for synthesis?
5. **Relevance**: Is all gathered context relevant to the query?

## Output Format

You MUST respond with valid JSON:
```json
{
  "passed": true/false,
  "score": 0.0-1.0,
  "issues": ["issue1", "issue2", ...],
  "suggestions": ["suggestion1", "suggestion2", ...]
}
```

## Pass Threshold

- Score >= 0.7: PASSED
- Score < 0.7: FAILED (provide specific issues and suggestions)

Be constructive in your feedback. Focus on what's missing or needs improvement.
"""

FLASH_EXPLORER_PROMPT = """You are a Flash Explorer, a fast research specialist.

Your job is to quickly gather information on a specific topic and return findings.

## Guidelines

1. **Focus**: Stay on the assigned topic
2. **Speed**: You have limited tool calls - use them wisely
3. **Concise**: Return findings, not explanations
4. **Cite**: Note sources for your findings

## Output Format

Return your findings in this structure:
```
## Key Findings
- Finding 1 [source]
- Finding 2 [source]

## Sources
[1] Source name: URL
[2] Source name: URL
```

Stop when you have 3-5 solid findings or after 3 searches.
"""


class PromptManager:
    """Manages prompt loading from LangFuse with local fallbacks."""

    def __init__(self, use_langfuse: bool = True):
        """Initialize prompt manager.

        Args:
            use_langfuse: Whether to try LangFuse first
        """
        self.use_langfuse = use_langfuse and bool(LANGFUSE_PUBLIC_KEY)
        self._prompts: dict[str, str] = {}

    def _get_langfuse_prompt(self, name: str, default: str) -> str:
        """Try to get prompt from LangFuse.

        Args:
            name: Prompt name in LangFuse
            default: Fallback prompt if not found

        Returns:
            Prompt string
        """
        if not self.use_langfuse:
            return default

        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
            )

            prompt = client.get_prompt(name)
            logger.info(f"Loaded prompt '{name}' from LangFuse")
            return prompt.prompt

        except Exception as e:
            logger.warning(f"Failed to load prompt '{name}' from LangFuse: {e}")
            return default

    def get_main_agent_prompt(self, session_id: str) -> str:
        """Get main agent system prompt.

        Args:
            session_id: Current session ID

        Returns:
            Formatted system prompt
        """
        if "main_agent" not in self._prompts:
            self._prompts["main_agent"] = self._get_langfuse_prompt(
                "context-engine-main-agent", MAIN_AGENT_PROMPT
            )

        return self._prompts["main_agent"].format(session_id=session_id)

    def get_critic_agent_prompt(self) -> str:
        """Get critic agent system prompt.

        Returns:
            System prompt string
        """
        if "critic_agent" not in self._prompts:
            self._prompts["critic_agent"] = self._get_langfuse_prompt(
                "context-engine-critic-agent", CRITIC_AGENT_PROMPT
            )

        return self._prompts["critic_agent"]

    def get_flash_explorer_prompt(self) -> str:
        """Get flash explorer system prompt.

        Returns:
            System prompt string
        """
        if "flash_explorer" not in self._prompts:
            self._prompts["flash_explorer"] = self._get_langfuse_prompt(
                "context-engine-flash-explorer", FLASH_EXPLORER_PROMPT
            )

        return self._prompts["flash_explorer"]

    def get_all_prompts(self) -> dict[str, str]:
        """Get all prompts as a dict.

        Returns:
            Dict of prompt name -> prompt content
        """
        return {
            "main_agent": MAIN_AGENT_PROMPT,
            "critic_agent": CRITIC_AGENT_PROMPT,
            "flash_explorer": FLASH_EXPLORER_PROMPT,
        }


# Singleton instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager(use_langfuse: bool = True) -> PromptManager:
    """Get or create the prompt manager singleton.

    Args:
        use_langfuse: Whether to use LangFuse

    Returns:
        PromptManager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager(use_langfuse=use_langfuse)
    return _prompt_manager


def load_prompts(use_langfuse: bool = True) -> dict[str, str]:
    """Load all prompts.

    Args:
        use_langfuse: Whether to try LangFuse first

    Returns:
        Dict of prompt name -> prompt content
    """
    manager = get_prompt_manager(use_langfuse)
    return manager.get_all_prompts()
