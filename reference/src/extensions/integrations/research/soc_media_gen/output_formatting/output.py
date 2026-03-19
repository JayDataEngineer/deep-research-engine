"""Twitter output formatter for research extension.

Formats LLM responses as Twitter threads with numbered tweets
and break markers for potential parallel processing.

Image prompts use simple <<image="description">> tags that LLMs write naturally.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR TWITTER OUTPUT VALIDATION
# =============================================================================

class Tweet(BaseModel):
    """A single tweet in a Twitter thread."""

    number: int = Field(description="Tweet number (N/X)")
    content: str = Field(description="Tweet text content")
    mentions: Optional[List[str]] = Field(default=None, description="@mentions in this tweet")
    hashtags: Optional[List[str]] = Field(default=None, description="Hashtags in this tweet")


class TwitterOutput(BaseModel):
    """Twitter thread output with numbered tweets."""

    topic: str = Field(description="Thread topic/title")
    tweet_count: int = Field(description="Total number of tweets")
    tweets: List[Tweet] = Field(description="Tweets in order")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# TWITTER FORMATTER
# =============================================================================

class TwitterFormatter:
    """Format output as a Twitter thread with numbered tweets.

    This formatter handles:
    - Twitter thread parsing from markdown format
    - Structured tweet validation
    - Simple <<image="description">> tag extraction
    - Tweet break markers for potential parallel processing
    """

    # Pattern to extract image prompts: <<image="description">>
    IMAGE_PATTERN = re.compile(r'<<image="([^"]*)">>', re.IGNORECASE)

    def format(self, response: str, config: Dict[str, Any]) -> str:
        """Format response as Twitter thread.

        Args:
            response: LLM response text (should be thread format)
            config: Output configuration

        Returns:
            Formatted Twitter thread
        """
        # If validation is enabled, try to parse as structured thread
        if config.get("validate", False):
            # Try to parse as JSON first (structured output)
            try:
                import json
                thread_data = json.loads(response)
                output = TwitterOutput(**thread_data)
                return self._render(output)
            except (json.JSONDecodeError, TypeError):
                # Parse as markdown-style thread
                output = self._parse_thread_format(response, config)
                return self._render(output)
        else:
            # Simple rendering - just add section breaks if needed
            return self._simple_render(response, config)

    def extract_image_prompts(self, text: str) -> List[str]:
        """Extract all <<image="description">> tags from text.

        Args:
            text: Text to extract image prompts from

        Returns:
            List of image prompt descriptions
        """
        return self.IMAGE_PATTERN.findall(text)

    def _parse_thread_format(self, response: str, config: Dict[str, Any]) -> TwitterOutput:
        """Parse markdown-style thread into structured TwitterOutput.

        Expected format:
        1/N [Hook tweet content]

        2/N [Body content]

        3/N [More content]

        Or plain numbered format.
        """
        lines = response.strip().split('\n')
        tweets = []
        current_tweet = []
        current_number = None
        total_tweets = None

        for line in lines:
            # Match tweet number pattern: "1/N", "2/10", etc.
            match = re.match(r'^(\d+)/(\d+)\s*(.*)$', line.strip())
            if match:
                # Save previous tweet
                if current_tweet and current_number is not None:
                    tweets.append(Tweet(
                        number=current_number,
                        content="\n".join(current_tweet).strip()
                    ))

                # Start new tweet
                current_number = int(match.group(1))
                total_tweets = int(match.group(2))
                current_tweet = []
                if match.group(3).strip():
                    current_tweet.append(match.group(3).strip())
            elif current_tweet:
                # Continue current tweet
                current_tweet.append(line.strip())

        # Don't forget the last tweet
        if current_tweet and current_number is not None:
            tweets.append(Tweet(
                number=current_number,
                content="\n".join(current_tweet).strip()
            ))

        # If no tweets parsed, treat entire response as single tweet
        if not tweets:
            tweets = [Tweet(number=1, content=response.strip())]
            total_tweets = 1

        return TwitterOutput(
            topic=config.get("title", "Twitter Thread"),
            tweet_count=total_tweets or len(tweets),
            tweets=tweets
        )

    def _render(self, output: TwitterOutput) -> str:
        """Render validated Twitter output."""
        lines = []

        # Header
        lines.append(f"## Thread: {output.topic}")
        lines.append("")
        lines.append(f"**Tweets:** {output.tweet_count}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Tweets - with unique section break marker for potential parallel processing
        for tweet in output.tweets:
            lines.append("<<<TWEET_BREAK>>>")
            lines.append(f"{tweet.number}/{output.tweet_count}")
            lines.append("")

            # Handle multi-line tweets (preserve line breaks within a tweet)
            for tweet_line in tweet.content.split('\n'):
                lines.append(tweet_line)

            lines.append("")

            # Extract and log any image prompts found
            image_prompts = self.extract_image_prompts(tweet.content)
            if image_prompts:
                lines.append("<<<IMAGE_PROMPTS>>>")
                for prompt in image_prompts:
                    lines.append(f"<<image=\"{prompt}\">>")
                lines.append("")
                lines.append("---")
                lines.append("")

        return "\n".join(lines)

    def _simple_render(self, response: str, config: Dict[str, Any]) -> str:
        """Simple rendering without validation."""
        # Just ensure thread format is preserved
        lines = []
        lines.append(f"## Thread: {config.get('title', 'Twitter Thread')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Add section break markers before each tweet number for potential parallel processing
        for line in response.split('\n'):
            # Match tweet number pattern: "1/N", "2/10", etc. at start of line
            if re.match(r'^\d+/\d+', line.strip()):
                lines.append("<<<TWEET_BREAK>>>")
            lines.append(line)

        return "\n".join(lines)


# =============================================================================
# FORMAT OUTPUT FUNCTION
# =============================================================================

def format_output(response: str, mode: str, config: Dict[str, Any]) -> str:
    """Format output according to mode.

    Args:
        response: LLM response text
        mode: Output mode (twitter, etc.)
        config: Output configuration

    Returns:
        Formatted output
    """
    if mode == "twitter":
        formatter = TwitterFormatter()
        return formatter.format(response, config)
    else:
        # Default: return as-is
        return response
