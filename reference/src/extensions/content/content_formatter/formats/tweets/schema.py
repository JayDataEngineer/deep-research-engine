"""Tweet Thread Pydantic Schema."""

from pydantic import BaseModel, Field
from typing import List, Optional


class Tweet(BaseModel):
    """A single tweet in a thread."""

    content: str = Field(
        description="Tweet content (max 280 characters)",
        max_length=280
    )
    hashtags: List[str] = Field(
        default_factory=list,
        description="Hashtags for this tweet"
    )
    media_hint: Optional[str] = Field(
        default=None,
        description="Suggested media type (image, video, None)"
    )


class TweetThread(BaseModel):
    """A thread of tweets."""

    tweets: List[Tweet] = Field(
        description="List of tweets in the thread",
        min_length=1,
        max_length=10
    )
    hook_strength: str = Field(
        description="Assessment of first tweet's hook strength (weak/okay/strong)",
        pattern="^(weak|okay|strong)$"
    )
    call_to_action: str = Field(
        description="Call to action or closing thought",
        max_length=500
    )
    total_char_count: int = Field(
        description="Total character count across all tweets"
    )
