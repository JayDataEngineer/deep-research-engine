
from pydantic import BaseModel, Field


class Tweet(BaseModel):
    text: str = Field(description="Content of the tweet")

class TwitterThread(BaseModel):
    tweets: list[Tweet] = Field(description="List of tweets in the thread")
