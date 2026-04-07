from pydantic import BaseModel, Field


class PodcastScript(BaseModel):
    title: str = Field(description="Title of the podcast script")
    content: str = Field(description="The script content")
