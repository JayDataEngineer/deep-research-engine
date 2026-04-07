from pydantic import BaseModel, Field


class SubstackArticle(BaseModel):
    title: str = Field(description="Title of the article")
    content: str = Field(description="The article content")
