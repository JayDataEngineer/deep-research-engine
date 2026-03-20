from pydantic import BaseModel, Field

class LLMsTxt(BaseModel):
    content: str = Field(description="The formatted text for LLMs consumption")
