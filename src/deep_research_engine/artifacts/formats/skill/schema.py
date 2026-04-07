
from pydantic import BaseModel, Field


class Step(BaseModel):
    action: str = Field(description="Action to perform")

class Skill(BaseModel):
    name: str = Field(description="Name of the skill")
    steps: list[Step] = Field(description="Steps to execute the skill")
