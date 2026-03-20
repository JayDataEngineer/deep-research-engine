from typing import List
from pydantic import BaseModel, Field

class Step(BaseModel):
    action: str = Field(description="Action to perform")

class Skill(BaseModel):
    name: str = Field(description="Name of the skill")
    steps: List[Step] = Field(description="Steps to execute the skill")
