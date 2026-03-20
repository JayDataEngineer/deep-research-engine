from pydantic import BaseModel, Field

class CodeSnippet(BaseModel):
    filename: str = Field(description="Name of the file")
    language: str = Field(description="Programming language")
    code: str = Field(description="Source code content")

class CodeProject(BaseModel):
    project_name: str = Field(description="Name of the project")
    snippets: list[CodeSnippet] = Field(description="List of code snippets")
