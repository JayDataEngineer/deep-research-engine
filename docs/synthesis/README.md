# Synthesis Module

The synthesis module adapts the **content_formatter** pattern for the deep research engine's synthesis stage. It transforms research findings from worker agents into structured, validated outputs using Pydantic schemas and Jinja2 templates.

## Overview

The synthesis stage:
1. Collects results from worker agents
2. Compiles findings with citations
3. Uses an LLM to format content according to a schema
4. Validates output with Pydantic
5. Retries with feedback on validation failure

## Architecture

```
Worker Results → SynthesisWorker → Format Config + Schema → LLM → Validated Output
```

### Key Components

- **SynthesisWorker**: Main worker that orchestrates synthesis
- **Format Configs**: YAML files with prompts and settings (`synthesis/formats/*/config.yaml`)
- **Schemas**: Pydantic models for validation (`synthesis/formats/*/schema.py`)
- **SynthesisNode**: LangGraph node wrapper for workflow integration

## Available Formats

### 1. Research Report (`research_report`)

Comprehensive research report with:
- Executive summary
- Key findings with confidence scores
- Organized sections
- Citations and references

**Schema**: `ResearchReport`
**Best for**: Detailed research outputs, formal reports

### 2. Markdown (`markdown`)

Simple markdown-formatted output with:
- Proper heading hierarchy
- Lists and formatting
- Source attribution

**Schema**: `MarkdownOutput`
**Best for**: Documentation, quick summaries, web content

### 3. Structured JSON (`structured_json`)

Programmatic JSON output with:
- Structured findings
- Metadata
- Source references

**Schema**: `StructuredJSON`
**Best for**: API responses, data pipelines, further processing

## Configuration

### YAML Configuration

```yaml
phases:
  - name: synthesis
    node: SynthesisNode
    config:
      worker_type: synthesis
      output:
        format: research_report  # research_report, markdown, structured_json
        citations:
          enabled: true
          format: numbered  # numbered, APA, MLA, chicago
      
      format_config:
        max_retries: 10
        include_metadata: true
```

### Programmatic Configuration

```python
from deep_research_engine import WorkerConfig, SynthesisWorker

config = WorkerConfig(
    worker_id="synthesis_0",
    worker_type="synthesis",
    task="Synthesize research findings",
    custom={
        "format": "research_report",
        "config": {
            "format": "research_report",
            "citations": {"enabled": True, "format": "numbered"},
            "max_retries": 10
        }
    }
)

worker = SynthesisWorker(config)
result = await worker.execute(state, config)
```

## Creating Custom Formats

### 1. Create Format Directory

```
synthesis/formats/my_format/
├── config.yaml
└── schema.py
```

### 2. Define Schema (`schema.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class MyFormatOutput(BaseModel):
    """My custom output format."""
    
    title: str = Field(description="Document title")
    content: str = Field(description="Main content")
    metadata: Optional[dict] = Field(default=None)
```

### 3. Define Config (`config.yaml`)

```yaml
format:
  name: my_format
  version: 1.0.0
  description: My custom format

model:
  name: qwen-35-27
  enable_thinking: true

max_retries: 10

system_prompt: |
  You are an expert formatter. Transform the findings into my custom format.
  
  **Requirements**:
  - Follow the schema exactly
  - Include all required fields
  
  **Input**:
  {{ findings }}
  
  **Task**:
  Create a formatted output following the schema.
```

### 4. (Optional) Add Extension Prompts (`extend.py`)

```python
"""Extension prompts for my_format."""

def get_prompts(**kwargs):
    """Return additional prompts for this format."""
    return [
        """
        ADDITIONAL INSTRUCTIONS:
        - Special formatting rule 1
        - Special formatting rule 2
        """
    ]
```

## How It Works

### 1. Compilation

Worker results are compiled into text:

```python
findings_text = worker._compile_findings(
    worker_results=[...],
    citations_enabled=True
)
```

### 2. Prompt Rendering

Jinja2 template renders with variables:

```python
rendered_prompt = worker._render_jinja_template(
    system_prompt,
    findings=findings_text,
    worker_count=3,
    citations_enabled=True,
    citations_format="numbered"
)
```

### 3. LLM Invocation

Deep agent generates structured output:

```python
agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt=rendered_prompt
)

result = await agent.ainvoke({...})
```

### 4. Validation & Retry

Pydantic validates with retry logic:

```python
for attempt in range(1, max_retries + 1):
    try:
        formatted_result = await extract_structured_output(...)
        return formatted_result  # Success
    except ValidationError as e:
        # Add error feedback and retry
        current_prompt = _add_retry_feedback(prompt, errors, attempt)
```

## Integration with Workflow

The synthesis node integrates with the LangGraph workflow:

```python
from deep_research_engine.nodes import synthesis_node

# In workflow
state = await synthesis_node(state)
```

The node:
1. Extracts worker results from state
2. Creates synthesis worker
3. Executes synthesis
4. Updates state with `synthesis_output` and `synthesis_metadata`

## Error Handling

### Validation Errors

If output fails Pydantic validation:
1. Error details are extracted
2. Feedback is added to prompt
3. LLM retries with corrections
4. Fails after max retries

### Runtime Errors

Non-validation errors crash immediately (crash-only philosophy).

## Testing

```bash
# Run synthesis tests
uv run pytest tests/unit/test_synthesis.py -v

# Run all unit tests
uv run pytest tests/unit/ -v
```

## Example Output

### Research Report

```json
{
  "title": "AI Trends in 2024",
  "executive_summary": "This report summarizes key AI trends...",
  "key_findings": [
    {
      "finding": "Multimodal models dominate",
      "confidence": 0.9,
      "supporting_evidence": ["Evidence 1", "Evidence 2"],
      "citations": ["1", "2"]
    }
  ],
  "sections": [...],
  "conclusion": "...",
  "citations": [
    {
      "id": "1",
      "title": "AI Trends 2024 Report",
      "url": "https://example.com/report"
    }
  ]
}
```

## Best Practices

1. **Choose the right format**: Match format to use case
2. **Enable citations**: Always attribute sources
3. **Set appropriate retries**: 10 retries for complex formats
4. **Validate schemas**: Test schemas with sample data
5. **Use Jinja2 variables**: Leverage template variables for flexibility

## Reference

- **Content Formatter**: `/home/ubuntu/Documents/programs/text/unified-llm-interface/backend/src/extensions/content/content_formatter`
- **Worker Base**: `src/deep_research_engine/workers/base.py`
- **State Types**: `src/deep_research_engine/state/types.py`
