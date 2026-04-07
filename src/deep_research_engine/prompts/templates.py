"""Local prompt templates for the Deep Research Engine.

This module provides local fallback templates for when LangFuse is not available
or when the PROMPT_SOURCE environment variable is set to 'local'.
"""

# =============================================================================
# PROMPT TEMPLATES ONLY - No config, no workflow logic
# =============================================================================

PROMPTS = {
    # ==========================================================================
    # BOSS NODE - Research Planning
    # ==========================================================================
    "research-planner": {
        "prompt": """You are an expert research planner. Your task is to analyze a research query and create a structured plan.

**Strategy**: {{ strategy }}

**Constraints**:
- Create between {{ min_workers }} and {{ max_workers }} research tasks
- Each task should be specific and actionable
- Tasks should cover different aspects of the query

**Output Format**:
Return a JSON object with:
{
    "tasks": ["task 1", "task 2", ...],
    "complexity": "simple|medium|complex",
    "reasoning": "Brief explanation of the plan"
}

**Query**: {{ query }}

Create a research plan that will guide worker agents to gather comprehensive information.""",
        "config": {
            "model": "qwen-35-27",
            "temperature": 0.7,
        },
    },

    # ==========================================================================
    # CRITIC NODE - Quality Evaluation
    # ==========================================================================
    "research-evaluator": {
        "prompt": """You are an expert research quality evaluator. Your task is to evaluate the quality of research outputs from worker agents.

**Evaluation Criteria**:
{{ criteria_text }}

**Worker Results**:
{{ results_text }}

**Task**:
Evaluate the research outputs against the criteria. Score each criterion from 0.0 to 1.0.

**Output Format**:
Return a JSON object with:
{
    "criteria_scores": {"completeness": 0.8, "accuracy": 0.7, ...},
    "overall_score": 0.75,
    "issues": ["List of specific problems found"],
    "suggestions": ["List of actionable improvements"]
}

Be objective and thorough in your evaluation.""",
        "config": {
            "model": "qwen-35-27",
            "temperature": 0.7,
        },
    },

    # ==========================================================================
    # SYNTHESIS NODE - Output Formatting
    # ==========================================================================
    "research-report": {
        "prompt": """You are an expert research analyst and technical writer. Your task is to synthesize the provided research findings into a comprehensive, well-structured research report.

**CRITICAL REQUIREMENTS:**

1. **Accuracy First**: Do not add new facts or information not present in the findings.
   - Base everything on the provided worker results
   - If information is uncertain, acknowledge it
   - Do not hallucinate citations or sources

2. **Structure**: Follow the exact schema with all required fields:
   - title: Clear, descriptive title
   - executive_summary: 2-4 paragraph overview
   - key_findings: 3-7 major insights with confidence scores
   - sections: Organized topical sections with markdown content
   - conclusion: Summary and actionable recommendations
   - citations: Complete bibliography

3. **Citations**: Use the citation IDs from the source materials.
   - Reference citations by their ID in section content
   - Ensure all claims are properly attributed
   - Format: [1], [2], etc. for numbered citations

4. **Writing Style**:
   - Professional, objective tone
   - Clear and concise language
   - Logical flow between sections
   - Appropriate technical depth

{% if citations_enabled %}
5. **Citation Format**: {{ citations_format }}
   - Include all sources from the worker results
   - Ensure proper attribution throughout
{% endif %}

**INPUT:**
The following research findings have been gathered from {{ worker_count }} research workers:

{{ findings }}

**YOUR TASK:**
Synthesize these findings into a structured research report following the schema exactly.
Focus on clarity, accuracy, and proper attribution.""",
        "config": {
            "model": "qwen-35-27",
            "temperature": 0.7,
            "enable_thinking": True,
        },
    },

    "markdown-format": {
        "prompt": """You are an expert technical writer specializing in markdown documentation.
Your task is to transform the provided research findings into a clean, well-organized markdown document.

**CRITICAL REQUIREMENTS:**

1. **Markdown Quality**:
   - Use proper markdown syntax (headers, lists, links, code blocks)
   - Maintain consistent heading hierarchy (H1 → H2 → H3)
   - Use appropriate formatting (bold, italic, code) for emphasis
   - Include tables where appropriate for structured data

2. **Structure**:
   - Clear H1 title
   - Executive summary or introduction
   - Organized sections with H2 headers
   - Bullet points and numbered lists for readability
   - Conclusion or key takeaways
   - Sources/references section

3. **Accuracy**:
   - Do not add new facts or information
   - Preserve the meaning from the original findings
   - Attribute information to sources correctly

{% if citations_enabled %}
4. **Citations**:
   - Use {{ citations_format }} format
   - Include inline citations as [1], [2], etc.
   - Add a "Sources" or "References" section at the end
{% endif %}

5. **Readability**:
   - Short paragraphs (3-5 sentences)
   - Clear topic sentences
   - Logical flow between sections
   - Professional, objective tone

**INPUT:**
The following research findings have been gathered from {{ worker_count }} research workers:

{{ findings }}

**YOUR TASK:**
Transform these findings into a polished markdown document following the schema exactly.
Focus on readability, proper formatting, and accurate attribution.""",
        "config": {
            "model": "qwen-35-27",
            "temperature": 0.7,
        },
    },

    "structured-json": {
        "prompt": """You are an expert data analyst specializing in structured data extraction.
Your task is to transform the provided research findings into a clean, structured JSON format.

**CRITICAL REQUIREMENTS:**

1. **Structure**: Follow the exact JSON schema:
   - title: Clear, descriptive title
   - summary: Brief 2-3 sentence overview
   - findings: Array of structured findings with confidence scores
   - sources: Array of all sources with metadata

2. **Findings Quality**:
   - Each finding should have a clear title and description
   - Confidence scores must be between 0.0 and 1.0
   - Include relevant tags for categorization
   - Link findings to their supporting sources

3. **Source Metadata**:
   - Include all available metadata (author, date, etc.)
   - Ensure URLs are valid
   - Use consistent ID format

4. **Accuracy**:
   - Do not add new facts or information
   - Preserve the meaning from original findings
   - Represent confidence levels honestly

{% if citations_enabled %}
5. **Citations**:
   - Link findings to sources by ID
   - Ensure all claims have source attribution
{% endif %}

**INPUT:**
The following research findings have been gathered from {{ worker_count }} research workers:

{{ findings }}

**YOUR TASK:**
Transform these findings into structured JSON following the schema exactly.
Focus on clarity, proper structure, and accurate attribution.""",
        "config": {
            "model": "qwen-35-27",
            "temperature": 0.7,
        },
    },

    # ==========================================================================
    # IDENTITY RESOLUTION - Multimodal Binding
    # ==========================================================================
    "identity-research": {
        "prompt": """You are a master identity resolution researcher. Your task is to cross-reference temporal metadata from audio, visual, and text sources to deduce the real names of participants.

**INPUT DATA**:
- ACOUSTIC DIARIZATION: {{ asr_summary }}
- VISUAL CLUSTERS: {{ visual_summary }}
- CONTEXTUAL TEXT: {{ text_summary }}

**DEDUCTION STEPS**:
1. Identify when specific Speaker IDs are active.
2. Cross-reference with Visual Cluster IDs active at the same time.
3. Analyze contextual text where individuals are addressed by name (e.g., "Hi John") and see which IDs are active immediately after.
4. Bind these IDs to the likely real name based on temporal overlap.

**OUTPUT**:
Provide a detailed report of your findings. Clearly list which Speaker IDs and Visual Cluster IDs map to which real names. Explain your reasoning for each mapping.""",
        "config": {
            "model": "openai:gpt-4o",
            "temperature": 0.0,
        },
    },

    "identity-format": {
        "prompt": """Based on the following research report, extract the identity mappings into a structured format.

**RESEARCH REPORT**:
{{ research_report }}

**REQUIRED OUTPUT**:
A JSON mapping from Speaker IDs (e.g., "Speaker 0") or Visual Cluster IDs (e.g., "Identity_Cluster_1") to their resolved real names.""",
        "config": {
            "model": "openai:gpt-4o-mini",
            "temperature": 0.0,
        },
    },

    "video-batch-analysis": {
        "prompt": """Generate a cohesive narrative for this video segment based on these key frames. Describe the actions, characters, and environment.""",
        "config": {
            "model": "openai:gpt-4o-mini",
            "temperature": 0.0,
        },
    },

    "video-synthesis": {
        "prompt": """Combine the following sequential segment descriptions into a single, cohesive narrative for the entire video:

{{ descriptions }}""",
        "config": {
            "model": "openai:gpt-4o-mini",
            "temperature": 0.3,
        },
    },

    "content-cluster": {
        "prompt": """You are a content clustering specialist. Group the following items into thematic clusters.

**RULES:**
1. Create 3-15 clusters covering all the content.
2. Each cluster has a concise name, a one-sentence summary, a list of suggested labels/tags, and key entities mentioned.
3. Group by topic, person, time period, or activity.
4. Items may belong to multiple clusters (use overlapping indices).
5. Prefer existing topic names when the content clearly fits.

{{ topics_section }}

**OUTPUT FORMAT — respond with ONLY valid JSON:**
```json
{
  "clusters": [
    {
      "name": "...",
      "summary": "...",
      "item_indices": [0, 5],
      "suggested_labels": ["..."],
      "key_entities": ["..."]
    }
  ]
}
```

**SOURCE TEXT (items separated by ---):**

{{ combined }}""",
        "config": {
            "model": "qwen-35-27",
            "temperature": 0.3,
        },
    },
}
