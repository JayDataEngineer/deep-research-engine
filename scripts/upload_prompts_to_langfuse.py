#!/usr/bin/env python3
"""Upload prompt templates to LangFuse.

This script uploads ONLY prompt templates to LangFuse.
Application config (models, workers, tools) stays in YAML files.

Usage:
    # Upload all prompts
    uv run python scripts/upload_prompts_to_langfuse.py
    
    # Upload specific prompt
    uv run python scripts/upload_prompts_to_langfuse.py --prompt research-planner
    
    # Dry run
    uv run python scripts/upload_prompts_to_langfuse.py --dry-run

Environment:
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=http://langfuse.local
"""

import argparse
import os
import sys
from typing import Any

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
}


# =============================================================================
# UPLOAD LOGIC
# =============================================================================


def get_langfuse_client() -> Any:
    """Initialize LangFuse client from environment."""
    from langfuse import Langfuse

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://langfuse.local")

    if not public_key or not secret_key:
        print("❌ Error: LangFuse credentials not found")
        print("\nSet the following environment variables:")
        print("  export LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("  export LANGFUSE_SECRET_KEY=sk-lf-...")
        print("  export LANGFUSE_HOST=http://langfuse.local  # Optional")
        sys.exit(1)

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )


def upload_prompt(
    client: Any, name: str, config: dict[str, Any], dry_run: bool = False
) -> bool:
    """Upload a single prompt to LangFuse."""
    prompt_text = config["prompt"]
    prompt_config = config.get("config", {})

    if dry_run:
        print(f"\n📋 Would upload prompt: {name}")
        print(f"   Model: {prompt_config.get('model', 'default')}")
        print(f"   Preview: {prompt_text[:150]}...")
        return True

    try:
        client.create_prompt(
            name=name,
            prompt=prompt_text,
            config=prompt_config,
            labels=["production"],
        )
        print(f"✅ Uploaded prompt: {name}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload prompt '{name}': {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Upload prompts to LangFuse")
    parser.add_argument(
        "--prompt", "-p",
        choices=list(PROMPTS.keys()),
        help="Upload specific prompt (default: all)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all prompts and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Prompts to upload:")
        for name in PROMPTS:
            print(f"  - {name}")
        sys.exit(0)

    client = get_langfuse_client()

    # Test connection
    try:
        health = client.client._client.health.get()
        print(f"✓ Connected to LangFuse: {health}")
    except Exception as e:
        print(f"⚠️  Could not verify LangFuse connection: {e}")

    prompts_to_upload = {args.prompt: PROMPTS[args.prompt]} if args.prompt else PROMPTS

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Uploading {len(prompts_to_upload)} prompt(s)...\n")

    success_count = sum(
        upload_prompt(client, name, config, dry_run=args.dry_run)
        for name, config in prompts_to_upload.items()
    )

    print(f"\n{'=' * 50}")
    print(f"Summary: {success_count}/{len(prompts_to_upload)} successful")

    if success_count == len(prompts_to_upload):
        print(f"\n🎉 All prompts {'would be' if args.dry_run else ''} uploaded!")
        if not args.dry_run:
            print(f"\nView prompts at: {os.getenv('LANGFUSE_HOST', 'http://langfuse.local')}/prompts")
    else:
        print(f"\n⚠️  Some prompts failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
