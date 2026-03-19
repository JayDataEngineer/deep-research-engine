"""Structured output schemas for deep research nodes.

This module provides Pydantic schemas for structured output from all
inference nodes (Boss, Critic, Synthesis, Architect).

Usage:
    from deep_research.graph.schemas import (
        BossDecision,
        CriticEvaluation,
        SynthesisOutput,
        ArchitectFindings,
    )
    
    # Use with LangChain structured output
    llm = ChatOpenAI(model="gpt-4")
    structured_llm = llm.with_structured_output(BossDecision)
    result = await structured_llm.ainvoke(messages)
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# =============================================================================
# BOSS DECISION SCHEMA
# =============================================================================

class BossDecision(BaseModel):
    """Boss decision output schema.
    
    The Boss node uses this schema to decide between:
    - spawn_architects: Start another research iteration
    - synthesize: Pass to synthesis (research complete)
    """
    action: Literal["spawn_architects", "synthesize"] = Field(
        description="Decision: 'spawn_architects' or 'synthesize'"
    )
    reasoning: str = Field(
        description="Explanation of the decision and current research state"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this decision (0-1)"
    )
    workers_to_spawn: List[str] = Field(
        default_factory=list,
        description="List of worker names to spawn (if spawning architects)"
    )
    research_brief_summary: Optional[str] = Field(
        default=None,
        description="Summary of current research findings (for next iteration)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "action": "spawn_architects",
                "reasoning": "Research covers philosophical angles but lacks practical examples",
                "confidence": 0.7,
                "workers_to_spawn": ["marketer", "industry_analyst"],
                "research_brief_summary": "Found 3 philosophical perspectives, need practical examples"
            }
        }


# =============================================================================
# CRITIC EVALUATION SCHEMA
# =============================================================================

class CriticEvaluation(BaseModel):
    """Critic evaluation output schema.
    
    The Critic node uses this schema to evaluate research completeness.
    """
    sufficient: bool = Field(
        description="Whether research is sufficient for synthesis"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this evaluation (0-1)"
    )
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall research quality score (0-1)"
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="List of identified gaps or missing information"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Guidance for next research iteration (if not sufficient)"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="List of research strengths identified"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sufficient": False,
                "confidence": 0.8,
                "quality_score": 0.6,
                "gaps": [
                    "Missing industry perspectives",
                    "No recent sources (most are 2+ years old)",
                    "Lacks practical implementation examples"
                ],
                "feedback": "Focus on finding recent industry case studies and practical examples",
                "strengths": [
                    "Strong theoretical foundation",
                    "Good coverage of philosophical perspectives"
                ]
            }
        }


# =============================================================================
# SYNTHESIS OUTPUT SCHEMA
# =============================================================================

class SynthesisCitation(BaseModel):
    """Citation schema for synthesis output."""
    title: str = Field(description="Source title")
    url: str = Field(description="Source URL")
    excerpt: Optional[str] = Field(
        default=None,
        description="Relevant excerpt from source"
    )
    tier: Optional[str] = Field(
        default=None,
        description="Source tier (A/B/C)"
    )


class SynthesisOutput(BaseModel):
    """Synthesis output schema with structured fields.
    
    The Synthesis node uses this schema to produce structured research output.
    """
    title: str = Field(
        description="Concise title for the synthesis"
    )
    summary: str = Field(
        description="Executive summary (2-3 sentences)"
    )
    content: str = Field(
        description="Main synthesis content with sections"
    )
    sections: List[str] = Field(
        default_factory=list,
        description="Content sections with headers"
    )
    citations: List[SynthesisCitation] = Field(
        default_factory=list,
        description="Source citations"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score (0-1)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "AI Safety: Perspectives and Approaches",
                "summary": "AI safety research encompasses alignment, capability, and governance approaches with ongoing debate about priorities.",
                "content": "# AI Safety Overview\n\n## Alignment...\n\n## Capability...\n\n## Governance...",
                "sections": [
                    "# AI Safety Overview",
                    "## Alignment Approaches",
                    "## Capability Approaches", 
                    "## Governance Frameworks",
                    "## Conclusion"
                ],
                "citations": [
                    {
                        "title": "AI Alignment Overview",
                        "url": "https://example.com/alignment",
                        "excerpt": "AI alignment focuses on...",
                        "tier": "A"
                    }
                ],
                "confidence": 0.85
            }
        }


# =============================================================================
# ARCHITECT FINDINGS SCHEMA
# =============================================================================

class ArchitectFindings(BaseModel):
    """Architect findings output schema.
    
    The Architect node uses this schema to report research findings.
    """
    research_angle: str = Field(
        description="The research angle investigated"
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Key findings from this research angle"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source URLs consulted"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in findings (0-1)"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for further research"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "research_angle": "Industry perspectives on AI safety",
                "findings": [
                    "Major tech companies have dedicated AI safety teams",
                    "Industry focuses on practical deployment challenges",
                    "Safety testing is becoming standardized"
                ],
                "sources": [
                    "https://example.com/industry-safety-1",
                    "https://example.com/industry-safety-2"
                ],
                "confidence": 0.75,
                "recommendations": [
                    "Investigate specific safety testing methodologies",
                    "Compare approaches across different companies"
                ]
            }
        }


# =============================================================================
# RESEARCH BRIEF SCHEMA
# =============================================================================

class ResearchBrief(BaseModel):
    """Research brief output schema.
    
    Used by Boss/Architect nodes to summarize research direction.
    """
    query: str = Field(
        description="Original or refined research query"
    )
    approach: str = Field(
        description="Research approach or strategy"
    )
    workers_needed: List[str] = Field(
        default_factory=list,
        description="Worker types needed for this research"
    )
    key_questions: List[str] = Field(
        default_factory=list,
        description="Key questions to answer"
    )
    success_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria for successful research"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the main approaches to AI safety?",
                "approach": "Multi-perspective analysis covering theoretical and practical angles",
                "workers_needed": ["philosopher", "industry_analyst", "policy_expert"],
                "key_questions": [
                    "What are the main theoretical frameworks?",
                    "How is AI safety implemented in practice?",
                    "What policies are being proposed?"
                ],
                "success_criteria": [
                    "Cover at least 3 distinct perspectives",
                    "Include both theoretical and practical sources",
                    "Cite recent sources (within 2 years)"
                ]
            }
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "BossDecision",
    "CriticEvaluation",
    "SynthesisOutput",
    "SynthesisCitation",
    "ArchitectFindings",
    "ResearchBrief",
]
