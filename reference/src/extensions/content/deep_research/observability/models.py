"""Data models and utilities for observability.

Contains:
- WorkerMetrics: Per-worker execution metrics
- RunMetrics: Run-level metrics collection
- Metadata extraction: LLM provider/model info
- Serialization: JSON output for worker outputs and sources
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from ..logging_utils import safe_get_logger
from .errors import raise_context_engine_error

logger = safe_get_logger(__name__, "models")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class WorkerMetrics:
    """Per-worker execution metrics for detailed performance analysis."""
    worker_id: str
    worker_type: str  # "generic", "deep_agent", "tool_calling", or specialized type
    task_description: str = ""

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    execution_time: float = 0.0

    # LLM calls
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Tool usage
    tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)
    tool_stats: Dict[str, int] = field(default_factory=dict)

    # Output quality
    output_length: int = 0
    source_count: int = 0
    confidence: float = 0.0

    # Status
    status: str = "success"  # "success", "error", "timeout"
    error_message: str = ""

    # Performance
    tokens_per_second: float = 0.0
    avg_llm_call_time: float = 0.0

    def calculate_performance_metrics(self) -> None:
        """Calculate derived performance metrics."""
        if self.execution_time > 0 and self.total_tokens > 0:
            self.tokens_per_second = self.total_tokens / self.execution_time
        if self.llm_calls > 0:
            self.avg_llm_call_time = self.execution_time / self.llm_calls


@dataclass
class RunMetrics:
    """Metrics collected during a research run.

    Attributes:
        session_id: Primary identifier for the research session (links all artifacts)
        timestamp: ISO format timestamp when the run started
        config: Configuration dictionary used for this run
        query: The original research query
        use_case: Use case type (rag, soc_media_gen, etc.)
        variant: Configuration variant
        overdrive: Whether overdrive mode was enabled
    """
    session_id: str  # Primary identifier
    timestamp: str
    config: Dict[str, Any] = field(default_factory=dict)
    query: str = ""
    use_case: str = "rag"
    variant: str = "default"
    overdrive: bool = False

    # Backward compatibility alias
    @property
    def run_id(self) -> str:
        """Alias for session_id for backward compatibility."""
        return self.session_id

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    total_time: float = 0.0

    # Phase timing
    worker_time: float = 0.0
    boss_time: float = 0.0
    lab_time: float = 0.0
    lab_wall_time: float = 0.0
    critic_time: float = 0.0
    synthesis_time: float = 0.0

    # Model calls by phase
    worker_calls: int = 0
    boss_calls: int = 0
    worker_calls: int = 0
    critic_calls: int = 0
    synthesis_calls: int = 0
    worker_tool_calls: int = 0

    # Tokens - total
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # Tokens - per phase
    worker_prompt_tokens: int = 0
    worker_completion_tokens: int = 0
    boss_prompt_tokens: int = 0
    boss_completion_tokens: int = 0
    worker_prompt_tokens: int = 0
    worker_completion_tokens: int = 0
    critic_prompt_tokens: int = 0
    critic_completion_tokens: int = 0
    synthesis_prompt_tokens: int = 0
    synthesis_completion_tokens: int = 0

    # Performance
    tokens_per_sec: float = 0.0
    avg_call_time: float = 0.0

    # Quality
    response_length: int = 0
    source_count: int = 0
    iterations: int = 0

    # Workers
    workers_spawned: int = 0
    worker_outputs: List[Dict[str, Any]] = field(default_factory=list)
    worker_metrics: List[WorkerMetrics] = field(default_factory=list)

    # Critique
    quality_score: float = 0.0
    passed_quality_gate: bool = False
    critique_history: List[Dict[str, Any]] = field(default_factory=list)

    # Details
    calls: List[Dict[str, Any]] = field(default_factory=list)

    # Tool call metrics
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    researcher_tool_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Prompt breakdown
    prompt_breakdown: Dict[str, Any] = field(default_factory=dict)
    boss_rationale: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# METADATA EXTRACTION
# =============================================================================

def extract_provider_from_base_url(base_url: str) -> str:
    """Extract human-readable provider name from base URL.

    Args:
        base_url: The base URL of the LLM API

    Returns:
        Human-readable provider name (e.g., "ZAI", "OpenAI", "Anthropic", "Local")
    """
    if not base_url:
        return "unknown"

    url_lower = base_url.lower()

    if "z.ai" in url_lower or "bigmodel" in url_lower:
        return "ZAI"
    elif "api.openai.com" in url_lower or "openai.com" in url_lower:
        return "OpenAI"
    elif "api.anthropic.com" in url_lower or "anthropic.com" in url_lower:
        return "Anthropic"
    elif "localhost" in url_lower or "127.0.0.1" in url_lower:
        return "Local"
    elif "googleapis.com" in url_lower:
        return "Google"
    elif "cohere.com" in url_lower:
        return "Cohere"
    else:
        try:
            parsed = urlparse(base_url)
            domain = parsed.netloc or base_url
            domain = domain.split(":")[0]
            domain = domain.replace("api.", "").replace("www.", "")
            return domain.capitalize()
        except Exception as e:
            logger.debug(f"Failed to parse base_url '{base_url[:50]}...': {e}")
            return "Unknown"


def extract_llm_metadata(llm) -> Dict[str, Any]:
    """Extract model metadata from LLM object for logging.

    Args:
        llm: BaseLLM instance

    Returns:
        Dict with model, provider, temperature, enable_thinking, max_tokens
    """
    if llm is None:
        return {
            "model": "unknown",
            "provider": "unknown",
            "temperature": 0.0,
            "enable_thinking": False,
            "max_tokens": None
        }

    model = getattr(llm, "model", "unknown")
    base_url = getattr(llm, "base_url", "")
    temperature = getattr(llm, "temperature", 0.0)
    enable_thinking = getattr(llm, "enable_thinking", False)
    max_tokens = getattr(llm, "max_tokens", None)

    provider = extract_provider_from_base_url(base_url)

    return {
        "model": model,
        "provider": provider,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "max_tokens": max_tokens
    }


# =============================================================================
# SERIALIZATION
# =============================================================================
# NO COMPROMISE: All output storage goes through context_engine tools
# No direct file I/O - everything is tracked and queryable
# =============================================================================

def save_worker_outputs_json(
    session_id: str,
    worker_outputs: List[Any],
) -> None:
    """Save worker outputs to context_engine storage.

    Uses save_output tool from context_engine registry.
    All outputs are tracked and queryable via context_engine.

    Args:
        session_id: Research session ID (for context_engine isolation)
        worker_outputs: List of worker output objects
    """
    outputs_data = []

    for wo in worker_outputs:
        if isinstance(wo, dict):
            outputs_data.append(wo)
        elif hasattr(wo, "worker_id"):
            # Convert WorkerResult object to dict
            citations_data = []
            if hasattr(wo, "citations"):
                for cit in wo.citations:
                    if isinstance(cit, dict):
                        cit_dict = cit.copy()
                        if "excerpt" in cit_dict and cit_dict["excerpt"]:
                            cit_dict["excerpt"] = cit_dict["excerpt"][:500]
                        citations_data.append(cit_dict)
                    else:
                        citations_data.append({
                            "source_type": cit.source_type,
                            "source_name": cit.source_name,
                            "url": cit.url,
                            "excerpt": cit.excerpt[:500] if cit.excerpt else "",
                            "worker_id": getattr(cit, "worker_id", ""),
                            "metadata": cit.metadata
                        })

            sources_list = []
            if hasattr(wo, "citations"):
                for cit in wo.citations:
                    cit_data = cit if isinstance(cit, dict) else {"url": cit.url, "source_name": cit.source_name}
                    if cit_data.get("url"):
                        sources_list.append(cit_data["url"])
                    elif cit_data.get("source_name"):
                        sources_list.append(cit_data["source_name"])

            worker_type = getattr(wo, "worker_type", None) or getattr(wo, "task_type", "generic")
            task_description = getattr(wo, "task_description", "") or getattr(wo, "metadata", {}).get("task_description", "")

            outputs_data.append({
                "worker_id": wo.worker_id,
                "worker_type": worker_type,
                "task_type": getattr(wo, "task_type", "research"),
                "task_description": task_description,
                "findings": wo.findings,
                "findings_full": getattr(wo, "findings_full", None),
                "confidence": wo.confidence,
                "sources": sources_list,
                "citations": citations_data,
                "citation_count": len(citations_data),
                "source_refs": getattr(wo, "source_refs", []),
                "source_refs_count": len(getattr(wo, "source_refs", [])),
                "summary_tokens": getattr(wo, "summary_tokens", 0),
                "full_tokens": getattr(wo, "full_tokens", 0),
                "manifest": getattr(wo, "manifest", None),
                "artifact_path": getattr(wo, "artifact_path", None),
                "execution_time": getattr(wo, "execution_time", 0.0),
                "timestamp": getattr(wo, "timestamp", "")
            })
        else:
            outputs_data.append({"raw": str(wo)})

    # Use context_engine save_output tool
    try:
        from src.shared.tools.registry import get_tool
        save_output = get_tool("save_output")

        content = json.dumps({
            "session_id": session_id,
            "run_id": session_id,  # Backward compat alias
            "timestamp": datetime.now().isoformat(),
            "workers_count": len(outputs_data),
            "worker_outputs": outputs_data
        }, indent=2)

        # Invoke save_output tool (synchronous wrapper)
        save_output.invoke({
            "session_id": session_id,
            "output_type": "worker_outputs",
            "content": content,
            "filename": "worker_outputs.json",
            "runtime": None
        })

        logger.debug(f"[OUTPUT] Saved {len(outputs_data)} worker outputs to context_engine")

    except Exception as e:
        # NO FALLBACK - context_engine is required
        raise_context_engine_error(
            "save_worker_outputs",
            e,
            f"Failed to save {len(outputs_data)} worker outputs",
            session_id=session_id
        )


def save_raw_sources_json(
    session_id: str,
    raw_sources: Dict[str, Any],
) -> None:
    """Save raw source content to context_engine storage.

    Uses save_output tool from context_engine registry.
    All sources are tracked and queryable via context_engine.

    Args:
        session_id: Research session ID (for context_engine isolation)
        raw_sources: Dictionary of source objects
    """
    sources_data = {}
    tier_counts = {"A": 0, "B": 0, "C": 0}

    for source_id, source_obj in raw_sources.items():
        if isinstance(source_obj, dict):
            sources_data[source_id] = source_obj
            tier = source_obj.get("tier", "C")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        elif hasattr(source_obj, "source_id"):
            source_dict = {
                "source_id": source_obj.source_id,
                "url": source_obj.url,
                "title": source_obj.title,
                "content": source_obj.content[:5000] if source_obj.content else "",
                "snippet": source_obj.snippet,
                "summary": source_obj.summary,
                "tier": source_obj.tier,
                "relevance_score": source_obj.relevance_score,
                "worker_id": source_obj.worker_id,
                "timestamp": source_obj.timestamp,
                "metadata": source_obj.metadata
            }
            sources_data[source_id] = source_dict
            tier_counts[source_obj.tier] += 1

    # Use context_engine save_output tool
    try:
        from src.shared.tools.registry import get_tool
        save_output = get_tool("save_output")

        content = json.dumps({
            "session_id": session_id,
            "run_id": session_id,  # Backward compat alias
            "timestamp": datetime.now().isoformat(),
            "total_sources": len(sources_data),
            "tier_distribution": tier_counts,
            "raw_sources": sources_data
        }, indent=2)

        # Invoke save_output tool (synchronous wrapper)
        save_output.invoke({
            "session_id": session_id,
            "output_type": "raw_sources",
            "content": content,
            "filename": "raw_sources.json",
            "runtime": None
        })

        logger.debug(
            f"[OUTPUT] Saved {len(sources_data)} raw sources to context_engine "
            f"(Tier A: {tier_counts['A']}, Tier B: {tier_counts['B']}, Tier C: {tier_counts['C']})"
        )

    except Exception as e:
        # NO FALLBACK - context_engine is required
        raise_context_engine_error(
            "save_raw_sources",
            e,
            f"Failed to save {len(sources_data)} raw sources",
            session_id=session_id,
            tier_distribution=tier_counts
        )
