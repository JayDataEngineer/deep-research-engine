# Deep Research Engine - Complete Implementation Plan

**Generated**: March 27, 2026  
**Based On**: `MISSING_FEATURES_REPORT.md`  
**Status**: Ready for Implementation

---

## Overview

This document provides **detailed, actionable implementation plans** for all 11 missing features identified in the audit. Each feature includes:

- Technical specifications
- Step-by-step implementation tasks
- Code templates
- Acceptance criteria
- Testing requirements

---

## Phase 1: Critical Fixes (Week 1-2) 🚨

### Feature 1: Critic Node Logic

**Priority**: P0 CRITICAL  
**File**: `src/deep_research_engine/nodes/critic.py`  
**Lines**: 29-35 (current placeholder)  
**Estimated Effort**: 4-6 hours

#### Problem Statement

The critic node currently uses a **hardcoded score of 0.8**, which always passes the quality gate. This defeats the entire purpose of having a critic phase in the 5-node pipeline.

#### Technical Specifications

**Input**:
- `state.worker_results` - List of worker outputs to evaluate
- `state.plan` - Original research plan/tasks
- `config.critic.criteria` - Evaluation criteria from config

**Output**:
```python
{
    "__goto__": "neo4j_builder" | "drafter",
    "critic_passed": bool,
    "critique": {
        "passed": bool,
        "score": float,  # 0-1
        "criteria_scores": {
            "completeness": float,
            "accuracy": float,
            "citation_quality": float,
            "coherence": float
        },
        "issues": list[str],
        "suggestions": list[str],
        "recommendation": "pass" | "revise" | "terminate"
    }
}
```

#### Implementation Tasks

##### Task 1.1: Define Critic Criteria Schema

**File**: `src/deep_research_engine/state/types.py`

```python
from pydantic import BaseModel, Field
from typing import Literal

class CriticEvaluation(BaseModel):
    """Structured critic evaluation output."""
    
    # Decision
    passed: bool = Field(description="Overall pass/fail decision")
    score: float = Field(ge=0, le=1, description="Overall quality score 0-1")
    
    # Per-criteria scores
    completeness: float = Field(ge=0, le=1, description="How complete is the output?")
    accuracy: float = Field(ge=0, le=1, description="Factual accuracy score")
    citation_quality: float = Field(ge=0, le=1, description="Quality of citations/sources")
    coherence: float = Field(ge=0, le=1, description="Logical flow and organization")
    
    # Qualitative feedback
    issues: list[str] = Field(default_factory=list, description="Specific problems found")
    suggestions: list[str] = Field(default_factory=list, description="Actionable improvements")
    
    # Routing
    recommendation: Literal["pass", "revise", "terminate"] = Field(
        description="Routing recommendation"
    )
    feedback_to: Literal["drafter", "workers", "artifact_maker"] = Field(
        description="Which node should receive feedback"
    )
```

##### Task 1.2: Implement LLM-Based Evaluation

**File**: `src/deep_research_engine/nodes/critic.py`

```python
from pydantic import BaseModel
from ..state import CriticEvaluation
from ..config import get_phase_config
from deepagents import create_deep_agent

@node_context("critic")
async def critic_node(state: EngineState) -> dict[str, Any]:
    """Critic node - Evaluates draft quality using LLM."""
    logger.info("Starting critic node")
    
    # 1. Get config
    config_data = state.get("context", {}).get("config", {})
    phase_config = get_phase_config(config_data, "critic")
    
    if not phase_config.get("enabled", True):
        logger.info("Critic node disabled. Skipping.")
        return {"__goto__": "neo4j_builder"}
    
    # 2. Gather evaluation context
    worker_results = state.get("worker_results", [])
    plan = state.get("plan", [])
    
    if not worker_results:
        logger.warning("No worker results to evaluate")
        return {"__goto__": "neo4j_builder", "critic_passed": True}
    
    # 3. Build evaluation prompt
    eval_prompt = build_critic_prompt(worker_results, plan, phase_config)
    
    # 4. Invoke critic agent with structured output
    critic_agent = create_deep_agent(
        model=phase_config.get("model", "openai:gpt-4o"),
        system_prompt=(
            "You are a rigorous academic critic. Your job is to evaluate "
            "research output quality across multiple dimensions. Be harsh but fair. "
            "Provide specific, actionable feedback."
        ),
    )
    
    # Use structured output for consistent evaluation
    structured_llm = critic_agent.with_structured_output(CriticEvaluation)
    
    result = await structured_llm.ainvoke([{"role": "user", "content": eval_prompt}])
    
    # 5. Parse evaluation
    if isinstance(result, CriticEvaluation):
        evaluation = result
    else:
        # Fallback if structured output fails
        evaluation = parse_fallback_evaluation(result)
    
    # 6. Count iterations
    iteration = len([p for p in state.get("phase_history", []) if p == "critic"])
    if not any(p == "critic" for p in state.get("phase_history", [])):
        iteration += 1
    
    max_retries = phase_config.get("max_retries", 3)
    
    # 7. Make routing decision
    if evaluation.passed or iteration >= max_retries:
        logger.info(f"Critic passed (score={evaluation.score:.2f}) or max retries reached")
        return {
            "__goto__": "neo4j_builder",
            "critic_passed": True,
            "critique": evaluation.model_dump()
        }
    else:
        logger.info(f"Critic failed (score={evaluation.score:.2f}). Retrying drafter.")
        return {
            "__goto__": "drafter",
            "critic_passed": False,
            "critique": evaluation.model_dump()
        }
```

##### Task 1.3: Build Critic Prompt Template

**File**: `src/deep_research_engine/nodes/critic.py`

```python
def build_critic_prompt(
    worker_results: list[dict],
    plan: list[str],
    config: dict
) -> str:
    """Build the critic evaluation prompt."""
    
    # Format worker results
    results_text = "\n\n".join([
        f"### Worker: {r.get('worker_type', 'unknown')}\n"
        f"Task: {r.get('task', 'N/A')}\n"
        f"Output: {r.get('output', '')[:500]}...\n"  # Truncate for context
        f"Confidence: {r.get('confidence', 0):.2f}\n"
        f"Sources: {r.get('source_count', 0)}"
        for r in worker_results
    ])
    
    # Format original plan
    plan_text = "\n".join([f"- {task}" for task in plan]) if plan else "No plan provided"
    
    # Get custom criteria from config
    criteria = config.get("criteria", {
        "completeness": "All planned tasks are addressed",
        "accuracy": "Claims are factually correct and verifiable",
        "citation_quality": "Sources are credible and properly cited",
        "coherence": "Output is well-organized and logically structured"
    })
    
    criteria_text = "\n".join([
        f"- **{k}**: {v}" for k, v in criteria.items()
    ])
    
    prompt = f"""You are evaluating research output quality.

## Original Plan
{plan_text}

## Worker Results
{results_text}

## Evaluation Criteria
{criteria_text}

## Instructions
Evaluate the worker results against the criteria. For each criterion:
1. Score from 0-1 (0=poor, 1=excellent)
2. Provide specific evidence for your score
3. List concrete issues found
4. Suggest actionable improvements

Be rigorous. A score of 0.7+ is required to pass.

Output your evaluation in the structured format provided."""
    
    return prompt
```

##### Task 1.4: Add Config Schema for Critic

**File**: `src/deep_research_engine/config.py`

```python
class CriticConfig(TypedDict, total=False):
    """Critic node configuration."""
    enabled: bool
    model: str
    max_retries: int
    threshold: float  # Pass threshold (default 0.7)
    criteria: dict[str, str]  # Custom evaluation criteria
```

#### Acceptance Criteria

- [ ] Critic uses LLM for evaluation (not hardcoded)
- [ ] Structured output with per-criteria scores
- [ ] Specific, actionable feedback generated
- [ ] Routing works correctly (pass → neo4j, fail → drafter)
- [ ] Max retries enforced
- [ ] Configurable criteria per workflow

#### Testing Requirements

```python
# tests/integration/test_critic_node.py
async def test_critic_passes_good_output():
    """Critic should pass high-quality output."""
    state = create_test_state(worker_results=[...], plan=[...])
    result = await critic_node(state)
    assert result["critic_passed"] == True
    assert result["__goto__"] == "neo4j_builder"

async def test_critic_fails_poor_output():
    """Critic should fail low-quality output."""
    state = create_test_state(worker_results=[...], plan=[...])
    result = await critic_node(state)
    assert result["critic_passed"] == False
    assert result["__goto__"] == "drafter"

async def test_critic_respects_max_retries():
    """Critic should pass after max retries even if poor."""
    state = create_test_state(phase_history=["critic"] * 3)
    result = await critic_node(state)
    assert result["critic_passed"] == True  # Forced pass
```

---

### Feature 2: Video Audio Extraction for ASR

**Priority**: P0 CRITICAL  
**File**: `src/deep_research_engine/utils/multimodal.py`  
**Line**: 346 (TODO comment)  
**Estimated Effort**: 2-3 hours

#### Problem Statement

Video files in Telegram exports cannot be transcribed because audio extraction is not implemented. Only VLM visual descriptions are generated, missing all spoken content.

#### Technical Specifications

**Input**: Video file (`.mp4`, `.mkv`, `.avi`, `.mov`, etc.)  
**Output**: Audio file (16kHz mono WAV for ASR compatibility)

**Requirements**:
- Extract audio track from video
- Convert to 16kHz sample rate
- Convert to mono channel
- Output as PCM 16-bit WAV (VibeVoice compatible)
- Preserve metadata (source video path, duration)

#### Implementation Tasks

##### Task 2.1: Implement Audio Extraction Function

**File**: `src/deep_research_engine/utils/multimodal.py`

```python
import subprocess
import tempfile
from pathlib import Path

async def extract_audio_from_video(
    video_path: Path,
    output_path: Path | None = None,
    sample_rate: int = 16000,
    channels: int = 1
) -> Path:
    """Extract audio track from video and convert to ASR-compatible format.
    
    Args:
        video_path: Path to input video file
        output_path: Optional output path (default: same name with .wav extension)
        sample_rate: Target sample rate in Hz (default: 16kHz for ASR)
        channels: Number of audio channels (default: 1 for mono)
    
    Returns:
        Path to extracted audio file
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        subprocess.CalledProcessError: If FFmpeg fails
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command for audio extraction
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", str(video_path),  # Input video
        "-vn",  # No video output
        "-acodec", "pcm_s16le",  # PCM 16-bit codec
        "-ar", str(sample_rate),  # Sample rate
        "-ac", str(channels),  # Channels (mono)
        str(output_path)  # Output file
    ]
    
    logger.info(f"Extracting audio from {video_path.name}...")
    
    # Run FFmpeg
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
        logger.error(f"Audio extraction failed: {error_msg}")
        raise subprocess.CalledProcessError(
            process.returncode,
            cmd,
            stdout,
            stderr
        )
    
    # Verify output
    if not output_path.exists():
        raise RuntimeError(f"Audio extraction failed: output file not created")
    
    logger.info(f"Audio extracted: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    
    return output_path


async def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using FFprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, _ = await process.communicate()
    
    if process.returncode != 0:
        return 0.0
    
    try:
        return float(stdout.decode().strip())
    except ValueError:
        return 0.0
```

##### Task 2.2: Integrate with Ingestion Pipeline

**File**: `src/deep_research_engine/nodes/ingestion.py`

```python
# In the video processing section, after frame extraction:

# Extract audio for ASR
video_duration = await get_video_duration(video_path)
if video_duration > 0:
    audio_output = output_dir / f"{video_path.stem}_audio.wav"
    try:
        extracted_audio = await extract_audio_from_video(video_path, audio_output)
        
        # Add to audio fragments for Mega-File processing
        audio_fragments.append({
            "path": extracted_audio,
            "source": "video",
            "source_video": video_path,
            "duration": video_duration,
            "timestamp": video_metadata.get("timestamp")
        })
        logger.info(f"Audio extracted from video: {extracted_audio}")
    except Exception as e:
        logger.warning(f"Audio extraction failed for {video_path}: {e}")
        # Continue with VLM-only processing
```

##### Task 2.3: Add Metadata Tracking

**File**: `src/deep_research_engine/utils/pipeline.py`

```python
class AudioMetadata(TypedDict, total=False):
    """Metadata for extracted audio."""
    source_type: Literal["video", "voice_note", "audio_file"]
    source_path: str  # Original video/audio path
    extracted_path: str  # Extracted WAV path
    duration: float  # Duration in seconds
    sample_rate: int
    channels: int
    timestamp: str | None  # Original timestamp from message
    speaker: str | None  # Speaker ID if known


def track_audio_metadata(
    extracted_path: Path,
    source_path: Path,
    duration: float,
    timestamp: str | None = None
) -> AudioMetadata:
    """Create metadata record for extracted audio."""
    return {
        "source_type": "video" if source_path.suffix in [".mp4", ".mkv", ".avi"] else "audio",
        "source_path": str(source_path),
        "extracted_path": str(extracted_path),
        "duration": duration,
        "sample_rate": 16000,
        "channels": 1,
        "timestamp": timestamp
    }
```

#### Acceptance Criteria

- [ ] `extract_audio_from_video()` function implemented
- [ ] FFmpeg audio extraction works for MP4, MKV, AVI, MOV
- [ ] Output is 16kHz mono WAV
- [ ] Audio added to Mega-File pipeline
- [ ] Metadata tracked for source tracing
- [ ] Graceful fallback if extraction fails

#### Testing Requirements

```python
# tests/unit/test_multimodal.py
async def test_extract_audio_from_video():
    """Test audio extraction from video."""
    video_path = Path("test_data/sample.mp4")
    output_path = await extract_audio_from_video(video_path)
    
    assert output_path.exists()
    assert output_path.suffix == ".wav"
    
    # Verify audio properties
    import soundfile as sf
    data, sr = sf.read(output_path)
    assert sr == 16000
    assert len(data.shape) == 1  # Mono

async def test_get_video_duration():
    """Test video duration extraction."""
    duration = await get_video_duration(Path("test_data/sample.mp4"))
    assert duration > 0
```

---

## Phase 2: Core Functionality (Week 2-4) 🔧

### Feature 3: Flash Subagent Orchestration

**Priority**: P1 HIGH  
**File**: `src/deep_research_engine/nodes/drafter.py`  
**Lines**: 1-147  
**Estimated Effort**: 8-12 hours

#### Problem Statement

Current drafter node has basic subagent orchestration but lacks the Flash Explorer pattern's autonomous research loops, specialist fan-out, and iterative refinement.

#### Implementation Plan

##### Task 3.1: Define Flash Agent Pattern

**File**: `src/deep_research_engine/agents/flash.py` (new file)

```python
"""Flash Agent - Autonomous research subagent with iterative refinement."""

from typing import Any
from deepagents import create_deep_agent
from ..state import WorkerResult

class FlashAgent:
    """Autonomous research agent with Flash Explorer pattern."""
    
    def __init__(
        self,
        name: str,
        task: str,
        model: str = "openai:gpt-4o",
        max_iterations: int = 3
    ):
        self.name = name
        self.task = task
        self.model = model
        self.max_iterations = max_iterations
    
    async def execute(self, state: dict[str, Any]) -> WorkerResult:
        """Execute research task with autonomous refinement."""
        
        # Iteration 1: Initial research
        agent = create_deep_agent(
            model=self.model,
            system_prompt=self._build_system_prompt(),
            subagents=self._get_specialists(),
        )
        
        result = await agent.ainvoke({
            "messages": [("user", self.task)]
        })
        
        # Check for gaps
        gaps = await self._identify_gaps(result)
        
        # Iteration 2+: Fill gaps
        for i in range(self.max_iterations - 1):
            if not gaps:
                break
            
            # Targeted gap-filling
            gap_result = await self._fill_gaps(gaps, result)
            result = self._merge_results(result, gap_result)
            gaps = await self._identify_gaps(result)
        
        return self._create_worker_result(result)
```

##### Task 3.2: Implement Specialist Fan-Out

**File**: `src/deep_research_engine/nodes/drafter.py`

```python
async def _create_specialist_subagents(
    worker_configs: list[dict]
) -> list[SubAgent]:
    """Create specialist sub-sub agents for specific tasks."""
    
    specialists = []
    
    for config in worker_configs:
        worker_type = config.get("type")
        
        if worker_type == "web_researcher":
            specialists.append({
                "name": "web_search_specialist",
                "description": "Expert at finding and evaluating web sources",
                "system_prompt": "You are a web search specialist...",
                "model": config.get("model", "openai:gpt-4o-mini")
            })
        
        elif worker_type == "chunk_processor":
            specialists.append({
                "name": "pdf_chunk_specialist",
                "description": "Expert at processing PDF chunks",
                "system_prompt": "You are a PDF processing specialist...",
                "model": config.get("model", "openai:gpt-4o-mini")
            })
        
        elif worker_type == "multimodal":
            specialists.append({
                "name": "multimodal_specialist",
                "description": "Expert at analyzing images and audio",
                "system_prompt": "You are a multimodal analysis specialist...",
                "model": config.get("model", "openai:gpt-4o-mini")
            })
    
    return specialists
```

#### Acceptance Criteria

- [ ] Flash agent pattern implemented
- [ ] Autonomous gap detection and filling
- [ ] Specialist sub-sub agents created
- [ ] Iterative refinement loops work
- [ ] Results merged correctly

---

### Feature 4: FFmpeg Metadata Parsing

**Priority**: P1 HIGH  
**File**: `src/deep_research_engine/utils/pipeline.py`  
**Function**: `extract_scene_frames()` (lines 62-130)  
**Estimated Effort**: 3-4 hours

#### Implementation

##### Task 4.1: Parse FFmpeg time.txt

**File**: `src/deep_research_engine/utils/pipeline.py`

```python
import re

async def parse_ffmpeg_metadata(time_file: Path) -> list[dict[str, Any]]:
    """Parse FFmpeg time.txt metadata to extract precise timestamps.
    
    FFmpeg generates time.txt like:
    frame=    1 pts=      1 time=00:00:01.23
    frame=    2 pts=     45 time=00:00:04.50
    
    Returns:
        List of frame metadata with precise timestamps
    """
    frames = []
    
    if not time_file.exists():
        logger.warning(f"FFmpeg metadata file not found: {time_file}")
        return frames
    
    with open(time_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            # Parse: frame=    1 pts=      1 time=00:00:01.23
            match = re.match(
                r'frame=\s*(\d+)\s+pts=\s*(\d+)\s+time=(\d+):(\d+):(\d+)\.(\d+)',
                line.strip()
            )
            
            if match:
                frame_num = int(match.group(1))
                pts = int(match.group(2))
                hours = int(match.group(3))
                minutes = int(match.group(4))
                seconds = int(match.group(5))
                milliseconds = int(match.group(6).ljust(3, '0')[:3])
                
                # Calculate total milliseconds
                total_ms = (
                    hours * 3600000 +
                    minutes * 60000 +
                    seconds * 1000 +
                    milliseconds
                )
                
                frames.append({
                    "frame_num": frame_num,
                    "pts": pts,
                    "timestamp_ms": total_ms,
                    "timestamp_formatted": f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}",
                    "line_num": line_num
                })
    
    logger.info(f"Parsed {len(frames)} frame timestamps from {time_file}")
    return frames
```

##### Task 4.2: Integrate with Scene Detection

**File**: `src/deep_research_engine/utils/pipeline.py`

```python
async def extract_scene_frames(
    video_path: Path,
    output_dir: Path,
    threshold: float = 0.3
) -> list[dict[str, Any]]:
    """Extract scene-change frames with precise timestamps."""
    
    # ... existing frame extraction code ...
    
    # Parse FFmpeg metadata
    time_file = output_dir / "time.txt"
    frame_metadata = await parse_ffmpeg_metadata(time_file)
    
    # Merge frame paths with timestamps
    result = []
    for frame_path in frame_files:
        frame_num = int(frame_path.stem.split("_")[-1])
        metadata = next(
            (m for m in frame_metadata if m["frame_num"] == frame_num),
            None
        )
        
        result.append({
            "path": str(frame_path),
            "frame_num": frame_num,
            "pts_time": metadata["pts"] / 90000 if metadata else 0,  # Convert from FFmpeg pts
            "timestamp_ms": metadata["timestamp_ms"] if metadata else 0,
            "timestamp_formatted": metadata["timestamp_formatted"] if metadata else "00:00:00.000",
            "scene_id": f"scene_{frame_num // 10}"  # Group frames by scene
        })
    
    return result
```

#### Acceptance Criteria

- [ ] FFmpeg time.txt parsed correctly
- [ ] Millisecond-accurate timestamps
- [ ] Timestamps merged with frame paths
- [ ] Scene IDs assigned

---

## Phase 3: Enhancements (Week 4-6) 🚀

### Feature 5: Multi-Mega-File Logic

**Priority**: P2 MEDIUM  
**File**: `src/deep_research_engine/utils/pipeline.py`  
**Function**: `concatenate_audio_fragments()` (lines 159-210)  
**Estimated Effort**: 6-8 hours

#### Implementation

##### Task 5.1: Add Recursive Chunking

```python
def concatenate_audio_fragments(
    audio_fragments: list[dict[str, Any]],
    output_dir: Path,
    max_duration: int = 3600  # 60 minutes
) -> list[tuple[Path, list[dict]]]:
    """Concatenate audio fragments into Mega-Files with recursive chunking.
    
    Returns:
        List of (mega_file_path, fragment_metadata_list) tuples
    """
    # Sort fragments chronologically
    sorted_fragments = sorted(
        audio_fragments,
        key=lambda x: x.get("timestamp", "") or ""
    )
    
    mega_files = []
    current_batch = []
    current_duration = 0.0
    mega_file_index = 0
    
    for fragment in sorted_fragments:
        fragment_duration = fragment.get("duration", 0)
        
        # Check if adding this fragment would exceed limit
        if current_duration + fragment_duration > max_duration:
            # Finalize current Mega-File
            if current_batch:
                mega_file = create_mega_file(
                    current_batch,
                    output_dir,
                    mega_file_index
                )
                mega_files.append(mega_file)
                mega_file_index += 1
            
            # Start new batch
            current_batch = [fragment]
            current_duration = fragment_duration
        else:
            current_batch.append(fragment)
            current_duration += fragment_duration
    
    # Finalize last Mega-File
    if current_batch:
        mega_file = create_mega_file(
            current_batch,
            output_dir,
            mega_file_index
        )
        mega_files.append(mega_file)
    
    logger.info(f"Created {len(mega_files)} Mega-Files from {len(audio_fragments)} fragments")
    return mega_files
```

---

### Feature 6: Cross-File Biometric Verification

**Priority**: P2 MEDIUM  
**File**: `src/deep_research_engine/utils/pipeline.py`  
**Function**: `unify_speakers()` (lines 297-356)  
**Estimated Effort**: 4-6 hours

#### Implementation

##### Task 6.1: Add Confidence Scoring

```python
async def unify_speakers(
    mega_file_results: list[tuple[Path, dict[str, Any]]],
    output_dir: Path,
    threshold: float = 0.8,
    min_confidence: float = 0.6
) -> dict[str, str]:
    """Unify Speaker IDs with confidence scoring."""
    
    # ... existing code ...
    
    for gid, gemb in speaker_embeddings.items():
        sim = calculate_speaker_similarity(emb, gemb)
        
        if sim >= threshold:
            # High confidence match
            match_id = gid
            confidence = sim
            break
        elif sim >= min_confidence:
            # Low confidence - flag for review
            low_confidence_matches.append({
                "speaker": f"mf{mf_idx}:{speaker_id}",
                "candidate": gid,
                "similarity": sim
            })
    
    # Log low-confidence matches for manual review
    if low_confidence_matches:
        logger.warning(f"Found {len(low_confidence_matches)} low-confidence speaker matches")
        save_low_confidence_matches(low_confidence_matches, output_dir)
```

---

## Phase 4: Polish (Week 6-8) ✨

### Feature 7: Storage Backend Verification

**Priority**: P3 LOW  
**Files**: `src/deep_research_engine/storage/backends/*.py`  
**Estimated Effort**: 8-10 hours

#### Tasks

- [ ] Verify Postgres pgvector implementation
- [ ] Verify MongoDB GridFS for large files
- [ ] Verify Neo4j Cypher queries
- [ ] Verify MinIO streaming
- [ ] Add integration tests

---

### Feature 8: Automated Testing

**Priority**: P3 LOW  
**Directory**: `tests/`  
**Estimated Effort**: 12-16 hours

#### Test Files to Create

```
tests/
├── unit/
│   ├── test_state.py
│   ├── test_worker_registry.py
│   ├── test_storage_registry.py
│   ├── test_config_loader.py
│   └── test_multimodal.py
├── integration/
│   ├── test_critic_node.py
│   ├── test_drafter_node.py
│   ├── test_ingestion_node.py
│   └── test_vibevoice_asr.py
└── e2e/
    ├── test_rag_workflow.py
    ├── test_pdf_workflow.py
    └── test_telegram_workflow.py
```

---

### Feature 9: Documentation

**Priority**: P3 LOW  
**Directory**: `docs/`  
**Estimated Effort**: 6-8 hours

#### Docs to Write

- [ ] `docs/workflow-configs/schema-reference.md`
- [ ] `docs/workers/custom-worker-guide.md`
- [ ] `docs/storage/backend-guide.md`
- [ ] `docs/examples/telegram-ingestion.md`
- [ ] `docs/examples/pdf-summarization.md`

---

## Implementation Timeline

| Week | Phase | Features | Effort |
|------|-------|----------|--------|
| 1-2 | Phase 1 | Critic Logic, Video Audio | 10-14 hours |
| 2-4 | Phase 2 | Flash Orchestration, FFmpeg Metadata | 14-20 hours |
| 4-6 | Phase 3 | Multi-Mega-File, Biometric | 10-14 hours |
| 6-8 | Phase 4 | Storage, Tests, Docs | 26-34 hours |

**Total Estimated Effort**: 60-82 hours

---

## Next Steps

1. **Start with P0 features** (Critic, Video Audio)
2. **Test each feature** before moving to next
3. **Update IMPLEMENTATION_STATUS.md** after each completion
4. **Run full workflow test** after each phase

---

**Ready to begin implementation?**
