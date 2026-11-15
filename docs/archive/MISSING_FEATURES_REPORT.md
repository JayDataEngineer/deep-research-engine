# Missing Features Report - Deep Research Engine

**Generated**: March 27, 2026  
**Source Files Analyzed**: 47 Python files, 23 Markdown documents  
**Status**: Comprehensive audit of implemented vs. documented features

---

## Executive Summary

The Deep Research Engine has **core infrastructure implemented** but is missing **7 critical features** that prevent full functionality as documented in `PHASED_INGESTION_SUMMARY.md`.

### Implementation Status by Phase

| Phase | Documented Status | Actual Status | Gap |
|-------|------------------|---------------|-----|
| Phase 1: Temporal Normalization | ✅ Complete | ✅ Complete | None |
| Phase 2: Visual Processing | ✅ Complete | ⚠️ Partial | FFmpeg metadata parsing missing |
| Phase 3: Acoustic Mega-Files | ✅ Complete | ⚠️ Partial | Multi-file logic incomplete |
| Phase 4: GPU Memory Management | ✅ Complete | ✅ Complete | None |
| Phase 5: Acoustic Diarization | ✅ Complete | ⚠️ Partial | Cross-file verification stub |
| Phase 6: Multimodal Synthesis | ✅ Complete | ❌ Incomplete | LLM identity resolution incomplete |

---

## Critical Missing Features (P0-P1)

### 1. Critic Node Logic ⚠️ P0 CRITICAL

**File**: [`src/deep_research_engine/nodes/critic.py`](src/deep_research_engine/nodes/critic.py)  
**Lines**: 29-35

**Current Implementation**:
```python
# (Placeholder for real critic logic)
score = 0.8  # HARDCODED - Always passes!
passed = score >= 0.7
```

**Documented Behavior** (from `IMPLEMENTATION_STATUS.md`):
> "Critic Logic - Implement structured evaluation with specific feedback to Drafter"

**What's Missing**:
- [ ] Structured LLM evaluation of drafter output quality
- [ ] Specific, actionable feedback generation for drafter improvements
- [ ] Quality metrics (completeness, accuracy, citation quality, coherence)
- [ ] Configurable evaluation criteria per workflow type

**Impact**: The critic node **always passes** with a hardcoded score of 0.8, providing no quality control. This defeats the entire purpose of the critic phase in the 5-node pipeline.

**Fix Required**: Implement LLM-based evaluation with structured output containing:
- Quality scores (1-10) for multiple dimensions
- Specific improvement suggestions
- Citation verification
- Fact-checking flags

---

### 2. Video Audio Extraction for ASR ⚠️ P0 CRITICAL

**File**: [`src/deep_research_engine/utils/multimodal.py`](src/deep_research_engine/utils/multimodal.py)  
**Line**: 346

**Current Implementation**:
```python
# TODO: Implement audio extraction from video for ASR
```

**Documented Behavior** (from `PHASED_INGESTION_SUMMARY.md`):
> "Phase 3: Acoustic Mega-File Strategy - Audio fragments (voice notes, video audio) are chronologically concatenated into contiguous 'Mega-Files'"

**What's Missing**:
- [ ] FFmpeg audio demuxing from video files (`.mp4`, `.mkv`, `.avi`, etc.)
- [ ] Conversion to ASR-compatible format (16kHz mono WAV)
- [ ] Integration with Mega-File concatenation pipeline
- [ ] Metadata tracking (which audio came from which video)

**Impact**: Video files in Telegram exports **cannot be transcribed via ASR**. Only VLM visual description is generated, missing all spoken content from videos.

**Fix Required**: Implement `extract_audio_from_video()` function:
```python
async def extract_audio_from_video(video_path: Path, output_path: Path) -> bool:
    """Extract audio track from video and convert to 16kHz mono WAV for ASR."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path)
    ]
    # ... execute and return success
```

---

### 3. Flash Subagent Orchestration ⚠️ P1 HIGH

**File**: [`src/deep_research_engine/nodes/drafter.py`](src/deep_research_engine/nodes/drafter.py)  
**Lines**: 1-147 (entire file)

**Current Implementation**:
- ✅ Basic subagent creation from worker config
- ✅ DeepAgents integration present
- ✅ King agent invocation works

**Documented Behavior** (from `TODO_FOR_JULES.md`):
> "Flash Subagent Orchestration: Implement the `web_researcher` worker wrapping the Flash Explorer pattern"
> "Specialist Fan-Out: Implement specific sub-agents for chunk_processor, multimodal, scraper"

**What's Missing**:
- [ ] Flash Explorer pattern implementation (autonomous research loops)
- [ ] Specialist sub-sub agent fan-out logic
- [ ] Dynamic task decomposition based on query type
- [ ] Inter-agent communication and result aggregation
- [ ] Quality-based subagent retry logic

**Impact**: Workers run sequentially but lack advanced orchestration. No autonomous research behavior, no specialist collaboration.

**Fix Required**: Implement Flash pattern with:
- Autonomous task planning
- Parallel specialist deployment
- Result synthesis and gap detection
- Iterative refinement loops

---

### 4. FFmpeg Metadata Parsing ⚠️ P1 HIGH

**File**: [`src/deep_research_engine/utils/pipeline.py`](src/deep_research_engine/utils/pipeline.py)  
**Function**: `extract_scene_frames()` (lines 62-130)

**Current Implementation**:
```python
async def extract_scene_frames(video_path: Path, output_dir: Path, threshold: float = 0.3) -> list[dict[str, Any]]:
    # ... extracts frames using FFmpeg scene detection
    # Returns: list of dicts with frame paths
    # MISSING: Precise timestamp mapping
```

**Documented Behavior** (from `PHASED_INGESTION_SUMMARY.md`):
> "Pending: Robust regex-based parsing of the `time.txt` metadata generated by FFmpeg to map frames to precise millisecond timestamps for ASR synchronization"

**What's Missing**:
- [ ] FFmpeg `time.txt` metadata file parsing
- [ ] Regex extraction of PTS (Presentation Time Stamp) values
- [ ] Mapping frame filenames to millisecond-accurate timestamps
- [ ] Synchronization with ASR transcript segments

**Current Output** (incomplete):
```python
{
    "path": "frames/frame_001.png",
    "pts_time": 1.234  # From FFmpeg output, but not parsed
}
```

**Expected Output** (complete):
```python
{
    "path": "frames/frame_001.png",
    "pts_time": 1.234,
    "timestamp_ms": 1234,
    "timestamp_formatted": "00:00:01.234",
    "scene_id": "scene_001"
}
```

**Impact**: Cannot synchronize video frames with audio transcripts. Identity resolution across modalities (audio speaker + visual person) fails without precise timestamp alignment.

**Fix Required**: Parse FFmpeg's `time.txt` output:
```python
# FFmpeg generates time.txt like:
# frame=    1 pts=      1 time=00:00:01.23
# frame=    2 pts=     45 time=00:00:04.50

async def parse_ffmpeg_metadata(time_file: Path) -> list[dict]:
    """Parse FFmpeg time.txt metadata to extract precise timestamps."""
    frames = []
    with open(time_file) as f:
        for line in f:
            # Parse: frame=    1 pts=      1 time=00:00:01.23
            match = re.match(r'frame=\s*(\d+)\s+pts=\s*(\d+)\s+time=(\d+:\d+:\d+\.\d+)', line)
            if match:
                frames.append({
                    "frame_num": int(match.group(1)),
                    "pts": int(match.group(2)),
                    "time": match.group(3)
                })
    return frames
```

---

## Medium Priority Missing Features (P2)

### 5. Batch VLM Video Analysis ⚠️ P2 MEDIUM

**File**: [`src/deep_research_engine/shared/multimodal.py`](src/deep_research_engine/shared/multimodal.py)  
**Function**: `batch_vlm_video_analysis()` (lines 147-200)

**Current Implementation**:
```python
async def batch_vlm_video_analysis(
    frame_items: list[dict[str, Any]],
    batch_size: int = 15  # Processes 15 frames at a time
) -> str:
    # Sorts frames chronologically
    # Batches frames to avoid token limits
    # Calls VLM for each batch
    # Synthesizes multi-batch results
```

**Status**: ✅ **IMPLEMENTED** (as of recent commits)

**What's Working**:
- ✅ Frame batching (15 frames per batch)
- ✅ Chronological sorting
- ✅ Multi-batch synthesis
- ✅ Error handling per batch

**Remaining Enhancements**:
- [ ] Configurable batch size based on model context window
- [ ] Attention mechanism for key frame selection
- [ ] Character consistency tracking across batches
- [ ] Temporal narrative coherence optimization

**Impact**: Feature is functional but could be optimized for longer videos.

---

### 6. Multi-Mega-File Logic ⚠️ P2 MEDIUM

**File**: [`src/deep_research_engine/utils/pipeline.py`](src/deep_research_engine/utils/pipeline.py)  
**Function**: `concatenate_audio_fragments()` (lines 159-210)

**Current Implementation**:
```python
def concatenate_audio_fragments(
    audio_fragments: list[Path],
    output_dir: Path,
    max_duration: int = 3600  # 60 minutes
) -> list[tuple[Path, list[Path]]]:
    """Concatenate audio fragments into multiple Mega-Files if duration exceeds limit."""
```

**Documented Behavior** (from `PHASED_INGESTION_SUMMARY.md`):
> "Pending: Recursive chunking logic to handle datasets exceeding 1 hour by spawning multiple sequential Mega-File passes"

**What's Implemented**:
- ✅ Basic concatenation with duration tracking
- ✅ Multiple file creation when limit exceeded
- ✅ Boundary map generation

**What's Missing**:
- [ ] **Recursive chunking** for very long exports (4+ hours)
- [ ] **Memory-efficient streaming** for large file lists
- [ ] **Progress tracking** across multiple Mega-Files
- [ ] **Parallel ASR processing** of multiple Mega-Files
- [ ] **Cross-file boundary map merging**

**Impact**: Telegram exports with >60 minutes of audio are split into multiple Mega-Files, but speaker identities are not reconciled across files.

**Related Function**: `unify_speakers()` (lines 302-356) - **IMPLEMENTED** but requires ECAPA-TDNN integration (see #7)

---

### 7. Cross-Mega-File Biometric Verification ⚠️ P2 MEDIUM

**File**: [`src/deep_research_engine/utils/pipeline.py`](src/deep_research_engine/utils/pipeline.py)  
**Function**: `unify_speakers()` (lines 297-356)  
**Helper**: `extract_speaker_embedding()` (lines 261-280)

**Current Implementation**:
```python
async def unify_speakers(
    mega_file_results: list[tuple[Path, dict[str, Any]]],
    output_dir: Path,
    threshold: float = 0.8
) -> dict[str, str]:
    """Unify Speaker IDs across multiple Mega-Files using acoustic embeddings."""
    
    # Extracts 30s audio snippets for each speaker
    # Generates ECAPA-TDNN embeddings
    # Clusters speakers by cosine similarity
    # Returns global speaker map
```

**Status**: ⚠️ **PARTIALLY IMPLEMENTED**

**What's Working**:
- ✅ Audio snippet extraction (5-30s segments)
- ✅ ECAPA-TDNN model loading (`speechbrain/spkrec-ecapa-voxceleb`)
- ✅ Embedding generation
- ✅ Cosine similarity comparison
- ✅ Global speaker ID assignment

**What's Missing**:
- [ ] **Automated snippet quality filtering** (currently takes first >5s segment)
- [ ] **Confidence scoring** for speaker matches
- [ ] **Manual verification interface** for low-confidence matches
- [ ] **Persistence** of speaker embeddings for future sessions
- [ ] **Integration with visual identity clusters**

**Impact**: Speaker identities can be reconciled across Mega-Files, but quality is variable without confidence filtering and visual cross-reference.

---

## Low Priority Missing Features (P3)

### 8. Automated LLM Identity Resolution ⚠️ P3 LOW

**File**: [`src/deep_research_engine/utils/pipeline.py`](src/deep_research_engine/utils/pipeline.py)  
**Function**: `resolve_identities()` (lines 367-460)

**Current Implementation**:
```python
async def resolve_identities(
    asr_segments: list[dict[str, Any]],
    visual_events: list[dict[str, Any]],
    text_messages: list[dict[str, Any]]
) -> dict[str, str]:
    """Use a two-pass approach to resolve identities across modalities.
    
    1. Research: DeepAgent deduces identities across modalities (Text output).
    2. Format: Standard LLM formats the research into a structured JSON mapping.
    """
```

**Status**: ✅ **IMPLEMENTED**

**What's Working**:
- ✅ Two-pass Research-then-Format pattern
- ✅ DeepAgents integration for research pass
- ✅ Structured output via `IdentityMapping` Pydantic model
- ✅ LangFuse prompt template integration
- ✅ Cross-modal temporal reasoning

**Remaining Enhancements**:
- [ ] **Confidence scoring** for each identity mapping
- [ ] **Evidence citation** (which timestamps support each mapping)
- [ ] **Conflicting evidence handling** (when modalities disagree)
- [ ] **User verification UI** for low-confidence mappings

**Impact**: Feature is functional but lacks transparency for uncertain mappings.

---

### 9. Storage Backend Implementations ⚠️ P3 LOW

**Files**:
- [`src/deep_research_engine/storage/backends/postgres.py`](src/deep_research_engine/storage/backends/postgres.py)
- [`src/deep_research_engine/storage/backends/mongodb.py`](src/deep_research_engine/storage/backends/mongodb.py)
- [`src/deep_research_engine/storage/backends/neo4j.py`](src/deep_research_engine/storage/backends/neo4j.py)
- [`src/deep_research_engine/storage/backends/minio.py`](src/deep_research_engine/storage/backends/minio.py)

**Status**: ⚠️ **NEEDS VERIFICATION**

**Documented Behavior** (from `IMPLEMENTATION_STATUS.md`):
> "Storage Backends - [ ] PostgresStorage, MongoDBStorage, Neo4jStorage, MinIOStorage"

**What Needs Verification**:
- [ ] Full CRUD operations for each backend
- [ ] Vector search implementation (Postgres pgvector)
- [ ] Graph operations (Neo4j Cypher queries)
- [ ] Large object streaming (MinIO)
- [ ] Media file chunking (MongoDB GridFS)

**Impact**: Storage backends exist but may be incomplete. Requires code review.

---

### 10. Automated Testing ⚠️ P3 LOW

**Directory**: [`tests/`](tests/)

**Documented Requirements** (from `TODO_FOR_JULES.md`):
> "Unit Tests: test_state.py, test_worker_registry.py, test_storage_registry.py, test_config_loader.py"
> "Integration Tests: Individual nodes (test_boss_node.py, test_worker_node.py, test_critic_node.py)"
> "End-to-End Workflow Tests: test_rag_workflow.py, test_pdf_workflow.py, test_telegram_workflow.py"

**Current Status**: ❌ **MISSING**

**What's Missing**:
- [ ] Unit tests for state management
- [ ] Unit tests for worker registry
- [ ] Unit tests for storage registry
- [ ] Unit tests for config loader
- [ ] Integration tests for each node
- [ ] E2E workflow tests
- [ ] VibeVoice ASR integration tests

**Impact**: No automated testing. Breaking changes can go undetected.

---

### 11. Documentation ⚠️ P3 LOW

**Directory**: [`docs/`](docs/)

**Documented Requirements** (from `TODO_FOR_JULES.md`):
> "Workflow Configs Guide: Document schemas and provide examples"
> "Workers Guide: Provide a guide on how to build custom workers"
> "Storage Guide: Document the custom storage backends interface"

**Current Status**: ⚠️ **INCOMPLETE**

**What Exists**:
- ✅ `docs/index.md` - Overview
- ✅ `docs/getting-started/` - Installation guide
- ✅ `docs/vibevoice-asr.md` - VibeVoice integration
- ✅ `docs/storage/` - Storage documentation (partial)

**What's Missing**:
- [ ] Complete workflow config schema reference
- [ ] Custom worker development guide
- [ ] Storage backend implementation guide
- [ ] API reference documentation
- [ ] Example workflows with sample data

---

## Summary Table

| # | Feature | Priority | File(s) | Status | Lines |
|---|---------|----------|---------|--------|-------|
| 1 | Critic Node Logic | P0 | `nodes/critic.py` | ❌ Placeholder | 29-35 |
| 2 | Video Audio Extraction | P0 | `utils/multimodal.py` | ❌ TODO | 346 |
| 3 | Flash Subagent Orchestration | P1 | `nodes/drafter.py` | ⚠️ Partial | 1-147 |
| 4 | FFmpeg Metadata Parsing | P1 | `utils/pipeline.py` | ❌ Stub | 62-130 |
| 5 | Batch VLM Video Analysis | P2 | `shared/multimodal.py` | ✅ Done | 147-200 |
| 6 | Multi-Mega-File Logic | P2 | `utils/pipeline.py` | ⚠️ Partial | 159-210 |
| 7 | Cross-File Biometric Verification | P2 | `utils/pipeline.py` | ⚠️ Partial | 297-356 |
| 8 | Automated LLM Identity Resolution | P3 | `utils/pipeline.py` | ✅ Done | 367-460 |
| 9 | Storage Backends | P3 | `storage/backends/*` | ⚠️ Unverified | - |
| 10 | Automated Testing | P3 | `tests/` | ❌ Missing | - |
| 11 | Documentation | P3 | `docs/` | ⚠️ Incomplete | - |

---

## Recommended Implementation Order

### Phase 1: Critical Fixes (Week 1-2)
1. **Critic Node Logic** - Implement LLM-based evaluation
2. **Video Audio Extraction** - Add FFmpeg demuxing

### Phase 2: Core Functionality (Week 2-4)
3. **Flash Subagent Orchestration** - Complete DeepAgents integration
4. **FFmpeg Metadata Parsing** - Add timestamp synchronization

### Phase 3: Enhancements (Week 4-6)
5. **Multi-Mega-File Logic** - Add recursive chunking
6. **Cross-File Biometric Verification** - Add confidence filtering

### Phase 4: Polish (Week 6-8)
7. **Storage Backend Verification** - Test all CRUD operations
8. **Automated Testing** - Write unit + integration tests
9. **Documentation** - Complete all guides

---

## Files Requiring Immediate Attention

### High Risk (Broken/Non-functional)
1. `src/deep_research_engine/nodes/critic.py` - Hardcoded score, no evaluation
2. `src/deep_research_engine/utils/multimodal.py:346` - TODO blocks video ASR

### Medium Risk (Partial Implementation)
3. `src/deep_research_engine/nodes/drafter.py` - Basic orchestration only
4. `src/deep_research_engine/utils/pipeline.py:62-130` - Missing metadata parsing
5. `src/deep_research_engine/utils/pipeline.py:297-356` - Needs confidence filtering

### Low Risk (Functional but Incomplete)
6. `src/deep_research_engine/storage/backends/*.py` - Needs verification
7. `tests/` - No tests exist
8. `docs/` - Documentation incomplete

---

**Report Generated By**: AI Analysis  
**Date**: March 27, 2026  
**Next Review**: After P0 fixes implemented
