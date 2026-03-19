# LLM Studio Extension - Development Plan

> **Status**: Planning Phase
> **Created**: 2026-01-31
> **Updated**: 2026-01-31 (Deep Agent Architecture)
> **Primary Use Case**: Video Podcast Generation Pipeline

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Deep Agent Planning System](#deep-agent-planning-system)
4. [Use Case: Video Podcast Pipeline](#use-case-video-podcast-pipeline)
5. [Component Specifications](#component-specifications)
6. [Implementation Phases](#implementation-phases)
7. [Configuration Schema](#configuration-schema)

---

## Overview

### Vision

The **LLM Studio** extension provides a composable, workflow-based system for building complex AI media pipelines. It extends the existing audio producer capabilities into a full multi-modal studio with:

- **Deep Agent Planning**: Single intelligent agent handles all planning (clip breaks, emotions, prompts, parameters)
- **Composable Functions**: Modular pipeline stages for generation
- **Prompt Engineering System**: Unified prompt management for the Deep Agent
- **Multi-Modal Support**: Text, Audio, Image, and Video generation
- **Two-Pass Architecture**: Rough generation → refinement with ground-truth data

### Key Design Principles

1. **Deep Agent Over Specialist Workers**: One capable agent with thinking > 10 specialist configs
2. **Prompt-First**: All LLM interactions go through managed prompt templates
3. **Manual Base Scene**: Studio base image is created manually (too important for AI)
4. **Composable Functions**: Modular pipeline stages
5. **Checkpointable**: All workflows support pause/resume

---

## Architecture

### Extension Structure

```
src/extensions/
├── features/
│   └── llm_studio/                    # Main feature extension
│       ├── extension.py
│       ├── config.yaml
│       └── cli.py
│
├── libraries/
│   └── llm_studio/                    # Core shared logic
│       ├── manager.py                 # StudioManager (main API)
│       ├── config_schemas.py          # Configuration classes
│       │
│       ├── core/                      # Core abstractions
│       │   ├── composable.py          # BaseComposableFunction
│       │   ├── prompt_manager.py      # Prompt template management
│       │   └── deep_agent.py         # Deep Agent wrapper
│       │
│       ├── functions/                 # Composable function implementations
│       │   ├── planning/              # Planning functions
│       │   │   ├── deep_agent_planner.py  # Deep Agent for all planning
│       │   │   └── json_to_markdown.py     # Convert agent JSON to markdown
│       │   │
│       │   ├── image/                 # Image generation functions
│       │   │   ├── qwen_edit.py       # Qwen image editing
│       │   │   └── comfyui_executor.py
│       │   │
│       │   ├── video/                 # Video processing functions
│       │   │   ├── ltx2_generation.py # LTX2 video generation
│       │   │   ├── ltx2_lipsync.py    # LTX2 lip sync refinement
│       │   │   └── stitching.py       # Video stitching
│       │   │
│       │   └── audio/                 # Audio processing functions
│       │       ├── whisperx_transcription.py  # Timestamps (TODO: replace)
│       │       ├── timestamp_alignment.py
│       │       └── indextts_generation.py     # IndexTTS with precise timing
│       │
│       ├── workflows/                 # Complete workflow implementations
│       │   └── video_podcast.py       # Primary use case
│       │
│       └── prompts/                   # Prompt templates
│           └── deep_agent/
│               └── mega_prompt.yaml   # The master planning prompt
│
└── services/
    └── comfyui_bridge/                # ComfyUI service extension
        ├── extension.py
        ├── api.py
        └── workflows/
            ├── qwen_edit.json
            ├── ltx2_video.json
            └── ltx2_lipsync.json
```

---

## Deep Agent Planning System

### The Core Innovation

Instead of complex multi-stage planning with specialist workers, we use a **single Deep Agent** with thinking mode to plan everything at once.

**Why this works:**
- Deep Agents have thinking mode for multi-step planning
- Can see the entire script and make coherent decisions
- Simpler - one prompt to maintain instead of many configs
- Faster - one LLM call instead of many
- More coherent - clip breaks, emotions, and prompts all align

### What the Deep Agent Plans

For each **clip** (segment between camera breaks), the agent outputs:

| Field | Description | Example |
|-------|-------------|---------|
| `clip_id` | Unique identifier | `clip_001` |
| `start_line` | First dialogue line | `5` |
| `end_line` | Last dialogue line | `12` |
| `speakers` | Speakers in this clip | `["EMMA", "HOST"]` |
| `camera_break_type` | Why this clip ends | `"speaker_change"` / `"scene_change"` / `"pause"` |
| `duration_frames` | Number of frames for LTX2 | `120` (~5 seconds @ 24fps) |
| `emotion_vectors` | Per-speaker emotion vectors | See below |
| `qwen_edit_prompt` | Prompt for Qwen Image Edit | `"Zoom in on EMMA, smiling warmly..."` |
| `qwen_negative_prompt` | Negative prompt for Qwen | `"(worst quality, blurry)..."` |
| `ltx2_prompt` | Prompt for LTX2 video | `"EMMA speaking excitedly, gesturing..."` |
| `ltx2_negative_prompt` | Negative prompt for LTX2 | `"(static, low quality)..."` |
| `camera_angle` | Camera positioning | `"medium_close_up"` |
| `additional_params` | Any extra parameters | `{...}` |

### Emotion Vector Format

```json
{
  "EMMA": {
    "happy": 0.7,
    "angry": 0.0,
    "sad": 0.0,
    "afraid": 0.0,
    "disgusted": 0.0,
    "melancholic": 0.0,
    "surprised": 0.3,
    "calm": 0.5,
    "intensity": 0.4
  }
}
```

### Output Format

The agent outputs a JSON structure:

```json
{
  "script_info": {
    "title": "Episode 1",
    "total_lines": 45,
    "num_clips": 8
  },
  "base_scene_reference": "base_studio_scene_v1.png",
  "clips": [
    {
      "clip_id": "clip_001",
      "start_line": 1,
      "end_line": 6,
      "speakers": ["EMMA"],
      "camera_break_type": "speaker_change",
      "duration_frames": 96,
      "emotion_vectors": {...},
      "qwen_edit_prompt": "...",
      "qwen_negative_prompt": "...",
      "ltx2_prompt": "...",
      "ltx2_negative_prompt": "...",
      "camera_angle": "close_up"
    },
    // ... more clips
  ]
}
```

This JSON is then converted to Markdown for human review/iteration.

---

## Use Case: Video Podcast Pipeline

### Complete Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VIDEO PODCAST PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

PREREQUISITE: BASE STUDIO SCENE (Manual)
┌──────────────────┐
│  User Creates    │────▶  base_studio_scene.png (saved for all videos)
│  Base Image      │       (Too important/complex for AI generation)
└──────────────────┘

PHASE 1: DEEP AGENT PLANNING
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Script +    │────▶│  Deep Agent  │────▶│  Plan JSON   │
│  Base Scene  │     │  (Thinking)  │     │  + Markdown  │
│  + Context   │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │  Agent Decides:  │
                   │  - Clip breaks   │
                   │  - Emotions      │
                   │  - Qwen prompts  │
                   │  - LTX2 prompts  │
                   │  - num_frames    │
                   └──────────────────┘

PHASE 2: QWEN IMAGE EDIT (Per Clip)
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Plan JSON   │────▶│  Qwen Image  │────▶│  Start       │
│  (per clip)  │     │  Edit        │     │  Frames      │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │  clip_001_    │
                                          │  start.png    │
                                          └──────────────┘

PHASE 3: LTX2 VIDEO GENERATION (Per Clip)
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Start Frame │────▶│  LTX2 Video  │────▶│  Clip Video  │
│  + LTX2      │     │  Generation  │     │  + LTX2      │
│  Prompt      │     │              │     │  Audio       │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │  clip_001_    │
                                          │  video.mp4    │
                                          └──────────────┘

PHASE 4: STITCH VIDEO
┌──────────────┐     ┌──────────────┐
│  All Clips   │────▶│  Rough       │
│  + Trans     │     │  Video       │
└──────────────┘     └──────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │  rough_       │
                   │  video.mp4    │
                   └──────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │  Extract LTX2│
                   │  Audio for   │
                   │  WhisperX    │
                   └──────────────┘

PHASE 5: SECOND PASS - PRECISION SYNC
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  WhisperX    │────▶│  Word-Level  │────▶│  IndexTTS    │
│  (LTX2 audio)│     │  Timestamps  │     │  (ONCE!)     │
└──────────────┘     └──────────────┘     └──────────────┘
       │                                            │
       │                                            ▼
       │                                    ┌──────────────┐
       │                                    │  FINAL Audio │
       │                                    │  (Precise    │
       │                                    │   timing)    │
       │                                    └──────────────┘
       │                                            │
       └───────────────────────────────────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │  LTX2 Lip    │
                   │  Sync with   │◀──── Uses FINAL IndexTTS audio
                   │  FINAL Audio │
                   └──────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │  final_      │
                   │  video.mp4    │
                   └──────────────┘
```

### Phase Breakdown

#### Prerequisite: Base Studio Scene (Manual)

| Step | Action | Note |
|------|--------|------|
| - | User creates base studio scene | Manual creation, too important for AI |
| - | Save as `base_studio_scene.png` | Referenced by all Qwen Edit operations |

**Why Manual**: The base studio scene sets the visual foundation for everything. AI generation could introduce inconsistencies that would compound through all clips.

#### Phase 1: Deep Agent Planning

| Step | Function | Input | Output |
|------|----------|-------|--------|
| 1.1 | Deep Agent Planner | Script + Base Scene + Context | Plan JSON |
| 1.2 | JSON to Markdown | Plan JSON | Reviewable Markdown |

**Key Point**: The Deep Agent sees the entire script and makes coherent decisions about:
- Where camera breaks should happen (speaker changes, scene changes, natural pauses)
- What emotions each speaker has in each clip
- What prompts to generate for Qwen Edit and LTX2
- How many frames each clip needs

#### Phase 2: Qwen Image Edit (Per Clip)

| Step | Function | Input | Output |
|------|----------|-------|--------|
| 2.1 | Execute Qwen Image Edit | Base Scene + Qwen Edit Prompt | Start Frame |

**Parallel Processing**: All clips can be processed in parallel since each is independent.

#### Phase 3: LTX2 Video Generation (Per Clip)

| Step | Function | Input | Output |
|------|----------|-------|--------|
| 3.1 | Execute LTX2 Video | Start Frame + LTX2 Prompt + num_frames | Clip Video + LTX2 Audio |

**Parallel Processing**: All clips can be generated in parallel.

#### Phase 4: Stitch Video

| Step | Function | Input | Output |
|------|----------|-------|--------|
| 4.1 | Stitch Video Clips | All clips + transition config | Rough Video |
| 4.2 | Extract LTX2 Audio | Rough Video | LTX2 Audio (for WhisperX) |

#### Phase 5: Second Pass - Precision Sync

| Step | Function | Input | Output |
|------|----------|-------|--------|
| 5.1 | WhisperX Transcription | LTX2 Audio | Word timestamps |
| 5.2 | Align Script to Timestamps | Timestamps + Original Script | Time-aligned dialogue |
| 5.3 | IndexTTS Generation (ONCE) | Aligned dialogue + Emotions | **Final Audio** |
| 5.4 | Execute LTX2 Lip Sync | Rough Video + Final Audio | Final Video |

**Key Point**:
- Step 5.1: WhisperX runs on LTX2's temporary audio just to get timing
- Step 5.3: IndexTTS runs **ONCE**, generating final audio with precise timing
- Step 5.4: LTX2 lip sync uses the IndexTTS final audio, not LTX2's original audio

---

## Component Specifications

### 1. Deep Agent Planner

```python
# src/extensions/libraries/llm_studio/functions/planning/deep_agent_planner.py

from typing import Any
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class ClipPlan(BaseModel):
    """Plan for a single clip."""
    clip_id: str = Field(description="Unique clip identifier")
    start_line: int = Field(description="First dialogue line number (1-indexed)")
    end_line: int = Field(description="Last dialogue line number (1-indexed)")
    speakers: list[str] = Field(description="Speakers in this clip")
    camera_break_type: str = Field(description="Why this clip ends: speaker_change, scene_change, pause, or natural_end")
    duration_frames: int = Field(description="Number of frames for LTX2 (24fps)")
    emotion_vectors: dict[str, dict[str, float]] = Field(description="Emotion vectors per speaker")
    qwen_edit_prompt: str = Field(description="Prompt for Qwen Image Edit")
    qwen_negative_prompt: str = Field(description="Negative prompt for Qwen")
    ltx2_prompt: str = Field(description="Prompt for LTX2 video generation")
    ltx2_negative_prompt: str = Field(description="Negative prompt for LTX2")
    camera_angle: str = Field(description="Camera angle: wide, medium, close_up, extreme_close_up")
    additional_params: dict[str, Any] = Field(default={}, description="Any additional parameters")

class VideoPodcastPlan(BaseModel):
    """Complete plan for video podcast generation."""
    script_info: dict[str, Any] = Field(description="Script metadata")
    base_scene_reference: str = Field(description="Path to base studio scene image")
    clips: list[ClipPlan] = Field(description="Plan for each clip")

class DeepAgentPlanner:
    """
    Deep Agent for planning video podcast generation.

    Uses thinking mode to plan everything in one pass:
    - Clip breaks
    - Emotion vectors
    - Qwen Image Edit prompts
    - LTX2 prompts
    - Additional parameters
    """

    def __init__(self, config: dict, prompt_manager):
        self.config = config
        self.prompt_manager = prompt_manager

        # Initialize LLM with thinking enabled
        self.llm = ChatOpenAI(
            model=config.get("model", "gpt-4"),
            temperature=config.get("temperature", 0.7),
        )

    async def plan_video_podcast(
        self,
        script: str,
        base_scene_path: str,
        context: dict,
    ) -> VideoPodcastPlan:
        """
        Generate complete plan for video podcast.

        Args:
            script: Markdown script content
            base_scene_path: Path to base studio scene image
            context: Additional context (show info, speaker profiles, etc.)

        Returns:
            VideoPodcastPlan with all clip information
        """
        # Load the mega prompt
        mega_prompt = self.prompt_manager.load("deep_agent/mega_prompt.yaml")

        # Prepare variables
        variables = {
            "script": script,
            "base_scene_description": context.get("base_scene_description", ""),
            "show_context": context.get("show_context", ""),
            "speaker_profiles": context.get("speaker_profiles", ""),
            "style_guidelines": context.get("style_guidelines", ""),
        }

        # Render prompts
        system_prompt = mega_prompt.system
        user_prompt = self.prompt_manager.render(mega_prompt, variables)

        # Execute with structured output
        structured_llm = self.llm.with_structured_output(VideoPodcastPlan)

        result = await structured_llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        # Add base scene reference
        result.base_scene_reference = base_scene_path

        return result
```

### 2. Mega Prompt Template

```yaml
# prompts/deep_agent/mega_prompt.yaml

version: "1.0"
name: "deep_agent_video_podcast_planner"
description: "Master planning prompt for video podcast generation"

system: |
  You are an expert video production planner specializing in AI-generated podcast videos.
  Your task is to analyze a podcast script and create a complete production plan.

  ## Your Responsibilities

  You will decide:
  1. **Clip Breaks**: Where to cut between video clips
     - Consider speaker changes
     - Consider scene/topic changes
     - Consider natural pause points
     - Keep clips between 3-10 seconds (72-240 frames at 24fps)

  2. **Emotion Vectors**: For each speaker in each clip
     - Use 9-dimensional format: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm, intensity]
     - Values: 0.0-1.0
     - Intensity should be 0.1-0.3 for normal conversation
     - Match the emotional content of the dialogue

  3. **Qwen Image Edit Prompts**: For generating start frames
     - Describe zoom/edit from base studio scene
     - Focus on speaker position and expression
     - Include camera angle information
     - Keep concise (under 100 words)

  4. **LTX2 Video Prompts**: For video generation
     - Describe the motion and expression changes
     - Match the emotional tone
     - Include any gestures or movements
     - Keep concise (under 100 words)

  5. **Parameters**: Additional generation parameters
     - duration_frames: Based on dialogue length
     - camera_angle: wide, medium, close_up, extreme_close_up

  ## Clip Break Guidelines

  - **speaker_change**: Cut when a different person starts speaking
  - **scene_change**: Cut when topic/location shifts
  - **pause**: Cut at significant pauses in conversation
  - **natural_end**: End of content

  Calculate frames: `duration_seconds * 24`
  Estimate duration: Read the dialogue aloud, count seconds

  ## Character Consistency

  All clips start from the same base studio scene.
  Qwen Image Edit applies zoom-in operations to show specific speakers.
  This ensures consistent backgrounds, lighting, and speaker appearance.

user: |
  # Task: Plan Video Podcast Production

  ## Script

  {{script}}

  ## Base Scene Description

  {{base_scene_description}}

  ## Show Context

  {{show_context}}

  ## Speaker Profiles

  {{speaker_profiles}}

  ## Style Guidelines

  {{style_guidelines}}

  ---

  Please analyze this script and create a complete production plan.
  Return your response as a JSON object with the structure specified.
```

### 3. JSON to Markdown Converter

```python
# src/extensions/libraries/llm_studio/functions/planning/json_to_markdown.py

def plan_to_markdown(plan: VideoPodcastPlan) -> str:
    """
    Convert plan JSON to human-readable markdown.

    Useful for:
    - Review and iteration
    - Documentation
    - Debugging
    """
    lines = [
        f"# Video Podcast Production Plan",
        f"",
        f"## Script Info",
        f"- Title: {plan.script_info.get('title', 'N/A')}",
        f"- Total Lines: {plan.script_info.get('total_lines', 0)}",
        f"- Number of Clips: {len(plan.clips)}",
        f"",
        f"## Base Scene",
        f"Reference: `{plan.base_scene_reference}`",
        f"",
        f"## Clips",
        f"",
    ]

    for clip in plan.clips:
        lines.extend([
            f"### {clip.clip_id}",
            f"",
            f"- **Lines**: {clip.start_line}-{clip.end_line}",
            f"- **Speakers**: {', '.join(clip.speakers)}",
            f"- **Duration**: {clip.duration_frames} frames (~{clip.duration_frames/24:.1f}s)",
            f"- **Camera**: {clip.camera_angle}",
            f"- **Break Type**: {clip.camera_break_type}",
            f"",
            f"#### Emotions",
        ])

        for speaker, emotions in clip.emotion_vectors.items():
            e_str = ", ".join([f"{k}:{v:.2f}" for k, v in emotions.items()])
            lines.append(f"- **{speaker}**: {e_str}")

        lines.extend([
            f"",
            f"#### Qwen Edit Prompt",
            f"```",
            f"{clip.qwen_edit_prompt}",
            f"```",
            f"",
            f"#### LTX2 Prompt",
            f"```",
            f"{clip.ltx2_prompt}",
            f"```",
            f"",
        ])

    return "\n".join(lines)
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal**: Core extension structure + Deep Agent planner

- [ ] Create extension structure (extension.py, config.yaml, cli.py)
- [ ] Implement prompt manager system
- [ ] Implement Deep Agent planner
- [ ] Create mega prompt template
- [ ] Implement JSON to Markdown converter
- [ ] Test Deep Agent on sample scripts

**Deliverables**:
- Extension loads and registers
- Deep Agent can plan a simple script
- JSON + Markdown output working

### Phase 2: ComfyUI Integration (Week 1-2)

**Goal**: Bridge to ComfyUI for Qwen Edit + LTX2

- [ ] Create ComfyUI Bridge Service
- [ ] Define ComfyUI workflows (qwen_edit.json, ltx2_video.json, ltx2_lipsync.json)
- [ ] Implement Qwen Image Edit function
- [ ] Implement LTX2 Video Generation function

**Deliverables**:
- ComfyUI bridge working
- Can generate start frames with Qwen Edit
- Can generate video clips with LTX2

### Phase 3: First Pass Pipeline (Week 2)

**Goal**: End-to-end rough video

- [ ] Implement video stitching
- [ ] Wire up: Plan → Qwen Edit → LTX2 → Stitch
- [ ] Test on full script
- [ ] Verify character consistency

**Deliverables**:
- Complete first pass pipeline
- Rough video with LTX2 audio

### Phase 4: Second Pass - Lip Sync (Week 2-3)

**Goal**: Precision audio + lip sync

- [ ] Implement WhisperX transcription
- [ ] Implement timestamp alignment
- [ ] Implement IndexTTS with precise timing
- [ ] Implement LTX2 lip sync

**Deliverables**:
- Lip-synced final video

### Phase 5: Polish & CLI (Week 3)

**Goal**: Production-ready

- [ ] CLI interface
- [ ] Configuration presets
- [ ] Error handling
- [ ] Documentation

**Deliverables**:
- Usable CLI commands
- Documentation

---

## Configuration Schema

### Extension Config

```yaml
# config.yaml

extension:
  name: llm_studio
  type: feature
  display_name: LLM Studio
  description: Composable LLM Studio for multi-modal content generation
  version: 1.0.0
  category: feature
  enabled: true

python:
  use_host_env: true

capabilities:
  input_type: [text, markdown]
  output_type: [video, audio]
  streaming: false

tags: [studio, video, podcast, multimodal]

# Deep Agent settings
deep_agent:
  model: gpt-4  # Or your preferred model
  temperature: 0.7
  max_tokens: 8000
  thinking_enabled: true

# Base scene (manual prerequisite)
base_scene:
  path: data/inputs/base_studio_scene.png  # User creates this manually
  description: "Professional podcast studio with warm lighting"

# ComfyUI settings
comfyui:
  api_base: http://localhost:8188
  timeout: 300
  enabled: true

# LTX2 settings
ltx2:
  fps: 24
  guidance_scale: 7.0
  num_inference_steps: 50
  resolution: "1080p"

# WhisperX settings
whisperx:
  model: large-v3
  device: cuda
  compute_type: float16

# Output settings
output:
  directory: data/outputs/llm_studio
  video_format: mp4
  video_codec: libx264
  audio_codec: aac

# Presets
presets:
  draft:
    resolution: "720p"
    ltx2_steps: 30
  standard:
    resolution: "1080p"
    ltx2_steps: 50
  premium:
    resolution: "4k"
    ltx2_steps: 50
```

---

## CLI Interface

```bash
# Main command
uv run backend/manage.py studio <command> [options]

# Plan a video (Deep Agent only)
uv run backend/manage.py studio plan \
    --input script.md \
    --base-scene data/inputs/base_studio_scene.png \
    --output plan.json

# Generate video (full pipeline)
uv run backend/manage.py studio generate \
    --input script.md \
    --plan plan.json \
    --output video.mp4 \
    --preset standard

# Generate from plan (skip planning phase)
uv run backend/manage.py studio generate-from-plan \
    --plan plan.json \
    --output video.mp4
```

---

## File Summary

### New Files to Create

#### Extension Structure
- `src/extensions/features/llm_studio/extension.py`
- `src/extensions/features/llm_studio/config.yaml`
- `src/extensions/features/llm_studio/cli.py`

#### Core Library
- `src/extensions/libraries/llm_studio/manager.py`
- `src/extensions/libraries/llm_studio/config_schemas.py`
- `src/extensions/libraries/llm_studio/core/composable.py`
- `src/extensions/libraries/llm_studio/core/prompt_manager.py`
- `src/extensions/libraries/llm_studio/core/deep_agent.py`

#### Functions (Planning)
- `src/extensions/libraries/llm_studio/functions/planning/deep_agent_planner.py` ⭐ KEY
- `src/extensions/libraries/llm_studio/functions/planning/json_to_markdown.py`

#### Functions (Image)
- `src/extensions/libraries/llm_studio/functions/image/qwen_edit.py`
- `src/extensions/libraries/llm_studio/functions/image/comfyui_executor.py`

#### Functions (Video)
- `src/extensions/libraries/llm_studio/functions/video/ltx2_generation.py`
- `src/extensions/libraries/llm_studio/functions/video/ltx2_lipsync.py`
- `src/extensions/libraries/llm_studio/functions/video/stitching.py`

#### Functions (Audio)
- `src/extensions/libraries/llm_studio/functions/audio/whisperx_transcription.py`
- `src/extensions/libraries/llm_studio/functions/audio/timestamp_alignment.py`
- `src/extensions/libraries/llm_studio/functions/audio/indextts_generation.py`

#### Workflows
- `src/extensions/libraries/llm_studio/workflows/video_podcast.py`

#### Prompts
- `src/extensions/libraries/llm_studio/prompts/deep_agent/mega_prompt.yaml` ⭐ KEY

#### Service Extension
- `src/extensions/services/comfyui_bridge/extension.py`
- `src/extensions/services/comfyui_bridge/api.py`
- `src/extensions/services/comfyui_bridge/workflows/qwen_edit.json`
- `src/extensions/services/comfyui_bridge/workflows/ltx2_video.json`
- `src/extensions/services/comfyui_bridge/workflows/ltx2_lipsync.json`

---

## Next Steps

1. ✅ **Plan approved with Deep Agent architecture**
2. **User creates base studio scene** (manual prerequisite)
3. **Begin Phase 1** - Foundation + Deep Agent Planner
4. **Create and test mega prompt** - This is critical

---

*End of Plan*
