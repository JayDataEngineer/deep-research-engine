# VibeVoice ASR Integration Guide

**VibeVoice ASR-7B** is a unified audio solution that provides transcription, speaker diarization, and timestamps in a single pass.

---

## Overview

### What is VibeVoice ASR?

VibeVoice ASR-7B is an advanced speech recognition model that handles:

| Feature | Traditional Stack | VibeVoice ASR |
|---------|------------------|---------------|
| Transcription | Whisper / Faster-Whisper | ✅ Built-in |
| Speaker Diarization | pyannote.audio (separate) | ✅ Built-in |
| Timestamps | Separate alignment step | ✅ Built-in |
| Models Required | 2-3 | 1 |
| Processing Passes | Multiple | Single |

### Architecture

```
Your Application ──► Traefik ──► LiteLLM ──► VibeVoice ASR
                                         │
                                         ├─► Transcription
                                         ├─► Speaker IDs
                                         └─► Timestamps
```

---

## Quick Start

### 1. Start Services

```bash
cd ~/Documents/programs/shared-docker-infra

# Start the stack
task up:traefik
task up:litellm
task up:vibevoice-asr
```

### 2. Configure Deep Research Engine

Add to your `.env` file:

```bash
# Use VibeVoice ASR (default)
ASR_MODEL=vibevoice-asr

# Optional: Speaker detection range
VIBEVOICE_MIN_SPEAKERS=1
VIBEVOICE_MAX_SPEAKERS=10
```

### 3. Run a Workflow

```bash
# Transcribe a podcast or meeting
uv run deep-research research "Meeting notes" \
  --config configs/podcast_vibevoice.yaml \
  --set AUDIO_PATH=/path/to/meeting.mp3
```

---

## Usage Examples

### Python API (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-litellm-master",
    base_url="http://litellm.local/v1"
)

with open("meeting.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="vibevoice-asr",
        file=f,
        response_format="verbose_json"  # Get timestamps + speakers
    )

print(result.text)
print(f"Speakers: {result.speakers}")
print(f"Segments: {len(result.segments)}")
```

### cURL

```bash
curl http://litellm.local/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-litellm-master" \
  -F "model=vibevoice-asr" \
  -F "file=@your-audio.mp3"
```

### Expected Output

```json
{
  "text": "Speaker 0: Welcome everyone. Speaker 1: Thanks for having me.",
  "segments": [
    {
      "speaker": "Speaker 0",
      "start": 0.0,
      "end": 2.5,
      "duration": 2.5,
      "text": "Welcome everyone."
    },
    {
      "speaker": "Speaker 1",
      "start": 3.1,
      "end": 5.8,
      "duration": 2.7,
      "text": "Thanks for having me."
    }
  ],
  "speakers": 2,
  "diarization": "native",
  "duration": 120.5
}
```

---

## Deep Research Engine Integration

### Using the VibeVoice Worker

The `VibeVoiceASRWorker` provides native diarization support:

```python
from deep_research_engine.workers.implementations import VibeVoiceASRWorker

# In your workflow config
config = {
    "worker_type": "vibevoice_asr",
    "custom": {
        "audio_path": "/path/to/audio.mp3",
        "min_speakers": 1,
        "max_speakers": 10,
        "language": "en",  # Optional hint
        "response_format": "verbose_json"
    }
}
```

### Using the ASR Utility Function

```python
from deep_research_engine.utils.multimodal import asr_transcribe
from pathlib import Path

# With diarization (returns dict)
result = await asr_transcribe(
    Path("meeting.mp3"),
    include_diarization=True
)

print(f"Speakers: {result['speakers']}")
print(f"Segments: {result['segments']}")

# Without diarization (returns string, compatible with Whisper)
text = await asr_transcribe(
    Path("meeting.mp3"),
    include_diarization=False
)
```

### Ingestion Node (Automatic)

The ingestion node automatically uses VibeVoice ASR when processing audio files:

```yaml
# configs/telegram_vibevoice.yaml
phases:
  - name: ingestion
    node: ingestion_node
    config:
      custom:
        asr_model: vibevoice-asr
        include_diarization: true
```

---

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_MODEL` | `vibevoice-asr` | ASR model to use |
| `VIBEVOICE_MIN_SPEAKERS` | `1` | Minimum speakers to detect |
| `VIBEVOICE_MAX_SPEAKERS` | `10` | Maximum speakers to detect |
| `LITELLM_BASE_URL` | `http://litellm.local/v1` | LiteLLM endpoint |
| `LITELLM_API_KEY` | `sk-litellm-master` | LiteLLM API key |

### Worker Configuration

```yaml
workers:
  - id: asr_worker
    type: vibevoice_asr
    custom:
      audio_path: "/path/to/audio.mp3"
      min_speakers: 2
      max_speakers: 5
      language: "en"  # Optional language hint
      response_format: "verbose_json"
```

---

## Use Cases

### 1. Podcast Transcription

```bash
uv run deep-research research "Podcast analysis" \
  --config configs/podcast_vibevoice.yaml \
  --set AUDIO_PATH=/path/to/podcast.mp3
```

**Output**: Full transcript with speaker labels, timestamps, and key quotes per speaker.

### 2. Meeting Notes

```bash
uv run deep-research research "Meeting transcription" \
  --config configs/telegram_vibevoice.yaml \
  --set EXPORT_PATH=/path/to/meeting/export
```

**Output**: Processed meeting notes with speaker identification and action items.

### 3. Interview Analysis

```python
from deep_research_engine import run_workflow

result = await run_workflow(
    query="Interview analysis",
    config="configs/podcast_vibevoice.yaml",
    context={"audio_path": "/path/to/interview.mp3"}
)

# Access speaker segments
for segment in result["transcription"]["segments"]:
    print(f"{segment['speaker']}: {segment['text']}")
```

---

## Output Format

### Structured Transcript

```markdown
# Transcript: meeting.mp3

**Duration**: 1800.5s
**Speakers**: 3

## Full Transcript

Speaker 0: Welcome everyone to the meeting...
Speaker 1: Thanks for having me...
Speaker 2: Let's get started...

## Speaker Segments

- **Speaker 0** [0.0s - 15.2s]: Welcome everyone to the meeting...
- **Speaker 1** [15.8s - 28.4s]: Thanks for having me...
- **Speaker 2** [29.0s - 45.6s]: Let's get started...
```

### Programmatic Access

```python
# Access full text
text = result["text"]

# Access individual segments
for segment in result["segments"]:
    speaker = segment["speaker"]
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    
# Access metadata
speakers = result["speakers"]
duration = result["duration"]
diarization_type = result["diarization"]  # "native"
```

---

## Comparison: VibeVoice vs Whisper

### VibeVoice ASR (Recommended)

✅ **Pros:**
- Built-in speaker diarization
- Precise timestamps
- Single API call
- No post-processing needed
- Accurate speaker separation

❌ **Cons:**
- Larger model (7B parameters)
- Slightly slower than Whisper

### Whisper

✅ **Pros:**
- Faster for simple transcription
- Smaller model footprint
- Well-established

❌ **Cons:**
- No built-in diarization
- Requires separate pyannote.audio
- Multiple processing passes
- Post-processing needed

---

## Troubleshooting

### Issue: No Speaker Diarization

**Solution**: Ensure you're using `vibevoice-asr` model and `response_format="verbose_json"`:

```python
result = client.audio.transcriptions.create(
    model="vibevoice-asr",  # Not "whisper-1"
    file=audio_file,
    response_format="verbose_json"  # Required for diarization
)
```

### Issue: Wrong Speaker Count

**Solution**: Adjust min/max speakers in config:

```yaml
custom:
  min_speakers: 1
  max_speakers: 10  # Increase if more speakers expected
```

### Issue: Audio File Not Found

**Solution**: Verify path exists and is accessible:

```python
from pathlib import Path

audio_path = Path("/path/to/audio.mp3")
assert audio_path.exists(), f"Audio file not found: {audio_path}"
```

---

## Performance

### Benchmarks

| Audio Length | Processing Time | Output |
|--------------|----------------|--------|
| 5 min | ~30s | Transcription + 2 speakers |
| 30 min | ~2-3 min | Transcription + 5 speakers |
| 60 min | ~5-6 min | Transcription + 8 speakers |

### Limitations

- Maximum audio length: 60 minutes
- Maximum speakers: 10 (configurable)
- Supported formats: MP3, WAV, OGG, M4A, FLAC

---

## Advanced Features

### Custom Speaker Naming

By default, speakers are labeled as "Speaker 0", "Speaker 1", etc. To map these to actual names:

```python
speaker_map = {
    "Speaker 0": "Alice",
    "Speaker 1": "Bob",
    "Speaker 2": "Charlie"
}

for segment in result["segments"]:
    speaker = speaker_map.get(segment["speaker"], segment["speaker"])
    print(f"{speaker}: {segment['text']}")
```

### Segment Filtering

Filter segments by duration or confidence:

```python
# Only segments longer than 5 seconds
long_segments = [
    seg for seg in result["segments"]
    if seg["duration"] > 5.0
]

# Only high-confidence segments
confident_segments = [
    seg for seg in result["segments"]
    if seg.get("confidence", 1.0) > 0.9
]
```

### Timeline Generation

```python
def generate_timeline(segments):
    timeline = []
    for seg in segments:
        timeline.append({
            "time": f"{seg['start']:.1f}s",
            "speaker": seg["speaker"],
            "summary": seg["text"][:50] + "..."
        })
    return timeline

timeline = generate_timeline(result["segments"])
```

---

## Integration with Other Tools

### LangFuse Tracing

VibeVoice ASR calls are automatically traced via LangFuse:

```python
from langfuse import Langfuse

langfuse = Langfuse()
trace = langfuse.trace(name="vibevoice-asr-call")
```

### Neo4j Knowledge Graph

Store speaker relationships in Neo4j:

```cypher
// Create speaker nodes
MERGE (s0:Speaker {id: "Speaker 0", name: "Alice"})
MERGE (s1:Speaker {id: "Speaker 1", name: "Bob"})

// Create conversation edges
MATCH (s0:Speaker), (s1:Speaker)
CREATE (s0)-[:SPOKE_AFTER]->(s1)
```

---

## Next Steps

1. **Try the demo**: Run `uv run deep-research` with `configs/podcast_vibevoice.yaml`
2. **Customize**: Adjust speaker detection parameters for your use case
3. **Integrate**: Use the `VibeVoiceASRWorker` in your custom workflows
4. **Explore**: Check out `configs/telegram_vibevoice.yaml` for chat processing

---

## References

- [VibeVoice ASR-7B Documentation](https://github.com/shared-docker-infra/docs/vibevoice)
- [LiteLLM Audio Transcription](https://docs.litellm.ai/docs/audio)
- [Deep Research Engine](../../README.md)
- [shared-docker-infra](../../shared-docker-infra/README.md)
