# VibeVoice ASR Integration - Quick Reference

## What Changed

✅ **New Worker**: `VibeVoiceASRWorker` - Dedicated worker for VibeVoice ASR-7B
✅ **Enhanced Utils**: `asr_transcribe()` now supports diarization
✅ **Updated Ingestion**: Automatically processes audio with speaker detection
✅ **New Configs**: `telegram_vibevoice.yaml` and `podcast_vibevoice.yaml`

## Files Modified

```
src/deep_research_engine/
├── workers/implementations/
│   ├── vibevoice_asr.py          # NEW: VibeVoice worker
│   └── __init__.py               # Updated exports
├── utils/
│   └── multimodal.py              # Enhanced asr_transcribe()
└── nodes/
    └── ingestion.py               # Audio processing with diarization

configs/
├── telegram_vibevoice.yaml        # NEW: Telegram + VibeVoice
└── podcast_vibevoice.yaml         # NEW: Podcast transcription

docs/
└── vibevoice-asr.md               # NEW: Complete guide

.env.example                       # Updated with ASR settings
```

## Quick Usage

### 1. Set Environment

```bash
# .env
ASR_MODEL=vibevoice-asr
VIBEVOICE_MIN_SPEAKERS=1
VIBEVOICE_MAX_SPEAKERS=10
```

### 2. Run Podcast Transcription

```bash
uv run deep-research research "Meeting notes" \
  --config configs/podcast_vibevoice.yaml \
  --set AUDIO_PATH=/path/to/meeting.mp3
```

### 3. Process Telegram Export

```bash
uv run deep-research research "Chat analysis" \
  --config configs/telegram_vibevoice.yaml \
  --set EXPORT_PATH=/path/to/telegram/export
```

## API Examples

### Using the Worker

```python
from deep_research_engine.workers.implementations import VibeVoiceASRWorker

worker = VibeVoiceASRWorker()
result = await worker.execute(
    state={"context": {"audio_path": "meeting.mp3"}},
    config=WorkerConfig(custom={
        "min_speakers": 2,
        "max_speakers": 5
    })
)

# Access results
print(f"Speakers: {result.output['speakers']}")
for seg in result.output['segments']:
    print(f"{seg['speaker']}: {seg['text']}")
```

### Using the Utility

```python
from deep_research_engine.utils.multimodal import asr_transcribe
from pathlib import Path

# With diarization
result = await asr_transcribe(Path("meeting.mp3"), include_diarization=True)
print(result)  # Dict with segments, speakers, timestamps

# Without diarization (Whisper compatibility)
text = await asr_transcribe(Path("meeting.mp3"), include_diarization=False)
print(text)  # Plain string
```

## Output Format

```python
{
    "text": "Full transcription text...",
    "segments": [
        {
            "speaker": "Speaker 0",
            "start": 0.0,      # Start time in seconds
            "end": 2.5,        # End time in seconds
            "duration": 2.5,   # Segment duration
            "text": "What was said",
            "confidence": 0.95
        },
        # ... more segments
    ],
    "speakers": 3,              # Number of unique speakers
    "speaker_list": ["Speaker 0", "Speaker 1", "Speaker 2"],
    "duration": 1800.5,         # Total audio duration
    "diarization": "native",    # Indicates VibeVoice native diarization
    "model": "vibevoice-asr"
}
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_MODEL` | `vibevoice-asr` | Model to use |
| `VIBEVOICE_MIN_SPEAKERS` | `1` | Min speakers to detect |
| `VIBEVOICE_MAX_SPEAKERS` | `10` | Max speakers to detect |

### Worker Config

```yaml
workers:
  - id: asr
    type: vibevoice_asr
    custom:
      audio_path: "/path/to/audio.mp3"
      min_speakers: 1
      max_speakers: 10
      language: "en"  # Optional
      response_format: "verbose_json"
```

## Benefits Over Whisper

| Feature | Whisper | VibeVoice ASR |
|---------|---------|---------------|
| Transcription | ✅ | ✅ |
| Speaker Diarization | ❌ (needs pyannote) | ✅ Built-in |
| Timestamps | ❌ (needs alignment) | ✅ Built-in |
| Processing Passes | 2-3 | 1 |
| Models Required | 2-3 | 1 |
| API Calls | Multiple | Single |

## Testing

```bash
# Check syntax
uv run ruff check src/deep_research_engine/workers/implementations/vibevoice_asr.py

# Run with test audio
uv run deep-research research "Test" \
  --config configs/podcast_vibevoice.yaml \
  --set AUDIO_PATH=./test_audio.mp3
```

## Troubleshooting

**No diarization?**
- Ensure `ASR_MODEL=vibevoice-asr` (not `whisper-1`)
- Use `response_format="verbose_json"`

**Wrong speaker count?**
- Adjust `VIBEVOICE_MIN_SPEAKERS` and `VIBEVOICE_MAX_SPEAKERS`

**Audio not found?**
- Verify path exists: `ls /path/to/audio.mp3`

## Documentation

- **Full Guide**: [`docs/vibevoice-asr.md`](docs/vibevoice-asr.md)
- **Telegram Config**: [`configs/telegram_vibevoice.yaml`](configs/telegram_vibevoice.yaml)
- **Podcast Config**: [`configs/podcast_vibevoice.yaml`](configs/podcast_vibevoice.yaml)

## Next Steps

1. ✅ Integration complete
2. 🔄 Test with your audio files
3. 📊 Review diarization quality
4. 🔧 Adjust speaker detection parameters if needed
5. 🎯 Integrate into your workflows

---

**Status**: ✅ Complete | **Version**: 1.0.0 | **Date**: 2026-03-25
