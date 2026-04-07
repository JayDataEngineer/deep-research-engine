# VibeVoice ASR Debug Notes

## Goal
Transcribe Telegram voice messages with speaker diarization for sharing with users without GPU.

## Issues Fixed So Far

### 1. VRAM Limit Too Strict
- **Problem**: Model manager reserves 10% VRAM overhead, so 24GB GPU shows as 21.6GB available
- **Fix**: Set `VIBEVOICE_MAX_VRAM=30` environment variable to override
- **File**: `services/tts/vibevoice/docker-compose.yml`

### 2. Missing Tokenizer Class
- **Problem**: `VibeVoiceASRTextTokenizerFast` doesn't exist, only `VibeVoiceTextTokenizerFast`
- **Fix**: Patch `vibevoice_asr_processor.py` to use `VibeVoiceTextTokenizerFast`
- **Location**: Line 110 and 142 in processor file

### 3. Missing speech_pad_id
- **Problem**: Qwen tokenizer doesn't have `<|speech_pad|>` token, returns None
- **Fix**: Add fallback token IDs in processor `_cache_special_tokens()` method
- **Code**:
  ```python
  self.speech_pad_id = self.tokenizer.convert_tokens_to_ids("<|speech_pad|>")
  if self.speech_pad_id is None:
      self.speech_pad_id = 151649  # Use unused token ID as fallback
  ```

### 4. verbose Argument Not Supported
- **Problem**: `verbose=False` passed to model.generate() but not supported
- **Fix**: Remove `verbose=False` from generate calls in api_server.py

### 5. is_final_chunk Not Supported
- **Problem**: Acoustic tokenizer's encode() method doesn't accept `is_final_chunk`
- **Fix**: Add parameter to encode method signature (even if unused)
- **File**: `vibevoice/modular/modular_vibevoice_tokenizer.py`

### 6. Current: Mask Shape Mismatch
- **Problem**: `mask [1, 677]` doesn't match `tensor [1, 676, 3584]`
- **Status**: Currently debugging...

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     VibeVoice ASR Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│  1. Load audio (sf.read) → 24kHz mono                               │
│  2. Processor converts to:                                           │
│     - input_ids (text tokens)                                        │
│     - speech_tensors (acoustic tokens)                               │
│     - acoustic_input_mask                                            │
│  3. Model generates transcription with native diarization            │
│  4. Parse output format: "Speaker 1: [0.0s - 2.5s] Hello..."          │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Files
- `/app/api_server.py` - FastAPI endpoints
- `/app/model_manager.py` - Model loading with VRAM management
- `/app/vibevoice/vibevoice/processor/vibevoice_asr_processor.py` - Audio processing
- `/app/vibevoice/vibevoice/modular/modeling_vibevoice_asr.py` - ASR model

## Running Tests
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.ogg" \
  -F "model=vibevoice-asr-7b" \
  -F "response_format=verbose_json"
```
