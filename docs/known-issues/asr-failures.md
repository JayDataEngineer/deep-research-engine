# Known Issues — Ingestion Pipeline

## 2026-04-08: llama.cpp Chat Completions — HTTP 500 (json parse error)

### Symptom
ALL chat completions via the orchestrator fail with HTTP 500.

### Error
```
Entity extraction failed: chat failed (HTTP 500):
{'error': {'code': 500, 'message': '[json.exception.parse_error.101] parse error
at line 1, column 1: attempting to parse an empty input; check that your input
string or stream contains the expected JSON', 'type': 'server_error'}}
```

### Root Cause (CONFIRMED)
llama.cpp build 8643 (f49e91787) crashes internally when processing any
`/v1/chat/completions` request. The server container logs show:
```
srv  operator(): got exception: {"error":{"code":500,...parse error...}}
```
The model loads fine, slots are idle, but every chat request triggers this
internal parse exception. Even a minimal `{"model":"gemma-4-e4b","messages":[...]}` fails.

### Impact
- Phase 1.5 NER extraction produces 0 entities (falls back gracefully)
- DeepAgents sub-agents (Sort, Entity Extractor, Entity Finder) will fail
- Drafter synthesis LLM calls via `infra.chat()` also fail

### Status
This is a llama.cpp/infrastructure issue, not application code. The orchestrator
is proxying correctly but llama.cpp's internal JSON handler crashes.

### Fix Needed
- [ ] Update llama.cpp container to a newer build
- [ ] Or investigate if specific request fields trigger the parse error

---

## 2026-04-08: VibeVoice ASR — m4a/librosa/audioread hang (RESOLVED)

### Root Cause (CONFIRMED)
VibeVoice ASR uses librosa for audio loading. For m4a/AAC files, PySoundFile
fails and librosa falls back to audioread which hangs indefinitely.

### Fixes Applied
1. Pre-convert all non-WAV files to 16kHz mono WAV via ffmpeg before ASR
2. Reduced fallback segment duration from 300s to 120s
3. Per-segment timeout via `asyncio.wait_for()` (segment_duration + 120s)
4. Reduced segment retries from 3 to 2
5. HTTP 500 fast-fail (skip retry on non-transient errors)
6. Full-file threshold lowered from 300s to 120s (larger files go straight to segments)

### Results (after redeployment + fixes)
- `audio_1.ogg`: 750 chars, 2 segments, 1 speaker ✅
- `audio_2.ogg`: 1846 chars, 10 segments, 1 speaker ✅
- `New Recording 34.m4a` (538s): 40 segments, 2 speakers ✅
- All m4a files now pre-converted to WAV before ASR

---

## 2026-04-08: InfraClient "Slot already loading" race condition (RESOLVED)

### Root Cause
When multiple `InfraClient` instances (or concurrent calls) try to load the same
model, the second call hits `HTTP 400: Slot asr is already loading` because the
orchestrator doesn't report which model is loading — only `loading: true` with
no `current_model` field.

### Fix Applied
Added handling in `InfraClient.load()` for the case where a slot is already
loading: instead of trying to load again or unload everything, wait for the
current load to complete via `_wait_healthy()`.
