#!/usr/bin/env python3
"""Test VibeVoice ASR inference following official documentation."""
import json
import torch
import time
import soundfile as sf

_original_default = json.encoder.JSONEncoder.default
def _patched_default(self, o):
    if isinstance(o, torch.dtype):
        return str(o)
    return _original_default(self, o)
json.encoder.JSONEncoder.default = _patched_default
json.encoder._default = _patched_default

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

model_path = "microsoft/VibeVoice-ASR"

print("Loading processor...", flush=True)
processor = VibeVoiceASRProcessor.from_pretrained(
    model_path,
    language_model_pretrained_name="Qwen/Qwen2.5-7B"
)
print("Processor loaded!", flush=True)

print("Loading model...", flush=True)
model = VibeVoiceASRForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True
)
model.eval()
print("Model loaded!", flush=True)

audio_path = "/data/ChatExport_2026-03-13/extracted_audio/IMG_5795.wav"
waveform, sr = sf.read(audio_path)
if len(waveform.shape) > 1:
    waveform = waveform.mean(axis=1)
print(f"Audio: {len(waveform)/sr:.1f}s at {sr}Hz", flush=True)

inputs = processor(
    audio=[waveform],
    sampling_rate=sr,
    return_tensors="pt",
    padding=True,
    add_generation_prompt=True
)
inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
print(f"Input IDs shape: {inputs['input_ids'].shape}", flush=True)

start = time.time()
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=8192,
        pad_token_id=processor.pad_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )
elapsed = time.time() - start
print(f"Generation took {elapsed:.1f}s", flush=True)

input_length = inputs["input_ids"].shape[1]
generated_ids = output_ids[0, input_length:]

eos_positions = (generated_ids == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
if len(eos_positions) > 0:
    generated_ids = generated_ids[:eos_positions[0]]

generated_text = processor.decode(generated_ids, skip_special_tokens=True)
print(f"\n=== RAW OUTPUT ({len(generated_text)} chars) ===", flush=True)
print(generated_text[:1000], flush=True)

segments = processor.post_process_transcription(generated_text)
print(f"\n=== SEGMENTS: {len(segments)} ===", flush=True)
for s in segments[:10]:
    speaker = s.get("Speaker", "?")
    start_t = s.get("Start", 0)
    end_t = s.get("End", 0)
    content = s.get("Content", "")
    print(f"  Speaker {speaker} [{start_t:.1f}-{end_t:.1f}]: {content[:80]}")
if len(segments) > 10:
    print(f"  ... ({len(segments)-10} more)")
