#!/usr/bin/env python3
"""Batch transcribe audio files using VibeVoice ASR with native diarization.

Follows official VibeVoice-ASR usage from:
https://github.com/microsoft/VibeVoice
"""

import json
import re
import time
import torch
import soundfile as sf
from pathlib import Path

# Patch JSON encoder for torch.dtype (needed for config loading)
_original_default = json.encoder.JSONEncoder.default
def _patched_default(self, o):
    if isinstance(o, torch.dtype):
        return str(o)
    return _original_default(self, o)
json.encoder.JSONEncoder.default = _patched_default
json.encoder._default = _patched_default

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

# Configuration
AUDIO_DIR = Path("/home/ubuntu/Documents/programs/deep-research-engine/data/ChatExport_2026-03-13/extracted_audio")
OUTPUT_DIR = Path("/home/ubuntu/Documents/programs/deep-research-engine/output/transcripts/vibevoice")
MODEL_PATH = "microsoft/VibeVoice-ASR"


def load_model():
    """Load model and processor following official instructions."""
    print("Loading processor...", flush=True)
    processor = VibeVoiceASRProcessor.from_pretrained(
        MODEL_PATH,
        language_model_pretrained_name="Qwen/Qwen2.5-7B"
    )
    print("Loading model...", flush=True)
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded! GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
    return model, processor


def transcribe_file(model, processor, audio_path):
    """Transcribe a single audio file with native diarization."""
    try:
        # Load audio
        waveform, sr = sf.read(audio_path)
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        duration = len(waveform) / sr

        # Process using official pattern
        inputs = processor(
            audio=[waveform],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True
        )
        inputs = {k: v.cuda() if hasattr(v, "to") else v for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=16384,
                pad_token_id=processor.pad_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,
            )

        # Decode only generated tokens (strip input prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]

        # Remove eos/pad tokens
        eos_pos = (generated_ids == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            generated_ids = generated_ids[:eos_pos[0]]

        generated_text = processor.decode(generated_ids, skip_special_tokens=True)

        # Post-process using official method
        try:
            segments = processor.post_process_transcription(generated_text)
        except Exception:
            # Fallback: extract JSON manually
            segments = extract_json_segments(generated_text)

        return {
            "success": True,
            "segments": segments,
            "raw_text": generated_text,
            "duration": duration
        }
    except Exception as e:
        return {"success": False, "error": str(e), "duration": 0}


def extract_json_segments(text):
    """Fallback JSON extraction from raw text."""
    match = re.search(r'\[\s*\{', text)
    if not match:
        return [{"Speaker": 0, "Start": 0, "End": 0, "Content": text}]
    remainder = text[match.start():]
    for end in re.finditer(r'\}\s*\]', remainder):
        try:
            return json.loads(remainder[:end.end()])
        except json.JSONDecodeError:
            continue
    return [{"Speaker": 0, "Start": 0, "End": 0, "Content": text}]


def format_transcript(result, filename):
    """Format transcription result as markdown."""
    lines = [f"## {filename}", ""]
    if not result["success"]:
        lines.append(f"**ERROR**: {result['error']}")
        return "\n".join(lines) + "\n"

    lines.append(f"**Duration**: {result['duration']:.2f}s")
    lines.append("")

    for seg in result["segments"]:
        speaker = seg.get("Speaker", 0)
        start = seg.get("Start", 0)
        end = seg.get("End", 0)
        content = seg.get("Content", "")
        lines.append(f"**Speaker {speaker}** [{start:.1f}s - {end:.1f}s]: {content}")

    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result["segments"], indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main():
    model, processor = load_model()

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    total = len(audio_files)
    print(f"\nFound {total} audio files to transcribe\n", flush=True)

    output_file = OUTPUT_DIR / "transcripts.md"
    json_output = OUTPUT_DIR / "transcripts.json"
    all_results = {}

    with open(output_file, "w") as f:
        f.write("# VibeVoice ASR Transcriptions (Native Diarization)\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

    for i, audio_path in enumerate(audio_files, 1):
        filename = audio_path.name
        print(f"[{i}/{total}] {filename} ", end="", flush=True)
        t0 = time.time()

        result = transcribe_file(model, processor, audio_path)
        all_results[filename] = result

        elapsed = time.time() - t0
        with open(output_file, "a") as f:
            f.write(format_transcript(result, filename))

        if result["success"]:
            speakers = set(s.get("Speaker", 0) for s in result["segments"])
            n_seg = len(result["segments"])
            print(f"OK {len(speakers)} speakers, {n_seg} segments, {result['duration']:.0f}s audio, {elapsed:.0f}s processing")
        else:
            print(f"ERROR: {result['error'][:100]}")

    with open(json_output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone! {output_file} and {json_output}")


if __name__ == "__main__":
    main()
