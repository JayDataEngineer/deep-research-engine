#!/usr/bin/env python3
"""Batch transcribe audio files using Qwen3-ASR-1.7B with timestamps.

Uses the qwen-asr Python package with transformers backend.
Note: Qwen3-ASR does NOT support speaker diarization - it outputs plain
transcription with optional word-level timestamps via the forced aligner.
"""

import json
import time
from pathlib import Path
import torch
from qwen_asr import Qwen3ASRModel

# Configuration
AUDIO_DIR = Path("/home/ubuntu/Documents/programs/deep-research-engine/data/ChatExport_2026-03-13/extracted_audio")
OUTPUT_DIR = Path("/home/ubuntu/Documents/programs/deep-research-engine/output/transcripts/qwen-asr")


def load_model():
    """Load Qwen3-ASR model with forced aligner for timestamps."""
    print("Loading Qwen3-ASR-1.7B...", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=8,
        max_new_tokens=2048,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
        ),
    )
    print("Model loaded!", flush=True)
    return model


def transcribe_file(model, audio_path):
    """Transcribe a single file with timestamps."""
    try:
        import soundfile as sf
        waveform, sr = sf.read(audio_path)
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        duration = len(waveform) / sr

        results = model.transcribe(
            audio=[(waveform, sr)],
            language=None,
            return_time_stamps=True,
        )

        r = results[0]
        segments = []
        if r.time_stamps:
            for ts in r.time_stamps:
                segments.append({
                    "text": ts.text if hasattr(ts, 'text') else str(ts),
                    "start": round(ts.start_time, 2) if hasattr(ts, 'start_time') else 0,
                    "end": round(ts.end_time, 2) if hasattr(ts, 'end_time') else 0,
                })

        return {
            "success": True,
            "language": r.language,
            "text": r.text,
            "duration": round(duration, 2),
            "segments": segments,
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{e}\n{traceback.format_exc()}", "duration": 0}


def format_transcript(result, filename):
    """Format as markdown."""
    lines = [f"## {filename}", ""]
    if not result["success"]:
        lines.append(f"**ERROR**: {result['error']}")
        return "\n".join(lines) + "\n"

    lines.append(f"**Language**: {result['language']}")
    lines.append(f"**Duration**: {result['duration']:.2f}s")
    lines.append("")
    lines.append(result["text"])
    lines.append("")

    if result["segments"]:
        lines.append("### Timestamps")
        for seg in result["segments"]:
            lines.append(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")

    lines.append("")
    return "\n".join(lines)


def main():
    model = load_model()

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    total = len(audio_files)
    print(f"\nFound {total} audio files\n", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "transcripts.md"
    json_output = OUTPUT_DIR / "transcripts.jsonl"
    all_results = {}

    with open(output_file, "w") as f:
        f.write("# Qwen3-ASR Transcriptions\n\n")
        f.write(f"Model: Qwen3-ASR-1.7B + ForcedAligner-0.6B\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

    with open(json_output, "w") as jf:
        for i, audio_path in enumerate(audio_files, 1):
            filename = audio_path.name
            print(f"[{i}/{total}] {filename} ", end="", flush=True)
            t0 = time.time()

            result = transcribe_file(model, audio_path)
            all_results[filename] = result

            elapsed = time.time() - t0
            with open(output_file, "a") as f:
                f.write(format_transcript(result, filename))

            jf.write(json.dumps({"file": filename, **result}, ensure_ascii=False) + "\n")

            if result["success"]:
                n_ts = len(result["segments"])
                print(f"OK {result['language']}, {n_ts} timestamps, "
                      f"{result['duration']:.0f}s audio, {elapsed:.0f}s proc")
            else:
                print(f"ERROR: {result['error'][:100]}")

    with open(OUTPUT_DIR / "transcripts.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {output_file}")


if __name__ == "__main__":
    main()
