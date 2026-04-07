#!/usr/bin/env python3
"""Batch transcribe audio files using Faster-Whisper with word-level timestamps.

Uses the faster-whisper library (CTranslate2 backend) for fast GPU inference.
Outputs word-level timestamps that can be combined with pyannote diarization.
"""

import json
import time
import csv
from pathlib import Path
from faster_whisper import WhisperModel

# Configuration
AUDIO_DIR = Path("/home/ubuntu/Documents/programs/deep-research-engine/data/ChatExport_2026-03-13/extracted_audio")
OUTPUT_DIR = Path("/home/ubuntu/Documents/programs/deep-research-engine/output/transcripts/faster-whisper")
MODEL_SIZE = "Systran/faster-whisper-base"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"


def load_model():
    """Load Faster-Whisper model."""
    print(f"Loading {MODEL_SIZE}...", flush=True)
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    print("Model loaded!", flush=True)
    return model


def transcribe_file(model, audio_path):
    """Transcribe a single file with word-level timestamps."""
    try:
        segments, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        result_segments = []
        result_words = []
        full_text_parts = []

        for seg in segments:
            seg_dict = {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
            result_segments.append(seg_dict)
            full_text_parts.append(seg.text.strip())

            for word in seg.words:
                result_words.append({
                    "word": word.word.strip(),
                    "start": round(word.start, 2),
                    "end": round(word.end, 2),
                    "probability": round(word.probability, 3),
                })

        return {
            "success": True,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
            "segments": result_segments,
            "words": result_words,
            "text": " ".join(full_text_parts),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "duration": 0}


def format_transcript(result, filename):
    """Format as markdown."""
    lines = [f"## {filename}", ""]
    if not result["success"]:
        lines.append(f"**ERROR**: {result['error']}")
        return "\n".join(lines) + "\n"

    lines.append(f"**Language**: {result['language']} ({result['language_probability']:.0%})")
    lines.append(f"**Duration**: {result['duration']:.2f}s")
    lines.append("")

    for seg in result["segments"]:
        lines.append(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")

    lines.append("")
    lines.append("### Word Timestamps")
    for w in result["words"]:
        lines.append(f"  {w['start']:.2f}-{w['end']:.2f} {w['word']}")

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
        f.write("# Faster-Whisper Transcriptions\n\n")
        f.write(f"Model: {MODEL_SIZE}\n")
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

            # Write JSONL (one line per file)
            jf.write(json.dumps({"file": filename, **result}) + "\n")

            if result["success"]:
                n_seg = len(result["segments"])
                n_words = len(result["words"])
                print(f"OK {result['language']} {n_seg} segs, {n_words} words, "
                      f"{result['duration']:.0f}s audio, {elapsed:.0f}s proc")
            else:
                print(f"ERROR: {result['error'][:100]}")

    # Also write full JSON
    with open(OUTPUT_DIR / "transcripts.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {output_file}")


if __name__ == "__main__":
    main()
