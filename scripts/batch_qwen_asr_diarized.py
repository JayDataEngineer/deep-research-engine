#!/usr/bin/env python3
"""Pipeline 3: Qwen3-ASR transcription + Pyannote speaker diarization.

Qwen3-ASR produces accurate ASR with word-level timestamps via ForcedAligner.
Pyannote identifies who spoke when.
Combined: full diarized transcription with speaker labels.

Prerequisites:
  - Qwen3-ASR-1.7B model cached locally
  - Qwen3-ForcedAligner-0.6B cached locally
  - pyannote/speaker-diarization-community-1 accepted on HuggingFace
  - HF_TOKEN set in environment or .env

Usage:
  python3 scripts/batch_qwen_asr_diarized.py
"""

import json
import csv
import time
import torch
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "data/ChatExport_2026-03-13/extracted_audio"
OUTPUT_DIR = BASE_DIR / "output/transcripts/qwen-asr-diarized"
MODEL_CACHE = Path(os.environ.get("HF_HOME", str(BASE_DIR / "models/cache/huggingface")))

# Load HF token
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    env_file = Path("/home/ubuntu/Documents/programs/shared-docker-infra/.env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                HF_TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found. Set it in environment or .env")
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_qwen_model():
    """Load Qwen3-ASR with ForcedAligner for word-level timestamps."""
    from qwen_asr import Qwen3ASRModel

    print("Loading Qwen3-ASR-1.7B + ForcedAligner...", flush=True)
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
    return model


def load_pyannote():
    """Load Pyannote speaker diarization pipeline."""
    from pyannote.audio import Pipeline

    print("Loading Pyannote speaker diarization...", flush=True)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=HF_TOKEN,
    )
    pipeline = pipeline.to(torch.device("cuda"))
    return pipeline


def get_speaker_turns(pyannote_pipeline, audio_path):
    """Run Pyannote diarization on an audio file, return speaker turns."""
    import soundfile as sf

    waveform, sr = sf.read(audio_path)
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    diarization_output = pyannote_pipeline(
        {"waveform": torch.FloatTensor(waveform).unsqueeze(0), "sample_rate": sr},
    )
    diarization = diarization_output.speaker_diarization

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
        })
    return turns


def assign_words_to_speakers(qwen_timestamps, speaker_turns):
    """Assign speaker labels to Qwen word timestamps based on Pyannote speaker turns.

    Each word gets the speaker whose turn has the most overlap with the word.
    """
    if not qwen_timestamps or not speaker_turns:
        return []

    sorted_turns = sorted(speaker_turns, key=lambda t: t["start"])

    word_assignments = []
    for ts in qwen_timestamps:
        word_start = ts.get("start_time", ts.get("start", 0))
        word_end = ts.get("end_time", ts.get("end", 0))
        word_mid = (word_start + word_end) / 2
        text = ts.get("text", "")

        best_speaker = "UNKNOWN"
        best_overlap = 0

        for turn in sorted_turns:
            overlap_start = max(word_start, turn["start"])
            overlap_end = min(word_end, turn["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

            if turn["start"] > word_end:
                break

        word_assignments.append({
            "speaker": best_speaker,
            "start": round(word_start, 2),
            "end": round(word_end, 2),
            "text": text,
        })

    return word_assignments


def merge_speaker_segments(word_assignments):
    """Merge consecutive words from the same speaker into segments."""
    if not word_assignments:
        return []

    segments = []
    current = dict(word_assignments[0])

    for w in word_assignments[1:]:
        if w["speaker"] == current["speaker"]:
            current["end"] = w["end"]
            current["text"] += " " + w["text"]
        else:
            segments.append(current)
            current = dict(w)

    segments.append(current)
    return segments


def transcribe_file(qwen_model, pyannote_pipeline, audio_path):
    """Transcribe with Qwen-ASR and diarize with Pyannote."""
    import soundfile as sf

    try:
        waveform, sr = sf.read(audio_path)
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        duration = len(waveform) / sr

        # Check for silence
        import numpy as np
        rms = np.sqrt(np.mean(waveform ** 2))
        if rms < 1e-6:
            return {
                "success": True,
                "duration": duration,
                "segments": [],
                "speakers": [],
                "note": "silent audio",
            }

        # Step 1: Qwen ASR with word-level timestamps
        results = qwen_model.transcribe(
            audio=str(audio_path),
            language=None,
            return_time_stamps=True,
        )

        if not results or not results[0].time_stamps:
            return {
                "success": True,
                "duration": duration,
                "segments": [],
                "speakers": [],
                "text": results[0].text if results else "",
                "note": "no timestamps",
            }

        qwen_text = results[0].text
        qwen_timestamps = []
        for ts in results[0].time_stamps:
            qwen_timestamps.append({
                "text": ts.text,
                "start": round(ts.start_time, 2),
                "end": round(ts.end_time, 2),
            })

        # Step 2: Pyannote diarization
        speaker_turns = get_speaker_turns(pyannote_pipeline, audio_path)

        # Step 3: Combine
        word_assignments = assign_words_to_speakers(qwen_timestamps, speaker_turns)
        segments = merge_speaker_segments(word_assignments)

        speakers = sorted(set(s["speaker"] for s in segments)) if segments else []

        return {
            "success": True,
            "duration": duration,
            "text": qwen_text,
            "language": results[0].language if results else "",
            "segments": segments,
            "speakers": speakers,
            "num_speakers": len(speakers),
            "speaker_turns": speaker_turns,
            "word_count": len(qwen_timestamps),
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"{e}\n{traceback.format_exc()}",
            "duration": 0,
            "segments": [],
            "speakers": [],
        }


def main():
    print("=" * 60)
    print("Pipeline 3: Qwen3-ASR + Pyannote Speaker Diarization")
    print("=" * 60)

    qwen_model = load_qwen_model()
    pyannote_pipeline = load_pyannote()

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    total = len(audio_files)
    print(f"\nFound {total} audio files\n", flush=True)

    all_results = {}
    json_output = OUTPUT_DIR / "transcripts.json"
    jsonl_output = OUTPUT_DIR / "transcripts.jsonl"
    md_output = OUTPUT_DIR / "transcripts.md"

    with open(md_output, "w") as f:
        f.write("# Qwen3-ASR + Pyannote Diarized Transcriptions\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

    with open(jsonl_output, "w") as jf:
        for i, audio_path in enumerate(audio_files, 1):
            filename = audio_path.name
            print(f"[{i}/{total}] {filename} ", end="", flush=True)
            t0 = time.time()

            result = transcribe_file(qwen_model, pyannote_pipeline, audio_path)
            all_results[filename] = result
            elapsed = time.time() - t0

            # Write markdown
            with open(md_output, "a") as f:
                f.write(f"## {filename}\n\n")
                if not result["success"]:
                    f.write(f"**ERROR**: {result['error'][:200]}\n\n")
                    continue

                f.write(f"**Duration**: {result['duration']:.1f}s | ")
                f.write(f"**Language**: {result.get('language','?')} | ")
                f.write(f"**Speakers**: {result.get('num_speakers',0)} | ")
                f.write(f"**Segments**: {len(result['segments'])}\n\n")

                for seg in result["segments"]:
                    f.write(f"- **{seg['speaker']}** [{seg['start']:.1f}-{seg['end']:.1f}s]: {seg['text']}\n")
                f.write("\n")

            # Write JSONL
            jf.write(json.dumps({"file": filename, **result}, ensure_ascii=False) + "\n")

            # Progress
            if result["success"] and result["segments"]:
                n_seg = len(result["segments"])
                n_spk = result.get("num_speakers", 0)
                print(f"OK {n_spk} speakers, {n_seg} segments, {result['duration']:.0f}s audio, {elapsed:.0f}s")
            elif result["success"]:
                note = result.get("note", "empty")
                print(f"EMPTY ({note}), {result['duration']:.0f}s, {elapsed:.0f}s")
            else:
                print(f"ERROR: {str(result['error'])[:100]}")

    # Write JSON
    with open(json_output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    ok = sum(1 for r in all_results.values() if r["success"] and r["segments"])
    silent = sum(1 for r in all_results.values() if r["success"] and not r["segments"] and r.get("note") == "silent audio")
    errors = sum(1 for r in all_results.values() if not r["success"])

    print(f"\n{'='*60}")
    print(f"DONE! {ok}/{total} diarized, {silent} silent, {errors} errors")
    print(f"Output: {OUTPUT_DIR}")
    print(f"JSON: {json_output}")


if __name__ == "__main__":
    main()
