#!/usr/bin/env python3
"""Batch speaker diarization using pyannote.community-1.

Combines with Faster-Whisper transcriptions for speaker-attributed text.
"""
import os
import json
import csv
import time
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

HF_TOKEN = os.environ.get("HF_TOKEN", "")
AUDIO_DIR = Path("/data/ChatExport_2026-03-13/extracted_audio")
WHISPER_JSON = Path("/output/transcripts/whisper/transcripts_faster-whisper_20260327_001647.jsonl")
OUTPUT_DIR = Path("/output/transcripts/pyannote")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_pipeline():
    print("Loading pyannote community-1 pipeline...", flush=True)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=HF_TOKEN if HF_TOKEN else None
    )
    pipeline.to(torch.device("cuda"))
    print("Pipeline loaded!", flush=True)
    return pipeline

def diarize_file(pipeline, audio_path):
    """Run diarization on a single file."""
    try:
        waveform, sr = sf.read(audio_path)
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        duration = len(waveform) / sr

        # pyannote expects dict with waveform tensor
        audio_dict = {
            "waveform": torch.FloatTensor(waveform).unsqueeze(0),
            "sample_rate": sr
        }

        t0 = time.time()
        with ProgressHook() as hook:
            diarization = pipeline(audio_dict, hook=hook)
        elapsed = time.time() - t0

        # Extract speaker turns
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker
            })

        speakers = list(set(t["speaker"] for t in turns))
        return {
            "success": True,
            "turns": turns,
            "speakers": speakers,
            "num_speakers": len(speakers),
            "num_turns": len(turns),
            "duration": round(duration, 2),
            "processing_time": round(elapsed, 1)
        }
    except Exception as e:
        return {"success": False, "error": str(e), "turns": [], "speakers": [],
                "num_speakers": 0, "num_turns": 0, "duration": 0, "processing_time": 0}

def main():
    pipeline = load_pipeline()

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    total = len(audio_files)
    print(f"\nFound {total} audio files\n", flush=True)

    all_results = {}
    output_file = OUTPUT_DIR / "diarization.json"
    md_file = OUTPUT_DIR / "diarization.md"

    with open(md_file, "w") as f:
        f.write("# Pyannote Speaker Diarization Results (community-1)\n\n")

    for i, audio_path in enumerate(audio_files, 1):
        name = audio_path.name
        print(f"[{i}/{total}] {name} ", end="", flush=True)
        result = diarize_file(pipeline, audio_path)
        all_results[name] = result

        if result["success"]:
            print(f"OK {result['num_speakers']} speakers, {result['num_turns']} turns, "
                  f"{result['duration']:.0f}s audio, {result['processing_time']:.0f}s")
            with open(md_file, "a") as f:
                f.write(f"## {name}\n\n")
                f.write(f"**Duration**: {result['duration']:.0f}s | **Speakers**: {result['num_speakers']} | **Turns**: {result['num_turns']}\n\n")
                for t in result["turns"][:20]:
                    f.write(f"- {t['speaker']} [{t['start']:.1f}-{t['end']:.1f}]\n")
                if len(result["turns"]) > 20:
                    f.write(f"- ... ({len(result['turns'])-20} more turns)\n")
                f.write("\n")
        else:
            print(f"ERROR: {result['error'][:80]}")

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone! {output_file}")

if __name__ == "__main__":
    main()
