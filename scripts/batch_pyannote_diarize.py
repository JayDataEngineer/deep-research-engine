#!/usr/bin/env python3
"""Batch diarize audio files using Pyannote + transcribe with Faster-Whisper.

Combines pyannote.audio speaker diarization with faster-whisper word-level
timestamps to produce speaker-attributed transcriptions.
"""

import json
import os
import time
import csv
import torch
import soundfile as sf
from pathlib import Path
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# Configuration
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", "data/extracted_audio"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output/transcripts/pyannote_new"))
HF_TOKEN = os.environ.get("HF_TOKEN")  # never hardcode — export HF_TOKEN

# Diarization config
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
NUM_SPEAKERS = None  # Auto-detect, or set to 2 for known conversations

# Whisper config for transcription
WHISPER_MODEL = "Systran/faster-whisper-base"


def load_models():
    """Load pyannote pipeline and faster-whisper model."""
    print("Loading pyannote pipeline...", flush=True)
    pipeline = Pipeline.from_pretrained(
        PYANNOTE_MODEL,
        token=HF_TOKEN,
    )
    pipeline.to(torch.device("cuda"))
    print("Pyannote loaded!", flush=True)

    print("Loading faster-whisper...", flush=True)
    whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    print("Whisper loaded!", flush=True)

    return pipeline, whisper


def diarize_and_transcribe(pipeline, whisper, audio_path):
    """Run diarization and transcription on a single file."""
    try:
        # Load audio
        waveform, sr = sf.read(audio_path)
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        duration = len(waveform) / sr

        # Step 1: Diarization with pyannote
        diarization_output = pipeline(
            {"waveform": torch.FloatTensor(waveform).unsqueeze(0), "sample_rate": sr},
            num_speakers=NUM_SPEAKERS,
        )

        # Collect speaker turns - new pyannote 4.0 API returns DiarizeOutput
        diarization = diarization_output.speaker_diarization
        speaker_turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_turns.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker,
            })

        # Step 2: Transcription with faster-whisper
        segments, info = whisper.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        words = []
        for seg in segments:
            for w in seg.words:
                words.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                })

        # Step 3: Assign words to speakers
        # For each word, find the speaker turn with the most overlap
        speaker_segments = []
        for word in words:
            word_mid = (word["start"] + word["end"]) / 2
            best_speaker = "UNKNOWN"
            for turn in speaker_turns:
                if turn["start"] <= word_mid <= turn["end"]:
                    best_speaker = turn["speaker"]
                    break
            word["speaker"] = best_speaker

        # Merge consecutive words from the same speaker into segments
        if words:
            current_speaker = words[0]["speaker"]
            current_start = words[0]["start"]
            current_words = [words[0]["word"]]

            for w in words[1:]:
                if w["speaker"] == current_speaker:
                    current_words.append(w["word"])
                else:
                    speaker_segments.append({
                        "speaker": current_speaker,
                        "start": current_start,
                        "end": words[words.index(w) - 1]["end"] if w != words[-1] else w["end"],
                        "text": " ".join(current_words),
                    })
                    current_speaker = w["speaker"]
                    current_start = w["start"]
                    current_words = [w["word"]]

            # Last segment
            speaker_segments.append({
                "speaker": current_speaker,
                "start": current_start,
                "end": words[-1]["end"],
                "text": " ".join(current_words),
            })

        speakers = sorted(set(t["speaker"] for t in speaker_turns))

        return {
            "success": True,
            "duration": round(duration, 2),
            "language": info.language,
            "speakers": speakers,
            "num_speakers": len(speakers),
            "speaker_turns": speaker_turns,
            "segments": speaker_segments,
            "words": words,
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

    lines.append(f"**Duration**: {result['duration']:.2f}s")
    lines.append(f"**Speakers**: {result['num_speakers']} ({', '.join(result['speakers'])})")
    lines.append(f"**Language**: {result['language']}")
    lines.append("")

    for seg in result["segments"]:
        lines.append(f"**{seg['speaker']}** [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}")

    lines.append("")
    return "\n".join(lines)


def main():
    pipeline, whisper = load_models()

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    total = len(audio_files)
    print(f"\nFound {total} audio files\n", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "transcripts.md"
    json_output = OUTPUT_DIR / "transcripts.jsonl"
    all_results = {}

    with open(output_file, "w") as f:
        f.write("# Pyannote Diarization + Faster-Whisper Transcriptions\n\n")
        f.write(f"Diarization: {PYANNOTE_MODEL}\n")
        f.write(f"ASR: {WHISPER_MODEL}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

    with open(json_output, "w") as jf:
        for i, audio_path in enumerate(audio_files, 1):
            filename = audio_path.name
            print(f"[{i}/{total}] {filename} ", end="", flush=True)
            t0 = time.time()

            result = diarize_and_transcribe(pipeline, whisper, audio_path)
            all_results[filename] = result

            elapsed = time.time() - t0
            with open(output_file, "a") as f:
                f.write(format_transcript(result, filename))

            jf.write(json.dumps({"file": filename, **result}) + "\n")

            if result["success"]:
                n_spk = result["num_speakers"]
                n_seg = len(result["segments"])
                print(f"OK {n_spk} speakers, {n_seg} segs, "
                      f"{result['duration']:.0f}s audio, {elapsed:.0f}s proc")
            else:
                print(f"ERROR: {result['error'][:100]}")

    with open(OUTPUT_DIR / "transcripts.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {output_file}")


if __name__ == "__main__":
    main()
