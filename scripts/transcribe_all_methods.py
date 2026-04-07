#!/usr/bin/env python3
"""
Generate transcripts using multiple ASR methods for comparison.

Creates three versions:
1. Whisper (faster-whisper) - plain transcription
2. Pyannote + Whisper - speaker diarization
3. VibeVoice ASR - native diarization (requires GPU server)

Output is a single markdown file with all three versions side-by-side.
"""

import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import os
import sys
import time

AUDIO_EXTENSIONS = {".ogg", ".m4a", ".mp3", ".wav", ".webm", ".flac", ".aac"}


def find_audio_files(data_dir: Path) -> list[Path]:
    """Find all unique audio files."""
    audio_files = []
    seen_names = set()
    for ext in AUDIO_EXTENSIONS:
        for f in data_dir.rglob(f"*{ext}"):
            if f.name not in seen_names:
                seen_names.add(f.name)
                audio_files.append(f)
    return sorted(audio_files)


# =============================================================================
# Method 1: Faster-Whisper (Plain transcription)
# =============================================================================

def transcribe_whisper(audio_path: Path, model_size: str = "small") -> Dict[str, Any]:
    """Transcribe using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("pip install faster-whisper")

    device = "cuda" if _has_cuda() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,
    )

    result = {
        "text": "",
        "segments": [],
        "language": info.language,
        "duration": info.duration,
        "backend": f"faster-whisper/{model_size}",
    }

    for seg in segments:
        result["text"] += seg.text + " "
        result["segments"].append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })

    return result


# =============================================================================
# Method 2: Pyannote + Whisper (Speaker Diarization)
# =============================================================================

def transcribe_pyannote_whisper(audio_path: Path, model_size: str = "small", hf_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe with speaker diarization using pyannote + faster-whisper.
    """
    try:
        from faster_whisper import WhisperModel
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}\npip install faster-whisper pyannote.audio")

    device = "cuda" if _has_cuda() else "cpu"

    # 1. Run Whisper transcription
    whisper_model = WhisperModel(model_size, device=device, compute_type="int8" if device == "cpu" else "float16")
    segments, info = whisper_model.transcribe(str(audio_path), word_timestamps=True, vad_filter=True)

    whisper_result = {"text": "", "segments": []}
    for seg in segments:
        whisper_result["text"] += seg.text + " "
        whisper_result["segments"].append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })

    # 2. Run pyannote diarization
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("    Warning: No HF_TOKEN, skipping diarization")
        whisper_result["backend"] = f"whisper/{model_size} (no diarization)"
        return whisper_result

    try:
        diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )

        # Run diarization
        dia_result = diarization(str(audio_path))

        # Extract speaker turns
        turns = []
        for turn, _, speaker in dia_result.itertracks(yield_label=True):
            turns.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        # 3. Merge transcription with diarization
        merged_segments = []
        for seg in whisper_result["segments"]:
            seg_mid = (seg["start"] + seg["end"]) / 2
            speaker = "Unknown"
            for turn in turns:
                if turn["start"] <= seg_mid <= turn["end"]:
                    speaker = turn["speaker"]
                    break
            merged_segments.append({**seg, "speaker": speaker})

        speakers = sorted(set(s["speaker"] for s in merged_segments))

        return {
            "text": whisper_result["text"],
            "segments": merged_segments,
            "speakers": len(speakers),
            "speaker_list": speakers,
            "language": info.language,
            "duration": info.duration,
            "backend": f"pyannote+whisper/{model_size}",
        }

    except Exception as e:
        print(f"    Warning: Diarization failed: {e}")
        whisper_result["backend"] = f"whisper/{model_size} (diarization failed)"
        return whisper_result


# =============================================================================
# Method 3: VibeVoice ASR (Native Diarization, requires GPU server)
# =============================================================================

def transcribe_vibevoice(audio_path: Path, api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Transcribe using VibeVoice ASR via API.
    Best quality with native speaker diarization.
    """
    import requests

    with open(audio_path, "rb") as f:
        response = requests.post(
            f"{api_url}/v1/audio/transcriptions",
            files={"file": (audio_path.name, f, "audio/ogg")},
            data={
                "model": "vibevoice-asr-7b",
                "response_format": "verbose_json",
            },
            timeout=600
        )

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text[:200]}")

    result = response.json()
    result["backend"] = "vibevoice-asr-7b"
    return result


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


# =============================================================================
# Formatting
# =============================================================================

def format_comparison_markdown(results: Dict[str, Dict], audio_name: str) -> str:
    """Format all three methods side-by-side."""
    lines = [f"## {audio_name}\n"]

    # Metadata
    duration = next((r.get("duration", 0) for r in results.values() if r), 0)
    lines.append(f"**Duration:** {duration:.1f}s\n")

    # Method summaries
    lines.append("### Method Comparison\n")
    lines.append("| Method | Backend | Speakers | Segments |")
    lines.append("|--------|---------|----------|----------|")
    for name, r in results.items():
        if r:
            speakers = r.get("speakers", "N/A")
            segments = len(r.get("segments", []))
            backend = r.get("backend", name)
            lines.append(f"| {name} | {backend} | {speakers} | {segments} |")
    lines.append("")

    # Full transcripts side-by-side
    lines.append("### Transcripts\n")

    for name, r in results.items():
        if not r:
            lines.append(f"#### {name}\n*Not available*\n")
            continue

        lines.append(f"#### {name} ({r.get('backend', 'unknown')})\n")

        if r.get("text"):
            lines.append("**Full text:**")
            lines.append(f"> {r['text'].strip()}\n")

        if r.get("segments"):
            lines.append("**Timestamped:**\n")
            current_speaker = None
            for seg in r["segments"]:
                speaker = seg.get("speaker")
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("text", "").strip()
                if not text:
                    continue

                if speaker and speaker != current_speaker:
                    lines.append(f"\n**{speaker}** [{start:.1f}s - {end:.1f}s]:")
                    current_speaker = speaker
                elif not speaker:
                    lines.append(f"[{start:.1f}s - {end:.1f}s]")

                lines.append(f"  {text}")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate transcripts using multiple ASR methods")
    parser.add_argument("--data-dir", type=Path,
        default=Path("/home/ubuntu/Documents/programs/deep-research-engine/data/ChatExport_2026-03-13"))
    parser.add_argument("--output", type=Path,
        default=Path("/home/ubuntu/Documents/programs/deep-research-engine/output/transcripts"))
    parser.add_argument("--whisper-model", default="small", help="Whisper model size")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for pyannote")
    parser.add_argument("--vibevoice-url", default="http://localhost:8000", help="VibeVoice API URL")
    parser.add_argument("--skip-whisper", action="store_true", help="Skip whisper transcription")
    parser.add_argument("--skip-pyannote", action="store_true", help="Skip pyannote diarization")
    parser.add_argument("--skip-vibevoice", action="store_true", help="Skip VibeVoice ASR")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files with existing transcripts")

    args = parser.parse_args()

    # Find audio files
    audio_files = find_audio_files(args.data_dir)
    print(f"\n{'='*60}")
    print(f"Multi-Method ASR Transcription Pipeline")
    print(f"{'='*60}")
    print(f"Found {len(audio_files)} audio files")

    if args.limit:
        audio_files = audio_files[:args.limit]
        print(f"Processing {len(audio_files)} files (limited)")

    args.output.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output / f"transcripts_all_methods_{timestamp}.md"

    with open(output_file, "w") as f:
        f.write("# Telegram Audio Transcripts - Multi-Method Comparison\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n")
        f.write(f"**Whisper Model:** {args.whisper_model}\n")
        f.write(f"**Methods:**\n")
        if not args.skip_whisper:
            f.write(f"- Whisper (faster-whisper)\n")
        if not args.skip_pyannote:
            f.write(f"- Pyannote + Whisper (speaker diarization)\n")
        if not args.skip_vibevoice:
            f.write(f"- VibeVoice ASR (native diarization)\n")
        f.write(f"\n**Total files:** {len(audio_files)}\n\n---\n\n")

    stats = {"whisper": 0, "pyannote": 0, "vibevoice": 0}

    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] {audio_path.name}")

        results = {}

        # Method 1: Whisper
        if not args.skip_whisper:
            try:
                print(f"  Running Whisper...")
                start = time.time()
                results["Whisper"] = transcribe_whisper(audio_path, args.whisper_model)
                print(f"    Done in {time.time()-start:.1f}s")
                stats["whisper"] += 1
            except Exception as e:
                print(f"    Error: {e}")
                results["Whisper"] = None

        # Method 2: Pyannote + Whisper
        if not args.skip_pyannote:
            try:
                print(f"  Running Pyannote+Whisper...")
                start = time.time()
                results["Pyannote"] = transcribe_pyannote_whisper(audio_path, args.whisper_model, args.hf_token)
                print(f"    Done in {time.time()-start:.1f}s")
                stats["pyannote"] += 1
            except Exception as e:
                print(f"    Error: {e}")
                results["Pyannote"] = None

        # Method 3: VibeVoice ASR
        if not args.skip_vibevoice:
            try:
                print(f"  Running VibeVoice ASR...")
                start = time.time()
                results["VibeVoice"] = transcribe_vibevoice(audio_path, args.vibevoice_url)
                print(f"    Done in {time.time()-start:.1f}s")
                stats["vibevoice"] += 1
            except Exception as e:
                print(f"    Error: {e}")
                results["VibeVoice"] = None

        # Append to output
        with open(output_file, "a") as f:
            f.write(format_comparison_markdown(results, audio_path.name))
            f.write("\n\n---\n\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Whisper: {stats['whisper']}/{len(audio_files)}")
    print(f"Pyannote: {stats['pyannote']}/{len(audio_files)}")
    print(f"VibeVoice: {stats['vibevoice']}/{len(audio_files)}")
    print(f"\nOutput: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
