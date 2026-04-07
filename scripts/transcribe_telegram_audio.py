#!/usr/bin/env python3
"""
Telegram Audio Transcription Pipeline
======================================

Provides multiple ASR backends for transcribing Telegram voice messages:
1. faster-whisper (CPU/GPU, reliable, good quality)
2. openai-whisper (CPU, original OpenAI model)
3. VibeVoice ASR via API (GPU required, best quality with diarization)

Outputs annotated transcripts that can be shared with users without GPU.
"""

import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import os
import sys

# Audio format support
AUDIO_EXTENSIONS = {".ogg", ".m4a", ".mp3", ".wav", ".webm", ".flac", ".aac"}


def find_audio_files(data_dir: Path) -> list[Path]:
    """Find all unique audio files in the dataset."""
    audio_files = []
    seen_names = set()

    for ext in AUDIO_EXTENSIONS:
        for f in data_dir.rglob(f"*{ext}"):
            if f.name not in seen_names:
                seen_names.add(f.name)
                audio_files.append(f)

    return sorted(audio_files)


def convert_to_wav(audio_path: Path) -> Path:
    """Convert audio to WAV format for processing (16kHz mono)."""
    wav_path = audio_path.with_suffix(".wav")
    if not wav_path.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(wav_path)],
            capture_output=True,
            check=True
        )
    return wav_path


# =============================================================================
# ASR Backend: faster-whisper (Recommended for CPU)
# =============================================================================

def transcribe_faster_whisper(
    audio_path: Path,
    model_size: str = "small",
    device: str = "auto",
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribe using faster-whisper (CTranslate2).

    This is the recommended backend for CPU-only systems.
    Models: tiny, base, small, medium, large-v2, large-v3
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")

    # Auto-detect device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"    Using faster-whisper/{model_size} on {device} ({compute_type})")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    result = {
        "text": "",
        "segments": [],
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "backend": f"faster-whisper/{model_size}",
    }

    for seg in segments:
        result["text"] += seg.text + " "
        result["segments"].append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": [
                {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                for w in (seg.words or [])
            ]
        })

    return result


# =============================================================================
# ASR Backend: openai-whisper (Original)
# =============================================================================

def transcribe_openai_whisper(
    audio_path: Path,
    model_size: str = "small",
    device: str = "auto",
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribe using OpenAI's original whisper.

    Slower than faster-whisper but may have slightly better accuracy.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("openai-whisper not installed. Run: pip install openai-whisper")

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"    Using openai-whisper/{model_size} on {device}")

    model = whisper.load_model(model_size, device=device)

    result = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
    )

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "words": seg.get("words", [])
        })

    return {
        "text": result["text"],
        "segments": segments,
        "language": result.get("language", "unknown"),
        "duration": result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0,
        "backend": f"openai-whisper/{model_size}",
    }


# =============================================================================
# ASR Backend: VibeVoice via API (GPU required)
# =============================================================================

def transcribe_vibevoice_api(
    audio_path: Path,
    api_url: str = "http://localhost:8000",
    model: str = "vibevoice-asr-7b"
) -> Dict[str, Any]:
    """
    Transcribe using VibeVoice ASR via API.

    Best quality with built-in speaker diarization.
    Requires GPU with 24GB+ VRAM.
    """
    import requests

    print(f"    Using VibeVoice ASR via {api_url}")

    with open(audio_path, "rb") as f:
        response = requests.post(
            f"{api_url}/v1/audio/transcriptions",
            files={"file": (audio_path.name, f)},
            data={
                "model": model,
                "response_format": "verbose_json",
            },
            timeout=600
        )

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    result = response.json()
    result["backend"] = "vibevoice-asr-7b"
    return result


# =============================================================================
# Output Formatting
# =============================================================================

def format_transcript_markdown(result: Dict[str, Any], audio_name: str) -> str:
    """Format transcription result as markdown."""
    lines = [f"## {audio_name}\n"]
    lines.append(f"**Duration:** {result.get('duration', 0):.1f}s")
    lines.append(f"**Language:** {result.get('language', 'unknown')}")

    if result.get("language_probability"):
        lines.append(f"**Language Confidence:** {result['language_probability']:.0%}")

    lines.append(f"**Backend:** {result.get('backend', 'unknown')}")

    if result.get("speakers"):
        lines.append(f"**Speakers Detected:** {result['speakers']}")

    lines.append("")

    # Full text
    if "text" in result and result["text"]:
        lines.append("### Full Transcript\n")
        lines.append(result["text"].strip())
        lines.append("")

    # Segments with speakers
    if "segments" in result and result["segments"]:
        lines.append("### Timestamped Segments\n")

        current_speaker = None
        for seg in result["segments"]:
            speaker = seg.get("speaker")
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()

            if not text:
                continue

            # Speaker header for diarized output
            if speaker and speaker != current_speaker:
                lines.append(f"\n**{speaker}** [{start:.1f}s - {end:.1f}s]:")
                current_speaker = speaker
            elif not speaker:
                lines.append(f"\n[{start:.1f}s - {end:.1f}s]")

            lines.append(f"  {text}")

    return "\n".join(lines)


def format_transcript_jsonl(results: List[Dict[str, Any]], output_path: Path):
    """Export results as JSONL for programmatic use."""
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Telegram audio with multiple ASR backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use faster-whisper (recommended for CPU)
  python transcribe_telegram_audio.py --backend faster-whisper --model small

  # Use larger model for better accuracy
  python transcribe_telegram_audio.py --backend faster-whisper --model medium

  # Use VibeVoice API (requires GPU server)
  python transcribe_telegram_audio.py --backend vibevoice --api-url http://localhost:8000
"""
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/ubuntu/Documents/programs/deep-research-engine/data/ChatExport_2026-03-13"),
        help="Path to Telegram export directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ubuntu/Documents/programs/deep-research-engine/output/transcripts"),
        help="Output directory for transcripts",
    )
    parser.add_argument(
        "--backend",
        choices=["faster-whisper", "openai-whisper", "vibevoice"],
        default="faster-whisper",
        help="ASR backend to use (default: faster-whisper)",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Model size: tiny, base, small, medium, large-v2, large-v3",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for inference",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g., 'en', 'es'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="VibeVoice API URL (for vibevoice backend)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have transcripts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )

    args = parser.parse_args()

    # Find audio files
    audio_files = find_audio_files(args.data_dir)
    print(f"\n{'='*60}")
    print(f"Telegram Audio Transcription Pipeline")
    print(f"{'='*60}")
    print(f"Found {len(audio_files)} unique audio files")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")

    if not audio_files:
        print("No audio files found!")
        return 1

    if args.limit:
        audio_files = audio_files[:args.limit]
        print(f"Processing {len(audio_files)} files (limited)")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Select backend
    if args.backend == "faster-whisper":
        transcribe = lambda p: transcribe_faster_whisper(p, args.model, args.device, args.language)
    elif args.backend == "openai-whisper":
        transcribe = lambda p: transcribe_openai_whisper(p, args.model, args.device, args.language)
    elif args.backend == "vibevoice":
        transcribe = lambda p: transcribe_vibevoice_api(p, args.api_url, args.model)

    # Process files
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_file = args.output / f"transcripts_{args.backend}_{timestamp}.md"
    jsonl_file = args.output / f"transcripts_{args.backend}_{timestamp}.jsonl"

    with open(transcript_file, "w") as f:
        f.write(f"# Telegram Audio Transcripts\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n")
        f.write(f"**Backend:** {args.backend}\n")
        f.write(f"**Model:** {args.model}\n")
        f.write(f"**Total files:** {len(audio_files)}\n\n")
        f.write("---\n\n")

    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] {audio_path.name}")

        # Check for existing transcript
        json_out = args.output / f"{audio_path.stem}.json"
        if args.skip_existing and json_out.exists():
            print(f"    Skipping (existing)")
            continue

        try:
            result = transcribe(audio_path)

            # Save JSON
            with open(json_out, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Append to markdown
            with open(transcript_file, "a") as f:
                f.write(format_transcript_markdown(result, audio_path.name))
                f.write("\n\n---\n\n")

            print(f"    Done: {result.get('duration', 0):.1f}s audio, {len(result.get('segments', []))} segments")

            results.append({
                "file": audio_path.name,
                "success": True,
                "duration": result.get("duration"),
                "segments": len(result.get("segments", [])),
            })

        except Exception as e:
            print(f"    Error: {e}")
            results.append({
                "file": audio_path.name,
                "success": False,
                "error": str(e),
            })

    # Save JSONL
    format_transcript_jsonl(results, jsonl_file)

    # Summary
    successful = sum(1 for r in results if r["success"])
    total_duration = sum(r.get("duration", 0) for r in results if r["success"])

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Completed: {successful}/{len(results)} files")
    print(f"Total audio: {total_duration/60:.1f} minutes")
    print(f"Transcript: {transcript_file}")
    print(f"JSONL: {jsonl_file}")
    print(f"Individual JSONs: {args.output}")

    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
