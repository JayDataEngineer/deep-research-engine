#!/usr/bin/env python3
"""Generate final 3-pipeline diarized comparison CSV.

Pipelines:
  1. Pyannote-Community + Faster-Whisper (diarization + ASR)
  2. VibeVoice ASR (native diarization, flash-attn Docker)
  3. Qwen3-ASR + Pyannote-Community diarization

Output: comparison_final.csv with speaker-labeled segments from all 3 pipelines.
"""

import json
import csv
import time
from pathlib import Path
from typing import List, Dict

BASE = Path("/home/ubuntu/Documents/programs/deep-research-engine/output/transcripts")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def normalize_segments(segments, method):
    """Normalize segments to: speaker, start, end, text."""
    normalized = []
    for s in segments:
        speaker = s.get("speaker", s.get("speaker_id", s.get("Speaker", "UNKNOWN")))
        start = s.get("start", s.get("start_time", s.get("Start", 0)))
        end = s.get("end", s.get("end_time", s.get("End", 0)))
        text = s.get("text", s.get("Content", ""))

        if isinstance(speaker, int):
            speaker = f"SPEAKER_{speaker:02d}"
        elif isinstance(speaker, str) and speaker.startswith("SPEAKER_"):
            pass
        elif isinstance(speaker, str):
            speaker = f"SPEAKER_{speaker}"

        normalized.append({
            "speaker": str(speaker),
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "text": str(text).strip(),
        })
    return normalized


def format_diarized_text(segments, max_len=600):
    if not segments:
        return ""
    parts = []
    for s in segments[:20]:
        parts.append(f"[{s['speaker']} {s['start']:.0f}-{s['end']:.0f}s] {s['text']}")
    text = " | ".join(parts)
    if len(segments) > 20:
        text += f" ... (+{len(segments)-20} more)"
    return text[:max_len]


def main():
    print("Loading results from all 3 pipelines...")

    pyannote = load_json(BASE / "pyannote_new" / "transcripts.json")
    vibevoice = load_json(BASE / "vibevoice_flash" / "transcripts.json")
    qwen = load_json(BASE / "qwen-asr-diarized" / "transcripts.json")

    all_files = sorted(
        set(list(pyannote.keys()) + list(vibevoice.keys()) + list(qwen.keys()))
    )
    print(f"Total files: {len(all_files)}")

    csv_path = BASE / "comparison_final.csv"
    md_path = BASE / "comparison_final.md"
    json_path = BASE / "comparison_final.json"

    all_results = {}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "File", "Duration_s",
            "PyannoteFW_Status", "PyannoteFW_Speakers", "PyannoteFW_Segments", "PyannoteFW_Text",
            "VibeVoice_Status", "VibeVoice_Speakers", "VibeVoice_Segments", "VibeVoice_Text",
            "QwenPyannote_Status", "QwenPyannote_Speakers", "QwenPyannote_Segments", "QwenPyannote_Text",
        ])

        for fname in all_files:
            row = [fname]
            duration = 0

            # Pipeline 1: Pyannote + Faster-Whisper
            p1 = pyannote.get(fname, {})
            p1_segs = normalize_segments(p1.get("segments", []), "pyannote")
            p1_status = "OK" if p1_segs else ("SILENT" if p1.get("success") else "ERROR")
            p1_speakers = sorted(set(s["speaker"] for s in p1_segs)) if p1_segs else []
            duration = max(duration, p1.get("duration", 0))

            # Pipeline 2: VibeVoice native
            p2 = vibevoice.get(fname, {})
            p2_segs = normalize_segments(p2.get("segments", []), "vibevoice")
            p2_status = "OK" if p2_segs else ("SILENT" if p2.get("success") else "ERROR")
            p2_speakers = sorted(set(s["speaker"] for s in p2_segs)) if p2_segs else []
            duration = max(duration, p2.get("duration", 0))

            # Pipeline 3: Qwen-ASR + Pyannote
            p3 = qwen.get(fname, {})
            p3_segs = normalize_segments(p3.get("segments", []), "qwen")
            p3_status = "OK" if p3_segs else ("SILENT" if p3.get("success") and p3.get("note") == "silent audio") else ("EMPTY" if p3.get("success") else "ERROR")
            p3_speakers = sorted(set(s["speaker"] for s in p3_segs)) if p3_segs else []
            duration = max(duration, p3.get("duration", 0))

            row.append(f"{duration:.1f}")
            for status, speakers, segs in [
                (p1_status, p1_speakers, p1_segs),
                (p2_status, p2_speakers, p2_segs),
                (p3_status, p3_speakers, p3_segs),
            ]:
                row.append(status)
                row.append(len(speakers))
                row.append(len(segs))
                row.append(format_diarized_text(segs))

            writer.writerow(row)

            all_results[fname] = {
                "duration": duration,
                "pyannote_fw": {"status": p1_status, "segments": p1_segs, "speakers": p1_speakers},
                "vibevoice": {"status": p2_status, "segments": p2_segs, "speakers": p2_speakers},
                "qwen_pyannote": {"status": p3_status, "segments": p3_segs, "speakers": p3_speakers},
            }

    # Write JSON
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Write markdown
    with open(md_path, "w") as f:
        f.write("# Diarized Transcription Comparison — 3 Pipelines\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| # | Pipeline | ASR | Diarization | Status |\n")
        f.write("|---|----------|-----|-------------|--------|\n")
        f.write("| 1 | Pyannote + Faster-Whisper | Faster-Whisper base | Pyannote-community-1 | Standard |\n")
        f.write("| 2 | VibeVoice | VibeVoice-ASR-7B | VibeVoice (native) | Flash-attn Docker |\n")
        f.write("| 3 | Qwen + Pyannote | Qwen3-ASR-1.7B | Pyannote-community-1 | Qwen ForcedAligner |\n")
        f.write("\n---\n\n")

        for fname in all_files:
            r = all_results[fname]
            f.write(f"## {fname} ({r['duration']:.0f}s)\n\n")

            for method, label in [
                ("pyannote_fw", "Pipeline 1: Pyannote + Faster-Whisper"),
                ("vibevoice", "Pipeline 2: VibeVoice (Native Diarization)"),
                ("qwen_pyannote", "Pipeline 3: Qwen3-ASR + Pyannote"),
            ]:
                m = r[method]
                f.write(f"### {label}\n\n")
                f.write(f"**Status**: {m['status']} | **Speakers**: {len(m['speakers'])} | **Segments**: {len(m['segments'])}\n\n")

                for seg in m["segments"][:30]:
                    f.write(f"- **{seg['speaker']}** [{seg['start']:.1f}-{seg['end']:.1f}s]: {seg['text']}\n")
                if len(m["segments"]) > 30:
                    f.write(f"\n... and {len(m['segments'])-30} more segments\n")
                f.write("\n")

            f.write("---\n\n")

    # Summary
    print(f"\n{'='*70}")
    print("FINAL 3-PIPELINE COMPARISON")
    print(f"{'='*70}")
    for method, label in [
        ("pyannote_fw", "Pipeline 1: Pyannote + Faster-Whisper"),
        ("vibevoice", "Pipeline 2: VibeVoice (Native)"),
        ("qwen_pyannote", "Pipeline 3: Qwen-ASR + Pyannote"),
    ]:
        ok = sum(1 for r in all_results.values() if r[method]["status"] == "OK")
        total_segs = sum(len(r[method]["segments"]) for r in all_results.values())
        total_speakers = sum(len(r[method]["speakers"]) for r in all_results.values())
        print(f"  {label}")
        print(f"    {ok}/{len(all_files)} files diarized, {total_segs} total segments, {total_speakers} total speaker-assignments")
        print()

    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
