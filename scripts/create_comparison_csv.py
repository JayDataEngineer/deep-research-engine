#!/usr/bin/env python3
"""Create comparison CSV from all ASR method outputs.

Reads JSONL/JSON results from:
- Faster-Whisper (ASR + word timestamps)
- Pyannote (diarization) + Faster-Whisper (ASR)
- Qwen3-ASR (ASR + forced aligner timestamps)
- VibeVoice (ASR + native diarization)
"""

import json
import csv
from pathlib import Path

BASE = Path("/home/ubuntu/Documents/programs/deep-research-engine")


def load_jsonl(path):
    """Load JSONL file into dict keyed by filename."""
    results = {}
    if not path.exists():
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                fname = data.pop("file", None)
                if fname:
                    results[fname] = data
            except json.JSONDecodeError:
                continue
    return results


def load_json(path):
    """Load JSON file (dict keyed by filename)."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def get_text(result):
    """Extract full text from result."""
    if not result.get("success", True):
        return f"ERROR: {result.get('error', 'unknown')[:80]}"
    return result.get("text", "")


def get_segments_text(segments, max_len=200):
    """Get a truncated summary of segments."""
    if not segments:
        return ""
    parts = []
    for s in segments[:5]:
        if isinstance(s, dict):
            text = s.get("text", s.get("Content", ""))
            speaker = s.get("speaker", s.get("Speaker", ""))
            start = s.get("start", s.get("Start", 0))
            end = s.get("end", s.get("End", 0))
            if speaker:
                parts.append(f"[{speaker} {start:.0f}-{end:.0f}s] {text}")
            else:
                parts.append(f"[{start:.0f}-{end:.0f}s] {text}")
    text = " | ".join(parts)
    if len(segments) > 5:
        text += f" ... (+{len(segments)-5} more)"
    return text[:max_len]


def main():
    # Load all results
    fw = load_jsonl(BASE / "output/transcripts/faster-whisper/transcripts.jsonl")
    py = load_jsonl(BASE / "output/transcripts/pyannote_new/transcripts.jsonl")
    qwen = load_jsonl(BASE / "output/transcripts/qwen-asr/transcripts.jsonl")

    # VibeVoice - try JSONL first, then JSON
    vv = load_jsonl(BASE / "output/transcripts/vibevoice/transcripts.jsonl")
    if not vv:
        vv_json = load_json(BASE / "output/transcripts/vibevoice/transcripts.json")
        if vv_json:
            vv = vv_json

    # Get all filenames
    all_files = sorted(set(list(fw.keys()) + list(py.keys()) + list(qwen.keys()) + list(vv.keys())))

    output_path = BASE / "output/transcripts/comparison_all_methods.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "File",
            "Duration_s",
            "FasterWhisper_Status",
            "FasterWhisper_Text",
            "FasterWhisper_Segments",
            "FasterWhisper_Words",
            "Pyannote_Status",
            "Pyannote_Speakers",
            "Pyannote_Segments",
            "Pyannote_Text",
            "QwenASR_Status",
            "QwenASR_Language",
            "QwenASR_Text",
            "QwenASR_Timestamps",
            "VibeVoice_Status",
            "VibeVoice_Speakers",
            "VibeVoice_Segments",
            "VibeVoice_Text",
        ])

        for fname in all_files:
            duration = 0
            row = [fname]

            # Faster-Whisper
            r = fw.get(fname, {})
            if r:
                duration = r.get("duration", duration)
                row.append(f"{duration:.1f}")
                row.append("OK" if r.get("success") else "ERROR")
                row.append(get_text(r)[:200])
                row.append(len(r.get("segments", [])))
                row.append(len(r.get("words", [])))
            else:
                row.extend(["", "", "", "", ""])

            # Pyannote
            r = py.get(fname, {})
            if r:
                duration = r.get("duration", duration)
                row.append("OK" if r.get("success") else "ERROR")
                row.append(r.get("num_speakers", 0))
                row.append(len(r.get("segments", [])))
                segs = r.get("segments", [])
                text = " ".join(s.get("text", "") for s in segs[:10])
                row.append(text[:200])
            else:
                row.extend(["", "", "", ""])

            # Qwen ASR
            r = qwen.get(fname, {})
            if r:
                duration = r.get("duration", duration)
                row.append("OK" if r.get("success") else "ERROR")
                row.append(r.get("language", ""))
                row.append(get_text(r)[:200])
                row.append(len(r.get("segments", [])))
            else:
                row.extend(["", "", "", ""])

            # VibeVoice
            r = vv.get(fname, {})
            if r:
                duration = r.get("duration", duration)
                speakers = set()
                segs = r.get("segments", [])
                for s in segs:
                    sp = s.get("Speaker", s.get("speaker_id", 0))
                    if sp:
                        speakers.add(sp)
                row.append("OK" if r.get("success") and segs else "EMPTY")
                row.append(len(speakers))
                row.append(len(segs))
                text = " ".join(s.get("Content", s.get("text", "")) for s in segs[:10])
                row.append(text[:200])
            else:
                row.extend(["", "", "", ""])

            # Update duration if not set from faster-whisper
            if row[1] == "" and duration:
                row[1] = f"{duration:.1f}"

            writer.writerow(row)

    print(f"Comparison CSV written to: {output_path}")
    print(f"Total files: {len(all_files)}")

    # Print summary
    fw_ok = sum(1 for v in fw.values() if v.get("success"))
    py_ok = sum(1 for v in py.values() if v.get("success"))
    qwen_ok = sum(1 for v in qwen.values() if v.get("success"))
    vv_ok = sum(1 for k, v in vv.items() if v.get("success") and v.get("segments"))

    print(f"\nSummary:")
    print(f"  Faster-Whisper: {fw_ok}/{len(all_files)} successful")
    print(f"  Pyannote+FW:    {py_ok}/{len(all_files)} successful")
    print(f"  Qwen3-ASR:      {qwen_ok}/{len(all_files)} successful")
    print(f"  VibeVoice ASR:  {vv_ok}/{len(all_files)} successful")


if __name__ == "__main__":
    main()
