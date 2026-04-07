#!/usr/bin/env python3
"""Create diarized comparison from all 4 ASR methods.

Applies Pyannote speaker diarization to methods that don't have native
diarization (Faster-Whisper, Qwen-ASR), producing speaker-attributed
segments for all methods.

Output: comparison_diarized.csv with speaker-labeled segments for ALL methods.
"""

import json
import csv
import re
from pathlib import Path
from collections import defaultdict

BASE = Path("/home/ubuntu/Documents/programs/deep-research-engine")


def load_json(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
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


def assign_words_to_speakers(words, speaker_turns):
    """Assign speaker labels to words based on speaker turn overlaps.

    words: list of dicts with 'start', 'end', 'text'/'word' keys
    speaker_turns: list of dicts with 'start', 'end', 'speaker' keys

    Returns: list of dicts with 'speaker', 'start', 'end', 'text' keys
    """
    if not words or not speaker_turns:
        return []

    # Sort turns by start time for efficient lookup
    sorted_turns = sorted(speaker_turns, key=lambda t: t["start"])

    word_assignments = []
    for w in words:
        word_start = w.get("start", 0)
        word_end = w.get("end", 0)
        word_mid = (word_start + word_end) / 2
        text = w.get("text", w.get("word", ""))

        best_speaker = "UNKNOWN"
        best_overlap = 0

        for turn in sorted_turns:
            # Check overlap
            overlap_start = max(word_start, turn["start"])
            overlap_end = min(word_end, turn["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

            # Early exit if past the word
            if turn["start"] > word_end:
                break

        word_assignments.append({
            "speaker": best_speaker,
            "start": word_start,
            "end": word_end,
            "text": text,
        })

    return word_assignments


def merge_consecutive_speaker_segments(word_assignments):
    """Merge consecutive words from the same speaker into segments."""
    if not word_assignments:
        return []

    segments = []
    current = {
        "speaker": word_assignments[0]["speaker"],
        "start": word_assignments[0]["start"],
        "end": word_assignments[0]["end"],
        "text": word_assignments[0]["text"],
    }

    for w in word_assignments[1:]:
        if w["speaker"] == current["speaker"]:
            current["end"] = w["end"]
            current["text"] += " " + w["text"]
        else:
            segments.append(current)
            current = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "text": w["text"],
            }

    segments.append(current)
    return segments


def normalize_segments(segments, method):
    """Normalize segments to common format: speaker, start, end, text."""
    normalized = []
    for s in segments:
        speaker = (
            s.get("speaker", s.get("speaker_id", s.get("Speaker", "UNKNOWN")))
        )
        start = s.get("start", s.get("start_time", s.get("Start", 0)))
        end = s.get("end", s.get("end_time", s.get("End", 0)))
        text = s.get("text", s.get("Content", ""))

        # Normalize speaker labels
        if isinstance(speaker, int):
            speaker = f"SPEAKER_{speaker:02d}"
        elif isinstance(speaker, str) and not speaker.startswith("SPEAKER"):
            speaker = f"SPEAKER_{speaker}"

        normalized.append({
            "speaker": str(speaker),
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "text": str(text),
        })
    return normalized


def format_diarized_text(segments, max_len=500):
    """Format diarized segments as readable text."""
    if not segments:
        return ""
    parts = []
    for s in segments[:15]:
        sp = s["speaker"]
        start = s["start"]
        end = s["end"]
        text = s["text"]
        parts.append(f"[{sp} {start:.0f}-{end:.0f}s] {text}")
    text = " | ".join(parts)
    if len(segments) > 15:
        text += f" ... (+{len(segments)-15} more)"
    return text[:max_len]


def main():
    print("Loading results from all methods...")

    # Load raw results
    fw = load_json(BASE / "output/transcripts/faster-whisper/transcripts.json")
    py = load_json(BASE / "output/transcripts/pyannote_new/transcripts.json")
    qwen = load_json(BASE / "output/transcripts/qwen-asr/transcripts.json")
    vv = load_json(BASE / "output/transcripts/vibevoice/transcripts.json")

    all_files = sorted(
        set(list(fw.keys()) + list(py.keys()) + list(qwen.keys()) + list(vv.keys()))
    )
    print(f"Total files: {len(all_files)}")

    # Build diarized output for each method
    results = {}

    for fname in all_files:
        file_results = {}

        # --- Method 1: Pyannote diarization + Faster-Whisper ASR ---
        py_data = py.get(fname, {})
        if py_data.get("success"):
            file_results["pyannote_fw"] = {
                "status": "OK",
                "duration": py_data.get("duration", 0),
                "num_speakers": py_data.get("num_speakers", 0),
                "speakers": py_data.get("speakers", []),
                "segments": normalize_segments(py_data.get("segments", []), "pyannote"),
            }
        else:
            file_results["pyannote_fw"] = {"status": "MISSING", "segments": []}

        # --- Method 2: VibeVoice native diarization ---
        vv_data = vv.get(fname, {})
        if vv_data.get("success"):
            vv_segs = vv_data.get("segments", [])
            if vv_segs:
                file_results["vibevoice"] = {
                    "status": "OK",
                    "duration": vv_data.get("duration", 0),
                    "segments": normalize_segments(vv_segs, "vibevoice"),
                }
            else:
                file_results["vibevoice"] = {
                    "status": "EMPTY",
                    "duration": vv_data.get("duration", 0),
                    "segments": [],
                }
        else:
            file_results["vibevoice"] = {"status": "MISSING", "segments": []}

        # --- Method 3: Pyannote diarization + Faster-Whisper words (re-diariated) ---
        # Use pyannote speaker turns + faster-whisper word timestamps
        fw_data = fw.get(fname, {})
        if fw_data.get("success") and py_data.get("success"):
            fw_words = fw_data.get("words", [])
            py_turns = py_data.get("speaker_turns", [])

            if fw_words and py_turns:
                word_assignments = assign_words_to_speakers(fw_words, py_turns)
                fw_diarized = merge_consecutive_speaker_segments(word_assignments)
                speakers = sorted(set(s["speaker"] for s in fw_diarized))
                file_results["fw_diarized"] = {
                    "status": "OK",
                    "duration": fw_data.get("duration", 0),
                    "num_speakers": len(speakers),
                    "speakers": speakers,
                    "segments": normalize_segments(fw_diarized, "fw_diarized"),
                }
            else:
                file_results["fw_diarized"] = {
                    "status": "NO_WORDS",
                    "segments": [],
                }
        else:
            file_results["fw_diarized"] = {"status": "MISSING", "segments": []}

        # --- Method 4: Pyannote diarization + Qwen-ASR timestamps ---
        qw_data = qwen.get(fname, {})
        if qw_data.get("success") and py_data.get("success"):
            qw_segs = qw_data.get("segments", [])
            py_turns = py_data.get("speaker_turns", [])

            if qw_segs and py_turns:
                # Qwen segments have start/end/text - treat like words
                qwen_words = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in qw_segs]
                word_assignments = assign_words_to_speakers(qwen_words, py_turns)
                qwen_diarized = merge_consecutive_speaker_segments(word_assignments)
                speakers = sorted(set(s["speaker"] for s in qwen_diarized))
                file_results["qwen_diarized"] = {
                    "status": "OK",
                    "duration": qw_data.get("duration", 0),
                    "num_speakers": len(speakers),
                    "speakers": speakers,
                    "segments": normalize_segments(qwen_diarized, "qwen_diarized"),
                }
            else:
                file_results["qwen_diarized"] = {
                    "status": "NO_TIMESTAMPS",
                    "segments": [],
                }
        else:
            file_results["qwen_diarized"] = {"status": "MISSING", "segments": []}

        results[fname] = file_results

    # Write comparison CSV
    output_path = BASE / "output/transcripts/comparison_diarized.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow([
            "File",
            "Duration_s",
            # Pyannote + FW
            "PyannoteFW_Status",
            "PyannoteFW_Speakers",
            "PyannoteFW_Segments",
            "PyannoteFW_DiarizedText",
            # VibeVoice
            "VibeVoice_Status",
            "VibeVoice_Speakers",
            "VibeVoice_Segments",
            "VibeVoice_DiarizedText",
            # FW diarized
            "FW_Diarized_Status",
            "FW_Diarized_Speakers",
            "FW_Diarized_Segments",
            "FW_Diarized_DiarizedText",
            # Qwen diarized
            "Qwen_Diarized_Status",
            "Qwen_Diarized_Speakers",
            "Qwen_Diarized_Segments",
            "Qwen_Diarized_DiarizedText",
        ])

        for fname in all_files:
            r = results[fname]
            duration = max(
                r.get("pyannote_fw", {}).get("duration", 0),
                r.get("vibevoice", {}).get("duration", 0),
                r.get("fw_diarized", {}).get("duration", 0),
                r.get("qwen_diarized", {}).get("duration", 0),
            )

            row = [fname, f"{duration:.1f}"]

            for method in ["pyannote_fw", "vibevoice", "fw_diarized", "qwen_diarized"]:
                m = r.get(method, {})
                row.append(m.get("status", "MISSING"))
                segs = m.get("segments", [])
                speakers = sorted(set(s["speaker"] for s in segs)) if segs else []
                row.append(len(speakers))
                row.append(len(segs))
                row.append(format_diarized_text(segs))

            writer.writerow(row)

    print(f"Diarized comparison CSV: {output_path}")

    # Also write detailed JSON with full diarized segments
    json_path = BASE / "output/transcripts/comparison_diarized.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Diarized comparison JSON: {json_path}")

    # Write human-readable markdown
    md_path = BASE / "output/transcripts/comparison_diarized.md"
    with open(md_path, "w") as f:
        f.write("# Diarized Transcription Comparison (All 4 Methods)\n\n")
        f.write("Each method has speaker-attributed segments (Speaker ID + timestamps + text).\n\n")
        f.write("---\n\n")

        for fname in all_files:
            r = results[fname]
            duration = max(
                r.get("pyannote_fw", {}).get("duration", 0),
                r.get("vibevoice", {}).get("duration", 0),
                r.get("fw_diarized", {}).get("duration", 0),
                r.get("qwen_diarized", {}).get("duration", 0),
            )

            f.write(f"## {fname} ({duration:.0f}s)\n\n")

            for method, label in [
                ("pyannote_fw", "Pyannote Diarization + Faster-Whisper ASR"),
                ("vibevoice", "VibeVoice ASR (Native Diarization)"),
                ("fw_diarized", "Faster-Whisper ASR + Pyannote Diarization"),
                ("qwen_diarized", "Qwen3-ASR + Pyannote Diarization"),
            ]:
                m = r.get(method, {})
                f.write(f"### {label}\n\n")
                f.write(f"**Status**: {m.get('status', 'MISSING')}\n\n")

                segs = m.get("segments", [])
                if not segs:
                    f.write("*No segments*\n\n")
                    continue

                speakers = sorted(set(s["speaker"] for s in segs))
                f.write(f"**Speakers**: {len(speakers)} ({', '.join(speakers)})\n")
                f.write(f"**Segments**: {len(segs)}\n\n")

                for s in segs:
                    f.write(f"- **{s['speaker']}** [{s['start']:.1f}s - {s['end']:.1f}s]: {s['text']}\n")
                f.write("\n")

            f.write("---\n\n")

    print(f"Diarized comparison MD: {md_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for method, label in [
        ("pyannote_fw", "Pyannote + Faster-Whisper"),
        ("vibevoice", "VibeVoice"),
        ("fw_diarized", "FW + Pyannote Diarization"),
        ("qwen_diarized", "Qwen-ASR + Pyannote Diarization"),
    ]:
        ok = sum(
            1 for r in results.values()
            if r.get(method, {}).get("status") == "OK"
        )
        total_segs = sum(
            len(r.get(method, {}).get("segments", []))
            for r in results.values()
        )
        print(f"  {label}: {ok}/{len(all_files)} files, {total_segs} total segments")


if __name__ == "__main__":
    main()
