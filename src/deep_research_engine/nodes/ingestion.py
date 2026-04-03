"""Ingestion Node - Optional entry point for non-text data.

Handles extracting and converting non-text data (e.g. PDFs, images, Telegram exports)
into nice, organized text files.

Supports both legacy state return and LangGraph Command return.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from bs4 import BeautifulSoup

try:
    import pymupdf
except ImportError:
    pymupdf = None

from ..config import get_phase_config
from ..shared.config import DEFAULT_MODEL, VISION_MODEL

# Note: multimodal.py in utils might still be needed for direct asr_transcribe etc.
from ..shared.multimodal import (
    _get_infra,
    asr_transcribe,
    batch_vlm_video_analysis,
    cluster_identities,
    create_mega_files,
    extract_audio_from_video,
    extract_scene_frames,
    face_recognize,
    get_video_duration,
    normalize_timestamp,
    resolve_identities,
    unify_speakers,
    unstitch_transcript,
    video_analyze,
    vlm_analyze,
)
from ..state import EngineState
from ..storage.backends.neo4j import Neo4jStorage
from ..utils.logging import get_loguru_logger
from ..utils.nlp import extract_entities
from ..utils.nodes import node_context

logger = get_loguru_logger("nodes.ingestion")


@node_context("ingestion")
async def ingestion_node(state: EngineState) -> dict[str, Any]:
    """Ingestion node for Telegram exports and other non-text data."""
    logger.info("Starting ingestion node")

    # Get configuration
    config_data = state.get("context", {}).get("config", {})
    phase_config = get_phase_config(config_data, "ingestion")

    if not phase_config.get("enabled", True):
        logger.info("Ingestion node disabled. Skipping.")
        return {"__goto__": "drafter"}

    # Get input path
    input_data = state.get("input_data", {})
    input_path = (
        input_data.get("path")
        or input_data.get("export_path")
        or input_data.get("source")
        or phase_config.get("source")
    )

    if not input_path:
        logger.info("No input path for ingestion. Skipping.")
        return {"__goto__": "drafter"}

    input_dir = Path(str(input_path))
    if not input_dir.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return {"__goto__": "drafter"}

    # Create output directory for ingested files
    output_dir = Path("./output/ingested") / state.get("session_id", "default")
    output_dir.mkdir(parents=True, exist_ok=True)

    ingested_files: list[Path] = []

    # --- PDF Processing (runs regardless of other data types) ---
    pdf_files = _find_pdfs(input_dir)
    if pdf_files:
        logger.info(f"Found {len(pdf_files)} PDF file(s) in {input_dir}")
        pdf_ingested = await _process_pdfs(pdf_files, output_dir)
        ingested_files.extend(pdf_ingested)
        logger.info(f"PDF processing complete: {len(pdf_ingested)} files extracted")

    # --- Telegram Processing ---
    messages_html = input_dir / "messages.html"

    if messages_html.exists():
        logger.info("Detected Telegram export. Processing using phased pipeline...")

        # 1. Parse structured items (Phase 1 normalization included)
        items = await _parse_telegram_export(input_dir)
        logger.info(f"Parsed {len(items)} items from Telegram export")

        # Separate items by type for batch processing
        audio_items = [it for it in items if it["type"] == "voice" and it.get("path")]
        video_items = [it for it in items if it["type"] == "video" and it.get("path")]
        photo_items = [it for it in items if it["type"] == "photo" and it.get("path")]

        # Output directory for pipeline artifacts
        pipeline_dir = output_dir / "pipeline"
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Phase 2: Batch Visual Processing
        # Explicitly load LLM model once for all VLM/face operations
        has_visual_work = photo_items or video_items
        if has_visual_work:
            logger.info(f"Phase 2: Loading LLM model ({VISION_MODEL}) for visual processing...")
            infra = _get_infra()
            infra.load("llm", VISION_MODEL, wait=True)
            logger.info(
                f"LLM loaded. Processing {len(photo_items)} photos and {len(video_items)} videos"
            )

        all_embeddings = []
        embedding_to_item = []  # Track which item each embedding belongs to
        video_frames = {}  # rel_path -> list of frames

        # 1. Collect embeddings from photos
        for it in photo_items:
            photo_path = input_dir / it["path"]
            if photo_path.exists():
                faces = await face_recognize(photo_path)
                for face in faces:
                    if face.get("embedding"):
                        all_embeddings.append(face["embedding"])
                        embedding_to_item.append(it)

        # 2. Extract frames and collect embeddings from videos
        audio_fragments_for_megafile = []  # Collect audio from videos for Mega-File processing

        for it in video_items:
            video_path = input_dir / it["path"]
            if video_path.exists():
                video_output_dir = pipeline_dir / f"frames_{video_path.stem}"
                frames = await extract_scene_frames(video_path, video_output_dir)
                video_frames[it["path"]] = frames  # Store for VLM analysis

                # Extract audio from video for ASR processing
                try:
                    video_duration = await get_video_duration(video_path)
                    if video_duration > 0:
                        audio_output = video_output_dir / f"{video_path.stem}_audio.wav"
                        extracted_audio = await extract_audio_from_video(video_path, audio_output)

                        # Add to audio fragments for Mega-File processing
                        audio_fragments_for_megafile.append(
                            {
                                "path": extracted_audio,
                                "source": "video",
                                "source_video": str(video_path),
                                "duration": video_duration,
                                "timestamp": it.get("timestamp"),
                            }
                        )
                        logger.info(
                            f"Audio extracted from video: {extracted_audio} ({video_duration:.1f}s)"
                        )
                except Exception as e:
                    logger.warning(
                        f"Audio extraction failed for {video_path}: {e}. "
                        f"Continuing with VLM-only processing."
                    )

                # Process frames for face recognition
                for frame in frames:
                    frame_path = Path(frame["path"])
                    faces = await face_recognize(frame_path)
                    for face in faces:
                        if face.get("embedding"):
                            all_embeddings.append(face["embedding"])
                            embedding_to_item.append({**it, "frame_path": str(frame_path)})

        # 3. Cluster Identities
        identity_map = {}  # item_path -> cluster_id
        if all_embeddings:
            logger.info(f"Clustering {len(all_embeddings)} face embeddings...")
            embeddings_np = np.array(all_embeddings)
            labels = cluster_identities(embeddings_np)

            for i, label in enumerate(labels):
                if label != -1:  # Ignore noise
                    item = embedding_to_item[i]
                    path = item["path"]
                    if path not in identity_map:
                        identity_map[path] = []
                    identity_map[path].append(f"Identity_Cluster_{label}")

            logger.info(f"Identified {len(set(labels[labels != -1]))} unique visual identities")

        # Phase 3: Batch Acoustic Processing (Mega-File Strategy)
        audio_results = {}  # abs_path_str -> list[seg]

        # Combine audio from audio_items and video audio extracts
        all_audio_items = audio_items.copy()

        # Add video audio fragments to the audio items list
        for audio_frag in audio_fragments_for_megafile:
            all_audio_items.append(
                {
                    "path": str(audio_frag["path"]),
                    "source": audio_frag["source"],
                    "source_video": audio_frag.get("source_video"),
                    "duration": audio_frag["duration"],
                    "timestamp": audio_frag.get("timestamp"),
                }
            )

        if all_audio_items:
            logger.info(
                f"Phase 3: Creating Mega-Files for {len(all_audio_items)} audio fragments "
                f"({len(audio_items)} audio + {len(audio_fragments_for_megafile)} from video)"
            )

            # Phase 3.5: Load vibevoice-asr (load() auto-unloads LLM first + waits for VRAM)
            logger.info("Loading vibevoice-asr for Phase 4 (will auto-unload LLM)...")
            infra = _get_infra()
            infra.load("asr", "vibevoice-asr", wait=True)
            logger.info("vibevoice-asr loaded and ready")
            audio_paths = []

            for it in all_audio_items:
                audio_path = Path(it["path"])
                if not audio_path.is_absolute():
                    # Voice items have relative paths like "voice_messages/foo.ogg"
                    audio_path = input_dir / audio_path
                if audio_path.exists():
                    audio_paths.append(audio_path)

            # Sort audio_paths by their timestamp in items if needed
            boundary_maps = await create_mega_files(audio_paths, pipeline_dir)

            mega_file_results = []

            for i, boundary_map in enumerate(boundary_maps):
                mega_file_path = Path(boundary_map["mega_file_path"])

                # Phase 4: Long-Form ASR
                logger.info(
                    f"Phase 4: Running VibeVoice ASR on {mega_file_path.name} (Batch {i + 1}/{len(boundary_maps)})"
                )

                asr_output = await asr_transcribe(mega_file_path, include_diarization=True)

                if isinstance(asr_output, dict):
                    # Save for speaker unification
                    mega_file_results.append((mega_file_path, asr_output))

                    # Un-stitch transcripts (locally within this mega-file)
                    batch_audio_results = unstitch_transcript(asr_output, boundary_map)

                    # Update local speaker IDs to include batch prefix
                    for segments in batch_audio_results.values():
                        for seg in segments:
                            seg["mf_speaker_id"] = f"mf{i}:{seg['speaker']}"

                    audio_results.update(batch_audio_results)
                    logger.info(f"Successfully unstitched ASR for batch {i + 1}")

            # Phase 5: Cross-Mega-File Acoustic Clustering
            if len(mega_file_results) > 1:
                logger.info("Phase 5: Reconciling speaker identities across mega-files...")
                global_speaker_map = await unify_speakers(mega_file_results, pipeline_dir)

                # Apply global IDs to all unstitched segments
                for segments in audio_results.values():
                    for seg in segments:
                        mf_id = seg.get("mf_speaker_id")
                        if mf_id in global_speaker_map:
                            seg["speaker"] = global_speaker_map[mf_id]
            else:
                # If only one mega-file, mf0:Speaker 0 is already good enough
                pass

        # Phase 6: Multimodal Synthesis & Identity Resolution
        resolved_identity_map = {}
        video_descriptions = {}  # rel_path -> description
        if identity_map or audio_results or video_frames:
            # Re-load LLM for identity resolution + VLM analysis
            logger.info("Phase 6: Loading LLM for identity resolution (will auto-unload ASR)...")
            infra = _get_infra()
            infra.load("llm", DEFAULT_MODEL, wait=True)
            logger.info("Phase 6: Performing final identity resolution and batch VLM analysis...")

            # Prepare metadata for LLM resolution
            asr_all_segments = []
            for _, asr_output in mega_file_results:
                asr_all_segments.extend(asr_output.get("segments", []))

            visual_events = []
            for path, clusters in identity_map.items():
                visual_events.extend(
                    [
                        {
                            "timestamp": "unknown",  # Correlate with item timestamp
                            "cluster": cluster,
                            "path": path,
                        }
                        for cluster in clusters
                    ]
                )

            # Use top 100 items as context for resolution
            text_context = [
                {"timestamp": it["timestamp"], "sender": it["sender"], "text": it.get("text", "")}
                for it in items
                if it.get("text")
            ][:100]

            resolved_identity_map = await resolve_identities(
                asr_all_segments, visual_events, text_context,
                neo4j_storage=await _try_init_neo4j(config_data),
                namespace=config_data.get("storage", {}).get("neo4j", {}).get("namespace", "default"),
            )

            # Batch VLM Video Analysis (Enhancement #4)
            if video_frames:
                logger.info(f"Running batch VLM analysis for {len(video_frames)} videos")
                for video_rel_path, frames in video_frames.items():
                    try:
                        desc = await batch_vlm_video_analysis(frames)
                        video_descriptions[video_rel_path] = desc
                    except RuntimeError:
                        logger.warning("VLM unavailable - skipping all video analysis")
                        break

        processed_messages: list[str] = []
        vlm_available = True  # Will be set False on first permanent VLM failure
        entity_extractions: list[dict[str, Any]] = []  # Collected NER results
        structured_media_items: list[dict[str, Any]] = []  # Items with structured data

        for item in items:
            msg_text = item.get("text") or ""
            media_path_rel = item.get("path")
            item_type = item["type"]
            sender = item.get("sender", "Unknown")
            timestamp = item.get("timestamp", "Unknown")

            content_parts = []
            if msg_text:
                content_parts.append(msg_text)

            if media_path_rel:
                media_path = input_dir / media_path_rel

                if item_type == "voice" and str(media_path) in audio_results:
                    # Use unstitched transcript from Phase 3/4
                    segments = audio_results[str(media_path)]

                    # Apply resolved names
                    for seg in segments:
                        raw_spk = seg["speaker"]
                        if raw_spk in resolved_identity_map:
                            seg["resolved_name"] = resolved_identity_map[raw_spk]
                        elif seg.get("mf_speaker_id") in resolved_identity_map:
                            seg["resolved_name"] = resolved_identity_map[seg["mf_speaker_id"]]

                    transcript_parts = []
                    for seg in segments:
                        name = seg.get("resolved_name", seg["speaker"])
                        transcript_parts.append(f"{name}: {seg['text']}")

                    transcript = " ".join(transcript_parts)
                    content_parts.append(f"[Voice Message Transcript: {transcript}]")

                    # Track unique speakers in this specific snippet
                    unique_speakers = {seg.get("resolved_name", seg["speaker"]) for seg in segments}
                    if len(unique_speakers) > 1:
                        content_parts.append(f"[Speakers in snippet: {', '.join(unique_speakers)}]")

                elif item_type == "photo":
                    # Use identified faces from identity_map
                    if media_path_rel in identity_map:
                        faces = identity_map[media_path_rel]
                        resolved_faces = [resolved_identity_map.get(f, f) for f in faces]
                        faces_str = ", ".join(set(resolved_faces))
                        content_parts.append(f"[People in Photo: {faces_str}]")

                    # VLM description (skip if model doesn't support images)
                    if vlm_available:
                        try:
                            desc = await vlm_analyze(media_path)
                            content_parts.append(f"[Photo Description: {desc}]")
                        except RuntimeError:
                            vlm_available = False
                            logger.warning("VLM unavailable - skipping photo descriptions")

                elif item_type == "video":
                    # Use identified faces from identity_map
                    if media_path_rel in identity_map:
                        faces = identity_map[media_path_rel]
                        resolved_faces = [resolved_identity_map.get(f, f) for f in faces]
                        faces_str = ", ".join(set(resolved_faces))
                        content_parts.append(f"[People in Video: {faces_str}]")

                    # Use batch video analysis if available
                    if media_path_rel in video_descriptions:
                        video_desc = video_descriptions[media_path_rel]
                    elif vlm_available:
                        try:
                            video_desc = await video_analyze(media_path)
                        except RuntimeError:
                            vlm_available = False
                            video_desc = (
                                "[Video analysis unavailable - VLM does not support images]"
                            )
                    else:
                        video_desc = "[Video analysis unavailable - VLM does not support images]"

                    content_parts.append(f"[Video Summary: {video_desc}]")

            item_combined = " ".join(content_parts)
            if item_combined:
                # Extract entities for extra context (optional, can be batched too)
                entities = await extract_entities(item_combined)
                if entities.get("people"):
                    content_parts.append(f"[Entities - People: {', '.join(entities['people'])}]")

                # Collect structured data for graph building
                has_structure = bool(entities.get("people")) or bool(entities.get("locations"))
                if has_structure:
                    entity_extractions.append(entities)

            # Track items that carry structured/ media data (not plain text)
            is_media_with_data = (
                item_type in ("voice", "video", "photo")
                or item.get("forwarded")
                or bool(entities.get("people"))
                if item_combined
                else False
            )
            if is_media_with_data:
                structured_item: dict[str, Any] = {
                    "type": item_type,
                    "path": media_path_rel,
                    "timestamp": timestamp,
                    "sender": sender,
                }
                if (
                    item_type == "voice" and str(input_dir / media_path_rel)
                    if media_path_rel
                    else "" in audio_results
                ):
                    abs_media = str(input_dir / media_path_rel) if media_path_rel else ""
                    segments = audio_results.get(abs_media, [])
                    structured_item["speakers"] = list(
                        {seg.get("resolved_name", seg["speaker"]) for seg in segments}
                    )
                    structured_item["transcript"] = " ".join(
                        seg.get("text", "") for seg in segments
                    )
                if item_type in ("video", "photo") and media_path_rel in identity_map:
                    structured_item["faces"] = identity_map[media_path_rel]
                if item_type == "video" and media_path_rel in video_descriptions:
                    structured_item["description"] = video_descriptions[media_path_rel]
                structured_media_items.append(structured_item)

            final_combined_content = " ".join(content_parts)
            if final_combined_content:
                processed_messages.append(f"**{sender}** [{timestamp}]: {final_combined_content}")

        # 3. Create ingested files
        chat_history_file = output_dir / "chat_history_processed.md"
        with chat_history_file.open("w", encoding="utf-8") as f:
            f.write("# Processed Telegram Chat History\n\n")
            f.write("\n\n".join(processed_messages))

        ingested_files.append(chat_history_file)

    else:
        # General file ingestion (generic files)
        logger.info(f"Ingesting generic files from {input_dir}")

        # Check for standalone audio files
        audio_extensions = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
        audio_files = [f for f in input_dir.iterdir() if f.suffix.lower() in audio_extensions]

        if audio_files:
            logger.info(f"Found {len(audio_files)} audio files. Processing with VibeVoice ASR...")

            for audio_file in audio_files:
                transcript_result = await asr_transcribe(audio_file, include_diarization=True)

                # Create transcript file
                transcript_file = output_dir / f"{audio_file.stem}_transcript.md"
                with transcript_file.open("w", encoding="utf-8") as f:
                    f.write(f"# Transcript: {audio_file.name}\n\n")

                    if isinstance(transcript_result, dict):
                        # VibeVoice with diarization
                        f.write(f"**Duration**: {transcript_result.get('duration', 0):.1f}s\n")
                        f.write(f"**Speakers**: {transcript_result.get('speakers', 0)}\n\n")
                        f.write("## Full Transcript\n\n")
                        f.write(transcript_result.get("text", ""))
                        f.write("\n\n## Speaker Segments\n\n")

                        for seg in transcript_result.get("segments", []):
                            speaker = seg.get("speaker", "Unknown")
                            start = seg.get("start", 0)
                            end = seg.get("end", 0)
                            text = seg.get("text", "")
                            f.write(f"- **{speaker}** [{start:.1f}s - {end:.1f}s]: {text}\n")
                    else:
                        # Simple transcription
                        f.write("## Transcript\n\n")
                        f.write(str(transcript_result))

                ingested_files.append(transcript_file)
                logger.info(f"Transcribed audio: {audio_file.name}")

        # (Placeholder for more generic ingestion)
        pass

    # Produce the "big text dump"
    big_dump_file = output_dir / "big_text_dump.md"
    _combine_ingested_files(ingested_files, big_dump_file)

    logger.info(f"Ingestion completed. Big text dump: {big_dump_file}")

    return {
        "ingested_items": [str(f) for f in ingested_files],
        "context": {
            **state.get("context", {}),
            "ingested_text_files": [str(f) for f in ingested_files],
            "big_text_dump": str(big_dump_file),
            "ingestion_structured": {
                "resolved_identity_map": resolved_identity_map,
                "entity_extractions": entity_extractions,
                "media_items": structured_media_items,
                "video_descriptions": video_descriptions,
            },
        },
        "__goto__": "drafter",
    }


def _extract_pdf_text(pdf_path: Path, page_chunk_size: int = 50) -> str:
    """Extract text from a PDF file using pymupdf.

    Returns markdown-formatted text with page headers.
    Handles large PDFs by extracting page-by-page.

    Args:
        pdf_path: Path to the PDF file.
        page_chunk_size: Number of pages to process at once (memory control).

    Returns:
        Markdown string with extracted text, or empty string on failure.
    """
    if pymupdf is None:
        logger.warning("pymupdf not installed - cannot extract PDF text")
        return f"[PDF text extraction unavailable: pymupdf not installed. File: {pdf_path.name}]"

    try:
        doc = pymupdf.open(str(pdf_path))
        total_pages = len(doc)
        logger.info(f"Extracting text from {pdf_path.name} ({total_pages} pages)")

        parts = [f"# {pdf_path.stem}\n"]
        parts.append(f"**Source**: {pdf_path.name}  \n")
        parts.append(f"**Pages**: {total_pages}\n\n")

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                parts.append(f"## Page {page_num + 1}\n\n")
                parts.append(text)
                parts.append("\n\n")

        doc.close()
        full_text = "".join(parts)
        logger.info(f"Extracted {len(full_text)} chars from {pdf_path.name}")
        return full_text

    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return f"[PDF extraction failed for {pdf_path.name}: {e}]"


def _find_pdfs(input_dir: Path) -> list[Path]:
    """Find all PDF files in the input directory (recursive)."""
    pdfs = list(input_dir.rglob("*.pdf"))
    # Also check for uppercase extension
    pdfs.extend(p for p in input_dir.rglob("*.PDF") if p not in pdfs)
    return sorted(pdfs)


async def _process_pdfs(pdf_files: list[Path], output_dir: Path) -> list[Path]:
    """Process PDF files into ingested markdown files.

    Args:
        pdf_files: List of PDF paths to process.
        output_dir: Directory to write extracted markdown files.

    Returns:
        List of output file paths.
    """
    ingested: list[Path] = []

    for pdf_path in pdf_files:
        logger.info(f"Processing PDF: {pdf_path.name}")

        text = _extract_pdf_text(pdf_path)

        # Write to output dir
        output_file = output_dir / f"{pdf_path.stem}_extracted.md"
        output_file.write_text(text, encoding="utf-8")
        ingested.append(output_file)

        logger.info(f"PDF extracted: {pdf_path.name} -> {output_file.name} ({len(text)} chars)")

    return ingested


async def _parse_telegram_export(input_dir: Path) -> list[dict[str, Any]]:
    """Parse Telegram export (messages.html or result.json)."""
    result_json = input_dir / "result.json"
    items = []

    if result_json.exists():
        logger.info(f"Detected Telegram JSON export: {result_json}")
        with result_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
            messages = data.get("messages", [])
            for msg in messages:
                # Noise filter: skip service messages (pin, group create, etc.)
                if msg.get("type") == "service":
                    continue

                # Basic JSON mapping
                timestamp = normalize_timestamp(msg.get("date", ""))
                sender = msg.get("from", "Unknown")
                text = msg.get("text", "")
                if isinstance(text, list):
                    text = "".join([t.get("text", "") if isinstance(t, dict) else t for t in text])

                media_path = msg.get("file", "") or msg.get("photo", "")
                item_type = msg.get("type", "message")
                forwarded = bool(msg.get("forwarded_from"))

                # Refine type
                if msg.get("media_type") == "voice_message":
                    item_type = "voice"
                elif msg.get("media_type") == "video_file":
                    item_type = "video"
                elif msg.get("photo"):
                    item_type = "photo"
                elif msg.get("sticker_emoji"):
                    # Sticker-only with no text = noise
                    if not text:
                        continue

                # Noise filter: skip items with no content at all
                if not text and not media_path:
                    continue

                items.append(
                    {
                        "type": item_type,
                        "path": media_path,
                        "text": text,
                        "timestamp": timestamp,
                        "sender": sender,
                        "forwarded": forwarded,
                    }
                )
        return items

    # Fallback to HTML
    messages_html = input_dir / "messages.html"
    if not messages_html.exists():
        return items

    with messages_html.open("r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    message_divs = soup.find_all("div", class_="message")

    for div in message_divs:
        # Skip service messages (date headers, group events, pin notifications)
        if "service" in (div.get("class") or []):
            continue

        # Extract metadata
        timestamp_raw = ""
        time_elem = div.find("div", class_="pull_right") or div.find("time")
        if time_elem:
            timestamp_raw = time_elem.get("title", "") or time_elem.text.strip()

        # Phase 1: Temporal Normalization
        timestamp = normalize_timestamp(timestamp_raw)

        sender = "Unknown"
        from_elem = div.find("div", class_="from_name")
        if from_elem:
            sender = from_elem.text.strip()

        text = None
        text_elem = div.find("div", class_="text")
        if text_elem:
            text = text_elem.text.strip()

        media_path = None
        item_type = "message"
        forwarded = False

        # Detect forwarded context
        fwd_body = div.find("div", class_="forwarded")
        if fwd_body:
            forwarded = True

        # Check for media linkages (multiple class patterns for forwarded items)
        photo_link = div.find("a", class_="photo_wrap")
        if photo_link:
            media_path = photo_link.get("href", "")
            item_type = "photo"

        # Voice: check both direct voice div and forwarded media_voice_message
        voice_elem = div.find("div", class_="voice") or div.find(
            "div", class_="media_voice_message"
        )
        if voice_elem:
            audio_src = voice_elem.find("source")
            if audio_src:
                media_path = audio_src.get("src", "")
            item_type = "voice"

        # Video: check both direct video div and forwarded video_file_wrap
        video_elem = div.find("div", class_="video") or div.find("div", class_="video_file_wrap")
        if video_elem:
            video_src = video_elem.find("source")
            if video_src:
                media_path = video_src.get("src", "")
            item_type = "video"

        # Noise filter: skip items with no meaningful content
        # - No text, no media, no sender = empty shell (forwarded body with no attachment)
        # - Sticker-only messages (img with sticker class, no text/audio/video)
        is_sticker = bool(div.find("img", class_="sticker") or div.find("div", class_="sticker"))
        if is_sticker and not text:
            continue
        if not text and not media_path:
            continue

        items.append(
            {
                "type": item_type,
                "path": media_path,
                "text": text,
                "timestamp": timestamp,
                "sender": sender,
                "forwarded": forwarded,
            }
        )

    # Find orphan media (files not in messages.html)
    existing_paths = {item["path"] for item in items if item.get("path")}

    media_dirs = {"photos": "photo", "voice_messages": "voice", "video_files": "video"}

    for dir_name, type_name in media_dirs.items():
        media_dir = input_dir / dir_name
        if media_dir.exists():
            for f in media_dir.iterdir():
                if f.is_file():
                    # Skip thumbnails and small companion images
                    if "_thumb." in f.name:
                        continue
                    rel_path = f"{dir_name}/{f.name}"
                    if rel_path not in existing_paths:
                        items.append(
                            {
                                "type": type_name,
                                "path": rel_path,
                                "text": None,
                                "timestamp": "Found in archive",
                                "sender": "Unknown",
                                "forwarded": False,
                            }
                        )

    return items


def _combine_ingested_files(files: list[Path], output_path: Path):
    """Combine multiple markdown files into one big text dump."""
    with output_path.open("w", encoding="utf-8") as out:
        out.write("# BIG TEXT DUMP - ARCHIVE\n\n")
        for f in files:
            content = f.read_text(encoding="utf-8")
            out.write(f"\n\n--- {f.name} ---\n\n")
            out.write(content)


async def _try_init_neo4j(config_data: dict[str, Any]) -> Neo4jStorage | None:
    """Try to create and initialize Neo4jStorage from config.

    Returns None if Neo4j is disabled or initialization fails.
    Caller is responsible for calling .cleanup() when done.
    """
    neo4j_config = config_data.get("storage", {}).get("neo4j", {})
    if not neo4j_config.get("enabled"):
        return None
    try:
        storage = Neo4jStorage(neo4j_config)
        await storage.initialize()
        return storage
    except Exception as e:
        logger.warning(f"Neo4j init failed for identity resolution (non-fatal): {e}")
        return None
