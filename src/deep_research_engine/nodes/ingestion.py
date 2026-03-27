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

from ..config import get_phase_config
from ..shared.multimodal import (
    batch_vlm_video_analysis,
    cluster_identities,
    create_mega_files,
    extract_audio_from_video,
    extract_scene_frames,
    get_video_duration,
    normalize_timestamp,
    purge_vram,
    resolve_identities,
    unify_speakers,
    unstitch_transcript,
)
from ..state import EngineState
from ..utils.logging import get_loguru_logger

# Note: multimodal.py in utils might still be needed for direct asr_transcribe etc.
from ..shared.multimodal import asr_transcribe, face_recognize, video_analyze, vlm_analyze
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
    input_path = input_data.get("path") or input_data.get("source") or phase_config.get("source")

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
        logger.info(f"Phase 2: Processing {len(photo_items)} photos and {len(video_items)} videos")
        
        all_embeddings = []
        embedding_to_item = [] # Track which item each embedding belongs to
        video_frames = {} # rel_path -> list of frames

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
                        audio_fragments_for_megafile.append({
                            "path": extracted_audio,
                            "source": "video",
                            "source_video": str(video_path),
                            "duration": video_duration,
                            "timestamp": it.get("timestamp")
                        })
                        logger.info(
                            f"Audio extracted from video: {extracted_audio} "
                            f"({video_duration:.1f}s)"
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
        identity_map = {} # item_path -> cluster_id
        if all_embeddings:
            logger.info(f"Clustering {len(all_embeddings)} face embeddings...")
            embeddings_np = np.array(all_embeddings)
            labels = cluster_identities(embeddings_np)
            
            for i, label in enumerate(labels):
                if label != -1: # Ignore noise
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
            all_audio_items.append({
                "path": str(audio_frag["path"]),
                "source": audio_frag["source"],
                "source_video": audio_frag.get("source_video"),
                "duration": audio_frag["duration"],
                "timestamp": audio_frag.get("timestamp")
            })
        
        if all_audio_items:
            logger.info(
                f"Phase 3: Creating Mega-Files for {len(all_audio_items)} audio fragments "
                f"({len(audio_items)} audio + {len(audio_fragments_for_megafile)} from video)"
            )
            audio_paths = []
            
            for it in all_audio_items:
                audio_path = Path(it["path"])
                if audio_path.exists():
                    audio_paths.append(audio_path)

            # Sort audio_paths by their timestamp in items if needed
            boundary_maps = await create_mega_files(audio_paths, pipeline_dir)
            
            mega_file_results = []
            
            for i, boundary_map in enumerate(boundary_maps):
                mega_file_path = Path(boundary_map["mega_file_path"])
                
                # Phase 4: Long-Form ASR
                purge_vram() # Ensure GPU is ready for ASR
                logger.info(f"Phase 4: Running VibeVoice ASR on {mega_file_path.name} (Batch {i+1}/{len(boundary_maps)})")
                
                asr_output = await asr_transcribe(
                    mega_file_path, 
                    include_diarization=True
                )
                
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
                    logger.info(f"Successfully unstitched ASR for batch {i+1}")

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
        video_descriptions = {} # rel_path -> description
        if identity_map or audio_results or video_frames:
            logger.info("Phase 6: Performing final identity resolution and batch VLM analysis...")
            
            # Prepare metadata for LLM resolution
            asr_all_segments = []
            for _, asr_output in mega_file_results:
                asr_all_segments.extend(asr_output.get("segments", []))
            
            visual_events = []
            for path, clusters in identity_map.items():
                visual_events.extend([
                    {
                        "timestamp": "unknown", # Correlate with item timestamp
                        "cluster": cluster,
                        "path": path
                    }
                    for cluster in clusters
                ])
            
            # Use top 100 items as context for resolution
            text_context = [
                {"timestamp": it["timestamp"], "sender": it["sender"], "text": it.get("text", "")}
                for it in items if it.get("text")
            ][:100]
            
            resolved_identity_map = await resolve_identities(
                asr_all_segments,
                visual_events,
                text_context
            )

            # Batch VLM Video Analysis (Enhancement #4)
            if video_frames:
                logger.info(f"Running batch VLM analysis for {len(video_frames)} videos")
                for video_rel_path, frames in video_frames.items():
                    desc = await batch_vlm_video_analysis(frames)
                    video_descriptions[video_rel_path] = desc

        processed_messages: list[str] = []

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
                    
                    # Fallback to single-item VLM for description
                    desc = await vlm_analyze(media_path)
                    content_parts.append(f"[Photo Description: {desc}]")
                    
                elif item_type == "video":
                    # Use identified faces from identity_map
                    if media_path_rel in identity_map:
                        faces = identity_map[media_path_rel]
                        resolved_faces = [resolved_identity_map.get(f, f) for f in faces]
                        faces_str = ", ".join(set(resolved_faces))
                        content_parts.append(f"[People in Video: {faces_str}]")
                        
                    # Use batch video analysis (Enhancement #4)
                    if media_path_rel in video_descriptions:
                        video_desc = video_descriptions[media_path_rel]
                    else:
                        # Fallback to single-item analysis if batching skipped it
                        video_desc = await video_analyze(media_path)
                    
                    content_parts.append(f"[Video Summary: {video_desc}]")

            item_combined = " ".join(content_parts)
            if item_combined:
                # Extract entities for extra context (optional, can be batched too)
                entities = await extract_entities(item_combined)
                if entities.get("people"):
                    content_parts.append(f"[Entities - People: {', '.join(entities['people'])}]")

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
                transcript_result = await asr_transcribe(
                    audio_file,
                    include_diarization=True
                )

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
        },
        "__goto__": "drafter"
    }




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
                # Basic JSON mapping
                timestamp = normalize_timestamp(msg.get("date", ""))
                sender = msg.get("from", "Unknown")
                text = msg.get("text", "")
                if isinstance(text, list):
                    text = "".join([t.get("text", "") if isinstance(t, dict) else t for t in text])

                media_path = msg.get("file", "") or msg.get("photo", "")
                item_type = msg.get("type", "message")
                
                # Refine type
                if msg.get("media_type") == "voice_message":
                    item_type = "voice"
                elif msg.get("media_type") == "video_file":
                    item_type = "video"
                elif msg.get("photo"):
                    item_type = "photo"

                if text or media_path:
                    items.append({
                        "type": item_type,
                        "path": media_path,
                        "text": text,
                        "timestamp": timestamp,
                        "sender": sender,
                        "msg_id": msg.get("id")
                    })
        return items

    # Fallback to HTML
    messages_html = input_dir / "messages.html"
    if not messages_html.exists():
        return items

    with messages_html.open("r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    message_divs = soup.find_all("div", class_="message")

    for div in message_divs:
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

        # Check for media linkages
        photo_link = div.find("a", class_="photo_wrap")
        if photo_link:
            media_path = photo_link.get("href", "")
            item_type = "photo"

        voice_elem = div.find("div", class_="voice")
        if voice_elem:
            audio_src = voice_elem.find("source")
            if audio_src:
                media_path = audio_src.get("src", "")
            item_type = "voice"

        video_elem = div.find("div", class_="video")
        if video_elem:
            video_src = video_elem.find("source")
            if video_src:
                media_path = video_src.get("src", "")
            item_type = "video"

        if text or media_path:
            items.append({
                "type": item_type,
                "path": media_path,
                "text": text,
                "timestamp": timestamp,
                "sender": sender,
            })

    # Find orphan media (files not in messages.html)
    existing_paths = {item["path"] for item in items if item.get("path")}

    media_dirs = {
        "photos": "photo",
        "voice_messages": "voice",
        "video_files": "video"
    }

    for dir_name, type_name in media_dirs.items():
        media_dir = input_dir / dir_name
        if media_dir.exists():
            for f in media_dir.iterdir():
                if f.is_file():
                    rel_path = f"{dir_name}/{f.name}"
                    if rel_path not in existing_paths:
                        items.append({
                            "type": type_name,
                            "path": rel_path,
                            "text": None,
                            "timestamp": "Found in archive",
                            "sender": "System",
                        })

    return items


def _combine_ingested_files(files: list[Path], output_path: Path):
    """Combine multiple markdown files into one big text dump."""
    with output_path.open("w", encoding="utf-8") as out:
        out.write("# BIG TEXT DUMP - ARCHIVE\n\n")
        for f in files:
            content = f.read_text(encoding="utf-8")
            out.write(f"\n\n--- {f.name} ---\n\n")
            out.write(content)


