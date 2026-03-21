"""Ingestion Node - Optional entry point for non-text data.

Handles extracting and converting non-text data (e.g. PDFs, images, Telegram exports)
into nice, organized text files.

Supports both legacy state return and LangGraph Command return.
"""

import asyncio
import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from langgraph.types import Command

from ..state import EngineState
from ..utils.logging import get_loguru_logger
from ..utils.nodes import node_context
from ..config import get_phase_config
from ..utils.multimodal import asr_transcribe, face_recognize, video_analyze, vlm_analyze
from ..utils.nlp import extract_entities

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

    input_path_obj = Path(str(input_path))
    if not input_path_obj.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return {"__goto__": "drafter"}

    # Create output directory for ingested files
    output_dir = Path("./output/ingested") / state.get("session_id", "default")
    output_dir.mkdir(parents=True, exist_ok=True)

    ingested_files: list[Path] = []

    if input_path_obj.is_dir() and (input_path_obj / "messages.html").exists():
        logger.info("Detected Telegram export. Processing...")
        ingested_files = await _process_telegram_export(input_path_obj, output_dir)
    else:
        logger.info(f"Processing generic input: {input_path}")
        ingested_files = await _process_generic_input(input_path_obj, output_dir)

    # Produce the "big text dump"
    big_dump_file = output_dir / "big_text_dump.md"
    _combine_ingested_files(ingested_files, big_dump_file)

    logger.info(f"Ingestion completed. {len(ingested_files)} files processed. Big text dump: {big_dump_file}")

    return {
        "ingested_items": [str(f) for f in ingested_files],
        "context": {
            **state.get("context", {}),
            "ingested_text_files": [str(f) for f in ingested_files],
            "big_text_dump": str(big_dump_file),
        },
        "__goto__": "drafter"
    }


async def _process_generic_input(input_path: Path, output_dir: Path) -> list[Path]:
    """Process a generic file or directory of files."""
    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    else:
        # Recursively find all files
        for ext in ["*.pdf", "*.jpg", "*.jpeg", "*.png", "*.webp", "*.mp3", "*.wav", "*.ogg", "*.m4a", "*.mp4", "*.mov", "*.webm"]:
            files_to_process.extend(list(input_path.rglob(ext)))

    logger.info(f"Found {len(files_to_process)} files to process")

    tasks = [_process_file(f, output_dir) for f in files_to_process]
    results = await asyncio.gather(*tasks)

    # Filter out None results (failed processing)
    return [f for f in results if f is not None]


async def _process_file(file_path: Path, output_dir: Path) -> Path | None:
    """Dispatch file to appropriate processor based on extension."""
    ext = file_path.suffix.lower()

    try:
        if ext == ".pdf":
            return await _process_pdf(file_path, output_dir)
        elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
            return await _process_image(file_path, output_dir)
        elif ext in [".mp3", ".wav", ".ogg", ".m4a"]:
            return await _process_audio(file_path, output_dir)
        elif ext in [".mp4", ".mov", ".webm"]:
            return await _process_video(file_path, output_dir)
        else:
            logger.warning(f"Unsupported file type: {ext} for {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


async def _process_pdf(file_path: Path, output_dir: Path) -> Path:
    """Extract text from PDF and save as Markdown."""
    logger.info(f"Processing PDF: {file_path}")

    output_file = output_dir / f"{file_path.stem}_pdf.md"

    try:
        import fitz  # PyMuPDF
        import pymupdf4llm

        # Use pymupdf4llm for better markdown conversion if available
        md_text = pymupdf4llm.to_markdown(str(file_path))

        with output_file.open("w", encoding="utf-8") as f:
            f.write(f"# Document: {file_path.name}\n")
            f.write(f"Source: {file_path}\n")
            f.write(f"Ingested: {datetime.datetime.now().isoformat()}\n\n")
            f.write(md_text)

    except ImportError:
        logger.warning("pymupdf or pymupdf4llm not installed, falling back to basic extraction")
        try:
            import fitz
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()

            with output_file.open("w", encoding="utf-8") as f:
                f.write(f"# Document: {file_path.name}\n\n")
                f.write(text)
        except ImportError:
             with output_file.open("w", encoding="utf-8") as f:
                f.write(f"# Document: {file_path.name}\n\n")
                f.write(f"[Error: PDF extraction requires pymupdf. File: {file_path}]")

    return output_file


async def _process_image(file_path: Path, output_dir: Path) -> Path:
    """Analyze image and save description as Markdown."""
    logger.info(f"Processing image: {file_path}")

    desc = await vlm_analyze(file_path)
    faces = await face_recognize(file_path)
    entities = await extract_entities(desc)

    output_file = output_dir / f"{file_path.stem}_image.md"
    with output_file.open("w", encoding="utf-8") as f:
        f.write(f"# Image Analysis: {file_path.name}\n")
        f.write(f"Source: {file_path}\n\n")
        f.write(f"## Description\n{desc}\n\n")

        if faces:
            face_names = [f["name"] for f in faces]
            f.write(f"## People Detected\n{', '.join(face_names)}\n\n")

        if entities.get("people") or entities.get("topics") or entities.get("locations"):
            f.write("## Extracted Entities\n")
            if entities.get("people"):
                f.write(f"- **People**: {', '.join(entities['people'])}\n")
            if entities.get("topics"):
                f.write(f"- **Topics**: {', '.join(entities['topics'])}\n")
            if entities.get("locations"):
                f.write(f"- **Locations**: {', '.join(entities['locations'])}\n")
            f.write("\n")

    return output_file


async def _process_audio(file_path: Path, output_dir: Path) -> Path:
    """Transcribe audio and save as Markdown."""
    logger.info(f"Processing audio: {file_path}")

    transcript = await asr_transcribe(file_path)
    entities = await extract_entities(transcript)

    output_file = output_dir / f"{file_path.stem}_audio.md"
    with output_file.open("w", encoding="utf-8") as f:
        f.write(f"# Audio Transcript: {file_path.name}\n")
        f.write(f"Source: {file_path}\n\n")
        f.write(f"## Transcript\n{transcript}\n\n")

        if entities.get("people") or entities.get("topics"):
            f.write("## Extracted Entities\n")
            if entities.get("people"):
                f.write(f"- **People**: {', '.join(entities['people'])}\n")
            if entities.get("topics"):
                f.write(f"- **Topics**: {', '.join(entities['topics'])}\n")
            f.write("\n")

    return output_file


async def _process_video(file_path: Path, output_dir: Path) -> Path:
    """Analyze video and save summary as Markdown."""
    logger.info(f"Processing video: {file_path}")

    video_desc = await video_analyze(file_path)
    entities = await extract_entities(video_desc)

    output_file = output_dir / f"{file_path.stem}_video.md"
    with output_file.open("w", encoding="utf-8") as f:
        f.write(f"# Video Analysis: {file_path.name}\n")
        f.write(f"Source: {file_path}\n\n")
        f.write(f"## Summary\n{video_desc}\n\n")

        if entities.get("people") or entities.get("topics"):
            f.write("## Extracted Entities\n")
            if entities.get("people"):
                f.write(f"- **People**: {', '.join(entities['people'])}\n")
            if entities.get("topics"):
                f.write(f"- **Topics**: {', '.join(entities['topics'])}\n")
            f.write("\n")

    return output_file


async def _process_telegram_export(input_dir: Path, output_dir: Path) -> list[Path]:
    """Process Telegram export and return list of ingested files."""
    # 1. Parse structured items
    items = await _parse_telegram_export(input_dir)
    logger.info(f"Parsed {len(items)} items from Telegram export")

    # 2. Process items (Multimodal + Entity Extraction)
    processed_messages: list[str] = []

    # Group items for batch processing where possible
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
            if media_path.exists():
                if item_type == "photo":
                    desc = await vlm_analyze(media_path)
                    faces = await face_recognize(media_path)
                    face_names = [f["name"] for f in faces]
                    content_parts.append(f"[Photo Description: {desc}]")
                    if face_names:
                        content_parts.append(f"[People in Photo: {', '.join(face_names)}]")

                elif item_type == "voice":
                    transcript = await asr_transcribe(media_path)
                    content_parts.append(f"[Voice Message Transcript: {transcript}]")

                elif item_type == "video":
                    video_desc = await video_analyze(media_path)
                    content_parts.append(f"[Video Summary: {video_desc}]")
            else:
                logger.warning(f"Media file missing: {media_path}")

        item_combined = " ".join(content_parts)
        if item_combined:
            # Extract entities for extra context
            entities = await extract_entities(item_combined)
            if entities.get("people"):
                content_parts.append(f"[Entities - People: {', '.join(entities['people'])}]")
            if entities.get("topics"):
                content_parts.append(f"[Entities - Topics: {', '.join(entities['topics'])}]")

        final_combined_content = " ".join(content_parts)
        if final_combined_content:
            processed_messages.append(f"**{sender}** [{timestamp}]: {final_combined_content}")

    # 3. Create ingested files
    chat_history_file = output_dir / "chat_history_processed.md"
    with chat_history_file.open("w", encoding="utf-8") as f:
        f.write("# Processed Telegram Chat History\n\n")
        f.write("\n\n".join(processed_messages))

    return [chat_history_file]


async def _parse_telegram_export(input_dir: Path) -> list[dict[str, Any]]:
    """Parse Telegram export (messages.html + orphan media)."""
    messages_html = input_dir / "messages.html"
    
    with messages_html.open("r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    items = []
    message_divs = soup.find_all("div", class_="message")
    
    for div in message_divs:
        # Extract metadata
        timestamp = ""
        time_elem = div.find("div", class_="pull_right") or div.find("time")
        if time_elem:
            timestamp = time_elem.get("title", "") or time_elem.text.strip()

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
        out.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
        for f in files:
            content = f.read_text(encoding="utf-8")
            out.write(f"\n\n--- {f.name} ---\n\n")
            out.write(content)
