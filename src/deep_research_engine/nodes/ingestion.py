"""Ingestion Node - Optional entry point for non-text data.

Handles extracting and converting non-text data (e.g. PDFs, images, Telegram exports)
into nice, organized text files.

Supports both legacy state return and LangGraph Command return.
"""

import asyncio
from pathlib import Path
from typing import Literal, Any

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
        logger.info("Detected Telegram export. Processing...")
        
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
        
        ingested_files.append(chat_history_file)

    else:
        # General file ingestion (generic files)
        logger.info(f"Ingesting generic files from {input_dir}")
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
        for f in files:
            content = f.read_text(encoding="utf-8")
            out.write(f"\n\n--- {f.name} ---\n\n")
            out.write(content)


