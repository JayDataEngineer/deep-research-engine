"""Ingestion Node - Optional entry point for non-text data.

Handles extracting and converting non-text data (e.g. PDFs, images, Telegram exports)
into nice, organized text files.

Supports both legacy state return and LangGraph Command return.
"""

import asyncio
from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup
from langgraph.types import Command

from ..state import EngineState
from ..utils.logging import get_loguru_logger
from ..utils.multimodal import asr_transcribe, face_recognize, video_analyze, vlm_analyze

logger = get_loguru_logger("nodes.ingestion")


async def ingestion_node(state: EngineState) -> Command[Literal["drafter"]]:
    """Ingestion node for Telegram exports and other non-text data.

    Returns Command for LangGraph routing.
    """
    logger.info("Starting ingestion node")

    # Get config and check if enabled
    config = state.get("context", {}).get("config", {})
    ingestion_config = config.get("ingestion", {})
    if not ingestion_config.get("enabled", True):
        logger.info("Ingestion node disabled. Skipping.")
        return Command(
            update={
                "current_phase": "drafter",
                "phase_history": state.get("phase_history", []) + ["drafter"],
            },
            goto="drafter",
        )

    # Get input path
    input_data = state.get("input_data", {})
    input_path = input_data.get("path", "")

    # Also check 'source' which is common in configs
    if not input_path:
        input_path = input_data.get("source", "")

    if not input_path:
        # Fallback to session data path if provided in config
        config = state.get("context", {}).get("config", {})
        input_path = config.get("ingestion", {}).get("source", "")

    if not input_path:
        logger.info("No input path for ingestion. Skipping.")
        return Command(
            update={
                "current_phase": "drafter",
                "phase_history": state.get("phase_history", []) + ["drafter"],
            },
            goto="drafter",
        )

    input_dir = Path(input_path)
    if not input_dir.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return Command(
            update={
                "fatal_error": f"Input path does not exist: {input_path}",
                "current_phase": "drafter",
            },
            goto="drafter",
        )

    # Create output directory for ingested files
    output_dir = Path("./output/ingested") / state.get("session_id", "default")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for Telegram export (look for messages.html)
    messages_html = input_dir / "messages.html"

    ingested_files: list[Path] = []

    if messages_html.exists():
        logger.info("Detected Telegram export. Processing...")
        # 1. Parse chat messages
        chat_text_file = output_dir / "chat_history.md"
        await _parse_telegram_messages(messages_html, chat_text_file)
        ingested_files.append(chat_text_file)

        # 2. Process Voice Messages
        voice_dir = input_dir / "voice_messages"
        if voice_dir.exists():
            voice_transcripts_file = output_dir / "voice_transcripts.md"
            await _process_audio_dir(voice_dir, voice_transcripts_file)
            ingested_files.append(voice_transcripts_file)

        # 3. Process Photos
        photos_dir = input_dir / "photos"
        if photos_dir.exists():
            photos_desc_file = output_dir / "photo_descriptions.md"
            await _process_images_dir(photos_dir, photos_desc_file)
            ingested_files.append(photos_desc_file)

        # 4. Process Videos
        videos_dir = input_dir / "video_files"
        if videos_dir.exists():
            videos_desc_file = output_dir / "video_descriptions.md"
            await _process_videos_dir(videos_dir, videos_desc_file)
            ingested_files.append(videos_desc_file)

    else:
        # General file ingestion (generic files)
        logger.info(f"Ingesting generic files from {input_dir}")
        # (Placeholder for more generic ingestion)
        pass

    # Produce the "big text dump"
    big_dump_file = output_dir / "big_text_dump.md"
    _combine_ingested_files(ingested_files, big_dump_file)

    logger.info(f"Ingestion completed. Big text dump: {big_dump_file}")

    return Command(
        update={
            "ingestion_results": [str(f) for f in ingested_files],
            "context": {
                **state.get("context", {}),
                "ingested_text_files": [str(f) for f in ingested_files],
                "big_text_dump": str(big_dump_file),
            },
            "current_phase": "drafter",
            "phase_history": state.get("phase_history", []) + ["drafter"],
        },
        goto="drafter",
    )


async def _parse_telegram_messages(html_path: Path, output_path: Path):
    """Parse Telegram messages.html into a markdown chat log."""
    logger.info(f"Parsing Telegram chat: {html_path}")

    with html_path.open("r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    messages = []
    for msg in soup.find_all("div", class_="message"):
        from_name = msg.find("div", class_="from_name")
        body = msg.find("div", class_="text")
        time = msg.find("div", class_="pull_right date details")

        if from_name and body:
            name = from_name.get_text(strip=True)
            content = body.get_text(strip=True)
            timestamp = time.get("title", "") if time else ""
            messages.append(f"**{name}** [{timestamp}]: {content}")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Telegram Chat History\n\n")
        f.write("\n".join(messages))


async def _process_audio_dir(audio_dir: Path, output_path: Path):
    """Process all audio files into a transcript summary."""
    logger.info(f"Processing audio from {audio_dir}")

    transcripts = []
    audio_files = (
        list(audio_dir.glob("*.ogg"))
        + list(audio_dir.glob("*.mp3"))
        + list(audio_dir.glob("*.wav"))
    )

    tasks = [asr_transcribe(f) for f in audio_files]
    results = await asyncio.gather(*tasks)

    for f, text in zip(audio_files, results, strict=False):
        transcripts.append(f"### {f.name}\n\n{text}\n")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Voice Message Transcripts\n\n")
        f.write("\n".join(transcripts))


async def _process_images_dir(image_dir: Path, output_path: Path):
    """Process all images into descriptions and face info."""
    logger.info(f"Processing images from {image_dir}")

    entries = []
    image_files = (
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.webp"))
    )

    vlm_tasks = [vlm_analyze(f) for f in image_files]
    face_tasks = [face_recognize(f) for f in image_files]

    vlm_results = await asyncio.gather(*vlm_tasks)
    face_results = await asyncio.gather(*face_tasks)

    for f, vlm_text, faces in zip(image_files, vlm_results, face_results, strict=False):
        face_list = ", ".join([p["name"] for p in faces]) if faces else "None detected"
        entries.append(
            f"### {f.name}\n\n**Description:** {vlm_text}\n\n**People detected:** {face_list}\n"
        )

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Photo Descriptions & People Identification\n\n")
        f.write("\n".join(entries))


async def _process_videos_dir(video_dir: Path, output_path: Path):
    """Process all videos into summaries."""
    logger.info(f"Processing videos from {video_dir}")

    entries = []
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mov"))

    tasks = [video_analyze(f) for f in video_files]
    results = await asyncio.gather(*tasks)

    for f, summary in zip(video_files, results, strict=False):
        entries.append(f"### {f.name}\n\n{summary}\n")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Video Summaries\n\n")
        f.write("\n".join(entries))


def _combine_ingested_files(files: list[Path], output_path: Path):
    """Combine multiple markdown files into one big text dump."""
    with output_path.open("w", encoding="utf-8") as out:
        out.write("# BIG TEXT DUMP - ARCHIVE\n\n")
        for f in files:
            content = f.read_text(encoding="utf-8")
            out.write(f"\n\n--- {f.name} ---\n\n")
            out.write(content)


def route_after_ingestion(state: EngineState) -> str:
    """Route to drafter after ingestion (legacy compatibility)."""
    if state.get("fatal_error"):
        return "error"
    return "drafter"
