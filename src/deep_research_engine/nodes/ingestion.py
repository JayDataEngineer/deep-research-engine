"""Ingestion Node - Unified entry point for all non-text data.

Handles extracting and converting non-text data (PDFs, images, audio, video,
office docs, Telegram exports) into organized text files via a single unified
pipeline.

Document extraction is delegated to Kreuzberg (91+ formats) via the
document_router. Audio/video/photo processing uses our 6-phase multimodal
pipeline.

Pipeline phases:
  0. Scan input + route documents through Kreuzberg + parse Telegram
  1.5 NER on text items (entity names fed into Phase 6)
  2. Batch Visual Processing (face recognition + VLM)
  3-4. Direct ASR (VibeVoice per-file transcription with diarization)
  5. Cross-File Speaker Unification
  6. Identity Resolution + NER integration
  7. Enrich + write outputs

Supports both legacy state return and LangGraph Command return.
"""

import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

try:
    import pymupdf
except ImportError:
    pymupdf = None

from ..config import get_phase_config
from ..ingestion.document_router import DocumentRouter
from ..ingestion.kreuzberg_extractor import KreuzbergConfig
from ..shared.config import INGESTION_MODEL
from ..shared.multimodal import (
    _get_infra,
    asr_transcribe,
    batch_vlm_video_analysis,
    cluster_identities,
    extract_audio_from_video,
    extract_scene_frames,
    face_recognize,
    get_video_duration,
    normalize_timestamp,
    resolve_identities,
    unify_speakers,
    video_analyze,
    vlm_analyze,
)
from ..state import EngineState
from ..storage.backends.neo4j import Neo4jStorage
from ..utils.logging import get_loguru_logger
from ..utils.nlp import extract_entities
from ..utils.nodes import node_context

logger = get_loguru_logger("nodes.ingestion")

# ---------------------------------------------------------------------------
# File extension sets (for scanning only; Kreuzberg handles all document formats)
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".wma", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv"}
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

# ---------------------------------------------------------------------------
# Scan result typed dict
# ---------------------------------------------------------------------------


class ScanResult(TypedDict):
    """Result of scanning an input directory."""

    telegram_json: bool
    telegram_html: bool
    audio: list[Path]
    video: list[Path]
    photos: list[Path]


# ---------------------------------------------------------------------------
# Phase checkpointing
# ---------------------------------------------------------------------------


def _save_checkpoint(pipeline_dir: Path, name: str, data: Any) -> None:
    """Save a phase checkpoint artifact as JSON."""
    path = pipeline_dir / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Checkpoint saved: {path}")


def _load_checkpoint(pipeline_dir: Path, name: str) -> Any | None:
    """Load a phase checkpoint artifact. Returns None if not found."""
    path = pipeline_dir / f"{name}.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Checkpoint loaded (skipping phase): {path}")
        return data
    return None


# ---------------------------------------------------------------------------
# Dead letter queue
# ---------------------------------------------------------------------------


def _write_dead_letter(pipeline_dir: Path, failed_items: list[dict[str, Any]]) -> None:
    """Append failed items to dead_letter_queue.json."""
    dlq_path = pipeline_dir / "dead_letter_queue.json"
    existing: list[dict[str, Any]] = []
    if dlq_path.exists():
        try:
            with dlq_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.extend(failed_items)
    with dlq_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2, default=str)
    logger.warning(f"Dead letter queue updated: {len(failed_items)} new failures ({dlq_path})")


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def _scan_input(input_dir: Path) -> ScanResult:
    """Scan input directory for all supported file types.

    Returns a typed dict categorising every discoverable file.
    Document files are handled by Kreuzberg via DocumentRouter — this scan
    only tracks audio/video/photo for the multimodal pipeline.
    """
    result: ScanResult = {
        "telegram_json": (input_dir / "result.json").exists(),
        "telegram_html": (input_dir / "messages.html").exists(),
        "audio": _find_by_extensions(input_dir, AUDIO_EXTENSIONS),
        "video": _find_by_extensions(input_dir, VIDEO_EXTENSIONS),
        "photos": _find_by_extensions(input_dir, PHOTO_EXTENSIONS),
    }
    # Exclude Telegram export sub-directories from generic scans
    tg_dirs = {"photos", "voice_messages", "video_files", "stickers", "files"}
    if result["telegram_json"] or result["telegram_html"]:
        for key in ("audio", "video", "photos"):
            result[key] = [p for p in result[key] if p.parent.name not in tg_dirs]
    return result


def _find_by_extensions(input_dir: Path, extensions: set[str]) -> list[Path]:
    """Recursively find files matching given extensions."""
    files: list[Path] = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(files)


# ---------------------------------------------------------------------------
# Generic file parsing (for standalone media files without metadata)
# ---------------------------------------------------------------------------


def _parse_generic_files(
    files: list[Path],
    item_type: str,
    input_dir: Path,
) -> list[dict[str, Any]]:
    """Convert standalone files into unified item schema.

    Uses file mtime as timestamp, "Unknown" as sender.
    """
    items: list[dict[str, Any]] = []
    for f in files:
        try:
            mtime = f.stat().st_mtime
            ts = normalize_timestamp(mtime)
        except OSError:
            ts = {"iso": "Unknown", "tz_offset": ""}
        rel_path = str(f.relative_to(input_dir)) if f.is_relative_to(input_dir) else str(f)
        items.append(
            {
                "type": item_type,
                "path": rel_path,
                "text": None,
                "timestamp": ts["iso"],
                "tz_offset": ts.get("tz_offset", ""),
                "sender": "Unknown",
                "forwarded": False,
            }
        )
    return items


# ---------------------------------------------------------------------------
# VLM circuit breaker
# ---------------------------------------------------------------------------


class VLMBatchCircuitBreaker:
    """Per-batch VLM circuit breaker.

    Tracks failures in windows of ``batch_size`` items.
    If failures exceed ``threshold`` in any window, VLM is disabled for that
    window only. Failed items are logged to the dead letter queue.
    """

    def __init__(
        self,
        pipeline_dir: Path,
        batch_size: int = 10,
        threshold: int = 5,
    ) -> None:
        self.pipeline_dir = pipeline_dir
        self.batch_size = batch_size
        self.threshold = threshold
        self._failures: list[dict[str, Any]] = []
        self._batch_fail_count = 0
        self._batch_item_count = 0

    def record_failure(self, item: dict[str, Any], error: str) -> None:
        """Record a VLM failure."""
        self._batch_fail_count += 1
        self._batch_item_count += 1
        self._failures.append(
            {"item": item.get("path", "unknown"), "error": error, "type": item.get("type")}
        )
        self._maybe_flush()

    def record_success(self) -> None:
        """Record a VLM success."""
        self._batch_item_count += 1
        self._maybe_flush()

    def is_tripped(self) -> bool:
        """Check if circuit breaker is tripped for current batch."""
        return self._batch_fail_count > self.threshold

    def _maybe_flush(self) -> None:
        """Flush batch window if full."""
        if self._batch_item_count >= self.batch_size:
            if self._batch_fail_count > 0:
                _write_dead_letter(self.pipeline_dir, self._failures)
                self._failures = []
            self._batch_fail_count = 0
            self._batch_item_count = 0

    def flush_remaining(self) -> None:
        """Flush any remaining failures."""
        if self._failures:
            _write_dead_letter(self.pipeline_dir, self._failures)
            self._failures = []


# ---------------------------------------------------------------------------
# Main ingestion node (unified pipeline)
# ---------------------------------------------------------------------------


@node_context("ingestion")
async def ingestion_node(state: EngineState) -> dict[str, Any]:
    """Ingestion node — unified pipeline for all data sources.

    Replaces the previous Telegram-vs-generic split with a single flow:
      scan → parse items → Phase 2-6 → enrich → write outputs
    """
    logger.info("Starting ingestion node (unified pipeline)")

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

    # Create output directories
    output_dir = Path("./output/ingested") / state.get("session_id", "default")
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    ingested_files: list[Path] = []

    # ================================================================
    # Phase 0: Scan input + route to extractors
    # ================================================================
    scan = _scan_input(input_dir)
    is_telegram = scan["telegram_json"] or scan["telegram_html"]

    logger.info(
        f"Scan results: telegram={is_telegram}, "
        f"audio={len(scan['audio'])}, video={len(scan['video'])}, "
        f"photos={len(scan['photos'])}"
    )

    items: list[dict[str, Any]] = []
    document_chunks: list[dict[str, Any]] = []

    # --- Route documents through Kreuzberg ---
    kreuzberg_config = KreuzbergConfig(
        chunking_enabled=True,
        chunk_max_chars=4000,
        chunk_max_overlap=200,
        output_format="markdown",
        ocr_enabled=True,
        table_extraction=True,
    )
    router = DocumentRouter(input_dir, kreuzberg_config)
    route_result = await router.route()

    if route_result.telegram_detected and route_result.telegram_items:
        items.extend(route_result.telegram_items)
        logger.info(f"Telegram: {route_result.total_telegram_messages} messages parsed")

    if route_result.documents:
        success_docs = [d for d in route_result.documents if d.success]
        failed_docs = [d for d in route_result.documents if not d.success]
        items.extend(route_result.pipeline_items)
        document_chunks = route_result.document_chunks
        ingested_files.extend([Path(d.file_path) for d in success_docs if d.file_path])
        logger.info(
            f"Kreuzberg: {len(success_docs)} docs extracted, "
            f"{route_result.total_chunks} chunks, "
            f"{route_result.total_tables} tables"
        )
        if failed_docs:
            for fail in failed_docs:
                logger.warning(f"Doc extraction failed: {fail.file_path}: {fail.error}")

    if route_result.errors:
        for err in route_result.errors:
            logger.warning(f"Routing error: {err['file']}: {err['error']}")

    # --- Generic audio/video/photo items (standalone, not from Telegram) ---
    for audio_path in scan["audio"]:
        items.extend(_parse_generic_files([audio_path], "voice", input_dir))
    for video_path in scan["video"]:
        items.extend(_parse_generic_files([video_path], "video", input_dir))
    for photo_path in scan["photos"]:
        items.extend(_parse_generic_files([photo_path], "photo", input_dir))

    logger.info(f"Total items after Phase 0: {len(items)}")

    if not items:
        logger.info("No items to process. Skipping pipeline.")
        return {"__goto__": "drafter"}

    # Separate items by type for phase routing
    audio_items = [it for it in items if it["type"] == "voice" and it.get("path")]
    video_items = [it for it in items if it["type"] == "video" and it.get("path")]
    photo_items = [it for it in items if it["type"] == "photo" and it.get("path")]

    # Initialize pipeline state variables (ensured defined regardless of code path)
    resolved_identity_map: dict[str, str] = {}
    video_descriptions: dict[str, str] = {}
    audio_fragments_for_megafile: list[dict[str, Any]] = []
    file_asr_results: list[tuple[Path, dict[str, Any]]] = []

    # ================================================================
    # Phase 1.5: Early NER on text items
    # ================================================================
    pre_named_entities: set[str] = set()

    checkpoint_ner = _load_checkpoint(pipeline_dir, "phase1.5_ner")
    if checkpoint_ner is None:
        logger.info("Phase 1.5: Running NER on text items...")
        for item in items:
            text = item.get("text")
            if text and len(text.strip()) > 10:
                try:
                    ents = await extract_entities(text[:4000], model=INGESTION_MODEL)
                    for p in ents.get("people", []):
                        pre_named_entities.add(p)
                except Exception:
                    pass
        _save_checkpoint(pipeline_dir, "phase1.5_ner", sorted(pre_named_entities))
        logger.info(f"Phase 1.5: Found {len(pre_named_entities)} unique person names via NER")
    else:
        pre_named_entities = set(checkpoint_ner)

    # ================================================================
    # Phase 2: Batch Visual Processing
    # ================================================================
    all_embeddings: list[Any] = []
    embedding_to_item: list[dict[str, Any]] = []
    video_frames: dict[str, list[dict[str, Any]]] = {}
    identity_map: dict[str, list[str]] = {}

    checkpoint_p2 = _load_checkpoint(pipeline_dir, "phase2_embeddings")
    has_visual_work = photo_items or video_items

    if has_visual_work and checkpoint_p2 is None:
        # Models are pre-loaded and managed by the orchestrator's VRAM management.
        # Just log what we're doing and proceed.
        logger.info(f"Phase 2: Processing {len(photo_items)} photos and {len(video_items)} videos")

        # Collect embeddings from photos
        for it in photo_items:
            photo_path = Path(it["path"])
            if not photo_path.is_absolute():
                photo_path = input_dir / it["path"]
            if photo_path.exists():
                faces = await face_recognize(photo_path)
                for face in faces:
                    if face.get("embedding"):
                        all_embeddings.append(face["embedding"])
                        embedding_to_item.append(it)

        # Extract frames and collect embeddings from videos
        audio_fragments_for_megafile: list[dict[str, Any]] = []

        for it in video_items:
            video_path = Path(it["path"])
            if not video_path.is_absolute():
                video_path = input_dir / it["path"]
            if not video_path.exists():
                continue

            video_output_dir = pipeline_dir / f"frames_{video_path.stem}"
            frames = await extract_scene_frames(video_path, video_output_dir)
            video_frames[it["path"]] = frames

            # Extract audio from video
            try:
                video_duration = await get_video_duration(video_path)
                if video_duration > 0:
                    audio_output = video_output_dir / f"{video_path.stem}_audio.wav"
                    extracted_audio = await extract_audio_from_video(video_path, audio_output)
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

        # Cluster identities
        if all_embeddings:
            logger.info(f"Clustering {len(all_embeddings)} face embeddings...")
            embeddings_np = np.array(all_embeddings)
            labels = cluster_identities(embeddings_np)

            for i, label in enumerate(labels):
                if label != -1:
                    item = embedding_to_item[i]
                    path = item["path"]
                    if path not in identity_map:
                        identity_map[path] = []
                    identity_map[path].append(f"Identity_Cluster_{label}")

            logger.info(f"Identified {len(set(labels[labels != -1]))} unique visual identities")

        _save_checkpoint(
            pipeline_dir,
            "phase2_embeddings",
            {"identity_map": identity_map, "video_frame_keys": list(video_frames.keys())},
        )
    elif checkpoint_p2 is not None:
        identity_map = checkpoint_p2.get("identity_map", {})
        # Video frames would need re-extraction for VLM but identity map is preserved
        logger.info("Phase 2 checkpoint loaded — video frames re-extracted for VLM if needed")

        # Re-extract video frames for VLM analysis even if checkpoint exists
        for it in video_items:
            video_path = Path(it["path"])
            if not video_path.is_absolute():
                video_path = input_dir / it["path"]
            if video_path.exists():
                video_output_dir = pipeline_dir / f"frames_{video_path.stem}"
                frames = await extract_scene_frames(video_path, video_output_dir)
                video_frames[it["path"]] = frames

        # Collect audio fragments from videos even on checkpoint load
        audio_fragments_for_megafile: list[dict[str, Any]] = []
        for it in video_items:
            video_path = Path(it["path"])
            if not video_path.is_absolute():
                video_path = input_dir / it["path"]
            if video_path.exists():
                try:
                    video_duration = await get_video_duration(video_path)
                    if video_duration > 0:
                        video_output_dir = pipeline_dir / f"frames_{video_path.stem}"
                        audio_output = video_output_dir / f"{video_path.stem}_audio.wav"
                        extracted_audio = await extract_audio_from_video(video_path, audio_output)
                        audio_fragments_for_megafile.append(
                            {
                                "path": extracted_audio,
                                "source": "video",
                                "source_video": str(video_path),
                                "duration": video_duration,
                                "timestamp": it.get("timestamp"),
                            }
                        )
                except Exception:
                    pass
    else:
        audio_fragments_for_megafile: list[dict[str, Any]] = []

    # ================================================================
    # Phase 3-4: Direct Per-File ASR (VibeVoice with diarization)
    # ================================================================
    audio_results: dict[str, list[dict[str, Any]]] = {}

    # Combine audio sources
    all_audio_items = audio_items.copy()
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

    # Track per-file ASR results for Phase 5 speaker unification
    # Format: list of (audio_path, asr_output_dict) — same as old mega_file_results
    file_asr_results: list[tuple[Path, dict[str, Any]]] = []

    if all_audio_items:
        logger.info(
            f"Phase 3-4: Running per-file ASR on {len(all_audio_items)} audio files "
            f"({len(audio_items)} voice notes + {len(audio_fragments_for_megafile)} from video)"
        )

        checkpoint_p4 = _load_checkpoint(pipeline_dir, "phase4_transcripts")
        checkpoint_p5 = _load_checkpoint(pipeline_dir, "phase5_speaker_map")

        if checkpoint_p4 is None:
            # Resolve audio paths
            for idx, it in enumerate(all_audio_items):
                audio_path = Path(it["path"])
                if not audio_path.is_absolute():
                    audio_path = input_dir / audio_path
                if not audio_path.exists():
                    logger.warning(f"Audio file not found, skipping: {audio_path}")
                    continue

                logger.info(
                    f"Phase 4: ASR on {audio_path.name} (file {idx + 1}/{len(all_audio_items)})"
                )

                try:
                    asr_output = await asr_transcribe(audio_path, include_diarization=True)
                except Exception as e:
                    logger.error(f"ASR failed for {audio_path.name}: {e}")
                    asr_output = {
                        "text": f"[ASR failed: {e}]",
                        "segments": [],
                        "speakers": 0,
                        "error": str(e),
                    }

                if isinstance(asr_output, dict):
                    file_asr_results.append((audio_path, asr_output))

                    # Assign file-scoped speaker IDs (no unstitching needed)
                    segments = asr_output.get("segments", [])
                    for seg in segments:
                        seg["mf_speaker_id"] = f"file{idx}:{seg['speaker']}"

                    # Map segments to the original file
                    audio_results[str(audio_path)] = segments
                    logger.info(
                        f"ASR complete for {audio_path.name}: "
                        f"{len(segments)} segments, {asr_output.get('speakers', 0)} speakers"
                    )
                elif isinstance(asr_output, str):
                    # Plain text result (no diarization) — store as single segment
                    audio_results[str(audio_path)] = [
                        {
                            "speaker": "Speaker_0",
                            "mf_speaker_id": f"file{idx}:Speaker_0",
                            "text": asr_output,
                            "start": 0.0,
                            "end": 0.0,
                        }
                    ]

            _save_checkpoint(
                pipeline_dir,
                "phase4_transcripts",
                {
                    "audio_results": dict(audio_results),
                    "file_count": len(file_asr_results),
                },
            )
        else:
            # Load from checkpoint
            audio_results = checkpoint_p4.get("audio_results", {})
            logger.info(f"Phase 4 checkpoint loaded: {len(audio_results)} transcripts")

        # Phase 5: Cross-File Speaker Unification
        if checkpoint_p5 is None and len(file_asr_results) > 1:
            logger.info("Phase 5: Reconciling speaker identities across files...")
            global_speaker_map = await unify_speakers(file_asr_results, pipeline_dir)

            for segments in audio_results.values():
                for seg in segments:
                    mf_id = seg.get("mf_speaker_id")
                    if mf_id in global_speaker_map:
                        seg["speaker"] = global_speaker_map[mf_id]

            _save_checkpoint(pipeline_dir, "phase5_speaker_map", global_speaker_map)
        elif checkpoint_p5 is not None:
            global_speaker_map = checkpoint_p5
            for segments in audio_results.values():
                for seg in segments:
                    mf_id = seg.get("mf_speaker_id")
                    if mf_id in global_speaker_map:
                        seg["speaker"] = global_speaker_map[mf_id]

    # ================================================================
    # Phase 6: Identity Resolution + NER integration
    # ================================================================
    checkpoint_p6 = _load_checkpoint(pipeline_dir, "phase6_identity_map")

    if checkpoint_p6 is None and (identity_map or audio_results or video_frames):
        # Reload LLM for identity resolution (was unloaded for ASR)
        logger.info("Reloading LLM for identity resolution...")
        infra = _get_infra()
        infra.load("llm", INGESTION_MODEL, wait=True)
        logger.info("Phase 6: Performing final identity resolution and batch VLM analysis...")

        # Prepare metadata for LLM resolution
        asr_all_segments: list[dict[str, Any]] = []
        for _, asr_output in file_asr_results:
            asr_all_segments.extend(asr_output.get("segments", []))

        visual_events: list[dict[str, Any]] = []
        for path, clusters in identity_map.items():
            visual_events.extend(
                [
                    {
                        "timestamp": "unknown",
                        "cluster": cluster,
                        "path": path,
                    }
                    for cluster in clusters
                ]
            )

        # Use top 100 text items as context, include pre-NER names
        text_context = [
            {
                "timestamp": it["timestamp"],
                "sender": it["sender"],
                "text": it.get("text", ""),
            }
            for it in items
            if it.get("text")
        ][:100]

        # Feed pre-NER entity names into identity resolution
        if pre_named_entities:
            logger.info(
                f"Phase 6: Including {len(pre_named_entities)} pre-NER names "
                f"for identity resolution context"
            )

        resolved_identity_map = await resolve_identities(
            asr_all_segments,
            visual_events,
            text_context,
            neo4j_storage=await _try_init_neo4j(config_data),
            namespace=config_data.get("storage", {}).get("neo4j", {}).get("namespace", "default"),
            pre_named_entities=sorted(pre_named_entities),
            model=INGESTION_MODEL,
        )

        # Batch VLM Video Analysis with circuit breaker
        cb = VLMBatchCircuitBreaker(pipeline_dir)
        if video_frames:
            logger.info(f"Running batch VLM analysis for {len(video_frames)} videos")
            for video_rel_path, frames in video_frames.items():
                if cb.is_tripped():
                    logger.warning("VLM circuit breaker tripped — skipping remaining videos")
                    break
                try:
                    desc = await batch_vlm_video_analysis(frames)
                    video_descriptions[video_rel_path] = desc
                    cb.record_success()
                except RuntimeError as e:
                    cb.record_failure({"path": video_rel_path, "type": "video"}, str(e))
                    logger.warning(f"VLM failed for {video_rel_path}: {e}")
                except Exception as e:
                    cb.record_failure({"path": video_rel_path, "type": "video"}, str(e))
                    logger.warning(f"VLM failed for {video_rel_path}: {e}")
            cb.flush_remaining()

        _save_checkpoint(pipeline_dir, "phase6_identity_map", resolved_identity_map)
    elif checkpoint_p6 is not None:
        resolved_identity_map = checkpoint_p6
        # Reload LLM for video VLM analysis (was unloaded for ASR)
        logger.info("Reloading LLM for VLM video analysis...")
        infra = _get_infra()
        infra.load("llm", INGESTION_MODEL, wait=True)
        # Still need VLM for video descriptions
        if video_frames:
            cb = VLMBatchCircuitBreaker(pipeline_dir)
            for video_rel_path, frames in video_frames.items():
                if cb.is_tripped():
                    break
                try:
                    desc = await batch_vlm_video_analysis(frames)
                    video_descriptions[video_rel_path] = desc
                    cb.record_success()
                except (RuntimeError, Exception) as e:
                    cb.record_failure({"path": video_rel_path, "type": "video"}, str(e))
            cb.flush_remaining()

    # ================================================================
    # Phase 7: Enrich items + write outputs
    # ================================================================
    processed_messages: list[str] = []
    entity_extractions = []
    structured_media_items = []

    # VLM circuit breaker for per-item VLM calls
    vlm_cb = VLMBatchCircuitBreaker(pipeline_dir)

    for item in items:
        msg_text = item.get("text") or ""
        media_path_rel = item.get("path")
        item_type = item["type"]
        sender = item.get("sender", "Unknown")
        timestamp = item.get("timestamp", "Unknown")

        content_parts: list[str] = []
        if msg_text:
            content_parts.append(msg_text)

        if media_path_rel:
            media_path = Path(media_path_rel)
            if not media_path.is_absolute():
                media_path = input_dir / media_path_rel

            if item_type == "voice" and str(media_path) in audio_results:
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

                unique_speakers = {seg.get("resolved_name", seg["speaker"]) for seg in segments}
                if len(unique_speakers) > 1:
                    content_parts.append(f"[Speakers in snippet: {', '.join(unique_speakers)}]")

            elif item_type == "photo":
                if media_path_rel in identity_map:
                    faces = identity_map[media_path_rel]
                    resolved_faces = [resolved_identity_map.get(f, f) for f in faces]
                    faces_str = ", ".join(set(resolved_faces))
                    content_parts.append(f"[People in Photo: {faces_str}]")

                if not vlm_cb.is_tripped() and media_path.exists():
                    try:
                        desc = await vlm_analyze(media_path)
                        content_parts.append(f"[Photo Description: {desc}]")
                        vlm_cb.record_success()
                    except RuntimeError:
                        vlm_cb.record_failure(item, "VLM unavailable for photo")
                        logger.warning("VLM unavailable - skipping photo descriptions")
                    except Exception as e:
                        vlm_cb.record_failure(item, str(e))

            elif item_type == "video":
                if media_path_rel in identity_map:
                    faces = identity_map[media_path_rel]
                    resolved_faces = [resolved_identity_map.get(f, f) for f in faces]
                    faces_str = ", ".join(set(resolved_faces))
                    content_parts.append(f"[People in Video: {faces_str}]")

                if media_path_rel in video_descriptions:
                    video_desc = video_descriptions[media_path_rel]
                elif not vlm_cb.is_tripped() and media_path.exists():
                    try:
                        video_desc = await video_analyze(media_path)
                        vlm_cb.record_success()
                    except RuntimeError:
                        vlm_cb.record_failure(item, "VLM unavailable for video")
                        video_desc = "[Video analysis unavailable - VLM circuit breaker active]"
                    except Exception as e:
                        vlm_cb.record_failure(item, str(e))
                        video_desc = "[Video analysis unavailable]"
                else:
                    video_desc = "[Video analysis unavailable]"

                content_parts.append(f"[Video Summary: {video_desc}]")

        item_combined = " ".join(content_parts)
        entities: dict[str, Any] = {}
        if item_combined:
            entities = await extract_entities(item_combined, model=INGESTION_MODEL)
            if entities.get("people"):
                content_parts.append(f"[Entities - People: {', '.join(entities['people'])}]")

            has_structure = bool(entities.get("people")) or bool(entities.get("locations"))
            if has_structure:
                entity_extractions.append(entities)

        # Track structured items
        is_media_with_data = (
            item_type in ("voice", "video", "photo")
            or item.get("forwarded")
            or bool(entities.get("people"))
            if item_combined
            else False
        )
        if is_media_with_data:
            s_item: dict[str, Any] = {
                "type": item_type,
                "path": media_path_rel,
                "timestamp": timestamp,
                "sender": sender,
            }
            if item_type == "voice" and media_path_rel:
                abs_media = str(input_dir / media_path_rel)
                segments = audio_results.get(abs_media, [])
                if segments:
                    s_item["speakers"] = list(
                        {seg.get("resolved_name", seg["speaker"]) for seg in segments}
                    )
                    s_item["transcript"] = " ".join(seg.get("text", "") for seg in segments)
            if item_type in ("video", "photo") and media_path_rel in identity_map:
                s_item["faces"] = identity_map[media_path_rel]
            if item_type == "video" and media_path_rel in video_descriptions:
                s_item["description"] = video_descriptions[media_path_rel]
            structured_media_items.append(s_item)

        final_combined = " ".join(content_parts)
        if final_combined:
            processed_messages.append(f"**{sender}** [{timestamp}]: {final_combined}")

    vlm_cb.flush_remaining()

    # Write outputs
    if processed_messages:
        chat_history_file = output_dir / "chat_history_processed.md"
        with chat_history_file.open("w", encoding="utf-8") as f:
            header = "# Processed Content Archive\n\n"
            f.write(header)
            f.write("\n\n".join(processed_messages))
        ingested_files.append(chat_history_file)

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
            "document_chunks": document_chunks,
            "ingestion_structured": {
                "resolved_identity_map": resolved_identity_map,
                "entity_extractions": entity_extractions,
                "media_items": structured_media_items,
                "video_descriptions": video_descriptions,
                "pre_named_entities": sorted(pre_named_entities),
            },
        },
        "__goto__": "drafter",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _combine_ingested_files(files: list[Path], output_path: Path) -> None:
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
