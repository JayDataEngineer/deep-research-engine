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
import re
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

try:
    import pymupdf
except ImportError:
    pymupdf = None

from ..agents.subagents import run_context_enrichment, run_identity_resolver
from ..config import get_phase_config
from ..ingestion.document_router import DocumentRouter
from ..ingestion.kreuzberg_extractor import KreuzbergConfig
from ..shared.config import DEFAULT_MODEL, INGESTION_MODEL
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


_VLM_GARBAGE_PHRASES = (
    "i am sorry",
    "i am a large language model",
    "i cannot describe",
    "i cannot view",
    "i cannot process",
    "please provide the image",
    "you have not provided an image",
    "unable to view",
    "unable to process",
    "safety guidelines",
    "violates my",
    "cannot be described",
    "as a large language model",
    "as an ai",
    "i'm unable to",
    "i am unable to",
    "i don't see any image",
)


def _is_valid_vlm_description(text: str | None, item_type: str = "") -> bool:
    """Sanity-check a VLM description. Returns False for garbage/refusals/roleplay."""
    if not text or len(text.strip()) < 20:
        return False
    # Reject error strings from failed VLM/ASR calls
    if text.strip().startswith("[Image analysis failed") or text.strip().startswith("[Video analysis failed"):
        return False
    if text.strip().startswith("[No frames") or text.strip().startswith("[VLM unavailable"):
        return False
    lower = text.lower().strip()
    if any(phrase in lower for phrase in _VLM_GARBAGE_PHRASES):
        return False
    return not (
        item_type == "video"
        and ("## narrative" in lower or "**setting:**" in lower or "**overall tone:**" in lower)
    )


def _load_checkpoint(pipeline_dir: Path, name: str) -> Any | None:
    """Load a phase checkpoint artifact. Returns None if not found or corrupted."""
    path = pipeline_dir / f"{name}.json"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Checkpoint loaded (skipping phase): {path}")
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Corrupted checkpoint {path}: {e} — deleting and re-running phase")
            path.unlink(missing_ok=True)
            return None
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
    tg_dirs = {"photos", "voice_messages", "video_files", "stickers", "files", "images"}
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
        self._total_fail_count = 0
        self._batch_item_count = 0

    def record_failure(self, item: dict[str, Any], error: str) -> None:
        """Record a VLM failure."""
        self._batch_fail_count += 1
        self._total_fail_count += 1
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
    global_speaker_map: dict[str, str] = {}

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
                    ents = await extract_entities(text[:4000], model=DEFAULT_MODEL)
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

            # Rebuild file_asr_results for Phase 5 speaker unification
            for path_str, segments in audio_results.items():
                if isinstance(segments, list) and segments:
                    speakers_set = {
                        seg.get("speaker", "") for seg in segments if seg.get("speaker")
                    }
                    asr_dict = {
                        "segments": segments,
                        "speakers": len(speakers_set),
                        "speaker_list": sorted(speakers_set),
                    }
                    file_asr_results.append((Path(path_str), asr_dict))

        # Phase 5: Cross-File Speaker Unification
        if checkpoint_p5 is None and len(file_asr_results) > 1:
            logger.info("Phase 5: Reconciling speaker identities across files...")
            # Unload all models to free VRAM for ECAPA-TDNN speaker embeddings
            infra = _get_infra()
            infra.unload_all()
            logger.info("All models unloaded — VRAM free for speaker embedding extraction")
            try:
                global_speaker_map = await unify_speakers(file_asr_results, pipeline_dir)

                for segments in audio_results.values():
                    for seg in segments:
                        mf_id = seg.get("mf_speaker_id")
                        if mf_id in global_speaker_map:
                            seg["speaker"] = global_speaker_map[mf_id]

                _save_checkpoint(pipeline_dir, "phase5_speaker_map", global_speaker_map)
            except Exception as e:
                logger.warning(f"Phase 5 speaker unification failed (non-fatal): {e}")
                logger.info("Continuing without cross-file speaker unification")
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
        # VLM goes direct to llama.cpp (Gemma 4 26B) — no orchestrator load needed.
        # Identity resolution uses InfraClient.chat() via LiteLLM (main model).
        logger.info("Phase 6: Performing final identity resolution and batch VLM analysis...")

        # Prepare metadata for LLM resolution
        asr_all_segments: list[dict[str, Any]] = []
        for _, asr_output in file_asr_results:
            asr_all_segments.extend(asr_output.get("segments", []))

        # Load Phase 7 VLM checkpoint (if available from a previous run)
        # to provide visual descriptions for face clusters to the identity resolver.
        cluster_descriptions: dict[str, list[str]] = {}
        checkpoint_p7 = _load_checkpoint(pipeline_dir, "phase7_vlm_enriched")
        if checkpoint_p7 and isinstance(checkpoint_p7, dict):
            p7_items = checkpoint_p7.get("items", [])
            for item in p7_items:
                desc = item.get("description", "")
                faces = item.get("faces", [])
                path = item.get("path", "")
                if desc and faces:
                    for face_id in faces:
                        cluster_descriptions.setdefault(face_id, []).append(
                            f"[{path}]: {desc[:300]}"
                        )
            if cluster_descriptions:
                logger.info(
                    f"Phase 6: Loaded {len(cluster_descriptions)} cluster descriptions "
                    f"from Phase 7 VLM checkpoint"
                )

        # If no Phase 7 checkpoint available, run inline VLM on a sample of
        # photos per cluster to provide visual context for identity resolution.
        if not cluster_descriptions and identity_map:
            # Build cluster_id -> list of photo paths mapping
            cluster_photos: dict[str, list[str]] = {}
            for photo_path, clusters in identity_map.items():
                for cid in clusters:
                    cluster_photos.setdefault(cid, []).append(photo_path)

            if cluster_photos:
                logger.info(
                    f"Phase 6: No Phase 7 checkpoint. Running inline VLM sampling "
                    f"for {len(cluster_photos)} face clusters..."
                )
                for cid, photo_paths in sorted(cluster_photos.items()):
                    # Sample up to 2 photos per cluster
                    sample_paths = photo_paths[:2]
                    for rel_path in sample_paths:
                        media_path = Path(rel_path)
                        if not media_path.is_absolute():
                            media_path = input_dir / rel_path
                        if not media_path.exists():
                            continue
                        try:
                            desc = await vlm_analyze(media_path)
                            if desc and not desc.startswith("[Image analysis failed"):
                                cluster_descriptions.setdefault(cid, []).append(
                                    f"[{rel_path}]: {desc[:300]}"
                                )
                        except Exception as e:
                            logger.debug(f"Inline VLM failed for {rel_path}: {e}")
                if cluster_descriptions:
                    logger.info(
                        f"Phase 6: Inline VLM produced {len(cluster_descriptions)} "
                        f"cluster descriptions"
                    )

        # Build cluster description summaries for the identity resolver
        cluster_desc_lines: list[str] = []
        for cid in sorted(cluster_descriptions):
            descs = cluster_descriptions[cid][:5]
            cluster_desc_lines.append(
                f"### {cid}\n" + "\n".join(f"- {d}" for d in descs)
            )
        cluster_description_text = "\n\n".join(cluster_desc_lines)

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

        # Phase 6a: Identity Resolver subagent (with graph_query tool)
        try:
            logger.info("Phase 6: Using Identity Resolver subagent (DeepAgents)...")
            resolved_identity_map = await run_identity_resolver(
                asr_segments=asr_all_segments,
                visual_events=visual_events,
                text_messages=text_context,
                pre_named_entities=sorted(pre_named_entities),
                model=DEFAULT_MODEL,
                config_data=config_data,
                cluster_descriptions=cluster_description_text,
            )
        except Exception as e:
            logger.warning(
                f"Identity Resolver subagent failed ({e}), falling back to direct resolution"
            )
            resolved_identity_map = await resolve_identities(
                asr_all_segments,
                visual_events,
                text_context,
                neo4j_storage=await _try_init_neo4j(config_data),
                namespace=config_data.get("storage", {})
                .get("neo4j", {})
                .get("namespace", "default"),
                pre_named_entities=sorted(pre_named_entities),
                model=DEFAULT_MODEL,
                cluster_descriptions=cluster_description_text,
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
        # VLM goes direct to llama.cpp — no orchestrator load needed
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
    entity_extractions: list[dict[str, Any]] = []
    structured_media_items: list[dict[str, Any]] = []

    # VLM circuit breaker for per-item VLM calls
    vlm_cb = VLMBatchCircuitBreaker(pipeline_dir)

    # Phase 7a: Run per-item VLM enrichment (I/O-bound, must stay per-item)
    # Checkpoint every 50 items so we can resume mid-VLM on restart.
    checkpoint_p7a = _load_checkpoint(pipeline_dir, "phase7_vlm_enriched")
    if checkpoint_p7a is not None:
        vlm_enriched_items = checkpoint_p7a.get("items", [])
        vlm_cb._batch_fail_count = checkpoint_p7a.get("vlm_fail_count", 0)
        vlm_cb._total_fail_count = checkpoint_p7a.get("vlm_fail_count", 0)
        logger.info(
            f"Phase 7a checkpoint loaded: {len(vlm_enriched_items)}/{len(items)} items enriched, resuming"
        )
    else:
        vlm_enriched_items = []

    checkpoint_interval = 50  # save every 50 items
    start_idx = len(vlm_enriched_items)

    # Build a robust lookup for voice items → audio_results.
    # audio_results keys are absolute paths from Phase 4; item paths may be relative.
    # We build a filename → (abs_path, segments) map for fuzzy matching.
    _audio_by_filename: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for _apath, _segs in audio_results.items():
        _fname = Path(_apath).name
        _audio_by_filename[_fname] = (_apath, _segs)
        # Also index by stem (without extension) for cross-format matches (.ogg vs .wav)
        _stem = Path(_apath).stem
        if _stem not in _audio_by_filename:
            _audio_by_filename[_stem] = (_apath, _segs)

    # Build the full speaker name resolver: file{idx}:SPEAKER_N → real name
    # Combines phase5_speaker_map + phase6_identity_map into a single lookup.
    _mf_id_to_name: dict[str, str] = {
        _mf_id: resolved_identity_map.get(_global_id, _global_id)
        for _mf_id, _global_id in global_speaker_map.items()
    }
    # Also add global IDs directly
    _mf_id_to_name.update(resolved_identity_map)

    def _resolve_voice_segments(
        item_path_rel: str, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Resolve speaker names in segments using the full mapping chain."""
        # Find the file index for this item
        _abs_path = (
            str(input_dir / item_path_rel)
            if not Path(item_path_rel).is_absolute()
            else item_path_rel
        )
        _resolved_segments: list[dict[str, Any]] = []
        for seg in segments:
            name = seg.get("resolved_name") or seg.get("speaker", "Unknown")
            # Try resolved_name first (already set by Phase 5/6), then mf_speaker_id
            if not seg.get("resolved_name"):
                _mf_id = seg.get("mf_speaker_id", "")
                if _mf_id in _mf_id_to_name:
                    name = _mf_id_to_name[_mf_id]
                elif seg.get("speaker", "") in _mf_id_to_name:
                    name = _mf_id_to_name[seg["speaker"]]
            # Sanitize any remaining raw speaker IDs
            if name and re.match(
                r"^(SPEAKER_\d+|Global_Speaker_\d+|Speaker_\d+)$", name
            ):
                name = "Unknown"
            _resolved_segments.append({**seg, "resolved_name": name})
        return _resolved_segments

    for idx, item in enumerate(items):
        if idx < start_idx:
            continue  # already processed

        msg_text = item.get("text") or ""
        media_path_rel = item.get("path")
        item_type = item["type"]
        sender = item.get("sender", "Unknown")
        timestamp = item.get("timestamp", "Unknown")

        content_parts: list[str] = []
        enriched: dict[str, Any] = {}  # Structured fields for detective report / file organizer

        if msg_text:
            content_parts.append(msg_text)

        if media_path_rel:
            media_path = Path(media_path_rel)
            if not media_path.is_absolute():
                media_path = input_dir / media_path_rel

            if item_type == "voice":
                # Robust lookup: try exact path, then filename, then stem
                segments: list[dict[str, Any]] | None = None
                abs_key = str(media_path)
                if abs_key in audio_results:
                    segments = audio_results[abs_key]
                else:
                    fname = media_path.name
                    if fname in _audio_by_filename:
                        segments = _audio_by_filename[fname][1]
                    else:
                        stem = media_path.stem
                        if stem in _audio_by_filename:
                            segments = _audio_by_filename[stem][1]

                if segments:
                    # Resolve speaker names via full chain
                    segments = _resolve_voice_segments(media_path_rel, segments)

                    transcript_parts = []
                    for seg in segments:
                        name = seg.get("resolved_name", seg["speaker"])
                        transcript_parts.append(f"{name}: {seg['text']}")

                    transcript = " ".join(transcript_parts)
                    content_parts.append(f"[Voice Message Transcript: {transcript}]")

                    unique_speakers = {seg.get("resolved_name", seg["speaker"]) for seg in segments}
                    if len(unique_speakers) > 1:
                        content_parts.append(f"[Speakers in snippet: {', '.join(unique_speakers)}]")

                    # Store structured fields
                    enriched["speakers"] = list(unique_speakers)
                    enriched["transcript"] = " ".join(seg.get("text", "") for seg in segments)

            elif item_type == "photo":
                if media_path_rel in identity_map:
                    faces = identity_map[media_path_rel]
                    resolved_faces = [resolved_identity_map.get(f, f) for f in faces]
                    faces_str = ", ".join(set(resolved_faces))
                    content_parts.append(f"[People in Photo: {faces_str}]")
                    enriched["faces"] = list(set(resolved_faces))

                photo_desc = None
                if not vlm_cb.is_tripped() and media_path.exists():
                    try:
                        photo_desc = await vlm_analyze(media_path)
                        if photo_desc and photo_desc.strip().startswith("[Image analysis failed"):
                            vlm_cb.record_failure(item, photo_desc[:200])
                            photo_desc = None
                        else:
                            vlm_cb.record_success()
                    except RuntimeError:
                        vlm_cb.record_failure(item, "VLM unavailable for photo")
                        logger.warning("VLM unavailable - skipping photo descriptions")
                    except Exception as e:
                        vlm_cb.record_failure(item, str(e))
                if _is_valid_vlm_description(photo_desc, "photo"):
                    content_parts.append(f"[Photo Description: {photo_desc}]")
                    enriched["description"] = photo_desc
                else:
                    photo_desc = None

            elif item_type == "video":
                if media_path_rel in identity_map:
                    faces = identity_map[media_path_rel]
                    resolved_faces = [resolved_identity_map.get(f, f) for f in faces]
                    faces_str = ", ".join(set(resolved_faces))
                    content_parts.append(f"[People in Video: {faces_str}]")
                    enriched["faces"] = list(set(resolved_faces))

                video_desc = None
                if media_path_rel in video_descriptions:
                    video_desc = video_descriptions[media_path_rel]
                elif not vlm_cb.is_tripped() and media_path.exists():
                    try:
                        video_desc = await video_analyze(media_path)
                        if video_desc and video_desc.strip().startswith("[Video analysis failed"):
                            vlm_cb.record_failure(item, video_desc[:200])
                            video_desc = None
                        else:
                            vlm_cb.record_success()
                    except RuntimeError:
                        vlm_cb.record_failure(item, "VLM unavailable for video")
                        video_desc = None
                    except Exception as e:
                        vlm_cb.record_failure(item, str(e))
                        video_desc = None

                if _is_valid_vlm_description(video_desc, "video"):
                    content_parts.append(f"[Video Summary: {video_desc}]")
                    enriched["description"] = video_desc
                elif video_desc:
                    logger.debug(f"Rejected invalid video description for {media_path_rel}")
                    video_desc = None

        vlm_enriched_items.append(
            {
                **item,
                **enriched,
                "_enriched_content": " ".join(content_parts),
            }
        )

        # Incremental checkpoint every N items
        if (idx + 1) % checkpoint_interval == 0:
            _save_checkpoint(
                pipeline_dir,
                "phase7_vlm_enriched",
                {
                    "items": vlm_enriched_items,
                    "vlm_fail_count": vlm_cb._total_fail_count,
                    "processed_count": len(vlm_enriched_items),
                    "total_count": len(items),
                },
            )
            logger.info(
                f"Phase 7a checkpoint: {len(vlm_enriched_items)}/{len(items)} items enriched"
            )

    vlm_cb.flush_remaining()

    # Save final Phase 7a checkpoint (complete)
    _save_checkpoint(
        pipeline_dir,
        "phase7_vlm_enriched",
        {
            "items": vlm_enriched_items,
            "vlm_fail_count": vlm_cb._total_fail_count,
            "processed_count": len(vlm_enriched_items),
            "total_count": len(items),
        },
    )
    logger.info(f"Phase 7a complete: {len(vlm_enriched_items)} items enriched")

    # Phase 7b: Batch entity extraction via Context Enrichment subagent
    try:
        logger.info("Phase 7: Running Context Enrichment subagent for batch NER...")
        enrichment_result = await run_context_enrichment(
            items=vlm_enriched_items,
            resolved_identity_map=resolved_identity_map,
            audio_results=audio_results,
            video_descriptions=video_descriptions,
            identity_map=identity_map,
            model=DEFAULT_MODEL,
            batch_size=20,
        )
        processed_messages = enrichment_result["processed_messages"]
        entity_extractions = enrichment_result["entity_extractions"]
        structured_media_items = enrichment_result["structured_media_items"]
        logger.info(
            f"Context Enrichment: {len(processed_messages)} messages, "
            f"{len(entity_extractions)} entity batches, "
            f"{len(structured_media_items)} structured items"
        )
    except Exception as e:
        import traceback

        logger.warning(f"Context Enrichment subagent failed ({e}), falling back to per-item NER")
        logger.warning(f"Traceback: {traceback.format_exc()}")
        # Fallback: per-item NER (original Phase 7 behavior)
        for item in vlm_enriched_items:
            item_combined = item.get("_enriched_content", "")
            sender = item.get("sender", "Unknown")
            timestamp = item.get("timestamp", "Unknown")
            media_path_rel = item.get("path")
            item_type = item.get("type", "text")

            entities: dict[str, Any] = {}
            content_parts = [item_combined] if item_combined else []
            if item_combined:
                entities = await extract_entities(item_combined, model=DEFAULT_MODEL)
                if entities.get("people"):
                    content_parts.append(f"[Entities - People: {', '.join(entities['people'])}]")
                has_structure = bool(entities.get("people")) or bool(entities.get("locations"))
                if has_structure:
                    entity_extractions.append(entities)

            if item_type in ("voice", "video", "photo", "message") or item_combined:
                s_item: dict[str, Any] = {
                    "type": item_type,
                    "path": media_path_rel,
                    "timestamp": timestamp,
                    "sender": sender,
                    "text": item.get("text"),
                }
                # Copy structured fields from Phase 7a enrichment
                for field in ("speakers", "transcript", "faces", "description"):
                    if field in item:
                        s_item[field] = item[field]
                # Fallback: re-derive from audio_results if not in enriched fields
                if item_type == "voice" and "transcript" not in s_item and media_path_rel:
                    abs_media = str(input_dir / media_path_rel)
                    segments = audio_results.get(abs_media, [])
                    if segments:
                        s_item["speakers"] = list(
                            {seg.get("resolved_name", seg["speaker"]) for seg in segments}
                        )
                        s_item["transcript"] = " ".join(seg.get("text", "") for seg in segments)
                structured_media_items.append(s_item)

            final_combined = " ".join(content_parts)
            if final_combined:
                processed_messages.append(f"**{sender}** [{timestamp}]: {final_combined}")

    vlm_cb.flush_remaining()

    # Write outputs — single file (chat_history_processed.md) is the source of truth
    big_dump_file = output_dir / "chat_history_processed.md"
    if processed_messages:
        with big_dump_file.open("w", encoding="utf-8") as f:
            f.write("# Processed Content Archive\n\n")
            f.write("\n\n".join(processed_messages))
    else:
        # No processed messages — combine Kreuzberg docs as fallback
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
    from ..tools.storage import get_neo4j_from_config

    return await get_neo4j_from_config(config_data)
