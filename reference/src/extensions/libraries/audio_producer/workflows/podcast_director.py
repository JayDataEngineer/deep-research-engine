"""Podcast Director Workflow - Orchestrates podcast audio production."""

import asyncio
import logging
import os
import re
import signal
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Union
from collections import defaultdict

import httpx

from ..config_schemas import AudioJobConfig
from ..core import DialogueLine, EmotionDetector, AudioMerger, EmotionVector
# Import from centralized content_formatter emotion module
from src.extensions.content.content_formatter.emotion import (
    DialogueLine as CFDialogueLine,
    EmotionDetector as CFEmotionDetector,
    EmotionVector as CFEmotionVector,
    detect_emotions,
)
from src.extensions.libraries.tts_engine.extension_process_manager import ExtensionProcessManager
from src.constants import BACKEND_ROOT, PROJECT_ROOT, TTS_EXTENSIONS_DIR, MAIN_API_URL, DATA_OUTPUTS_DIR
from src.shared.utils.venv_manager import get_venv_python

# Import podcast configuration system
from .podcast_config import (
    PodcastConfig,
    ChunkingConfig,
    ContextConfig,
    PostProcessorConfig,
    PRESET_BALANCED,
    get_preset,
    list_presets,
)
from .emotion_post_processor import EmotionPostProcessor

logger = logging.getLogger(__name__)


class PodcastDirector:
    """Orchestrates podcast audio production workflow."""

    # Flexible pattern that accepts ANY uppercase speaker name
    # Speakers should be uppercase words (EMMA, OLIVIA, NARRATOR, ALICE, BOB, etc.)
    DIALOGUE_PATTERN = re.compile(
        # Match: SPEAKER (direction): dialogue text (any uppercase word)
        r'^(?P<speaker>[A-Z][A-Z0-9_]*)(\s*(?P<direction>\([^)]*\)))?:\s*(?P<text>.+)$|'
        # Also support: **SPEAKER:** dialogue format (for compatibility)
        r'^\*\*(?P<speaker2>[A-Z][A-Z0-9_]*)(\s*(?P<direction2>\([^)]*\)))?:\*\*\s*(?P<text2>.+)$|'
        # Match scene/section descriptions as NARRATOR: **[SCENE: ...]** or **[...]**
        r'^\*\*\[(?P<narrator_type>SCENE:)?(?P<narrator_text>.+?)\]\*\*$|'
        # Match section headers as NARRATOR: ### Story 1: ... or ## Episode: ...
        r'^#{1,6}\s+(?P<section_text>.+)$'
    )

    def __init__(
        self,
        config: AudioJobConfig,
        prompts_dir: str,
        podcast_config: Union[PodcastConfig, str, None] = None
    ):
        """Initialize PodcastDirector with optional podcast pipeline configuration.

        Args:
            config: AudioJobConfig for TTS settings
            prompts_dir: Directory containing prompt templates
            podcast_config: PodcastConfig instance, preset name (e.g., "balanced", "natural"),
                           or None to use PRESET_BALANCED
        """
        self.config = config
        self.prompts_dir = prompts_dir
        # Use centralized emotion detector from content_formatter
        self._cf_emotion_detector: Optional[CFEmotionDetector] = None
        self.merger = AudioMerger()
        self._process_manager: Optional[ExtensionProcessManager] = None

        # Handle podcast configuration
        if podcast_config is None:
            self.podcast_config = PRESET_BALANCED
        elif isinstance(podcast_config, str):
            # It's a preset name
            self.podcast_config = get_preset(podcast_config)
        else:
            # It's a PodcastConfig instance
            self.podcast_config = podcast_config

        # Validate configuration
        self.podcast_config.validate()

        # Initialize post-processor
        self.post_processor = EmotionPostProcessor(self.podcast_config.post_processing)

        logger.info(f"PodcastDirector initialized with config: {self.podcast_config.description}")
        logger.info(f"  Chunking: mode={self.podcast_config.chunking.mode}, window_size={self.podcast_config.chunking.window_size}")
        logger.info(f"  Context: mode={self.podcast_config.context.mode}, window_size={self.podcast_config.context.window_size}")
        logger.info(f"  Post-Processing: {len(self.podcast_config.post_processing)} processors")

    async def _get_emotion_detector(self) -> CFEmotionDetector:
        """Get the centralized emotion detector (lazy initialization)."""
        if self._cf_emotion_detector is None:
            # Get from content_formatter extension if available
            try:
                from src.extensions.core.manager.loader import get_extension_manager
                manager = get_extension_manager()
                if manager:
                    formatter_ext = manager.get_extension("content_formatter")
                    if formatter_ext and formatter_ext.instance:
                        self._cf_emotion_detector = formatter_ext.instance.get_emotion_detector()
            except Exception as e:
                logger.warning(f"Could not get emotion detector from content_formatter: {e}")

            # Fallback: create directly
            if self._cf_emotion_detector is None:
                self._cf_emotion_detector = CFEmotionDetector()

        return self._cf_emotion_detector

    async def _execute_parallel_segments(self, input_text: str, output_path: Optional[Path] = None) -> dict:
        """Process multiple segments when SEGMENT_BREAK markers are present using QUEUE-BASED PIPELINE.

        TRUE PIPELINE ARCHITECTURE:
        - LLM Producer: Processes segments in parallel, pushes to TTS queue AS SOON AS each completes
        - TTS Consumer: Pulls from queue immediately, processes as workers become available
        - NO WAITING for all segments to complete LLM before starting TTS

        Each segment flows independently: LLM → Queue → TTS → Done
        """
        import time
        import logging

        start_time = time.time()

        # Create timestamped run directory
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DATA_OUTPUTS_DIR / "audio" / "podcast" / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create audio chunks directory
        audio_chunks_dir = output_dir / "audio_chunks"
        audio_chunks_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging to output directory
        # Note: File logging uses central config format
        log_file = output_dir / "podcast_generation.log"
        logger.info(f"Logging to file: {log_file}")
        logger.info(f"Output directory: {output_dir}")

        # Determine final output path
        final_path = output_path or (output_dir / f"podcast_{timestamp}.wav")

        # Split by SEGMENT_BREAK markers
        segments = input_text.split("<<<SEGMENT_BREAK>>>")
        segments = [s.strip() for s in segments if s.strip()]
        num_segments = len(segments)

        # Get LLM parallel concurrency from config
        llm_concurrency = self.config.emotion.parallel_concurrency

        logger.info(f"=== MULTI-SEGMENT PIPELINE MODE ===")
        logger.info(f"  Split input into {num_segments} independent segments")
        logger.info(f"  LLM parallel concurrency: {llm_concurrency} workers")
        logger.info(f"  PIPELINE: LLM → Queue → TTS (no waiting for all segments)")

        # Parse all segments into dialogue lines first
        all_segment_lines = []
        for segment_text in segments:
            dialogue_lines = self.parse_dialogue(segment_text)
            all_segment_lines.append(dialogue_lines)

        # Track statistics for output.md
        stats = {
            "total_chunks": 0,
            "speaker_counts": {},
            "total_audio_duration": 0.0,
            "failed_chunks": 0,
            "failures": [],
        }
        llm_failures = []

        # ============ TWO-STAGE PIPELINE WITH QUEUES ============
        # Stage 1: LLM Queue (segments waiting for LLM processing)
        llm_queue = asyncio.Queue()
        # Stage 2: TTS Queue (segments that completed LLM, waiting for TTS)
        tts_queue = asyncio.Queue()

        # Worker counts
        num_llm_workers = min(llm_concurrency, num_segments)  # LLM workers
        num_tts_workers = min(2, num_segments)  # TTS workers

        logger.info(f"=== TWO-STAGE PIPELINE ARCHITECTURE ===")
        logger.info(f"  Stage 1: LLM Queue → {num_llm_workers} Workers (Semantic Chunking + Emotion Tagging)")
        logger.info(f"  Stage 2: TTS Queue → {num_tts_workers} Workers")
        logger.info(f"  BOTH STAGES RUN IN PARALLEL via queues")

        # ============ STAGE 0: Populate LLM Queue ============
        for seg_idx in range(num_segments):
            await llm_queue.put((seg_idx, all_segment_lines[seg_idx]))
        logger.info(f"[LLM Queue] Populated with {num_segments} segments")

        # ============ STAGE 1: Start IndexTTS vLLM (single instance, 16x concurrency) ============
        # First, ensure ANY orphaned vLLM processes are killed (including child processes)
        import subprocess

        def kill_vllm_processes():
            """Kill all vLLM-related processes including child processes."""
            # Method 1: Use pkill to kill all processes matching index_tts_vllm
            try:
                subprocess.run(["pkill", "-9", "-f", "index_tts_vllm"],
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except FileNotFoundError:
                pass

            # Method 2: Kill by port
            try:
                subprocess.run(["fuser", "-k", "-9", "9000/tcp"],
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except FileNotFoundError:
                pass

            # Method 3: Use lsof to find PIDs and kill individually
            try:
                result = subprocess.run(["lsof", "-ti", "9000"],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                if result.stdout.strip():
                    for pid in result.stdout.strip().split('\n'):
                        try:
                            subprocess.run(["kill", "-9", pid], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                        except:
                            pass
            except FileNotFoundError:
                pass

        kill_vllm_processes()
        await asyncio.sleep(2.0)  # Give time for all processes to terminate

        # Use shared ExtensionProcessManager for TTS (DRY)
        multi_manager = ExtensionProcessManager()

        logger.info("Initializing IndexTTS vLLM via ExtensionProcessManager...")
        start_result = await multi_manager.start_extension("index_tts_vllm")
        if start_result.get("status") != "ready":
            raise RuntimeError(f"Failed to start IndexTTS vLLM: {start_result.get('message', 'Unknown error')}")

        # ============ STAGE 2: LLM Workers (pull from LLM Queue, push to TTS Queue) ============
        llm_start = time.time()
        segments_completed_llm = 0
        llm_results = []

        async def llm_worker(worker_id: int):
            """LLM Worker: Pull segment from LLM Queue, process (chunking + emotion tagging), push to TTS Queue."""
            nonlocal segments_completed_llm

            logger.info(f"[LLM Worker {worker_id}] Starting (waiting for LLM queue)...")

            # Get emotion detector
            emotion_detector = await self._get_emotion_detector()

            while True:
                try:
                    # Pull from LLM Queue (blocks until item available)
                    segment_idx, dialogue_lines = await asyncio.wait_for(
                        llm_queue.get(),
                        timeout=1.0
                    )

                    logger.info(f"[LLM Worker {worker_id}] Pulled Segment {segment_idx + 1} from LLM queue ({len(dialogue_lines)} lines)")

                    segment_start = time.time()

                    try:
                        # Process: Emotion Tagging using centralized content_formatter detector
                        detected_lines, llm_info = await emotion_detector.detect(dialogue_lines)
                        segment_time = time.time() - segment_start

                        logger.info(f"[LLM Worker {worker_id}] Segment {segment_idx + 1} COMPLETE → pushing to TTS queue ({segment_time:.2f}s)")

                        # Push to TTS Queue
                        await tts_queue.put((segment_idx, detected_lines, llm_info, None))
                        segments_completed_llm += 1
                        llm_results.append((segment_idx, detected_lines, llm_info, None))

                    except Exception as e:
                        logger.error(f"[LLM Worker {worker_id}] Segment {segment_idx + 1} FAILED: {e}")
                        logger.error(f"[LLM Worker {worker_id}] Response preview: {str(e)[:200]}")
                        llm_failures.append({"segment": segment_idx + 1, "error": str(e)})

                        # CRITICAL: NO SILENT FALLBACKS - Raise immediately to stop processing
                        raise RuntimeError(f"Segment {segment_idx + 1} emotion detection failed: {e}") from e

                except asyncio.TimeoutError:
                    # LLM Queue empty - we're done
                    logger.info(f"[LLM Worker {worker_id}] LLM queue empty, exiting")
                    break

        # Launch LLM workers
        llm_worker_tasks = []
        for worker_id in range(num_llm_workers):
            task = asyncio.create_task(llm_worker(worker_id))
            llm_worker_tasks.append(task)

        logger.info(f"[LLM Workers] Launched {num_llm_workers} parallel LLM workers")

        # ============ STAGE 3: TTS Workers (multiple consumers pulling from queue) ============
        # Multiple TTS workers process segments concurrently from the queue
        segment_audio_files = []
        total_duration = 0
        total_lines = 0
        tts_start = time.time()
        segments_completed_tts = 0
        # vLLM has 16x concurrency internally, so 2-4 workers is sufficient
        num_tts_workers = min(num_segments, 4)  # Up to 4 workers (vLLM batches internally)

        async def tts_worker(worker_id: int):
            """One TTS worker that pulls segments from queue and processes them."""
            nonlocal segments_completed_tts, total_duration, total_lines

            logger.info(f"[TTS Worker {worker_id}] Starting (waiting for segments from queue)...")

            while True:
                # Pull from queue (blocks until item available or all done)
                try:
                    # Wait for item with timeout to check if we're done
                    segment_idx, detected_lines, llm_info, error = await asyncio.wait_for(
                        tts_queue.get(),
                        timeout=1.0
                    )

                    segments_completed_tts += 1
                    logger.info(f"[TTS Worker {worker_id}] Pulled Segment {segment_idx + 1} from queue ({len(detected_lines)} sentences)")

                    # Generate TTS for this segment
                    segment_start = time.time()
                    segment_audio_path = output_dir / f"segment_{segment_idx:02d}_temp.wav"

                    try:
                        await self._generate_tts_for_segment(
                            detected_lines,
                            multi_manager,
                            audio_chunks_dir / f"segment_{segment_idx:02d}",
                            segment_audio_path
                        )

                        segment_time = time.time() - segment_start
                        logger.info(f"[TTS Worker {worker_id}] Segment {segment_idx + 1} TTS complete in {segment_time:.2f}s")

                        # Collect result
                        if segment_audio_path.exists() and segment_audio_path.stat().st_size > 1000:
                            segment_audio_files.append((segment_idx, segment_audio_path))
                            duration = await self.merger.get_audio_duration(segment_audio_path)
                            total_duration += duration
                            total_lines += len(detected_lines)
                            stats["total_chunks"] += len(detected_lines)
                        else:
                            logger.warning(f"[TTS Worker {worker_id}] Segment {segment_idx + 1} audio file too small or empty")
                            stats["failed_chunks"] += len(detected_lines)

                    except Exception as e:
                        logger.error(f"[TTS Worker {worker_id}] Segment {segment_idx + 1} TTS failed: {e}")
                        stats["failed_chunks"] += len(detected_lines)
                        stats["failures"].append({"segment": segment_idx + 1, "error": str(e)})

                except asyncio.TimeoutError:
                    # Queue empty - check if all LLM producers are done
                    if segments_completed_llm >= num_segments:
                        logger.info(f"[TTS Worker {worker_id}] No more segments, exiting")
                        break
                    # Otherwise, continue waiting for more segments
                    continue

        # Launch multiple TTS workers
        tts_tasks = []
        for worker_id in range(num_tts_workers):
            task = asyncio.create_task(tts_worker(worker_id))
            tts_tasks.append(task)

        logger.info(f"[TTS Workers] Launched {num_tts_workers} parallel TTS workers")

        # ============ STAGE 4: Wait for LLM workers to complete ============
        # Process all LLM worker tasks (they push to TTS queue as they complete)
        await asyncio.gather(*llm_worker_tasks)

        llm_time = time.time() - llm_start
        logger.info(f"=== LLM WORKERS COMPLETE in {llm_time:.2f}s ===")
        logger.info(f"  All {num_segments} segments pushed to TTS queue")

        # ============ STAGE 5: Wait for all TTS workers to finish processing queue ============
        try:
            await asyncio.gather(*tts_tasks)

        finally:
            # Cleanup IndexTTS vLLM to free GPU memory (ALWAYS runs)
            try:
                await multi_manager.stop_current()
                logger.info("IndexTTS vLLM stopped, GPU memory freed")
            except Exception as e:
                logger.warning(f"Error stopping IndexTTS vLLM: {e}")

        tts_time = time.time() - tts_start
        logger.info(f"=== TTS WORKERS COMPLETE in {tts_time:.2f}s ({num_tts_workers} parallel workers) ===")

        # Sort segment audio files by index and extract paths
        segment_audio_files.sort(key=lambda x: x[0])  # Sort by segment index
        segment_audio_paths = [path for _, path in segment_audio_files]

        logger.info(f"=== MERGING {len(segment_audio_paths)} SEGMENT AUDIO FILES (in order) ===")

        await AudioMerger.merge_audio_files(
            audio_files=segment_audio_paths,
            output_path=final_path,
            add_silence=True,
            silence_duration=1.0,
            speaker_aware=False,
            normalize=True,
            cleanup_source=True
        )

        total_time = time.time() - start_time
        processing_time = total_time

        logger.info(f"=== PIPELINE PROCESSING COMPLETE ===")
        logger.info(f"  Processed {num_segments} segments")
        logger.info(f"  LLM time: {llm_time:.2f}s (parallel producers)")
        logger.info(f"  TTS time: {tts_time:.2f}s ({num_tts_workers} parallel workers)")
        logger.info(f"  Pipeline efficiency: LLM and TTS ran in parallel via queue")
        logger.info(f"  Total: {total_duration:.2f}s audio, {total_lines} lines")
        logger.info(f"  Output: {final_path}")
        logger.info(f"  Total time: {processing_time:.2f}s")

        # Generate output.md
        stage_times = {
            "llm_emotion_detection": llm_time,
            "tts_generation": tts_time,
            "merging": processing_time - llm_time - tts_time,
        }

        output_md = self._generate_output_md(
            input_text=input_text,
            final_path=final_path,
            duration=total_duration,
            num_lines=total_lines,
            processing_time=processing_time,
            stage_times=stage_times,
            tts_stats=stats,
            llm_info={"reasoning": "Parallel segment mode", "llm_failures": llm_failures},
            llm_results=llm_results,
        )

        # Write output.md next to audio file
        output_md_path = final_path.with_suffix('.md')
        output_md_path.write_text(output_md)
        logger.info(f"Generated output report: {output_md_path}")

        # Close file handler
        file_handler.close()
        root_logger.removeHandler(file_handler)

        # Build result to return after cleanup
        result = {
            "audio_path": str(final_path),
            "output_md_path": str(output_md_path),
            "duration": total_duration,
            "num_lines": total_lines,
            "num_segments": num_segments,
            "processing_time": processing_time,
            "llm_time": llm_time,
            "tts_time": tts_time,
            "real_time_factor": processing_time / total_duration if total_duration > 0 else 0,
        }

        # Return result after cleanup in finally block
        return result


    async def execute(self, input_text: str, output_path: Optional[Path] = None) -> dict:
        """Execute the full podcast production workflow with LLM→TTS pipelining."""
        import asyncio
        import time
        import datetime
        import json
        from collections import defaultdict
        start_time = time.time()

        # Display configuration
        self._log_config()
        logger.info("Starting Podcast Director workflow (PIPELINED mode)")

        # CHECK FOR PARALLEL SEGMENT PROCESSING
        # If input contains <<<SEGMENT_BREAK>>>, process segments in parallel
        if "<<<SEGMENT_BREAK>>>" in input_text:
            logger.info("=== DETECTED SEGMENT_BREAK MARKERS - ENABLE PARALLEL SEGMENT PROCESSING ===")
            return await self._execute_parallel_segments(input_text, output_path)

        # Stage timing tracking
        stage_times = {}

        # Step 1: Parse dialogue
        stage_start = time.time()
        dialogue_lines = self.parse_dialogue(input_text)
        stage_times["parsing"] = time.time() - stage_start
        logger.info(f"Parsed {len(dialogue_lines)} dialogue lines in {stage_times['parsing']:.2f}s")

        # Create a queue for pipelining chunks from LLM to TTS
        chunk_queue = asyncio.Queue()
        all_split_lines = []
        llm_complete = asyncio.Event()

        # Step 2: Start IndexTTS vLLM (single instance with 16x concurrency)
        multi_manager = ExtensionProcessManager()

        logger.info("Initializing IndexTTS vLLM via ExtensionProcessManager...")
        start_result = await multi_manager.start_extension("index_tts_vllm")
        if start_result.get("status") != "ready":
            raise RuntimeError(f"Failed to start IndexTTS vLLM: {start_result.get('message', 'Unknown error')}")

        logger.info("IndexTTS vLLM ready, starting parallel pipelined processing...")

        generation_params = self.config.get_generation_params()

        # Create audio chunks directory for persistence
        output_dir = output_path.parent if output_path else (DATA_OUTPUTS_DIR / "audio" / "podcast")
        audio_chunks_dir = output_dir / "audio_chunks"
        audio_chunks_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving individual audio clips to: {audio_chunks_dir}")

        # Setup file logging to output directory
        log_file = output_dir / "podcast_generation.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        # Add file handler to the root logger so all modules log to file
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

        # Step 3: Multiple parallel TTS workers - process chunks as they arrive from LLM
        tts_stats = {
            "total_chunks": 0,
            "speaker_counts": defaultdict(int),
            "total_audio_duration": 0.0,
            "failed_chunks": 0,
            "failures": [],  # Track individual failures for output.md
        }
        llm_failures = []  # Track LLM parsing failures
        audio_chunks_dict = {}
        tts_complete = asyncio.Event()
        chunk_index_counter = 0  # Thread-safe counter via atomic operations in asyncio

        async def tts_worker(worker_id: int):
            """Worker that processes TTS chunks as they arrive from LLM.

            Each worker uses its own IndexTTS instance for parallel generation.
            """
            nonlocal chunk_index_counter
            try:
                while True:
                    # Get chunk from queue
                    chunk = await chunk_queue.get()

                    if chunk is None:  # Sentinel value - LLM is done
                        logger.info(f"TTS worker {worker_id}: LLM complete, finishing...")
                        break

                    # Process this chunk of dialogue lines
                    logger.info(f"TTS worker {worker_id}: Processing chunk with {len(chunk)} dialogue lines")

                    for line in chunk:
                        # NARRATOR lines are section headers that should be spoken
                        # (Scene descriptions are already filtered out during parsing)
                        # Retry logic with exponential backoff
                        max_retries = 5
                        base_delay = 1.0  # seconds
                        audio_bytes = None
                        duration = None
                        last_error = None

                        for attempt in range(max_retries):
                            try:
                                voice_id = self.config.voices.get_voice_id(line.speaker, "index2_ref_neutral")
                                reference_audio = self.config.voices.get_reference_audio(line.speaker)

                                # Validate and convert to IndexTTS2 format - NO FALLBACKS
                                from src.extensions.libraries.audio_producer.config_schemas import IndexTTS2EmotionVector
                                if line.emotion:
                                    validated = IndexTTS2EmotionVector.from_list(line.emotion.to_list())
                                    emo_vec = validated.to_indextts_format()
                                    logger.debug(f"Validated emotion for {line.speaker}: calm={validated.calm:.2f}, angry={validated.angry:.2f}")
                                else:
                                    raise RuntimeError(f"Line {line.speaker}: '{line.text[:50]}...' has NO emotion data. LLM failed to generate emotions. ABORTING.")

                                params = {"emo_vector": emo_vec, **generation_params}

                                if reference_audio:
                                    params["ref_audio_path"] = reference_audio

                                # Use shared ExtensionProcessManager (DRY)
                                audio_bytes, duration = await multi_manager.generate_audio(
                                    text=line.text,
                                    voice_id=voice_id,
                                    **params,
                                )

                                # Success! Break out of retry loop
                                break

                            except Exception as e:
                                last_error = e
                                is_timeout = "timeout" in str(e).lower() or "connection" in str(e).lower() or "timed out" in str(e).lower()

                                # Note: ExtensionProcessManager is more stable than the old adapter system
                                # and typically doesn't require restart_hung_instance
                                if attempt < max_retries - 1:
                                    # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                                    delay = base_delay * (2 ** attempt)
                                    logger.warning(f"TTS worker {worker_id}: Attempt {attempt + 1}/{max_retries} failed for '{line.text[:50]}...': {e}. Retrying in {delay:.1f}s...")
                                    await asyncio.sleep(delay)
                                else:
                                    # Final attempt failed - exit application
                                    logger.error(f"TTS worker {worker_id}: CRITICAL - Failed to generate audio after {max_retries} attempts for '{line.text[:50]}...': {e}")
                                    logger.error("Application cannot continue - unable to generate required audio clip.")
                                    logger.error("Please check IndexTTS status and restart.")
                                    raise RuntimeError(f"Failed to generate audio for '{line.text[:50]}...' after {max_retries} attempts: {e}")

                        # Atomically increment chunk index
                        current_chunk_index = chunk_index_counter
                        chunk_index_counter += 1

                        # Save to audio_chunks directory for persistence
                        chunk_path = audio_chunks_dir / f"chunk_{current_chunk_index:04d}_{line.speaker}.wav"
                        chunk_path.write_bytes(audio_bytes)
                        audio_chunks_dict[current_chunk_index] = chunk_path
                        tts_stats["total_chunks"] += 1
                        tts_stats["speaker_counts"][line.speaker] += 1
                        tts_stats["total_audio_duration"] += duration

                        logger.info(f"TTS worker {worker_id}: Generated chunk {current_chunk_index} ({duration:.2f}s)")

                logger.info(f"TTS worker {worker_id}: Complete!")

            except Exception as e:
                logger.error(f"TTS worker {worker_id} error: {e}")

        # Number of TTS workers - vLLM has 16x concurrency internally
        num_segments = len(dialogue_lines)
        num_tts_workers = min(num_segments, 4)  # Up to 4 workers (vLLM batches internally)

        # Start multiple TTS workers in background
        tts_tasks = [
            asyncio.create_task(tts_worker(i))
            for i in range(num_tts_workers)
        ]

        # Step 4: Detect emotions and stream chunks to TTS worker
        stage_start = time.time()
        logger.info("Detecting emotions and splitting sentences (streaming to TTS)...")

        # For parallel/single mode with pipelining, stream chunks to TTS as they complete
        output_dir = output_path.parent if output_path else (DATA_OUTPUTS_DIR / "audio" / "podcast")

        # Determine which mode to use based on podcast_config
        chunking_mode = self.podcast_config.chunking.mode

        # Collect all detected lines for report generation
        all_detected_lines_pipelined = []

        if chunking_mode == "rolling_window":
            # Rolling window batch mode - process batches of N lines with overlap
            all_detected_lines_pipelined = await self._detect_and_stream_rolling_batch(dialogue_lines, chunk_queue, output_dir, llm_failures)
        elif chunking_mode == "single":
            # Single line mode - process one line at a time with context
            all_detected_lines_pipelined = await self._detect_and_stream_single(dialogue_lines, chunk_queue, output_dir, llm_failures)
        elif chunking_mode == "whole":
            # Single LLM call for all lines (single_call mode)
            logger.info("Using whole/single_call mode (one LLM call for all lines)")
            emotion_detector = await self._get_emotion_detector()
            dialogue_lines, llm_info = await emotion_detector.detect(dialogue_lines)
            all_detected_lines_pipelined = dialogue_lines
            for line in dialogue_lines:
                await chunk_queue.put([line])
        else:
            raise ValueError(f"Unknown chunking mode: {chunking_mode}")

        # Signal LLM is done - send sentinel for each worker
        for _ in range(num_tts_workers):
            await chunk_queue.put(None)
        llm_complete.set()

        stage_times["emotion_detection"] = time.time() - stage_start
        logger.info(f"Emotion detection complete in {stage_times['emotion_detection']:.2f}s")

        # Wait for all TTS workers to finish
        await asyncio.gather(*tts_tasks)
        tts_complete.set()
        stage_times["tts_generation"] = time.time() - stage_start - (stage_times["emotion_detection"])
        logger.info(f"TTS generation complete in {stage_times['tts_generation']:.2f}s (overlapped with LLM)")

        # Stop all IndexTTS subprocesses
        logger.info("Stopping all IndexTTS subprocesses...")
        await multi_manager.stop_current()

        # Reconstruct audio_chunks in original order
        audio_chunks = [audio_chunks_dict[i] for i in sorted(audio_chunks_dict.keys())]

        # Step 5: Combine audio
        stage_start = time.time()
        if output_path is None:
            output_path = self._get_default_output_path()

        final_path = await self.merger.merge_audio_files(
            audio_chunks,
            output_path,
            add_silence=self.config.output.add_silence,
            silence_duration=self.config.output.silence_duration,
            speaker_aware=True,  # Enable speaker-aware gaps
            speaker_gap_duration=0.4,  # 400ms when speaker changes
            normal_gap_duration=0.15,  # 150ms between same speaker
            normalize=True,  # Enable audio normalization
            cleanup_source=False,  # Keep individual chunks for persistence
        )
        stage_times["merging"] = time.time() - stage_start

        # Calculate duration
        duration = await self.merger.get_audio_duration(final_path)
        processing_time = time.time() - start_time

        logger.info(f"Podcast production complete in {processing_time:.2f}s")

        # Format detected lines as llm_results for report generation
        llm_results_pipelined = [(0, all_detected_lines_pipelined, {}, None)] if all_detected_lines_pipelined else []

        # Generate output.md
        output_md = self._generate_output_md(
            input_text=input_text,
            final_path=final_path,
            duration=duration,
            num_lines=tts_stats["total_chunks"],
            processing_time=processing_time,
            stage_times=stage_times,
            tts_stats=tts_stats,
            llm_info={"reasoning": "Pipelined mode - LLM reasoning not captured", "llm_failures": llm_failures},
            llm_results=llm_results_pipelined,
        )

        # Write output.md next to audio file
        output_md_path = final_path.with_suffix('.md')
        output_md_path.write_text(output_md)
        logger.info(f"Generated output report: {output_md_path}")

        # Close file handler
        file_handler.close()
        root_logger.removeHandler(file_handler)

        return {
            "audio_path": str(final_path),
            "output_md_path": str(output_md_path),
            "duration": duration,
            "num_lines": tts_stats["total_chunks"],
            "processing_time": processing_time,
            "stage_times": stage_times,
        }

    def parse_dialogue(self, text: str) -> List[DialogueLine]:
        """Parse markdown text into dialogue lines."""
        dialogue_lines = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = self.DIALOGUE_PATTERN.match(line)
            if match:
                # Check for section headers as NARRATOR first: ### Story 1: ... or ## Episode: ...
                if match.group("section_text"):
                    section_text = match.group("section_text").strip()
                    # Section headers are narrated by NARRATOR
                    if section_text and section_text not in ['[...]', '(silence)', '(pause)', '(scene change)']:
                        dialogue_lines.append(DialogueLine("NARRATOR", section_text, ""))
                # Check for narrator/scene description: **[SCENE: ...]** or **[...]**
                elif match.group("narrator_type") or match.group("narrator_text"):
                    narrator_type = match.group("narrator_type")
                    narrator_text = match.group("narrator_text").strip() if match.group("narrator_text") else ""

                    # Skip scene descriptions entirely (metadata only, not spoken)
                    if narrator_type == "SCENE:" or narrator_text.startswith("[SCENE:"):
                        logger.info(f"Skipping scene description (metadata): '{narrator_text[:50]}...'")
                        continue

                    # Other bracketed text is narrated by NARRATOR
                    if narrator_text and narrator_text not in ['[...]', '(silence)', '(pause)', '(scene change)']:
                        dialogue_lines.append(DialogueLine("NARRATOR", narrator_text, ""))
                elif match.group("speaker"):
                    speaker = match.group("speaker").strip()
                    direction = match.group("direction") or ""
                    text = match.group("text").strip()
                    if text and text not in ['[...]', '(silence)', '(pause)', '(scene change)']:
                        dialogue_lines.append(DialogueLine(speaker, text, direction))
                else:
                    speaker = match.group("speaker2").strip()
                    direction = match.group("direction2") or ""
                    text = match.group("text2").strip()
                    if text and text not in ['[...]', '(silence)', '(pause)', '(scene change)']:
                        dialogue_lines.append(DialogueLine(speaker, text, direction))

        return dialogue_lines

    async def generate_tts(self, dialogue_lines: List[DialogueLine]) -> tuple[List[Path], dict]:
        """Generate TTS for all dialogue lines.

        Optimized to process lines grouped by speaker to leverage
        IndexTTS's speaker embedding cache (avoids reprocessing reference audio).

        Returns:
            Tuple of (audio_chunks, tts_stats)
        """
        from collections import defaultdict

        tts_stats = {
            "total_chunks": len(dialogue_lines),
            "speaker_counts": {},
            "total_audio_duration": 0.0,
            "failed_chunks": 0,
        }

        if self._process_manager is None:
            self._process_manager = ExtensionProcessManager()

        # Start IndexTTS extension
        logger.info("Starting IndexTTS extension subprocess...")
        start_result = await self._process_manager.start_extension("index_tts")

        if start_result.get("status") != "ready":
            raise RuntimeError(f"Failed to start IndexTTS: {start_result.get('message', 'Unknown error')}")

        logger.info("IndexTTS ready, generating audio...")

        generation_params = self.config.get_generation_params()

        # Group lines by speaker for optimized processing
        # This leverages IndexTTS's speaker embedding cache
        speaker_groups = defaultdict(list)
        for i, line in enumerate(dialogue_lines):
            speaker_groups[line.speaker].append((i, line))

        logger.info(f"Grouped {len(dialogue_lines)} lines by {len(speaker_groups)} speakers:")
        for speaker, lines in speaker_groups.items():
            logger.info(f"  {speaker}: {len(lines)} lines")
            tts_stats["speaker_counts"][speaker] = len(lines)

        # Process each speaker group
        audio_chunks_dict = {}
        for speaker, lines_with_indices in speaker_groups.items():
            logger.info(f"Processing {len(lines_with_indices)} lines for speaker: {speaker}")

            for idx, line in lines_with_indices:
                try:
                    voice_id = self.config.voices.get_voice_id(line.speaker, "index2_ref_neutral")
                    reference_audio = self.config.voices.get_reference_audio(line.speaker)

                    logger.info(f"Generating TTS {idx+1}/{len(dialogue_lines)}: {line.speaker}")

                    # Validate and convert to IndexTTS2 format - NO FALLBACKS
                    from src.extensions.libraries.audio_producer.config_schemas import IndexTTS2EmotionVector
                    if line.emotion:
                        validated = IndexTTS2EmotionVector.from_list(line.emotion.to_list())
                        emo_vec = validated.to_indextts_format()
                        logger.debug(f"Validated emotion for {line.speaker}: calm={validated.calm:.2f}, angry={validated.angry:.2f}")
                    else:
                        raise RuntimeError(f"Line {line.speaker}: '{line.text[:50]}...' has NO emotion data. LLM failed. ABORTING.")

                    params = {"emo_vector": emo_vec, **generation_params}

                    if reference_audio:
                        params["reference_audio"] = reference_audio

                    audio_bytes, duration = await self._process_manager.generate_audio(
                        text=line.text,
                        voice_id=voice_id,
                        **params,
                    )

                    chunk_path = Path(tempfile.gettempdir()) / f"podcast_chunk_{idx:04d}.wav"
                    chunk_path.write_bytes(audio_bytes)
                    audio_chunks_dict[idx] = chunk_path
                    tts_stats["total_audio_duration"] += duration
                    logger.info(f"Generated chunk {idx+1}: {chunk_path} ({duration:.2f}s)")

                except Exception as e:
                    logger.error(f"Failed to generate TTS for line {idx+1}: {e}")
                    tts_stats["failed_chunks"] += 1

        # Stop the subprocess
        logger.info("Stopping IndexTTS subprocess...")
        await self._process_manager.stop_current()

        # Reconstruct audio_chunks in original order
        audio_chunks = [audio_chunks_dict[i] for i in sorted(audio_chunks_dict.keys())]
        return audio_chunks, tts_stats

    def _get_default_output_path(self) -> Path:
        """Get default output path for podcast audio."""
        import datetime

        output_dir = DATA_OUTPUTS_DIR / "audio" / "podcast"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return output_dir / f"podcast_{timestamp}.wav"

    def _log_config(self):
        """Log the current configuration."""
        logger.info("=" * 60)
        logger.info("PODCAST DIRECTOR CONFIGURATION")
        logger.info("=" * 60)

        # Emotion detection config
        logger.info(f"Emotion Detection:")
        logger.info(f"  Enabled: {self.config.emotion.enabled}")
        logger.info(f"  Quality Mode: {self.config.emotion.quality_mode}")
        logger.info(f"  Processing Mode: {self.config.emotion.processing_mode}")

        # Voice config
        logger.info(f"Voices:")
        for speaker, voice_config in self.config.voices.voices.items():
            logger.info(f"  {speaker}: {voice_config.voice_id}")
            if voice_config.reference_audio:
                logger.info(f"    Reference: {voice_config.reference_audio}")

        # Generation config
        gen_params = self.config.get_generation_params()
        logger.info(f"Generation Parameters:")
        for key, value in gen_params.items():
            if key != "emo_vector":
                logger.info(f"  {key}: {value}")

        # Output config
        logger.info(f"Output:")
        logger.info(f"  Format: {self.config.output.format}")
        logger.info(f"  Sample Rate: {self.config.output.sample_rate}")
        logger.info(f"  Add Silence: {self.config.output.add_silence}")
        if self.config.output.add_silence:
            logger.info(f"  Silence Duration: {self.config.output.silence_duration}s")

        logger.info("=" * 60)

    def _generate_output_md(self, input_text: str, final_path: Path, duration: float,
                           num_lines: int, processing_time: float,
                           stage_times: dict, tts_stats: dict, llm_info: dict,
                           llm_results: list) -> str:
        """Generate output markdown report."""
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# Podcast Generation Report

**Generated:** {timestamp}
**Audio Output:** `{final_path}`
**Duration:** {duration:.2f} seconds

---

## Configuration

### Emotion Detection
- **Enabled:** {self.config.emotion.enabled}
- **Quality Mode:** {self.config.emotion.quality_mode}
- **Processing Mode:** {self.config.emotion.processing_mode}

### Voices
"""
        for speaker, voice_config in self.config.voices.voices.items():
            md += f"- **{speaker}:** `{voice_config.voice_id}`"
            if voice_config.reference_audio:
                md += f" (custom reference: `{voice_config.reference_audio}`)"
            md += "\n"

        # Add Podcast Pipeline Configuration
        md += f"""

### Podcast Pipeline Configuration
- **Preset:** {self.podcast_config.description}
- **Chunking Mode:** {self.podcast_config.chunking.mode}
  - Window Size: {self.podcast_config.chunking.window_size}
- **Context Mode:** {self.podcast_config.context.mode}
  - Window Size: {self.podcast_config.context.window_size}
  - Include Emotions: {self.podcast_config.context.include_emotions}
- **Post-Processing:** {len(self.podcast_config.post_processing)} processors
"""
        for i, processor in enumerate(self.podcast_config.post_processing, 1):
            md += f"  {i}. {processor.name}"
            # Add key config details
            if hasattr(processor, 'alpha'):
                md += f" (alpha={processor.alpha})"
            if hasattr(processor, 'power_curve'):
                md += f" (power_curve={processor.power_curve})"
            md += "\n"

        gen_params = self.config.get_generation_params()
        md += f"""
### Generation Parameters
"""
        for key, value in gen_params.items():
            if key != "emo_vector":
                md += f"- **{key}:** {value}\n"

        md += f"""
### Output Settings
- **Format:** {self.config.output.format}
- **Sample Rate:** {self.config.output.sample_rate} Hz
- **Add Silence:** {self.config.output.add_silence}
"""
        if self.config.output.add_silence:
            md += f"\n- **Silence Duration:** {self.config.output.silence_duration} seconds"

        md += f"""

---

## Performance Profiling

### Stage Timing
| Stage | Time (s) | Percentage |
|-------|----------|------------|
"""
        total_time = processing_time
        for stage, time_taken in stage_times.items():
            percentage = (time_taken / total_time) * 100
            md += f"| {stage.replace('_', ' ').title()} | {time_taken:.2f}s | {percentage:.1f}% |\n"

        md += f"| **Total** | **{total_time:.2f}s** | **100%** |\n"

        md += f"""
### Generation Statistics
- **Total Segments:** {num_lines}
- **Speaker Breakdown:**
"""
        for speaker, count in tts_stats["speaker_counts"].items():
            md += f"  - {speaker}: {count} segments\n"

        md += f"""
- **Total Audio Duration:** {tts_stats['total_audio_duration']:.2f}s
- **Failed Chunks:** {tts_stats['failed_chunks']}
- **Real-time Factor:** {f"{processing_time / duration:.2f}x (processing time / audio duration)" if duration > 0 else "N/A (no audio generated)"}

---

## LLM Response (Emotion Detection)
"""
        # Add LLM thinking/reasoning if available
        reasoning = llm_info.get("reasoning", "")
        if reasoning:
            # Truncate if too long
            if len(reasoning) > 5000:
                reasoning = reasoning[:5000] + "\n\n... (truncated, full reasoning in logs)"
            md += f"""
### Thinking Process
```
{reasoning}
```
"""
        else:
            md += """
*No thinking content available (fast mode or thinking disabled)*
"""

        # Add emotion vectors by line
        if llm_results:
            md += """

---

## Emotion Vectors by Line

| Seg | Line | Speaker | Text | Emotion Vector |
|-----|------|---------|------|----------------|
"""
            for segment_idx, detected_lines, segment_llm_info, error in sorted(llm_results, key=lambda x: x[0]):
                if error:
                    md += f"| {segment_idx + 1} | - | ERROR | {error[:50]}... | - |\n"
                    continue
                for line_idx, line in enumerate(detected_lines):
                    if line.emotion:
                        emo_list = line.emotion.to_list()
                        # Get IndexTTS format for display
                        emo_indextts = line.emotion.to_indextts_format()
                        # Truncate text for display
                        text_preview = line.text[:60].replace("\n", " ").replace("|", "\\|")
                        if len(line.text) > 60:
                            text_preview += "..."
                        # Format emotion vector compactly (IndexTTS format: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm])
                        emo_str = f"[{emo_indextts[0]:.2f}, {emo_indextts[1]:.2f}, {emo_indextts[2]:.2f}, {emo_indextts[3]:.2f}, {emo_indextts[4]:.2f}, {emo_indextts[5]:.2f}, {emo_indextts[6]:.2f}, {emo_indextts[7]:.2f}]"
                        md += f"| {segment_idx + 1} | {line_idx + 1} | {line.speaker} | {text_preview} | {emo_str} |\n"

        md += f"""

---

## Failures and Errors
"""
        # Combine TTS and LLM failures
        all_failures = tts_stats.get("failures", []) + llm_info.get("llm_failures", [])

        if all_failures:
            md += f"**Total Failures:** {len(all_failures)}\n\n"
            for i, failure in enumerate(all_failures, 1):
                md += f"### {i}. {failure.get('type', 'Unknown')} Error\n"
                md += f"- **Speaker:** {failure.get('speaker', 'N/A')}\n"
                md += f"- **Text:** {failure.get('text', 'N/A')}\n"
                md += f"- **Error:** {failure.get('error', 'N/A')}\n"
                if 'line_index' in failure:
                    md += f"- **Line Index:** {failure['line_index']}\n"
                if 'worker_id' in failure:
                    md += f"- **Worker ID:** {failure['worker_id']}\n"
                md += "\n"
        else:
            md += "*No failures reported - all chunks processed successfully*\n"

        md += f"""

---

## Input Preview

First 500 characters of input:
```
{input_text[:500]}{"..." if len(input_text) > 500 else ""}
```

---

*Generated by Audio Producer - Podcast Director Workflow*
"""
        return md

    async def _detect_and_stream_parallel(self, dialogue_lines: List[DialogueLine], chunk_queue: asyncio.Queue, output_dir: Optional[Path] = None):
        """Detect emotions in parallel mode and stream chunks to TTS worker as they complete.

        Uses the centralized content_formatter emotion detector.
        Chunks with emotions are saved to disk for persistence if IndexTTS fails.

        Args:
            dialogue_lines: List of dialogue lines to process
            chunk_queue: Queue to stream completed chunks to TTS worker
            output_dir: Optional directory to save chunks for persistence
        """
        import json
        import datetime

        # Process ONE dialogue line at a time (sentence-by-sentence mode)
        # This prevents IndexTTS from getting overwhelmed with huge batches
        chunk_size = 1

        logger.info(f"Detecting emotions for {len(dialogue_lines)} dialogue lines (SENTENCE-BY-SENTENCE mode, chunk_size={chunk_size}, concurrency={self.config.emotion.parallel_concurrency})...")

        # Get emotion detector
        emotion_detector = await self._get_emotion_detector()

        # Create chunks directory for persistence
        chunks_dir = None
        if output_dir:
            chunks_dir = output_dir / "llm_chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving LLM chunks to: {chunks_dir}")

        # Split into chunks (1 line per chunk)
        chunks = []
        for i in range(0, len(dialogue_lines), chunk_size):
            chunks.append(dialogue_lines[i:i + chunk_size])

        logger.info(f"Split into {len(chunks)} chunks for parallel processing with streaming")

        # Process chunks in parallel batches
        for i in range(0, len(chunks), self.config.emotion.parallel_concurrency):
            batch_chunks = chunks[i:i + self.config.emotion.parallel_concurrency]

            async def process_chunk(chunk: List[DialogueLine], chunk_index: int):
                """Process one chunk using centralized detector and stream to queue."""
                try:
                    # Use centralized emotion detector
                    detected_lines, llm_info = await emotion_detector.detect(chunk)

                    # Save chunk to disk for persistence (before TTS)
                    if chunks_dir:
                        chunk_data = {
                            "chunk_index": chunk_index,
                            "original_lines": len(chunk),
                            "detected_lines": len(detected_lines),
                            "timestamp": datetime.datetime.now().isoformat(),
                            "lines": [
                                {
                                    "speaker": line.speaker,
                                    "text": line.text,
                                    "direction": line.direction,
                                    "emotion": line.emotion.to_list() if line.emotion else None
                                }
                                for line in detected_lines
                            ]
                        }
                        chunk_path = chunks_dir / f"chunk_{chunk_index:04d}.json"
                        chunk_path.write_text(json.dumps(chunk_data, indent=2))
                        logger.info(f"Saved chunk {chunk_index} to {chunk_path}")

                    # Stream completed chunk to TTS worker immediately
                    logger.info(f"LLM chunk complete: {len(chunk)} → {len(detected_lines)} lines, streaming to TTS...")
                    await chunk_queue.put(detected_lines)

                    return len(detected_lines)
                except Exception as e:
                    logger.warning(f"Failed to detect emotion for chunk: {e}, using neutral emotions")
                    for line in chunk:
                        line.emotion = EmotionVector.neutral()
                    await chunk_queue.put(chunk)
                    return len(chunk)

            # Process batch in parallel with chunk index
            tasks = [
                process_chunk(chunk, i + j)
                for j, chunk in enumerate(batch_chunks)
            ]
            results = await asyncio.gather(*tasks)

            logger.info(f"Processed batch {i//self.config.emotion.parallel_concurrency + 1}/{(len(chunks) + self.config.emotion.parallel_concurrency - 1)//self.config.emotion.parallel_concurrency}")

        logger.info("All LLM chunks complete, pipelining complete")

    async def _detect_and_stream_single(self, dialogue_lines: List[DialogueLine], chunk_queue: asyncio.Queue, output_dir: Optional[Path] = None, llm_failures: Optional[list] = None):
        """Detect emotions in single mode and stream chunks to TTS worker as they complete.

        Uses the centralized content_formatter emotion detector (deep_agent mode).
        Each completed line is immediately streamed to TTS for pipelining.

        Args:
            dialogue_lines: List of dialogue lines to process
            chunk_queue: Queue to stream completed chunks to TTS worker
            output_dir: Optional directory to save chunks for persistence
            llm_failures: Optional list to track LLM parsing failures
        """
        if llm_failures is None:
            llm_failures = []
        import json
        import datetime

        logger.info(f"Detecting emotions for {len(dialogue_lines)} dialogue lines (SINGLE mode with pipelining)...")

        # Get emotion detector
        emotion_detector = await self._get_emotion_detector()

        # Create chunks directory for persistence
        chunks_dir = None
        if output_dir:
            chunks_dir = output_dir / "llm_chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving LLM chunks to: {chunks_dir}")

        # Track processed lines with their emotions
        processed_lines: List[DialogueLine] = []

        # Process lines one at a time
        for i, line in enumerate(dialogue_lines):
            try:
                # Detect emotion for this single line
                detected_lines, llm_info = await emotion_detector.detect([line])

                # Apply post-processing to all detected lines
                for sl in detected_lines:
                    if sl.emotion:
                        # Get previous emotion for context (if available)
                        prev_emotion = None
                        if processed_lines:
                            last_line = processed_lines[-1]
                            prev_emotion = last_line.emotion.to_list() if last_line.emotion else None

                        # Apply post-processing pipeline
                        raw_emotion = sl.emotion.to_list()
                        processed_emotion_list = self.post_processor.process(
                            emotions=raw_emotion,
                            speaker=sl.speaker,
                            text=sl.text,
                            previous_emotions=prev_emotion
                        )
                        sl.emotion = EmotionVector(processed_emotion_list)

                        logger.debug(f"Post-processed emotion for '{sl.text[:30]}...': {raw_emotion} -> {processed_emotion_list}")

                # Add processed lines to context
                for sl in detected_lines:
                    processed_lines.append(sl)

                # Save chunk to disk for persistence (before TTS)
                if chunks_dir:
                    chunk_data = {
                        "chunk_index": i,
                        "original_lines": 1,
                        "detected_lines": len(detected_lines),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "lines": [
                            {
                                "speaker": l.speaker,
                                "text": l.text,
                                "direction": l.direction,
                                "emotion": l.emotion.to_list() if l.emotion else None
                            }
                            for l in detected_lines
                        ]
                    }
                    chunk_path = chunks_dir / f"chunk_{i:04d}.json"
                    chunk_path.write_text(json.dumps(chunk_data, indent=2))
                    logger.info(f"Saved chunk {i} to {chunk_path}")

                # Stream completed chunk to TTS worker immediately
                logger.info(f"LLM chunk complete: 1 → {len(detected_lines)} lines, streaming to TTS...")
                await chunk_queue.put(detected_lines)

            except Exception as e:
                logger.warning(f"Failed to detect emotion for line {i+1}: {e}, using neutral emotions")
                llm_failures.append({
                    "type": "LLM_Parsing",
                    "speaker": line.speaker,
                    "text": line.text[:100] + "..." if len(line.text) > 100 else line.text,
                    "error": str(e),
                    "line_index": i
                })
                line.emotion = EmotionVector.neutral()
                processed_lines.append(line)
                await chunk_queue.put([line])

        logger.info("All LLM chunks complete (single mode), pipelining complete")

        # Return all processed lines for report generation
        return processed_lines

    async def _detect_and_stream_rolling_batch(
        self,
        dialogue_lines: List[DialogueLine],
        chunk_queue: asyncio.Queue,
        output_dir: Optional[Path] = None,
        llm_failures: Optional[list] = None
    ):
        """Detect emotions in rolling window batch mode.

        Processes lines in batches using the centralized content_formatter emotion detector.
        Much faster than single mode while maintaining conversation awareness.

        Example for 74 lines with window_size=7:
        - Batch 1: Lines 1-7   → LLM returns emotions  → Stream all
        - Batch 2: Lines 8-14  → LLM returns emotions  → Stream all
        - Batch 3: Lines 15-21 → LLM returns emotions  → Stream all
        ...
        Total: ~11 LLM calls instead of 74

        Args:
            dialogue_lines: List of dialogue lines to process
            chunk_queue: Queue to stream completed chunks to TTS worker
            output_dir: Optional directory to save chunks for persistence
            llm_failures: Optional list to track LLM parsing failures
        """
        if llm_failures is None:
            llm_failures = []
        import json
        import datetime

        # Get configuration
        window_size = self.podcast_config.chunking.window_size

        logger.info(f"Detecting emotions for {len(dialogue_lines)} dialogue lines (ROLLING WINDOW BATCH mode)")
        logger.info(f"  Batch size: {window_size} lines")
        logger.info(f"  Expected LLM calls: ~{(len(dialogue_lines) + window_size - 1) // window_size}")

        # Get emotion detector
        emotion_detector = await self._get_emotion_detector()

        # Create chunks directory for persistence
        chunks_dir = None
        if output_dir:
            chunks_dir = output_dir / "llm_chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving LLM chunks to: {chunks_dir}")

        # Track all processed lines for context
        all_processed_lines: List[DialogueLine] = []
        batch_index = 0

        # Process in batches
        for batch_start in range(0, len(dialogue_lines), window_size):
            batch_end = min(batch_start + window_size, len(dialogue_lines))
            batch_lines = dialogue_lines[batch_start:batch_end]

            logger.info(f"Processing batch {batch_index + 1}: lines {batch_start + 1}-{batch_end} ({len(batch_lines)} lines)")

            try:
                # Detect emotions for this batch using centralized detector
                detected_lines, llm_info = await emotion_detector.detect(batch_lines)

                # Apply post-processing to all detected lines
                for sl in detected_lines:
                    if sl.emotion:
                        # Get previous emotion for context
                        prev_emotion = None
                        if all_processed_lines:
                            last_line = all_processed_lines[-1]
                            prev_emotion = last_line.emotion.to_list() if last_line.emotion else None

                        # Apply post-processing
                        raw_emotion = sl.emotion.to_list()
                        processed_emotion = self.post_processor.process(
                            emotions=raw_emotion,
                            speaker=sl.speaker,
                            text=sl.text,
                            previous_emotions=prev_emotion
                        )
                        sl.emotion = EmotionVector(processed_emotion)

                        logger.debug(f"Post-processed: '{sl.text[:30]}...': {raw_emotion} -> {processed_emotion}")

                # Add processed lines to history
                for line in detected_lines:
                    all_processed_lines.append(line)

                # Save chunk to disk
                if chunks_dir:
                    chunk_data = {
                        "batch_index": batch_index,
                        "batch_range": f"{batch_start + 1}-{batch_end}",
                        "original_lines": len(batch_lines),
                        "detected_lines": len(detected_lines),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "lines": [
                            {
                                "speaker": l.speaker,
                                "text": l.text,
                                "direction": l.direction,
                                "emotion": l.emotion.to_list() if l.emotion else None
                            }
                            for l in detected_lines
                        ]
                    }
                    chunk_path = chunks_dir / f"batch_{batch_index:04d}.json"
                    chunk_path.write_text(json.dumps(chunk_data, indent=2))
                    logger.info(f"Saved batch {batch_index} to {chunk_path}")

                # Stream batch to TTS
                logger.info(f"LLM batch {batch_index} complete: {len(batch_lines)} → {len(detected_lines)} lines, streaming to TTS...")
                await chunk_queue.put(detected_lines)

            except Exception as e:
                logger.error(f"Failed to process batch {batch_index}: {e}")
                llm_failures.append({
                    "type": "LLM_Call_Failed",
                    "batch_index": batch_index,
                    "batch_range": f"{batch_start + 1}-{batch_end}",
                    "error": str(e)
                })
                # Fall back to neutral emotions
                for line in batch_lines:
                    line.emotion = EmotionVector.neutral()
                    all_processed_lines.append(line)
                await chunk_queue.put(batch_lines)

            batch_index += 1

        logger.info(f"All LLM batches complete (rolling window mode, {batch_index} batches total), pipelining complete")

        # Return all processed lines for report generation
        return all_processed_lines

    async def _generate_tts_for_segment(
        self,
        dialogue_lines: List[DialogueLine],
        multi_manager: ExtensionProcessManager,
        audio_chunks_dir: Path,
        output_path: Path
    ):
        """Generate TTS audio for a single segment using parallel TTS workers.

        Args:
            dialogue_lines: List of dialogue lines with emotions already detected
            multi_manager: ExtensionProcessManager with running IndexTTS vLLM instance
            audio_chunks_dir: Directory to save individual audio chunks
            output_path: Final output path for merged segment audio
        """
        # CRITICAL: Create the audio chunks directory if it doesn't exist
        audio_chunks_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TTS output directory: {audio_chunks_dir}")

        generation_params = self.config.get_generation_params()
        audio_chunks_dict = {}
        chunk_index_counter = 0

        # Create queue for TTS workers
        chunk_queue = asyncio.Queue()

        async def tts_worker(worker_id: int):
            """Worker that processes TTS chunks from queue."""
            nonlocal chunk_index_counter
            try:
                while True:
                    line = await chunk_queue.get()

                    if line is None:  # Sentinel - done
                        break

                    # Retry logic
                    max_retries = 5
                    base_delay = 1.0

                    for attempt in range(max_retries):
                        try:
                            voice_id = self.config.voices.get_voice_id(line.speaker, "index2_ref_neutral")
                            reference_audio = self.config.voices.get_reference_audio(line.speaker)

                            if line.emotion:
                                emo_vec = line.emotion.to_list()
                            else:
                                emo_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1]

                            params = {"emo_vector": emo_vec, **generation_params}

                            if reference_audio:
                                params["ref_audio_path"] = reference_audio

                            audio_bytes, duration = await multi_manager.generate_audio(
                                text=line.text,
                                voice_id=voice_id,
                                **params,
                            )

                            # Save chunk
                            current_chunk_index = chunk_index_counter
                            chunk_index_counter += 1

                            chunk_path = audio_chunks_dir / f"chunk_{current_chunk_index:04d}_{line.speaker}.wav"
                            chunk_path.write_bytes(audio_bytes)
                            audio_chunks_dict[current_chunk_index] = chunk_path

                            logger.info(f"TTS worker {worker_id}: Generated chunk {current_chunk_index} ({duration:.2f}s)")
                            break

                        except Exception as e:
                            is_timeout = "timeout" in str(e).lower() or "connection" in str(e).lower()

                            # Note: ExtensionProcessManager is more stable than the old adapter system
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt)
                                logger.warning(f"TTS worker {worker_id}: Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s...")
                                await asyncio.sleep(delay)
                            else:
                                raise RuntimeError(f"Failed to generate audio after {max_retries} attempts: {e}")

            except Exception as e:
                logger.error(f"TTS worker {worker_id} error: {e}")

        # Start TTS workers (ExtensionProcessManager uses single instance with internal concurrency)
        # vLLM has 16x concurrency internally, so 4 workers is sufficient
        num_workers = 4
        tts_tasks = [
            asyncio.create_task(tts_worker(i))
            for i in range(num_workers)
        ]

        # Queue all dialogue lines for TTS
        # (Scene descriptions are already filtered out during parsing)
        for line in dialogue_lines:
            await chunk_queue.put(line)

        # Signal completion
        for _ in range(num_workers):
            await chunk_queue.put(None)

        # Wait for all workers to complete
        await asyncio.gather(*tts_tasks)

        # Reconstruct chunks in order
        audio_chunks = [audio_chunks_dict[i] for i in sorted(audio_chunks_dict.keys())]

        # Merge audio chunks
        await self.merger.merge_audio_files(
            audio_files=audio_chunks,
            output_path=output_path,
            add_silence=self.config.output.add_silence,
            silence_duration=self.config.output.silence_duration,
            speaker_aware=True,
            speaker_gap_duration=0.4,
            normal_gap_duration=0.15,
            normalize=True,
            cleanup_source=False,
        )

        logger.info(f"Segment TTS complete: {len(audio_chunks)} chunks merged to {output_path}")

    async def cleanup(self):
        """Cleanup resources."""
        if self._process_manager:
            try:
                await self._process_manager.stop_current()
            except Exception:
                pass


def create_podcast_director(
    config: AudioJobConfig,
    prompts_dir: str,
    podcast_config: Union[PodcastConfig, str, None] = None
) -> PodcastDirector:
    """Create a Podcast Director instance.

    Args:
        config: AudioJobConfig for TTS settings
        prompts_dir: Directory containing prompt templates
        podcast_config: PodcastConfig instance, preset name (e.g., "balanced", "natural", "quality"),
                       or None to use default (balanced)

    Returns:
        Configured PodcastDirector instance

    Example:
        # Use default (balanced)
        director = create_podcast_director(config, prompts_dir)

        # Use preset by name
        director = create_podcast_director(config, prompts_dir, "natural")

        # Use custom config
        from podcast_config import PodcastConfig, DynamicsCompressionConfig
        custom_config = PodcastConfig(
            post_processing=[DynamicsCompressionConfig(power_curve=2.5)]
        )
        director = create_podcast_director(config, prompts_dir, custom_config)
    """
    return PodcastDirector(config, prompts_dir, podcast_config)
