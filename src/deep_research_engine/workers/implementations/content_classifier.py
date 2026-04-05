"""Content Classifier Worker - LLM classification and enrichment of text chunks.

This worker receives chunks from multiple sources:
- Kreuzberg document chunks (PDF, DOCX, HTML, Telegram, etc.)
- Audio/video transcripts (VibeVoice ASR with diarization)
- Web scraping results
- VLM image descriptions

For each chunk, it:
1. Classifies content (filler vs substantive vs mixed)
2. Extracts core content, key points, topics, entities, summary
3. Optionally filters filler content

Usage:
    worker = ContentClassifierWorker(config)
    result = await worker.execute(state, config)

Note: This replaces the old chunk_processor.py. The name changed because
Kreuzberg handles the chunking - this worker only classifies and enriches.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Any

from ...state import EngineState, WorkerResult
from ...utils.logging import get_loguru_logger
from ..base import BaseWorker, WorkerConfig, WorkerRegistry

logger = get_loguru_logger("workers.content_classifier")


@WorkerRegistry.register("content_classifier")
class ContentClassifierWorker(BaseWorker):
    """LLM-based content classifier and enricher.

    Classifies text chunks as filler, substantive, or mixed, then extracts
    structured information (key points, topics, entities, summary).

    Input: List of text chunks from Kreuzberg, ASR transcripts, or scraping.
    Output: Classified and enriched chunks ready for content clustering.
    """

    worker_type = "content_classifier"
    description = "Classifies text chunks and extracts structured content metadata"
    required_tools = ["llm_classify", "llm_extract"]

    # Batch processing limits
    MAX_CHUNKS_PER_BATCH = 50
    MAX_CHARS_PER_CHUNK = 4000
    MAX_CONCURRENT_CLASSIFICATIONS = 5

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.chunks: list[dict[str, Any]] = config.custom.get("chunks", [])
        self.enable_classification = config.custom.get("enable_classification", True)
        self.enable_extraction = config.custom.get("enable_extraction", True)
        self.filter_filler = config.custom.get("filter_filler", True)
        self.concurrency = config.custom.get("concurrency", self.MAX_CONCURRENT_CLASSIFICATIONS)

    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Classify and enrich a batch of text chunks.

        Args:
            state: Current engine state
            config: Worker configuration with chunks list

        Returns:
            WorkerResult with classified and enriched chunks
        """
        self._start_timing()

        try:
            # Get chunks from config or state
            chunks = self.chunks or state.get("context", {}).get("document_chunks", [])
            if not chunks:
                # Also check worker_results for transcript chunks
                for wr in state.get("worker_results", []):
                    output = wr.get("output", {})
                    if isinstance(output, dict):
                        transcript = output.get("transcription", {})
                        if isinstance(transcript, dict):
                            text = transcript.get("text", "")
                            if text:
                                chunks.append({
                                    "text": text,
                                    "source": wr.get("worker_type", "unknown"),
                                    "source_file": output.get("media_path", ""),
                                })

            if not chunks:
                return self._create_result(
                    success=False,
                    output=None,
                    error="No chunks to classify. Provide chunks in config.custom['chunks'] "
                          "or state.context['document_chunks'] or worker_results with text.",
                )

            # Normalize chunks to our schema
            normalized = self._normalize_chunks(chunks)

            # Process chunks in batches with concurrency limit
            classified = await self._process_batch(
                state=state,
                chunks=normalized,
            )

            # Filter filler if enabled
            if self.filter_filler:
                filtered = [c for c in classified if not c.get("is_filler", False)]
                filler_count = len(classified) - len(filtered)
            else:
                filtered = classified
                filler_count = 0

            # Build result
            result = {
                "classified_chunks": classified,
                "filtered_chunks": filtered,
                "total_chunks": len(classified),
                "filler_filtered": filler_count,
                "substantive_remaining": len(filtered),
                "categories": self._summarize_categories(classified),
                "processed_at": datetime.now().isoformat(),
            }

            self._log(
                f"Content classification complete: "
                f"{len(classified)} chunks classified, "
                f"{filler_count} filler filtered, "
                f"{len(filtered)} substantive remaining"
            )

            return self._create_result(
                success=True,
                output=result,
                output_type="content_classification",
                confidence=0.85,
            )

        except Exception as e:
            self._log(f"Content classification failed: {e}", level="error")
            return self._create_result(
                success=False,
                output=None,
                error=str(e),
            )

    def _normalize_chunks(
        self, raw_chunks: list[dict[str, Any]] | list[str]
    ) -> list[dict[str, Any]]:
        """Normalize chunks to a consistent schema.

        Handles chunks from different sources:
        - Kreuzberg: {"text": ..., "index": ..., "source_file": ...}
        - Transcripts: {"text": ..., "speaker": ..., "source": ...}
        - Raw strings: wrapped into dict
        """
        normalized = []
        for i, chunk in enumerate(raw_chunks[: self.MAX_CHUNKS_PER_BATCH]):
            if isinstance(chunk, str):
                normalized.append({
                    "text": chunk[: self.MAX_CHARS_PER_CHUNK],
                    "chunk_id": f"chunk_{i}",
                    "source": "raw_text",
                })
            elif isinstance(chunk, dict):
                text = (
                    chunk.get("text", "")
                    or chunk.get("content", "")
                    or str(chunk.get("output", ""))
                )
                normalized.append({
                    "text": text[: self.MAX_CHARS_PER_CHUNK],
                    "chunk_id": chunk.get("chunk_id", chunk.get("id", f"chunk_{i}")),
                    "index": chunk.get("index", i),
                    "source_file": chunk.get(
                        "source_file", chunk.get("source", "unknown")
                    ),
                    "source": chunk.get("source", "unknown"),
                    "heading_context": chunk.get("heading_context", ""),
                    "page": chunk.get("page"),
                    "speaker": chunk.get("speaker"),
                    **{
                        k: v
                        for k, v in chunk.items()
                        if k
                        not in (
                            "text",
                            "content",
                            "chunk_id",
                            "id",
                            "index",
                            "source_file",
                            "source",
                            "heading_context",
                            "page",
                            "speaker",
                        )
                    },
                })
        return normalized

    async def _process_batch(
        self,
        state: EngineState,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Process a batch of chunks with concurrency limiting.

        Uses a semaphore to limit concurrent LLM calls.

        Args:
            state: Current engine state
            chunks: Normalized chunk list

        Returns:
            List of classified and enriched chunks.
        """
        semaphore = asyncio.Semaphore(self.concurrency)
        results: list[dict[str, Any]] = [{}] * len(chunks)

        async def _process_one(idx: int, chunk: dict[str, Any]) -> None:
            async with semaphore:
                classification = None
                extraction = None

                if self.enable_classification:
                    classification = await self._classify_chunk(
                        state=state, chunk=chunk["text"]
                    )

                if self.enable_extraction:
                    extraction = await self._extract_core_content(
                        state=state,
                        chunk=chunk["text"],
                        classification=classification,
                    )

                results[idx] = {
                    **chunk,
                    "classification": classification,
                    "extraction": extraction,
                    "is_filler": classification.get("is_filler", False)
                    if classification
                    else False,
                }

        tasks = [_process_one(i, chunk) for i, chunk in enumerate(chunks)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions from gather
        final_results = []
        for i, result in enumerate(results):
            if not result:
                # Task failed or wasn't processed
                final_results.append({
                    **chunks[i],
                    "classification": {
                        "category": "unknown",
                        "confidence": 0.0,
                        "reasoning": "Processing failed",
                        "is_filler": False,
                    },
                    "extraction": None,
                    "is_filler": False,
                })
            else:
                final_results.append(result)

        return final_results

    async def _classify_chunk(
        self,
        state: EngineState,
        chunk: str,
    ) -> dict[str, Any]:
        """Classify a single text chunk via LLM.

        Args:
            state: Current engine state
            chunk: Text to classify

        Returns:
            Classification dict with category, confidence, reasoning.
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate

            llm = await self._get_llm(state, role="worker")

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a content classifier. Classify text chunks into categories:

CATEGORIES:
- filler: Navigation, ads, boilerplate, copyright, privacy policies, low-value content
- substantive: Core content, valuable information, main article content
- mixed: Contains both filler and substantive parts

CLASSIFY based on:
1. Is this the main content or supporting material?
2. Would this be valuable in a summary?
3. Is this repetitive or unique information?

Respond with JSON ONLY:
{"category": "filler|substantive|mixed", "confidence": 0.0-1.0, "reasoning": "Brief explanation"}""",
                    ),
                    ("user", "Classify this chunk:\n\n{chunk}"),
                ]
            )

            chain = prompt | llm
            response = await chain.ainvoke({"chunk": chunk[: self.MAX_CHARS_PER_CHUNK]})

            content = response.content if hasattr(response, "content") else str(response)
            return self._parse_json_response(content)

        except Exception as e:
            self._log(f"Classification failed: {e}", level="warning")
            return {
                "category": "mixed",
                "confidence": 0.5,
                "reasoning": f"Classification error: {e}",
                "is_filler": False,
            }

    async def _extract_core_content(
        self,
        state: EngineState,
        chunk: str,
        classification: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract structured information from a text chunk via LLM.

        Args:
            state: Current engine state
            chunk: Text to extract from
            classification: Optional classification result

        Returns:
            Extraction dict with core content, key points, topics, entities, summary.
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate

            llm = await self._get_llm(state, role="worker")

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a content extraction specialist. Extract core information from text chunks.

EXTRACT:
1. Core content: Main information without filler
2. Key points: 3-5 bullet points of key information
3. Topics: Main topics/themes (3-5 keywords)
4. Entities: Named entities (people, organizations, products)
5. Summary: 1-2 sentence summary

Respond with JSON ONLY:
{"core_content": "...", "key_points": [...], "topics": [...], "entities": {"people": [], "organizations": [], "products": []}, "summary": "..."}""",
                    ),
                    ("user", "Extract core content from this chunk:\n\n{chunk}"),
                ]
            )

            chain = prompt | llm
            response = await chain.ainvoke({"chunk": chunk[: self.MAX_CHARS_PER_CHUNK]})

            content = response.content if hasattr(response, "content") else str(response)
            parsed = self._parse_json_response(content)

            return {
                "core_content": parsed.get("core_content", chunk),
                "key_points": parsed.get("key_points", []),
                "topics": parsed.get("topics", []),
                "entities": parsed.get("entities", {}),
                "summary": parsed.get("summary", ""),
            }

        except Exception as e:
            self._log(f"Extraction failed: {e}", level="warning")
            return {
                "core_content": chunk,
                "key_points": [],
                "topics": [],
                "entities": {},
                "summary": "",
            }

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """Parse JSON from LLM response.

        Handles:
        - Raw JSON
        - JSON in markdown code blocks
        - Partial JSON with surrounding text
        """
        text = content.strip()

        # Try extracting from code blocks
        for marker in ["```json", "```"]:
            if marker in text:
                try:
                    text = text.split(marker, 1)[1].split("```", 1)[0].strip()
                    return json.loads(text)
                except (json.JSONDecodeError, IndexError):
                    continue

        # Try finding JSON object
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.debug(f"Could not parse JSON from: {content[:200]}")
        return {}

    def _summarize_categories(
        self, classified: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Count chunks by classification category.

        Args:
            classified: List of classified chunks.

        Returns:
            Dict mapping category to count.
        """
        counts: dict[str, int] = {}
        for chunk in classified:
            cat = chunk.get("classification", {}).get("category", "unknown")
            counts[cat] = counts.get(cat, 0) + 1
        return counts
