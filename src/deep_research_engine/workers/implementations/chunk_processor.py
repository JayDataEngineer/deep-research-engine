"""Chunk Processor Worker - Process text chunks for classification and extraction.

This worker processes text chunks (from PDFs, documents, web pages) to:
1. Classify if content is filler or substantive
2. Extract core content and key information
3. Enrich with metadata (topics, entities, summary)
4. Filter out low-value content

Key Features:
- LLM-based classification
- Core content extraction
- Topic and entity extraction
- Configurable filtering thresholds

Usage:
    worker = ChunkProcessorWorker(config)
    result = await worker.execute(state, config)
"""

from datetime import datetime
from typing import Any

from ...state import EngineState, WorkerResult
from ..base import BaseWorker, WorkerConfig, WorkerRegistry


@WorkerRegistry.register("chunk_processor")
class ChunkProcessorWorker(BaseWorker):
    """Process text chunks for classification and extraction.

    This worker:
    1. Receives a text chunk (from PDF, document, or web page)
    2. Classifies if it's filler or substantive content
    3. Extracts core content and key information
    4. Returns structured chunk data

    Classification Categories:
    - filler: Navigation, ads, boilerplate, low-value content
    - substantive: Core content, valuable information
    - mixed: Contains both filler and substantive parts
    """

    worker_type = "chunk_processor"
    description = "Classifies and extracts core content from text chunks"
    required_tools = ["llm_classify", "llm_extract"]

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.chunk = config.custom.get("chunk", "")
        self.chunk_id = config.custom.get("chunk_id", f"chunk_{datetime.now().isoformat()}")
        self.source_url = config.custom.get("source_url", "")
        self.enable_extraction = config.custom.get("enable_extraction", True)
        self.enable_classification = config.custom.get("enable_classification", True)
        self.filter_filler = config.custom.get("filter_filler", True)
        self.extraction_schema = config.custom.get("extraction_schema", None)

    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Process a text chunk.

        Args:
            state: Current engine state
            config: Worker configuration

        Returns:
            WorkerResult with processed chunk data
        """
        self._start_timing()

        try:
            # Get chunk from config or state
            chunk_text = self.chunk or state.get("input_data", {}).get("chunk", "")
            if not chunk_text:
                return self._create_result(
                    success=False,
                    output=None,
                    error="No chunk text provided",
                )

            # Get YAML config for LLM
            yaml_config = state.get("context", {}).get("config", {})

            # Classify chunk
            classification = None
            if self.enable_classification:
                classification = await self._classify_chunk(
                    chunk=chunk_text,
                    yaml_config=yaml_config,
                )

                # Filter filler if enabled
                if self.filter_filler and classification.get("is_filler", False):
                    self._log("Chunk classified as filler, filtering")
                    return self._create_result(
                        success=True,
                        output={
                            "chunk_id": self.chunk_id,
                            "is_filler": True,
                            "classification": classification,
                            "filtered": True,
                        },
                        output_type="chunk_classification",
                        confidence=classification.get("confidence", 0.8),
                    )

            # Extract core content
            extraction = None
            if self.enable_extraction:
                extraction = await self._extract_core_content(
                    chunk=chunk_text,
                    classification=classification,
                    yaml_config=yaml_config,
                )

            # Build result
            result = {
                "chunk_id": self.chunk_id,
                "source_url": self.source_url,
                "text": chunk_text,
                "classification": classification,
                "extraction": extraction,
                "metadata": {
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "processed_at": datetime.now().isoformat(),
                },
            }

            self._log(
                f"Chunk processed: {classification.get('category', 'unknown')}, "
                f"{len(chunk_text)} chars"
            )

            return self._create_result(
                success=True,
                output=result,
                output_type="processed_chunk",
                confidence=classification.get("confidence", 0.8) if classification else 0.7,
            )

        except Exception as e:
            self._log(f"Chunk processing failed: {e}", level="error")
            return self._create_result(
                success=False,
                output=None,
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _classify_chunk(
        self,
        chunk: str,
        yaml_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Classify a text chunk.

        Args:
            chunk: Text chunk to classify
            yaml_config: LLM configuration

        Returns:
            Classification dict with category and confidence
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate

            # Get LLM from config
            llm = await self._create_llm(yaml_config, role="worker")

            # Classification prompt
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

Respond with JSON:
{{
  "category": "filler|substantive|mixed",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}""",
                    ),
                    ("user", "Classify this chunk:\n\n{chunk}"),
                ]
            )

            # Invoke LLM
            chain = prompt | llm
            response = await chain.ainvoke({"chunk": chunk[:4000]})  # Limit context

            # Parse response
            import json
            import re

            content = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from response
            json_match = re.search(r"\{[^}]*\}", content, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group())
                return {
                    "category": classification.get("category", "mixed"),
                    "confidence": float(classification.get("confidence", 0.5)),
                    "reasoning": classification.get("reasoning", ""),
                    "is_filler": classification.get("category") == "filler",
                }

            # Fallback
            return {
                "category": "mixed",
                "confidence": 0.5,
                "reasoning": "Could not parse classification",
                "is_filler": False,
            }

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
        chunk: str,
        classification: dict[str, Any] | None = None,
        yaml_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract core content from chunk.

        Args:
            chunk: Text chunk
            classification: Optional classification result
            yaml_config: LLM configuration

        Returns:
            Extraction dict with core content and metadata
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate

            # Get LLM from config
            llm = await self._create_llm(yaml_config, role="worker")

            # Extraction prompt
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

Respond with JSON:
{{
  "core_content": "Cleaned text with only substantive content",
  "key_points": ["point 1", "point 2", ...],
  "topics": ["topic1", "topic2", ...],
  "entities": {{"people": [], "organizations": [], "products": []}},
  "summary": "Brief summary"
}}""",
                    ),
                    ("user", "Extract core content from this chunk:\n\n{chunk}"),
                ]
            )

            # Invoke LLM
            chain = prompt | llm
            response = await chain.ainvoke({"chunk": chunk[:4000]})  # Limit context

            # Parse response
            import json
            import re

            content = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from response
            json_match = re.search(r"\{[^}]*\}", content, re.DOTALL)
            if json_match:
                extraction = json.loads(json_match.group())
                return {
                    "core_content": extraction.get("core_content", chunk),
                    "key_points": extraction.get("key_points", []),
                    "topics": extraction.get("topics", []),
                    "entities": extraction.get("entities", {}),
                    "summary": extraction.get("summary", ""),
                }

            # Fallback
            return {
                "core_content": chunk,
                "key_points": [],
                "topics": [],
                "entities": {},
                "summary": "",
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

    async def _create_llm(
        self,
        yaml_config: dict[str, Any],
        role: str = "worker",
    ) -> Any:
        """Create LLM from config.

        Args:
            yaml_config: YAML configuration
            role: LLM role ("worker", "boss", "critic")

        Returns:
            LLM instance
        """
        try:
            from src.shared.chat_llm import get_chat_llm

            model_config = yaml_config.get("model", {})
            model_name = model_config.get("name", "qwen-35-27")
            enable_thinking = model_config.get("enable_thinking", False)

            return await get_chat_llm(model=model_name, enable_thinking=enable_thinking)

        except ImportError:
            # Fallback to langchain
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
            )
