"""Artifact Maker Worker - Formats research findings into structured artifacts.

This worker adapts the content_formatter pattern for the artifact maker stage.
It supports multiple output formats (research_report, findings, markdown, json, etc.)
with Pydantic-validated structured outputs.

Usage:
    artifact_maker_worker = ArtifactMakerWorker(config)
    result = await artifact_maker_worker.execute(state, config)
"""

import asyncio
import importlib.util
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from ...prompts import PromptRegistry, get_prompt
from ...state import EngineState, WorkerResult
from ..base import BaseWorker, WorkerConfig, WorkerRegistry


@WorkerRegistry.register("artifact_maker")
class ArtifactMakerWorker(BaseWorker):
    """Artifact maker worker that formats research findings.

    Uses the content_formatter pattern:
    - Format configs in artifacts/formats/*/config.yaml
    - Pydantic schemas in artifacts/formats/*/schema.py
    - Jinja2 templates for prompts
    - Retry logic with validation feedback
    """

    worker_type = "artifact_maker"
    description = "Formats research findings into structured artifacts"
    required_tools: list = []

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.formats_dir = Path(__file__).parent.parent.parent / "artifacts" / "formats"
        self.max_retries = config.custom.get("max_retries", 10)
        self.output_format = config.custom.get("format", "research_report")

    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Execute artifact making on worker results.

        Args:
            state: Current engine state with worker_results
            config: Worker configuration

        Returns:
            WorkerResult with artifact output
        """
        self._start_timing()

        try:
            # Extract worker findings
            worker_results = state.get("worker_results", [])
            if not worker_results:
                return self._create_result(
                    success=False, output=None, error="No worker results to turn into an artifact"
                )

            # Get artifact config
            artifact_config = config.custom.get("config", {})
            output_format = artifact_config.get("format", self.output_format)
            citations_enabled = artifact_config.get("citations", {}).get("enabled", True)
            citations_format = artifact_config.get("citations", {}).get("format", "numbered")

            # Format content
            formatted_output = await self._format_findings(
                worker_results=worker_results,
                format_name=output_format,
                citations_enabled=citations_enabled,
                citations_format=citations_format,
                session_id=state.get("session_id"),
            )

            # Store in state
            state["artifact_output"] = (
                formatted_output.model_dump_json()
                if isinstance(formatted_output, BaseModel)
                else str(formatted_output)
            )
            state["artifact_metadata"] = {
                "format": output_format,
                "worker_count": len(worker_results),
                "citations_enabled": citations_enabled,
            }

            return self._create_result(
                success=True,
                output=formatted_output,
                output_type=output_format,
                confidence=0.9,
            )

        except Exception as e:
            self._log(f"Artifact generation failed: {e}", level="error")
            return self._create_result(
                success=False,
                output=None,
                error=str(e),
            )

    async def _format_findings(
        self,
        worker_results: list[dict[str, Any]],
        format_name: str,
        citations_enabled: bool = True,
        citations_format: str = "numbered",
        session_id: str | None = None,
    ) -> BaseModel:
        """Format research findings using content_formatter pattern.

        Args:
            worker_results: List of worker result dicts
            format_name: Output format (research_report, findings, markdown, etc.)
            citations_enabled: Whether to include citations
            citations_format: Citation format (numbered, APA, MLA, etc.)
            session_id: Optional session ID for Context Engine access

        Returns:
            Validated Pydantic model instance
        """
        # Resolve format alias to prompt name
        format_to_prompt = {
            "research_report": PromptRegistry.RESEARCH_REPORT,
            "markdown": PromptRegistry.MARKDOWN_FORMAT,
            "structured_json": PromptRegistry.STRUCTURED_JSON,
        }

        prompt_name = format_to_prompt.get(format_name.lower())
        if not prompt_name:
            available = list(format_to_prompt.keys())
            raise ValueError(f"Format '{format_name}' not found. Available formats: {available}")

        # 1. Get prompt from LangFuse
        prompt = await get_prompt(prompt_name)

        # 2. Load schema
        schema_class = self._load_schema_from_prompt_name(format_name)

        # 3. Compile findings
        findings_text = self._compile_findings(worker_results, citations_enabled)

        # 4. Compile prompt with variables
        system_prompt = prompt.compile(
            findings=findings_text,
            worker_count=len(worker_results),
            citations_enabled=citations_enabled,
            citations_format=citations_format,
        )

        # 5. Get model config from prompt
        model_config = prompt.get_model_config()
        llm = await self._create_llm(model_config)

        # 6. Invoke with retry loop
        current_prompt = system_prompt
        last_error: ValidationError | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                agent_output = await llm.ainvoke(
                    [
                        {"role": "system", "content": current_prompt},
                        {"role": "user", "content": current_prompt},
                    ]
                )
                agent_output = agent_output.content

                # Extract structured output
                formatted_result = await self._extract_structured_output(
                    llm=llm,
                    agent_output=agent_output,
                    pydantic_model=schema_class,
                )

                self._log(f"Successfully formatted '{format_name}' on attempt {attempt}")
                return formatted_result

            except ValidationError as e:
                last_error = e

                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Format validation failed after {self.max_retries} attempts for '{format_name}'. "
                        f"Errors: {self._format_validation_errors(e)}"
                    ) from e

                error_details = self._format_validation_errors(e)
                current_prompt = self._add_retry_feedback(system_prompt, error_details, attempt)
                self._log(
                    f"Validation failed on attempt {attempt}/{self.max_retries} for '{format_name}'. "
                    f"Errors: {error_details}. Retrying...",
                    level="warning",
                )

                await asyncio.sleep(0.5 * attempt)

            except Exception as e:
                raise RuntimeError(
                    f"Unexpected error during formatting '{format_name}' on attempt {attempt}: {e}"
                ) from e

        raise RuntimeError(
            f"Unexpected end of retry loop for '{format_name}'. "
            f"Last validation error: {self._format_validation_errors(last_error) if last_error else 'None'}"
        )

    def _compile_findings(
        self, worker_results: list[dict[str, Any]], citations_enabled: bool = True
    ) -> str:
        """Compile worker findings into text for artifact creation.

        Args:
            worker_results: List of worker result dicts
            citations_enabled: Whether to include citation markers

        Returns:
            Compiled findings text
        """
        findings_parts = []

        for idx, result in enumerate(worker_results):
            if not result.get("success", False):
                continue

            output = result.get("output", "")
            task = result.get("task", f"Task {idx + 1}")
            sources = result.get("sources", [])

            findings_parts.append(f"## {task}\n\n{output}")

            if citations_enabled and sources:
                citations = "\n".join(
                    f"[{i + 1}] {src.get('title', 'Unknown')}: {src.get('url', 'Unknown')}"
                    for i, src in enumerate(sources)
                )
                findings_parts.append(f"\n**Sources:**\n{citations}\n")

        return "\n\n".join(findings_parts)

    def _load_schema_from_prompt_name(self, format_name: str) -> type[BaseModel]:
        """Dynamically load Pydantic schema from format directory.

        Args:
            format_name: Format name (research_report, markdown, structured_json)

        Returns:
            Pydantic BaseModel class for the format
        """
        # Map format names to schema modules
        schema_map = {
            "research_report": "research_report.schema",
            "markdown": "markdown.schema",
            "structured_json": "structured_json.schema",
            "podcast_script": "podcast_script.schema",
            "substack_article": "substack_article.schema",
            "twitter_thread": "twitter_thread.schema",
            "skill": "skill.schema",
            "llms_txt": "llms_txt.schema",
            "code": "code.schema",
        }

        schema_module = schema_map.get(format_name.lower())
        if not schema_module:
            available = list(schema_map.keys())
            raise ValueError(f"Schema not found for format '{format_name}'. Available: {available}")

        # Try to import from artifacts.formats package
        try:
            module = importlib.import_module(
                f".artifacts.formats.{format_name}.schema", package="deep_research_engine"
            )
        except (ImportError, ModuleNotFoundError):
            # Fallback: load from file path
            schema_path = self.formats_dir / format_name / "schema.py"
            if not schema_path.exists():
                raise ValueError(f"Schema file not found: {schema_path}")

            spec = importlib.util.spec_from_file_location(schema_module, schema_path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Cannot load schema from {schema_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        # Find the BaseModel class in the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseModel) and attr != BaseModel:
                return attr

        raise ValueError(f"No Pydantic BaseModel found in schema module {schema_module}")

    async def _create_llm(self, model_config: dict[str, Any]) -> Any:
        """Create LLM from model configuration.

        Routes directly to llama.cpp (bypasses orchestrator) to avoid
        the orchestrator restarting the container.
        """
        from langchain_openai import ChatOpenAI

        # Config uses "model" key, not "name"
        model_name = model_config.get("model") or model_config.get(
            "name", "qwen-35-35b-a3b-vision-fast"
        )

        # Point directly at llama.cpp — same as InfraClient.chat() and _make_chat_model()
        base_url = os.environ.get(
            "LLAMACPP_DIRECT_URL",
            "http://172.17.0.7:8001/v1/chat/completions",
        ).rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]

        api_model = os.environ.get(
            "LLAMACPP_DIRECT_MODEL",
            "gemma-4-26B-A4B-it-UD-IQ4_NL.gguf",
        )

        return ChatOpenAI(
            model=api_model,
            base_url=base_url,
            api_key="not-needed",
            temperature=0.3,
            max_tokens=16000,
        )

    async def _extract_structured_output(
        self, llm: Any, agent_output: str, pydantic_model: type[BaseModel]
    ) -> BaseModel:
        """Extract structured output from LLM response using structured output."""
        structured_llm = llm.with_structured_output(pydantic_model)
        return await structured_llm.ainvoke([{"role": "user", "content": agent_output}])

    def _format_validation_errors(self, error: ValidationError) -> str:
        """Extract readable error details from Pydantic ValidationError."""
        return "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in error.errors()
        )

    def _add_retry_feedback(self, prompt: str, errors: str, attempt: int) -> str:
        """Add validation feedback to prompt for retry."""
        return f"""{prompt}

[RETRY ATTEMPT {attempt}: Previous response had validation errors]
Errors found: {errors}
Please fix these issues and respond with valid JSON that matches the required schema."""
