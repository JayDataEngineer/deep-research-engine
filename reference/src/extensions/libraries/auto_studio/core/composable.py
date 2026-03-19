"""Composable Functions for Auto Studio.

Provides a scikit-learn-style pipeline system for chaining together
multi-modal generation functions.

Each function is independent and can be:
- Used standalone
- Chained in a Pipeline
- Checkpointed and resumed

Example:
    pipeline = Pipeline([
        ("qwen_edit", QwenImageEditFunction(config)),
        ("ltx2_video", LTX2VideoFunction(config)),
    ])

    result = await pipeline.run(input_data)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FunctionResult:
    """Result from a composable function execution.

    Attributes:
        success: Whether the function completed successfully
        data: The output data (if successful)
        error: Error message (if failed)
        checkpoint_data: Data for checkpointing/resuming
        metadata: Additional metadata about the execution
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    checkpoint_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionInput:
    """Input to a composable function.

    Attributes:
        clip_id: Identifier for the clip being processed
        plan_data: The relevant section of the VideoPodcastPlan
        input_paths: Paths to input files (images, audio, etc.)
        context: Additional context for execution
        config: Function-specific configuration overrides
    """
    clip_id: str
    plan_data: Dict[str, Any]
    input_paths: Dict[str, Path] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


class BaseComposableFunction(ABC):
    """Base class for composable generation functions.

    Similar to sklearn's Transformer API but designed for:
    - Multi-modal generation (image, video, audio)
    - GPU-intensive operations that can't run simultaneously
    - Checkpointing and resuming
    - Progress streaming

    Each function must implement:
    - validate_input(): Check if inputs are valid
    - execute(): Run the actual generation
    - get_output_paths(): Return paths to generated files
    """

    # Function metadata
    name: str = "base_function"
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    requires_gpu: bool = True

    def __init__(self, config: Dict[str, Any]):
        """Initialize the function with configuration.

        Args:
            config: Function configuration (from extension config.yaml)
        """
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate required config parameters. Override in subclasses."""
        pass

    @abstractmethod
    async def validate_input(self, input_data: FunctionInput) -> Tuple[bool, Optional[str]]:
        """Validate input data before execution.

        Args:
            input_data: The input to validate

        Returns:
            (is_valid, error_message) tuple
        """
        pass

    @abstractmethod
    async def execute(self, input_data: FunctionInput) -> FunctionResult:
        """Execute the generation function.

        Args:
            input_data: The input data

        Returns:
            FunctionResult with output data and paths
        """
        pass

    def get_output_paths(self, result: FunctionResult) -> Dict[str, Path]:
        """Extract output file paths from result.

        Args:
            result: The function result

        Returns:
            Dict mapping output names to Paths
        """
        if result.data and "output_paths" in result.data:
            return {k: Path(v) for k, v in result.data["output_paths"].items()}
        return {}

    async def run(self, input_data: FunctionInput) -> FunctionResult:
        """Run the function with validation.

        Args:
            input_data: Input data

        Returns:
            FunctionResult
        """
        logger.info(f"[{self.name}] Validating input for clip {input_data.clip_id}")

        # Validate input
        is_valid, error_msg = await self.validate_input(input_data)
        if not is_valid:
            logger.error(f"[{self.name}] Input validation failed: {error_msg}")
            return FunctionResult(
                success=False,
                error=f"Input validation failed: {error_msg}",
                metadata={"clip_id": input_data.clip_id}
            )

        # Execute
        logger.info(f"[{self.name}] Executing for clip {input_data.clip_id}")
        try:
            result = await self.execute(input_data)
            if result.success:
                logger.info(f"[{self.name}] Completed clip {input_data.clip_id}")
            else:
                logger.error(f"[{self.name}] Failed clip {input_data.clip_id}: {result.error}")
            return result
        except Exception as e:
            logger.exception(f"[{self.name}] Exception during execution: {e}")
            return FunctionResult(
                success=False,
                error=f"Execution exception: {str(e)}",
                metadata={"clip_id": input_data.clip_id}
            )


class Pipeline:
    """Pipeline for chaining composable functions.

    Similar to sklearn.pipeline.Pipeline but for async multi-modal generation.

    Functions run sequentially and can't run simultaneously due to GPU constraints.
    Each function's output becomes the next function's input.

    Example:
        pipeline = Pipeline([
            ("qwen_edit", QwenImageEditFunction(config)),
            ("ltx2_video", LTX2VideoFunction(config)),
        ])

        result = await pipeline.run(input_data)

    The pipeline maintains state between steps, allowing intermediate results
    to be passed forward.
    """

    def __init__(
        self,
        steps: List[Tuple[str, BaseComposableFunction]],
        checkpoint_dir: Optional[Path] = None
    ):
        """Initialize the pipeline.

        Args:
            steps: List of (name, function) tuples
            checkpoint_dir: Optional directory for checkpointing
        """
        self.steps = steps
        self.checkpoint_dir = checkpoint_dir
        self._state: Dict[str, Any] = {}

    @property
    def step_names(self) -> List[str]:
        """Get names of pipeline steps."""
        return [name for name, _ in self.steps]

    def get_step(self, name: str) -> Optional[BaseComposableFunction]:
        """Get a pipeline step by name."""
        for step_name, step_func in self.steps:
            if step_name == name:
                return step_func
        return None

    async def run(
        self,
        input_data: FunctionInput,
        resume_from: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> FunctionResult:
        """Run the pipeline.

        Args:
            input_data: Initial input data
            resume_from: Optional step name to resume from (for checkpointing)
            progress_callback: Optional callback for progress updates

        Returns:
            Final FunctionResult with all outputs
        """
        logger.info(f"[Pipeline] Starting pipeline with {len(self.steps)} steps")
        logger.info(f"[Pipeline] Steps: {', '.join(self.step_names)}")

        # Find resume index if specified
        start_idx = 0
        if resume_from:
            start_idx = self.step_names.index(resume_from)
            logger.info(f"[Pipeline] Resuming from step: {resume_from}")

        current_data = input_data
        results: Dict[str, FunctionResult] = {}

        # Run each step
        for step_name, step_func in self.steps[start_idx:]:
            logger.info(f"[Pipeline] Running step: {step_name}")

            # Update context with previous results
            current_data.context["_pipeline_results"] = results
            current_data.context["_current_step"] = step_name

            # Run the step
            result = await step_func.run(current_data)
            results[step_name] = result

            # Progress callback
            if progress_callback:
                await progress_callback(step_name, result)

            # Check for failure
            if not result.success:
                logger.error(f"[Pipeline] Step {step_name} failed: {result.error}")
                return FunctionResult(
                    success=False,
                    error=f"Pipeline failed at step '{step_name}': {result.error}",
                    checkpoint_data={"results": results},
                    metadata={"failed_step": step_name, "results": results}
                )

            # Prepare input for next step
            # Pass through original input + add outputs from this step
            current_data = self._prepare_next_step_input(current_data, result)

            # Checkpoint if enabled
            if self.checkpoint_dir:
                self._save_checkpoint(step_name, result)

        logger.info("[Pipeline] All steps completed successfully")

        # Combine all outputs
        all_output_paths = {}
        for step_result in results.values():
            if step_result.data and "output_paths" in step_result.data:
                all_output_paths.update(step_result.data["output_paths"])

        return FunctionResult(
            success=True,
            data={
                "output_paths": all_output_paths,
                "step_results": {k: v.data for k, v in results.items()}
            },
            checkpoint_data={"results": results},
            metadata={"completed_steps": list(results.keys())}
        )

    def _prepare_next_step_input(
        self,
        original_input: FunctionInput,
        previous_result: FunctionResult
    ) -> FunctionInput:
        """Prepare input for the next pipeline step.

        Args:
            original_input: The original pipeline input
            previous_result: Result from the previous step

        Returns:
            New FunctionInput for the next step
        """
        # Update input_paths with outputs from previous step
        new_paths = original_input.input_paths.copy()
        if previous_result.data and "output_paths" in previous_result.data:
            for key, path in previous_result.data["output_paths"].items():
                new_paths[key] = Path(path)

        return FunctionInput(
            clip_id=original_input.clip_id,
            plan_data=original_input.plan_data,
            input_paths=new_paths,
            context=original_input.context,
            config=original_input.config
        )

    def _save_checkpoint(self, step_name: str, result: FunctionResult):
        """Save a checkpoint after a step completes.

        Args:
            step_name: Name of the completed step
            result: Result from the step
        """
        if not self.checkpoint_dir:
            return

        checkpoint_path = self.checkpoint_dir / f"{step_name}_checkpoint.json"
        # TODO: Implement checkpoint serialization
        logger.debug(f"[Pipeline] Checkpoint saved: {checkpoint_path}")
