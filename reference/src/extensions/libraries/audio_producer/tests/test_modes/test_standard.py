"""Test Standard (Fast) Mode - Rule-based chunking with emotion detection.

STANDARD (fast) mode uses rule-based chunking with single-pass LLM emotion detection.
No semantic chunking, no refinement phases.

Benefits:
- Fastest processing
- Simple architecture
- Good baseline quality
"""

from test_utility import run_pipeline_test, get_output_dir, load_test_input


async def test_standard_mode():
    """Test standard (fast) mode on the pipeline."""
    mode_name = "Standard (Fast) Mode"
    quality_mode = "fast"

    # Load test input
    input_text = load_test_input("short")

    # Get output directory
    output_dir = get_output_dir("fast")

    # Run test
    result = await run_pipeline_test(
        mode_name=mode_name,
        quality_mode=quality_mode,
        input_text=input_text,
        output_dir=output_dir
    )

    return result
