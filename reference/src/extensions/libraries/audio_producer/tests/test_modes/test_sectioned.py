"""Test Sectioned Mode - Section-based two-phase quality pipeline.

SECTIONED mode works like OVERDRIVE but processes entire sections at once
instead of using semantic chunking.

Benefits:
- Fewer LLM calls (1 per section instead of multiple semantic chunks)
- Faster processing while maintaining quality
- Same "Subtle Director" prompting for quality
"""

from test_utility import run_pipeline_test, get_output_dir, load_test_input


async def test_sectioned_mode():
    """Test sectioned mode on the pipeline."""
    mode_name = "Sectioned Mode"
    quality_mode = "sectioned"

    # Load test input
    input_text = load_test_input("short")

    # Get output directory
    output_dir = get_output_dir("sectioned")

    # Run test
    result = await run_pipeline_test(
        mode_name=mode_name,
        quality_mode=quality_mode,
        input_text=input_text,
        output_dir=output_dir
    )

    return result
