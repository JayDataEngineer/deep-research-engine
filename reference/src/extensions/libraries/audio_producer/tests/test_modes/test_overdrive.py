"""Test Overdrive Mode - Semantic chunking two-phase quality pipeline.

OVERDRIVE mode uses semantic chunking to group lines into emotional beats,
then refines each chunk with "Subtle Director" prompting.

Benefits:
- Semantic-aware emotion detection
- Better emotional continuity
- High quality with targeted refinement
"""

from test_utility import run_pipeline_test, get_output_dir, load_test_input


async def test_overdrive_mode():
    """Test overdrive mode on the pipeline."""
    mode_name = "Overdrive Mode"
    quality_mode = "overdrive"

    # Load test input
    input_text = load_test_input("short")

    # Get output directory
    output_dir = get_output_dir("overdrive")

    # Run test
    result = await run_pipeline_test(
        mode_name=mode_name,
        quality_mode=quality_mode,
        input_text=input_text,
        output_dir=output_dir
    )

    return result
