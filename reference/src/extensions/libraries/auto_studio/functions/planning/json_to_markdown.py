"""Convert VideoPodcastPlan JSON to human-readable Markdown."""

from typing import Any


def plan_to_markdown(plan: dict[str, Any]) -> str:
    """
    Convert plan JSON to human-readable markdown.

    Useful for:
    - Review and iteration
    - Documentation
    - Debugging

    Args:
        plan: VideoPodcastPlan as dict (from model_dump())

    Returns:
        Markdown string
    """
    lines = [
        "# Video Podcast Production Plan",
        "",
        "## Script Info",
    ]

    script_info = plan.get("script_info", {})
    lines.extend([
        f"- **Title**: {script_info.get('title', 'N/A')}",
        f"- **Total Lines**: {script_info.get('total_lines', 0)}",
        f"- **Number of Clips**: {script_info.get('num_clips', 0)}",
        f"- **Estimated Duration**: {script_info.get('estimated_duration_seconds', 0):.1f} seconds",
        "",
        f"## Base Scene",
        f"Reference: `{plan.get('base_scene_reference', 'N/A')}`",
        "",
        f"## Clips",
        f"",
    ])

    for clip in plan.get("clips", []):
        lines.extend([
            f"### {clip.get('clip_id', 'unknown')}",
            "",
            f"- **Lines**: {clip.get('start_line', '?')}-{clip.get('end_line', '?')}",
            f"- **Speakers**: {', '.join(clip.get('speakers', []))}",
            f"- **Duration**: {clip.get('duration_frames', 0)} frames (~{clip.get('duration_frames', 0)/24:.1f}s @ 24fps)",
            f"- **Camera**: {clip.get('camera_angle', 'N/A')}",
            f"- **Break Type**: {clip.get('camera_break_type', 'N/A')}",
            "",
            f"#### Emotions",
        ])

        # Format emotion vectors
        emotion_vectors = clip.get("emotion_vectors", {})
        for speaker, emotions in emotion_vectors.items():
            if isinstance(emotions, dict):
                e_str = ", ".join([f"{k}:{v:.2f}" for k, v in emotions.items()])
                lines.append(f"- **{speaker}**: {e_str}")

        # Qwen Edit prompt
        lines.extend([
            "",
            f"#### Qwen Image Edit Prompt",
            "```",
            clip.get("qwen_edit_prompt", ""),
            "```",
        ])

        if clip.get("qwen_negative_prompt"):
            lines.extend([
                f"**Negative**: {clip.get('qwen_negative_prompt')}",
                "",
            ])

        # LTX2 prompt
        lines.extend([
            f"#### LTX2 Video Prompt",
            "```",
            clip.get("ltx2_prompt", ""),
            "```",
        ])

        if clip.get("ltx2_negative_prompt"):
            lines.extend([
                f"**Negative**: {clip.get('ltx2_negative_prompt')}",
                "",
            ])

        # Additional params
        additional = clip.get("additional_params", {})
        if additional:
            lines.extend([
                f"#### Additional Parameters",
                "```json",
                str(additional),
                "```",
            ])

        lines.append("")  # Blank line between clips

    return "\n".join(lines)
