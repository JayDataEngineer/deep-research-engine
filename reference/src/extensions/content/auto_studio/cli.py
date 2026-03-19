"""CLI commands for Auto Studio extension."""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def register(parser):
    """Register Auto Studio CLI commands.

    Usage:
        # Config mode (Deep Agent planning)
        uv run backend/manage.py studio plan --input script.md --output plan.json
        uv run backend/manage.py studio generate --input script.md --output video.mp4
        uv run backend/manage.py studio generate-from-plan --plan plan.json --output video.mp4

        # Fast mode (Real-time conversational)
        uv run backend/manage.py studio fast --message "Hello, how are you?" --output video.mp4
        uv run backend/manage.py studio fast-prompt --message "Hello, how are you?" --output prompt.json
    """
    subparsers = parser.add_subparsers(dest="studio_command", help="Auto Studio commands")

    # Plan command (Config mode)
    plan_parser = subparsers.add_parser("plan", help="Plan video podcast production using Deep Agent (Config mode)")
    plan_parser.add_argument("--input", "-i", required=True, help="Input markdown script file")
    plan_parser.add_argument("--base-scene", "-b", help="Path to base studio scene image")
    plan_parser.add_argument("--output", "-o", required=True, help="Output plan JSON file")
    plan_parser.add_argument("--markdown", "-m", help="Also save markdown plan to this file")
    plan_parser.add_argument("--context", "-c", help="JSON file with additional context")
    plan_parser.set_defaults(func=cmd_plan)

    # Generate command (Config mode)
    gen_parser = subparsers.add_parser("generate", help="Generate video podcast from script (Config mode)")
    gen_parser.add_argument("--input", "-i", required=True, help="Input markdown script file")
    gen_parser.add_argument("--base-scene", "-b", help="Path to base studio scene image")
    gen_parser.add_argument("--plan", "-p", help="Use pre-generated plan JSON file")
    gen_parser.add_argument("--output", "-o", required=True, help="Output video file")
    gen_parser.add_argument("--preset", choices=["draft", "standard", "premium"], default="standard",
                           help="Quality preset (default: standard)")
    gen_parser.add_argument("--context", "-c", help="JSON file with additional context")
    gen_parser.set_defaults(func=cmd_generate)

    # Generate from plan command (Config mode)
    gen_plan_parser = subparsers.add_parser("generate-from-plan", help="Generate video from pre-generated plan (Config mode)")
    gen_plan_parser.add_argument("--plan", "-p", required=True, help="Plan JSON file")
    gen_plan_parser.add_argument("--output", "-o", required=True, help="Output video file")
    gen_plan_parser.add_argument("--preset", choices=["draft", "standard", "premium"], default="standard",
                                  help="Quality preset (default: standard)")
    gen_plan_parser.set_defaults(func=cmd_generate_from_plan)

    # Fast mode command - generate video from user message
    fast_parser = subparsers.add_parser("fast", help="Generate video from user message in real-time (Fast mode)")
    fast_parser.add_argument("--message", "-m", required=True, help="User message to convert to video")
    fast_parser.add_argument("--base-scene", "-b", help="Path to base studio scene image")
    fast_parser.add_argument("--output", "-o", required=True, help="Output video file")
    fast_parser.add_argument("--context", "-c", help="JSON file with additional context")
    fast_parser.add_argument("--save-prompt", "-p", help="Save generated prompt to this file")
    fast_parser.set_defaults(func=cmd_fast)

    # Fast mode command - generate prompt only
    fast_prompt_parser = subparsers.add_parser("fast-prompt", help="Generate LTX2 prompt from user message (Fast mode)")
    fast_prompt_parser.add_argument("--message", "-m", required=True, help="User message to convert to LTX2 prompt")
    fast_prompt_parser.add_argument("--output", "-o", required=True, help="Output prompt JSON file")
    fast_prompt_parser.add_argument("--context", "-c", help="JSON file with additional context")
    fast_prompt_parser.set_defaults(func=cmd_fast_prompt)


async def cmd_plan(args):
    """Plan video podcast production."""
    from src.extensions.content.auto_studio.extension import AutoStudioExtension
    from src.extensions.libraries.auto_studio.functions.planning.deep_agent_planner import DeepAgentPlanner
    from src.extensions.libraries.auto_studio.functions.planning.json_to_markdown import plan_to_markdown
    from src.extensions.libraries.auto_studio.core.prompt_manager import ConfigPromptManager
    import yaml

    # Load script
    script_path = Path(args.input)
    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        return

    script = script_path.read_text()

    # Load context if provided
    context = {}
    if args.context:
        context_path = Path(args.context)
        if context_path.exists():
            context = json.loads(context_path.read_text())

    # Load extension config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get base scene path
    base_scene_path = Path(args.base_scene) if args.base_scene else None
    if base_scene_path is None:
        base_scene_path = Path(config.get("base_scene", {}).get("path", "data/inputs/base_studio_scene.png"))

    if not base_scene_path.exists():
        logger.error(f"Base scene not found at {base_scene_path}. Please create a base studio scene image or provide --base-scene path.")
        return

    # Initialize prompt manager and planner
    prompt_manager = ConfigPromptManager(config)
    planner = DeepAgentPlanner(
        config=config.get("deep_agent", {}),
        prompt_manager=prompt_manager
    )

    # Add base scene description to context
    if "base_scene_description" not in context:
        context["base_scene_description"] = config.get("base_scene", {}).get("description", "")

    # Plan
    logger.info("Planning video podcast production...")
    plan = await planner.plan_video_podcast(
        script=script,
        base_scene_path=str(base_scene_path),
        context=context,
    )

    # Convert to markdown
    markdown = plan_to_markdown(plan.model_dump())

    # Save plan JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan.model_dump(), indent=2))
    logger.info(f"Plan saved to: {output_path}")

    # Save markdown if requested
    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown)
        logger.info(f"Markdown plan saved to: {markdown_path}")

    # Print summary
    num_clips = len(plan.clips)
    logger.info(f"Planning complete! Generated {num_clips} clips.")


async def cmd_generate(args):
    """Generate video podcast from script."""
    from src.extensions.core.context import get_extension_context

    # Load script
    script_path = Path(args.input)
    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        return

    script = script_path.read_text()

    # Load context if provided
    context = {}
    if args.context:
        context_path = Path(args.context)
        if context_path.exists():
            context = json.loads(context_path.read_text())

    # Get extension
    ext_context = get_extension_context()
    auto_studio = ext_context.get_extension("auto_studio")

    if auto_studio is None:
        logger.error("Auto Studio extension not found")
        return

    # Get base scene path
    base_scene_path = Path(args.base_scene) if args.base_scene else None

    # Load plan if provided
    plan = None
    if args.plan:
        plan_path = Path(args.plan)
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            logger.info(f"Using pre-generated plan from: {plan_path}")

    # Generate
    logger.info(f"Generating video podcast (preset: {args.preset})...")
    async for update in auto_studio.generate_video_podcast(
        script=script,
        output_path=Path(args.output),
        plan=plan,
    ):
        data = json.loads(update)
        if data.get("status") == "planning":
            logger.info("Planning video production...")
        elif data.get("status") == "planned":
            logger.info(f"Planning complete! Generated {len(data['plan']['clips'])} clips")
        elif data.get("status") == "processing":
            logger.info(data.get("message", "Processing..."))
        elif data.get("status") == "complete":
            logger.info(f"Generation complete! Video saved to: {args.output}")


async def cmd_generate_from_plan(args):
    """Generate video from pre-generated plan."""
    from src.extensions.core.context import get_extension_context

    # Load plan
    plan_path = Path(args.plan)
    if not plan_path.exists():
        logger.error(f"Plan file not found: {plan_path}")
        return

    plan = json.loads(plan_path.read_text())
    logger.info(f"Loaded plan with {len(plan.get('clips', []))} clips")

    # Get extension
    ext_context = get_extension_context()
    auto_studio = ext_context.get_extension("auto_studio")

    if auto_studio is None:
        logger.error("Auto Studio extension not found")
        return

    # Generate (script not needed if plan is complete)
    script = plan.get("script", "")

    # Generate
    logger.info(f"Generating video from plan (preset: {args.preset})...")
    async for update in auto_studio.generate_video_podcast(
        script=script,
        output_path=Path(args.output),
        plan=plan,
    ):
        data = json.loads(update)
        if data.get("status") == "processing":
            logger.info(data.get("message", "Processing..."))
        elif data.get("status") == "complete":
            logger.info(f"Generation complete! Video saved to: {args.output}")


async def cmd_fast_prompt(args):
    """Generate LTX2 prompt from user message (Fast mode)."""
    from src.extensions.content.auto_studio.extension import AutoStudioExtension
    import yaml

    # Load context if provided
    context = {}
    if args.context:
        context_path = Path(args.context)
        if context_path.exists():
            context = json.loads(context_path.read_text())

    # Load extension config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Add base scene description to context
    if "base_scene_description" not in context:
        context["base_scene_description"] = config.get("base_scene", {}).get("description", "")

    # Initialize extension
    extension_path = Path(__file__).parent
    auto_studio = AutoStudioExtension(config, extension_path)

    # Generate prompt
    logger.info(f"Generating LTX2 prompt from message: '{args.message[:50]}...'")
    result = await auto_studio.generate_fast_prompt(
        user_message=args.message,
        context=context,
    )

    # Save prompt
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result["prompt"], indent=2))
    logger.info(f"Prompt saved to: {output_path}")

    # Print summary
    prompt_data = result["prompt"]
    logger.info(f"LTX2 Prompt: {prompt_data['ltx2_prompt'][:100]}...")
    logger.info(f"Duration: {prompt_data['duration_frames']} frames ({prompt_data['duration_frames']/24:.1f}s @ 24fps)")
    logger.info(f"Camera Angle: {prompt_data['camera_angle']}")


async def cmd_fast(args):
    """Generate video from user message (Fast mode)."""
    from src.extensions.core.context import get_extension_context

    # Load context if provided
    context = {}
    if args.context:
        context_path = Path(args.context)
        if context_path.exists():
            context = json.loads(context_path.read_text())

    # Get extension
    ext_context = get_extension_context()
    auto_studio = ext_context.get_extension("auto_studio")

    if auto_studio is None:
        logger.error("Auto Studio extension not found")
        return

    # Get base scene path
    base_scene_path = Path(args.base_scene) if args.base_scene else None

    # Generate video
    logger.info(f"Generating video from message: '{args.message[:50]}...'")
    async for update in auto_studio.generate_fast_video(
        user_message=args.message,
        base_scene_path=base_scene_path,
        output_path=Path(args.output),
        context=context,
    ):
        data = json.loads(update)
        if data.get("status") == "generating_prompt":
            logger.info("Generating LTX2 prompt...")
        elif data.get("status") == "prompt_ready":
            prompt_data = data["prompt"]
            logger.info(f"Prompt ready: {prompt_data['ltx2_prompt'][:100]}...")
            logger.info(f"Duration: {prompt_data['duration_frames']} frames ({prompt_data['duration_frames']/24:.1f}s @ 24fps)")

            # Save prompt if requested
            if args.save_prompt:
                prompt_path = Path(args.save_prompt)
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                prompt_path.write_text(json.dumps(prompt_data, indent=2))
                logger.info(f"Prompt saved to: {prompt_path}")
        elif data.get("status") == "complete":
            logger.info(f"Generation complete! Video saved to: {args.output}")
