"""ComfyUI CLI Commands

Provides CLI commands for ComfyUI extensions through the manage.py interface.
"""

import sys


def register(parser):
    """Register ComfyUI CLI commands.

    Args:
        parser: ArgumentParser subparsers object
    """
    comfy_parser = parser.add_parser("comfyui", help="ComfyUI image generation and editing")
    comfy_subparsers = comfy_parser.add_subparsers(dest="comfyui_command")

    # Generate image
    generate_cmd = comfy_subparsers.add_parser("generate", help="Generate an image from text")
    generate_cmd.add_argument("prompt", help="Image description/prompt")
    generate_cmd.add_argument("-w", "--width", type=int, default=1024, help="Image width")
    generate_cmd.add_argument("-h", "--height", type=int, default=1024, help="Image height")
    generate_cmd.add_argument("--steps", type=int, default=9, help="Sampling steps")
    generate_cmd.add_argument("--seed", type=int, default=-1, help="Random seed")
    generate_cmd.add_argument("-o", "--output", help="Output file path")
    generate_cmd.set_defaults(func=generate_image)

    # Edit image
    edit_cmd = comfy_subparsers.add_parser("edit", help="Edit an image with a prompt")
    edit_cmd.add_argument("image", help="Path to source image")
    edit_cmd.add_argument("prompt", help="Edit instruction")
    edit_cmd.add_argument("-s", "--strength", type=float, default=0.5, help="Edit strength (0-1)")
    edit_cmd.add_argument("-o", "--output", help="Output file path")
    edit_cmd.set_defaults(func=edit_image)

    # Server status
    status_cmd = comfy_subparsers.add_parser("status", help="Check ComfyUI server status")
    status_cmd.set_defaults(func=get_status)


def generate_image(args):
    """Generate an image from text prompt.

    Args:
        args: Parsed arguments
    """
    print("=" * 60)
    print("ComfyUI Image Generation")
    print("=" * 60)
    print(f"\nPrompt: {args.prompt}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed if args.seed != -1 else 'random'}")
    print()

    # Import the engine
    from .shared import ComfyUIEngine
    from src.constants import PROJECT_ROOT

    engine = ComfyUIEngine(
        comfy_address="http://127.0.0.1:18188",
        workflow_path=PROJECT_ROOT / "data" / "inputs" / "workflows" / "image_z_image_turbo.json",
        outputs_dir=PROJECT_ROOT / "data" / "outputs" / "generated_images",
    )

    result = engine.generate(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        output_path=args.output,
    )

    if result.get("success"):
        print(f"\nSuccess!")
        print(f"Output: {result['output_path']}")
        print(f"Size: {result['size']} bytes")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)

    print("\n" + "=" * 60)


def edit_image(args):
    """Edit an image with a prompt.

    Args:
        args: Parsed arguments
    """
    print("=" * 60)
    print("ComfyUI Image Editing")
    print("=" * 60)
    print(f"\nImage: {args.image}")
    print(f"Prompt: {args.prompt}")
    print(f"Strength: {args.strength}")
    print()

    # Import the engine
    from .shared import ComfyUIEngine
    from src.constants import PROJECT_ROOT

    engine = ComfyUIEngine(
        comfy_address="http://127.0.0.1:18188",
        workflow_path=PROJECT_ROOT / "data" / "inputs" / "workflows" / "Qwen_GGUF_API.json",
        outputs_dir=PROJECT_ROOT / "data" / "outputs" / "generated_images",
    )

    result = engine.edit(
        image=args.image,
        prompt=args.prompt,
        strength=args.strength,
        output_path=args.output,
    )

    if result.get("success"):
        print(f"\nSuccess!")
        print(f"Output: {result['output_path']}")
        print(f"Size: {result['size']} bytes")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)

    print("\n" + "=" * 60)


def get_status(args):
    """Check ComfyUI server status.

    Args:
        args: Parsed arguments
    """
    print("=" * 60)
    print("ComfyUI Server Status")
    print("=" * 60)

    # Import the engine
    from .shared import ComfyUIEngine

    engine = ComfyUIEngine()
    status = engine.get_status()

    print(f"\nURL: {status['url']}")
    print(f"Connected: {'Yes' if status.get('connected') else 'No'}")

    if status.get("connected"):
        system = status.get("system", {})
        print(f"\nSystem Info:")
        if "devices" in system:
            print(f"  Devices: {', '.join(system['devices'])}")
        if "python_version" in system:
            print(f"  Python: {system['python_version']}")
        if "version" in system:
            print(f"  ComfyUI: {system['version']}")
    else:
        error = status.get("error", "Unknown error")
        print(f"\nError: {error}")
        print("\nMake sure ComfyUI server is running:")
        print("  make start-all    # Starts ComfyUI with other services")

    print("\n" + "=" * 60)
