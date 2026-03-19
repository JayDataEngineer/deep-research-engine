"""ComfyUI CLI Plugin

Provides commands for managing ComfyUI workflows and image generation.

Usage:
    uv run backend/manage.py comfyui workflows
    uv run backend/manage.py comfyui generate "<prompt>" --workflow z_image_turbo
    uv run backend/manage.py comfyui edit <image_path> "<prompt>"
    uv run backend/manage.py comfyui models
    uv run backend/manage.py comfyui status
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def get_manager():
    """Get the ComfyUIManager instance."""
    from src.extensions.libraries.comfyui.manager import ComfyUIManager
    return ComfyUIManager()


def cmd_workflows(args):
    """List available workflows."""
    manager = get_manager()

    workflows = manager.list_workflows()

    if not workflows:
        print("\nNo workflows found.")
        print(f"  Workflows directory: {manager.workflows_dir}")
        return {"workflows": []}

    print(f"\nAvailable Workflows ({len(workflows)}):")
    print("-" * 60)

    for wf in workflows:
        print(f"\n  {wf['name']}")
        print(f"    Type: {wf['type']}")
        print(f"    Nodes: {wf['nodes']}")
        print(f"    Path: {wf['path']}")

    return {"workflows": workflows}


def cmd_generate(args):
    """Generate an image."""
    manager = get_manager()

    print(f"\nGenerating image...")
    print(f"  Prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
    print(f"  Workflow: {args.workflow}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  Steps: {args.steps}")

    result = manager.generate(
        prompt=args.prompt,
        workflow=args.workflow,
        width=args.width,
        height=args.height,
        steps=args.steps,
        negative_prompt=args.negative or "",
        seed=args.seed,
        output_path=args.output,
    )

    if result.get("success"):
        print(f"\n  Image generated successfully!")
        print(f"  Output: {result['output_path']}")
        print(f"  Size: {result['size']} bytes")
    else:
        print(f"\n  Generation failed: {result.get('error')}")

    return result


def cmd_edit(args):
    """Edit an image."""
    manager = get_manager()

    print(f"\nEditing image...")
    print(f"  Source: {args.image}")
    print(f"  Prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
    print(f"  Strength: {args.strength}")

    result = manager.edit_image(
        image_path=args.image,
        prompt=args.prompt,
        strength=args.strength,
        workflow=args.workflow,
        output_path=args.output,
    )

    if result.get("success"):
        print(f"\n  Image edited successfully!")
        print(f"  Output: {result['output_path']}")
        print(f"  Size: {result['size']} bytes")
    else:
        print(f"\n  Edit failed: {result.get('error')}")

    return result


def cmd_models(args):
    """List available models."""
    manager = get_manager()

    models = manager.list_models()

    print("\nAvailable Models:")
    print("-" * 60)

    for model_type, model_list in models.items():
        print(f"\n{model_type.upper()} ({len(model_list)}):")
        if model_list:
            for model in sorted(model_list)[:10]:
                print(f"  - {model}")
            if len(model_list) > 10:
                print(f"  ... and {len(model_list) - 10} more")
        else:
            print(f"  (none found)")

    return {"models": models}


def cmd_status(args):
    """Check ComfyUI server status."""
    manager = get_manager()

    status = manager.get_status()

    print(f"\nComfyUI Status:")
    print("-" * 40)
    print(f"  URL: {status['url']}")

    if status.get("connected"):
        print(f"  Status: Connected")

        if "system" in status:
            sys_info = status["system"]
            if "devices" in sys_info:
                print(f"\n  Devices:")
                for device in sys_info["devices"]:
                    print(f"    - {device.get('name', 'unknown')}")
                    if "vram_total" in device:
                        vram_gb = device["vram_total"] / (1024**3)
                        print(f"      VRAM: {vram_gb:.1f} GB")
    else:
        print(f"  Status: Disconnected")
        if "error" in status:
            print(f"  Error: {status['error']}")
        print(f"\n  Hint: Make sure ComfyUI is running at {status['url']}")

    return status


def cmd_outputs(args):
    """List recent generated images."""
    manager = get_manager()

    outputs = manager.list_outputs(limit=args.limit)

    if not outputs:
        print("\nNo generated images found.")
        print(f"  Outputs directory: {manager.outputs_dir}")
        return {"outputs": []}

    print(f"\nRecent Generated Images ({len(outputs)}):")
    print("-" * 60)

    for out in outputs:
        size_kb = out["size"] / 1024
        print(f"\n  {out['name']}")
        print(f"    Path: {out['path']}")
        print(f"    Size: {size_kb:.1f} KB")

    return {"outputs": outputs}


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(dest="comfyui_subcommand", help="ComfyUI commands")

    # workflows command
    workflows_parser = subparsers.add_parser("workflows", help="List available workflows")
    workflows_parser.set_defaults(func=cmd_workflows)

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate an image")
    gen_parser.add_argument("prompt", help="Image description prompt")
    gen_parser.add_argument("--workflow", "-w", default="z_image_turbo", help="Workflow name")
    gen_parser.add_argument("--width", type=int, default=1024, help="Image width")
    gen_parser.add_argument("--height", type=int, default=1024, help="Image height")
    gen_parser.add_argument("--steps", "-s", type=int, default=9, help="Sampling steps")
    gen_parser.add_argument("--negative", "-n", help="Negative prompt")
    gen_parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    gen_parser.add_argument("--output", "-o", help="Output file path")
    gen_parser.set_defaults(func=cmd_generate)

    # edit command
    edit_parser = subparsers.add_parser("edit", help="Edit an image")
    edit_parser.add_argument("image", help="Source image path")
    edit_parser.add_argument("prompt", help="Edit instruction prompt")
    edit_parser.add_argument("--strength", type=float, default=1.0, help="Edit strength (0-1)")
    edit_parser.add_argument("--workflow", "-w", default="qwen_gguf_api", help="Workflow name")
    edit_parser.add_argument("--output", "-o", help="Output file path")
    edit_parser.set_defaults(func=cmd_edit)

    # models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.set_defaults(func=cmd_models)

    # status command
    status_parser = subparsers.add_parser("status", help="Check ComfyUI server status")
    status_parser.set_defaults(func=cmd_status)

    # outputs command
    outputs_parser = subparsers.add_parser("outputs", help="List recent generated images")
    outputs_parser.add_argument("--limit", "-l", type=int, default=20, help="Max items")
    outputs_parser.set_defaults(func=cmd_outputs)
