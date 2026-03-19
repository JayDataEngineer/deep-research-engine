"""Instruct Mode CLI Plugin

Provides commands for managing instruct mode templates.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all instruct mode templates."""
    from src.extensions.features.instruct_mode.manager import InstructModeManager
    
    try:
        manager = InstructModeManager()
        templates = manager.list_templates()
        
        print(f"\nInstruct Mode Templates ({len(templates)} found):")
        for template in templates:
            print(f"  - {template}")
        
        return {"templates": templates}
    except Exception as e:
        print(f"Error listing instruct mode templates: {e}")
        return {"error": str(e)}


def cmd_set(args):
    """Set an instruct mode template."""
    from src.extensions.features.instruct_mode.manager import InstructModeManager
    
    try:
        manager = InstructModeManager()
        success = manager.set_template(args.template_name)
        
        if success:
            print(f"Set instruct mode template to '{args.template_name}'")
            return {"success": True}
        else:
            print(f"Failed to set instruct mode template to '{args.template_name}'")
            return {"error": "Set failed"}
    except Exception as e:
        print(f"Error setting instruct mode template: {e}")
        return {"error": str(e)}


def cmd_export(args):
    """Export an instruct mode template."""
    from src.extensions.features.instruct_mode.manager import InstructModeManager
    
    try:
        manager = InstructModeManager()
        template_data = manager.get_template(args.template_name)
        
        if template_data:
            output_file = args.output or f"{args.template_name}_template.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(template_data)
            
            print(f"Exported template '{args.template_name}' to {output_file}")
            return {"success": True, "output": output_file}
        else:
            print(f"Template '{args.template_name}' not found")
            return {"error": "Template not found"}
    except Exception as e:
        print(f"Error exporting template: {e}")
        return {"error": str(e)}


def register(parser):
    """Register instruct mode commands with the parser."""
    subparsers = parser.add_subparsers(dest="instruct_subcommand", help="Instruct mode commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all instruct mode templates")
    list_parser.set_defaults(func=cmd_list)
    
    # Set command
    set_parser = subparsers.add_parser("set", help="Set an instruct mode template")
    set_parser.add_argument("template_name", help="Name of the template to set")
    set_parser.set_defaults(func=cmd_set)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export an instruct mode template")
    export_parser.add_argument("template_name", help="Name of the template to export")
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.set_defaults(func=cmd_export)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Instruct Mode CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()