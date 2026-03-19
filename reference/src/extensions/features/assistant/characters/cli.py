"""Characters CLI Plugin

Provides commands for managing character cards.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all character cards."""
    from src.extensions.features.assistant.characters.manager import CharacterManager
    
    try:
        manager = CharacterManager()
        characters = manager.list_characters()
        
        print(f"\nCharacters ({len(characters)} found):")
        for char in characters:
            print(f"  - {char}")
        
        return {"characters": characters}
    except Exception as e:
        print(f"Error listing characters: {e}")
        return {"error": str(e)}


def cmd_import(args):
    """Import a character card."""
    from src.extensions.features.assistant.characters.manager import CharacterManager
    
    try:
        manager = CharacterManager()
        
        # Determine file type based on extension
        file_path = Path(args.file)
        if file_path.suffix.lower() in ['.png']:
            # Handle PNG character cards
            success = manager.import_png_character(str(file_path))
        else:
            # Handle JSON/Tavern character cards
            with open(args.file, 'r', encoding='utf-8') as f:
                char_data = f.read()
            success = manager.import_tavern_character(char_data)
        
        if success:
            print(f"Imported character from {args.file}")
            return {"success": True}
        else:
            print(f"Failed to import character from {args.file}")
            return {"error": "Import failed"}
    except Exception as e:
        print(f"Error importing character: {e}")
        return {"error": str(e)}


def cmd_export(args):
    """Export a character card."""
    from src.extensions.features.assistant.characters.manager import CharacterManager
    
    try:
        manager = CharacterManager()
        char_data = manager.get_character(args.character_name)
        
        if char_data:
            output_file = args.output or f"{args.character_name}_export.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(char_data)
            
            print(f"Exported character '{args.character_name}' to {output_file}")
            return {"success": True, "output": output_file}
        else:
            print(f"Character '{args.character_name}' not found")
            return {"error": "Character not found"}
    except Exception as e:
        print(f"Error exporting character: {e}")
        return {"error": str(e)}


def register(parser):
    """Register character commands with the parser."""
    subparsers = parser.add_subparsers(dest="character_subcommand", help="Character commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all characters")
    list_parser.set_defaults(func=cmd_list)
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import a character card")
    import_parser.add_argument("file", help="Character file to import (JSON, Tavern, or PNG)")
    import_parser.set_defaults(func=cmd_import)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a character card")
    export_parser.add_argument("character_name", help="Name of the character to export")
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.set_defaults(func=cmd_export)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Characters CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()