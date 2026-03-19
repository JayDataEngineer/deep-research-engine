"""Chat History CLI Plugin

Provides commands for managing chat history.
"""

import argparse
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List chat histories."""
    from src.extensions.features.assistant.chat_history.manager import ChatHistoryManager
    
    try:
        manager = ChatHistoryManager()
        histories = manager.list_histories()
        
        print(f"\nChat Histories ({len(histories)} found):")
        for hist in histories:
            print(f"  - {hist}")
        
        return {"histories": histories}
    except Exception as e:
        print(f"Error listing chat histories: {e}")
        return {"error": str(e)}


def cmd_export(args):
    """Export a chat history."""
    from src.extensions.features.assistant.chat_history.manager import ChatHistoryManager
    
    try:
        manager = ChatHistoryManager()
        history_data = manager.export_history(args.history_name)
        
        if history_data:
            output_file = args.output or f"{args.history_name}_export.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            print(f"Exported chat history '{args.history_name}' to {output_file}")
            return {"success": True, "output": output_file}
        else:
            print(f"Chat history '{args.history_name}' not found")
            return {"error": "History not found"}
    except Exception as e:
        print(f"Error exporting chat history: {e}")
        return {"error": str(e)}


def cmd_import(args):
    """Import a chat history."""
    from src.extensions.features.assistant.chat_history.manager import ChatHistoryManager
    
    try:
        manager = ChatHistoryManager()
        
        with open(args.file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        
        success = manager.import_history(args.history_name, history_data)
        
        if success:
            print(f"Imported chat history '{args.history_name}' from {args.file}")
            return {"success": True}
        else:
            print(f"Failed to import chat history '{args.history_name}'")
            return {"error": "Import failed"}
    except Exception as e:
        print(f"Error importing chat history: {e}")
        return {"error": str(e)}


def register(parser):
    """Register chat history commands with the parser."""
    subparsers = parser.add_subparsers(dest="chat_subcommand", help="Chat history commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all chat histories")
    list_parser.set_defaults(func=cmd_list)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a chat history")
    export_parser.add_argument("history_name", help="Name of the history to export")
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.set_defaults(func=cmd_export)
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import a chat history")
    import_parser.add_argument("history_name", help="Name for the imported history")
    import_parser.add_argument("file", help="Input file to import")
    import_parser.set_defaults(func=cmd_import)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Chat History CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()