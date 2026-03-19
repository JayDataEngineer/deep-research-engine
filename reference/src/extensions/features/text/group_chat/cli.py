"""Group Chat CLI Plugin

Provides commands for managing group chats.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all group chats."""
    from src.extensions.features.text.group_chat.manager import GroupChatManager
    
    try:
        manager = GroupChatManager()
        groups = manager.list_groups()
        
        print(f"\nGroup Chats ({len(groups)} found):")
        for group in groups:
            print(f"  - {group}")
        
        return {"groups": groups}
    except Exception as e:
        print(f"Error listing group chats: {e}")
        return {"error": str(e)}


def cmd_create(args):
    """Create a new group chat."""
    from src.extensions.features.text.group_chat.manager import GroupChatManager
    
    try:
        manager = GroupChatManager()
        success = manager.create_group(args.name, args.members or [])
        
        if success:
            print(f"Created group chat '{args.name}' with members: {args.members or []}")
            return {"success": True}
        else:
            print(f"Failed to create group chat '{args.name}'")
            return {"error": "Creation failed"}
    except Exception as e:
        print(f"Error creating group chat: {e}")
        return {"error": str(e)}


def cmd_members(args):
    """Manage group chat members."""
    from src.extensions.features.text.group_chat.manager import GroupChatManager
    
    try:
        manager = GroupChatManager()
        
        if args.action == "add":
            success = manager.add_member(args.group_name, args.member)
            action_str = "added"
        elif args.action == "remove":
            success = manager.remove_member(args.group_name, args.member)
            action_str = "removed"
        elif args.action == "list":
            members = manager.get_members(args.group_name)
            print(f"Members of '{args.group_name}': {members}")
            return {"members": members}
        else:
            print(f"Unknown action: {args.action}")
            return {"error": "Unknown action"}
        
        if success:
            print(f"{action_str.capitalize()} member '{args.member}' to group '{args.group_name}'")
            return {"success": True}
        else:
            print(f"Failed to {action_str} member '{args.member}' to group '{args.group_name}'")
            return {"error": f"{action_str.capitalize()} failed"}
    except Exception as e:
        print(f"Error managing group chat members: {e}")
        return {"error": str(e)}


def cmd_chat(args):
    """Start a group chat."""
    from src.extensions.features.text.group_chat.manager import GroupChatManager
    
    try:
        manager = GroupChatManager()
        
        # Start the group chat
        print(f"Starting group chat '{args.group_name}'...")
        print("Enter messages (type 'quit' to exit):")
        
        while True:
            message = input(f"[{args.group_name}] > ")
            if message.lower() == 'quit':
                break
            
            # Send message to group
            result = manager.send_message(args.group_name, message)
            print(f"Message sent: {result}")
        
        return {"success": True}
    except Exception as e:
        print(f"Error starting group chat: {e}")
        return {"error": str(e)}


def register(parser):
    """Register group chat commands with the parser."""
    subparsers = parser.add_subparsers(dest="group_subcommand", help="Group chat commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all group chats")
    list_parser.set_defaults(func=cmd_list)
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new group chat")
    create_parser.add_argument("name", help="Name of the group chat")
    create_parser.add_argument("--members", "-m", nargs="*", help="Initial members of the group")
    create_parser.set_defaults(func=cmd_create)
    
    # Members command
    members_parser = subparsers.add_parser("members", help="Manage group chat members")
    members_parser.add_argument("group_name", help="Name of the group chat")
    members_parser.add_argument("action", choices=["add", "remove", "list"], help="Action to perform")
    members_parser.add_argument("--member", "-m", help="Member to add/remove")
    members_parser.set_defaults(func=cmd_members)
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start a group chat")
    chat_parser.add_argument("group_name", help="Name of the group chat to join")
    chat_parser.set_defaults(func=cmd_chat)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Group Chat CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()