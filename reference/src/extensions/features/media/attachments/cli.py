"""Attachments CLI Plugin

Provides commands for managing attachments.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_list(args):
    """List all attachments."""
    from src.extensions.features.media.attachments.manager import AttachmentManager
    
    try:
        manager = AttachmentManager()
        attachments = manager.list_attachments()
        
        print(f"\nAttachments ({len(attachments)} found):")
        for attachment in attachments:
            print(f"  - {attachment}")
        
        return {"attachments": attachments}
    except Exception as e:
        print(f"Error listing attachments: {e}")
        return {"error": str(e)}


def cmd_scrape(args):
    """Scrape attachments from a URL."""
    from src.extensions.features.media.attachments.manager import AttachmentManager
    
    try:
        manager = AttachmentManager()
        
        print(f"Scraping attachments from: {args.url}")
        
        # Scrape attachments from URL
        result = manager.scrape_attachments(args.url)
        
        print(f"Scraped {len(result)} attachments from {args.url}")
        
        return {"result": result}
    except Exception as e:
        print(f"Error scraping attachments: {e}")
        return {"error": str(e)}


def cmd_view(args):
    """View an attachment."""
    from src.extensions.features.media.attachments.manager import AttachmentManager
    
    try:
        manager = AttachmentManager()
        
        # View attachment
        content = manager.view_attachment(args.attachment_id)
        
        if content:
            print(f"Attachment {args.attachment_id}:")
            print(content)
        else:
            print(f"Attachment {args.attachment_id} not found")
        
        return {"content": content}
    except Exception as e:
        print(f"Error viewing attachment: {e}")
        return {"error": str(e)}


def register(parser):
    """Register attachment commands with the parser."""
    subparsers = parser.add_subparsers(dest="attachment_subcommand", help="Attachment commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all attachments")
    list_parser.set_defaults(func=cmd_list)
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape attachments from a URL")
    scrape_parser.add_argument("url", help="URL to scrape attachments from")
    scrape_parser.set_defaults(func=cmd_scrape)
    
    # View command
    view_parser = subparsers.add_parser("view", help="View an attachment")
    view_parser.add_argument("attachment_id", help="ID of the attachment to view")
    view_parser.set_defaults(func=cmd_view)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Attachments CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()