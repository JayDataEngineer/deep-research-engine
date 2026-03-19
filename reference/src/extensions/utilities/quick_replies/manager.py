"""Quick Replies Manager

Manages quick response buttons functionality.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.constants import ROOT_DATA_DIR


class QuickReplyManager:
    """Manages quick response buttons functionality."""

    def __init__(self):
        """Initialize the quick reply manager."""
        self.replies_file = ROOT_DATA_DIR / "quick_replies.json"
        # Initialize with default replies if file doesn't exist
        if not self.replies_file.exists():
            self._save_replies({})
    
    def _load_replies(self) -> Dict[str, str]:
        """Load replies from file."""
        if self.replies_file.exists():
            with open(self.replies_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_replies(self, replies: Dict[str, str]):
        """Save replies to file."""
        self.replies_file.parent.mkdir(exist_ok=True)
        with open(self.replies_file, 'w', encoding='utf-8') as f:
            json.dump(replies, f, indent=2, ensure_ascii=False)
    
    def list_replies(self) -> Dict[str, str]:
        """
        List all quick replies.
        
        Returns:
            Dictionary of reply name to text
        """
        return self._load_replies()
    
    def add_reply(self, name: str, text: str) -> bool:
        """
        Add a new quick reply.
        
        Args:
            name: Name of the quick reply
            text: Text of the quick reply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            replies = self._load_replies()
            replies[name] = text
            self._save_replies(replies)
            return True
        except Exception:
            return False
    
    def remove_reply(self, name: str) -> bool:
        """
        Remove a quick reply.
        
        Args:
            name: Name of the quick reply to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            replies = self._load_replies()
            if name in replies:
                del replies[name]
                self._save_replies(replies)
                return True
            return False
        except Exception:
            return False