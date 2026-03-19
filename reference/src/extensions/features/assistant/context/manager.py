"""Context Manager

Manages context/conversation functionality.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.constants import ROOT_DATA_DIR


class ContextManager:
    """Manages context/conversation functionality."""

    def __init__(self):
        """Initialize the context manager."""
        self.context_file = ROOT_DATA_DIR / "current_context.json"
        # Initialize with default context if file doesn't exist
        if not self.context_file.exists():
            self._save_context({
                "messages": [],
                "current_length": 0,
                "max_length": 4096,
                "segment_count": 0
            })
    
    def _load_context(self) -> Dict[str, Any]:
        """Load context from file."""
        if self.context_file.exists():
            with open(self.context_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"messages": [], "current_length": 0, "max_length": 4096, "segment_count": 0}
    
    def _save_context(self, context: Dict[str, Any]):
        """Save context to file."""
        self.context_file.parent.mkdir(exist_ok=True)
        with open(self.context_file, 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=2, ensure_ascii=False)
    
    def get_context_info(self) -> Dict[str, Any]:
        """
        Get context information.
        
        Returns:
            Context information
        """
        context = self._load_context()
        return {
            "current_length": context.get("current_length", 0),
            "max_length": context.get("max_length", 4096),
            "segment_count": context.get("segment_count", 0),
            "status": "active" if context.get("messages") else "empty"
        }
    
    def get_segments(self) -> List[Dict[str, Any]]:
        """
        Get context segments.
        
        Returns:
            List of context segments
        """
        context = self._load_context()
        messages = context.get("messages", [])
        segments = []
        for i, msg in enumerate(messages):
            segments.append({
                "index": i,
                "summary": f"{msg.get('role', 'user')}: {msg.get('content', '')[:50]}...",
                "length": len(msg.get('content', ''))
            })
        return segments
    
    def trim_context(self, target_length: int) -> bool:
        """
        Trim context to target length.
        
        Args:
            target_length: Target length in tokens
            
        Returns:
            True if successful, False otherwise
        """
        try:
            context = self._load_context()
            # This is a simplified implementation - in reality, you'd need proper token counting
            messages = context.get("messages", [])
            
            # Keep only the most recent messages that fit within target length
            trimmed_messages = []
            current_length = 0
            
            # Go backwards through messages to keep most recent ones
            for msg in reversed(messages):
                msg_length = len(msg.get('content', ''))
                if current_length + msg_length <= target_length:
                    trimmed_messages.insert(0, msg)  # Insert at beginning to maintain order
                    current_length += msg_length
                else:
                    break
            
            # Update context
            context["messages"] = trimmed_messages
            context["current_length"] = current_length
            self._save_context(context)
            
            return True
        except Exception:
            return False