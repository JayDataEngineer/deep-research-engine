"""Chat History Manager

Manages chat history functionality.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.constants import ROOT_DATA_DIR


class ChatHistoryManager:
    """Manages chat history functionality."""

    def __init__(self):
        """Initialize the chat history manager."""
        self.histories_dir = ROOT_DATA_DIR / "chat_histories"  # Default location
        self.histories_dir.mkdir(exist_ok=True)
    
    def list_histories(self) -> List[str]:
        """
        List all chat histories.
        
        Returns:
            List of history names
        """
        histories = []
        for file_path in self.histories_dir.glob("*.json"):
            histories.append(file_path.stem)
        return histories
    
    def export_history(self, history_name: str) -> Optional[Dict[str, Any]]:
        """
        Export a chat history.
        
        Args:
            history_name: Name of the history to export
            
        Returns:
            History data if found, None otherwise
        """
        history_file = self.histories_dir / f"{history_name}.json"
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def import_history(self, history_name: str, history_data: Dict[str, Any]) -> bool:
        """
        Import a chat history.
        
        Args:
            history_name: Name for the imported history
            history_data: History data to import
            
        Returns:
            True if successful, False otherwise
        """
        try:
            history_file = self.histories_dir / f"{history_name}.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False