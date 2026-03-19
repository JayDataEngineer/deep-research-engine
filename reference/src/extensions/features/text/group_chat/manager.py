"""Group Chat Manager

Manages group chat functionality.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.constants import ROOT_DATA_DIR


class GroupChatManager:
    """Manages group chat functionality."""

    def __init__(self):
        """Initialize the group chat manager."""
        self.groups_dir = ROOT_DATA_DIR / "group_chats"  # Default location
        self.groups_dir.mkdir(exist_ok=True)
    
    def list_groups(self) -> List[str]:
        """
        List all group chats.
        
        Returns:
            List of group names
        """
        groups = []
        for file_path in self.groups_dir.glob("*.json"):
            groups.append(file_path.stem)
        return groups
    
    def create_group(self, name: str, members: List[str]) -> bool:
        """
        Create a new group chat.
        
        Args:
            name: Name of the group
            members: Initial members of the group
            
        Returns:
            True if successful, False otherwise
        """
        try:
            group_data = {
                "name": name,
                "members": members,
                "created_at": "now",  # In a real implementation, use actual timestamp
                "messages": []
            }
            
            group_file = self.groups_dir / f"{name}.json"
            with open(group_file, 'w', encoding='utf-8') as f:
                json.dump(group_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    def get_members(self, group_name: str) -> List[str]:
        """
        Get members of a group.
        
        Args:
            group_name: Name of the group
            
        Returns:
            List of members
        """
        group_file = self.groups_dir / f"{group_name}.json"
        if group_file.exists():
            with open(group_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("members", [])
        return []
    
    def add_member(self, group_name: str, member: str) -> bool:
        """
        Add a member to a group.
        
        Args:
            group_name: Name of the group
            member: Member to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            group_file = self.groups_dir / f"{group_name}.json"
            if not group_file.exists():
                return False
                
            with open(group_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            members = data.get("members", [])
            if member not in members:
                members.append(member)
                data["members"] = members
                
                with open(group_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception:
            return False
    
    def remove_member(self, group_name: str, member: str) -> bool:
        """
        Remove a member from a group.
        
        Args:
            group_name: Name of the group
            member: Member to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            group_file = self.groups_dir / f"{group_name}.json"
            if not group_file.exists():
                return False
                
            with open(group_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            members = data.get("members", [])
            if member in members:
                members.remove(member)
                data["members"] = members
                
                with open(group_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception:
            return False
    
    def send_message(self, group_name: str, message: str) -> str:
        """
        Send a message to a group.
        
        Args:
            group_name: Name of the group
            message: Message to send
            
        Returns:
            Status message
        """
        try:
            group_file = self.groups_dir / f"{group_name}.json"
            if not group_file.exists():
                return f"Group '{group_name}' does not exist"
                
            with open(group_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add message to the group
            data.setdefault("messages", []).append({
                "message": message,
                "timestamp": "now",  # In a real implementation, use actual timestamp
                "sender": "user"  # In a real implementation, get actual sender
            })
            
            with open(group_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return f"Message sent to group '{group_name}'"
        except Exception as e:
            return f"Error sending message: {str(e)}"