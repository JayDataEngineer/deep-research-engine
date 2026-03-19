"""Backgrounds Manager

Manages background images functionality.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.constants import ROOT_DATA_DIR


class BackgroundManager:
    """Manages background images functionality."""

    def __init__(self):
        """Initialize the background manager."""
        self.backgrounds_dir = ROOT_DATA_DIR / "backgrounds"  # Default location
        self.current_bg_file = ROOT_DATA_DIR / "current_background.json"
        self.backgrounds_dir.mkdir(exist_ok=True)
        
        # Initialize with default if needed
        if not self.current_bg_file.exists():
            self._save_current_background("default")
    
    def _save_current_background(self, bg_name: str):
        """Save the current background."""
        self.current_bg_file.parent.mkdir(exist_ok=True)
        with open(self.current_bg_file, 'w', encoding='utf-8') as f:
            json.dump({"current": bg_name}, f)
    
    def _get_current_background(self) -> str:
        """Get the current background."""
        if self.current_bg_file.exists():
            with open(self.current_bg_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("current", "default")
        return "default"
    
    def list_backgrounds(self) -> List[str]:
        """
        List all backgrounds.
        
        Returns:
            List of background names
        """
        backgrounds = ["default"]  # Always include default
        for file_path in self.backgrounds_dir.glob("*"):
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                backgrounds.append(file_path.stem)
        return backgrounds
    
    def set_background(self, name: str) -> bool:
        """
        Set a background.
        
        Args:
            name: Name of the background to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if background exists or is valid
            if name == "default":
                self._save_current_background(name)
                return True
            
            # Check if it's a file in backgrounds dir
            bg_file = self.backgrounds_dir / f"{name}.jpg"
            if not bg_file.exists():
                bg_file = self.backgrounds_dir / f"{name}.png"
            if not bg_file.exists():
                bg_file = self.backgrounds_dir / f"{name}.jpeg"
            
            if bg_file.exists():
                self._save_current_background(name)
                return True
            return False
        except Exception:
            return False
    
    def set_random_background(self) -> Optional[str]:
        """
        Set a random background.
        
        Returns:
            Name of the background set, or None if failed
        """
        import random
        backgrounds = self.list_backgrounds()
        if backgrounds:
            random_bg = random.choice(backgrounds)
            if self.set_background(random_bg):
                return random_bg
        return None