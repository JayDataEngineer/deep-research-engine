"""Instruct Mode Manager

Manages instruct mode templates functionality.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.constants import ROOT_DATA_DIR


class InstructModeManager:
    """Manages instruct mode templates functionality."""

    def __init__(self):
        """Initialize the instruct mode manager."""
        self.templates_dir = ROOT_DATA_DIR / "instruct_templates"  # Default location
        self.current_template_file = ROOT_DATA_DIR / "current_instruct_template.json"
        self.templates_dir.mkdir(exist_ok=True)
        
        # Initialize with default if needed
        if not self.current_template_file.exists():
            self._save_current_template("default")
    
    def _save_current_template(self, template_name: str):
        """Save the current template."""
        self.current_template_file.parent.mkdir(exist_ok=True)
        with open(self.current_template_file, 'w', encoding='utf-8') as f:
            json.dump({"current": template_name}, f)
    
    def _get_current_template(self) -> str:
        """Get the current template."""
        if self.current_template_file.exists():
            with open(self.current_template_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("current", "default")
        return "default"
    
    def list_templates(self) -> List[str]:
        """
        List all instruct mode templates.
        
        Returns:
            List of template names
        """
        templates = ["default"]  # Always include default
        for file_path in self.templates_dir.glob("*.txt"):
            templates.append(file_path.stem)
        return templates
    
    def set_template(self, template_name: str) -> bool:
        """
        Set an instruct mode template.
        
        Args:
            template_name: Name of the template to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if template exists or is valid
            if template_name == "default":
                self._save_current_template(template_name)
                return True
            
            # Check if it's a file in templates dir
            template_file = self.templates_dir / f"{template_name}.txt"
            if template_file.exists():
                self._save_current_template(template_name)
                return True
            return False
        except Exception:
            return False
    
    def get_template(self, template_name: str) -> Optional[str]:
        """
        Get an instruct mode template.
        
        Args:
            template_name: Name of the template to get
            
        Returns:
            Template content if found, None otherwise
        """
        template_file = self.templates_dir / f"{template_name}.txt"
        if template_file.exists():
            with open(template_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None