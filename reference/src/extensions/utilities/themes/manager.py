"""Themes Manager

Manages UI theme configurations.
Backend storage for theme settings (frontend would apply these).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Theme:
    """A UI theme configuration."""
    id: str
    name: str
    display_name: str
    colors: Dict[str, str]  # CSS color variables
    fonts: Dict[str, str]  # Font settings
    custom_css: str = ""
    is_dark: bool = False
    is_builtin: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Theme":
        return cls(**data)


class ThemesManager:
    """Manages UI theme configurations."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the themes manager.

        Args:
            base_dir: Base directory for theme storage
        """
        if base_dir is None:
            from src.constants import PROJECT_ROOT
            base_dir = PROJECT_ROOT / "data" / "inputs" / "themes"

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.themes_file = self.base_dir / "themes.json"
        self.active_file = self.base_dir / "active.txt"

        self._themes: Dict[str, Theme] = {}
        self._active_theme: Optional[str] = None

        self._load_builtin_themes()
        self._load()

    def _get_builtin_themes(self) -> List[Theme]:
        """Get built-in theme definitions."""
        return [
            Theme(
                id="light",
                name="light",
                display_name="Light",
                is_dark=False,
                is_builtin=True,
                colors={
                    "bg-primary": "#ffffff",
                    "bg-secondary": "#f3f4f6",
                    "bg-tertiary": "#e5e7eb",
                    "text-primary": "#111827",
                    "text-secondary": "#6b7280",
                    "accent": "#3b82f6",
                    "accent-hover": "#2563eb",
                    "border": "#d1d5db",
                },
                fonts={
                    "sans": "Inter, system-ui, sans-serif",
                    "mono": "JetBrains Mono, monospace",
                },
            ),
            Theme(
                id="dark",
                name="dark",
                display_name="Dark",
                is_dark=True,
                is_builtin=True,
                colors={
                    "bg-primary": "#111827",
                    "bg-secondary": "#1f2937",
                    "bg-tertiary": "#374151",
                    "text-primary": "#f9fafb",
                    "text-secondary": "#9ca3af",
                    "accent": "#3b82f6",
                    "accent-hover": "#60a5fa",
                    "border": "#374151",
                },
                fonts={
                    "sans": "Inter, system-ui, sans-serif",
                    "mono": "JetBrains Mono, monospace",
                },
            ),
            Theme(
                id="midnight",
                name="midnight",
                display_name="Midnight",
                is_dark=True,
                is_builtin=True,
                colors={
                    "bg-primary": "#0a0a0f",
                    "bg-secondary": "#13131a",
                    "bg-tertiary": "#1a1a24",
                    "text-primary": "#e4e4e7",
                    "text-secondary": "#71717a",
                    "accent": "#8b5cf6",
                    "accent-hover": "#a78bfa",
                    "border": "#27272a",
                },
                fonts={
                    "sans": "Inter, system-ui, sans-serif",
                    "mono": "JetBrains Mono, monospace",
                },
            ),
        ]

    def _load_builtin_themes(self):
        """Load built-in themes."""
        for theme in self._get_builtin_themes():
            self._themes[theme.id] = theme

    def _load(self):
        """Load themes from storage."""
        # Load custom themes
        if self.themes_file.exists():
            try:
                with open(self.themes_file) as f:
                    data = json.load(f)
                for theme_data in data:
                    theme = Theme.from_dict(theme_data)
                    self._themes[theme.id] = theme
            except Exception as e:
                logger.warning(f"Failed to load themes: {e}")

        # Load active theme
        if self.active_file.exists():
            try:
                self._active_theme = self.active_file.read_text().strip()
            except Exception as e:
                logger.warning(f"Failed to load active theme: {e}")

        logger.info(f"Themes Manager initialized: {len(self._themes)} themes")

    def _save(self):
        """Save themes to storage."""
        # Save only custom themes
        custom_themes = [
            theme.to_dict()
            for theme in self._themes.values()
            if not theme.is_builtin
        ]

        with open(self.themes_file, "w") as f:
            json.dump(custom_themes, f, indent=2)

        # Save active theme
        if self._active_theme:
            with open(self.active_file, "w") as f:
                f.write(self._active_theme)

    # Theme Management

    def list_themes(self) -> List[Theme]:
        """List all themes."""
        return sorted(self._themes.values(), key=lambda t: (t.is_builtin, t.name))

    def get_theme(self, theme_id: str) -> Optional[Theme]:
        """Get a theme by ID."""
        return self._themes.get(theme_id)

    def create_theme(
        self,
        id: str,
        name: str,
        display_name: str,
        colors: Dict[str, str],
        fonts: Optional[Dict[str, str]] = None,
        custom_css: str = "",
        is_dark: bool = False,
    ) -> Theme:
        """Create a new custom theme.

        Args:
            id: Theme ID
            name: Internal name
            display_name: Display name
            colors: Color dictionary
            fonts: Font settings (optional)
            custom_css: Custom CSS (optional)
            is_dark: Whether this is a dark theme

        Returns:
            Created Theme
        """
        if id in self._themes:
            raise ValueError(f"Theme already exists: {id}")

        theme = Theme(
            id=id,
            name=name,
            display_name=display_name,
            colors=colors,
            fonts=fonts or {},
            custom_css=custom_css,
            is_dark=is_dark,
            is_builtin=False,
        )

        self._themes[id] = theme
        self._save()
        return theme

    def update_theme(
        self,
        theme_id: str,
        name: Optional[str] = None,
        display_name: Optional[str] = None,
        colors: Optional[Dict[str, str]] = None,
        fonts: Optional[Dict[str, str]] = None,
        custom_css: Optional[str] = None,
    ) -> Optional[Theme]:
        """Update a theme.

        Args:
            theme_id: Theme ID
            name: New name (optional)
            display_name: New display name (optional)
            colors: New colors (optional)
            fonts: New fonts (optional)
            custom_css: New custom CSS (optional)

        Returns:
            Updated Theme or None
        """
        theme = self._themes.get(theme_id)
        if not theme:
            return None

        if theme.is_builtin:
            raise ValueError("Cannot modify built-in themes")

        if name is not None:
            theme.name = name
        if display_name is not None:
            theme.display_name = display_name
        if colors is not None:
            theme.colors.update(colors)
        if fonts is not None:
            theme.fonts.update(fonts)
        if custom_css is not None:
            theme.custom_css = custom_css

        self._save()
        return theme

    def delete_theme(self, theme_id: str) -> bool:
        """Delete a custom theme.

        Args:
            theme_id: Theme ID

        Returns:
            True if deleted
        """
        theme = self._themes.get(theme_id)
        if not theme or theme.is_builtin:
            return False

        del self._themes[theme_id]

        if self._active_theme == theme_id:
            self._active_theme = None

        self._save()
        return True

    # Active Theme

    def set_active_theme(self, theme_id: str) -> bool:
        """Set the active theme.

        Args:
            theme_id: Theme ID

        Returns:
            True if set
        """
        if theme_id not in self._themes:
            return False

        self._active_theme = theme_id
        self._save()
        return True

    def get_active_theme(self) -> Optional[Theme]:
        """Get the active theme."""
        if self._active_theme:
            return self._themes.get(self._active_theme)
        return None

    # Export/Import

    def export_theme(self, theme_id: str, output_path: str) -> bool:
        """Export a theme to a file.

        Args:
            theme_id: Theme ID
            output_path: Output file path

        Returns:
            True if exported
        """
        theme = self._themes.get(theme_id)
        if not theme:
            return False

        with open(output_path, "w") as f:
            json.dump(theme.to_dict(), f, indent=2)

        return True

    def import_theme(self, filepath: str) -> Optional[Theme]:
        """Import a theme from a file.

        Args:
            filepath: Path to theme JSON file

        Returns:
            Imported Theme
        """
        with open(filepath) as f:
            data = json.load(f)

        theme = Theme.from_dict(data)

        # Generate new ID if conflict
        if theme.id in self._themes:
            import re
            counter = 1
            new_id = f"{theme.id}_{counter}"
            while new_id in self._themes:
                counter += 1
                new_id = f"{theme.id}_{counter}"
            theme.id = new_id

        self._themes[theme.id] = theme
        self._save()
        return theme

    def to_css(self, theme_id: str) -> str:
        """Convert a theme to CSS variables.

        Args:
            theme_id: Theme ID

        Returns:
            CSS string
        """
        theme = self._themes.get(theme_id)
        if not theme:
            return ""

        css_lines = [":root {"]
        for key, value in theme.colors.items():
            css_lines.append(f"  --{key}: {value};")
        for key, value in theme.fonts.items():
            css_lines.append(f"  --font-{key}: {value};")
        css_lines.append("}")

        if theme.custom_css:
            css_lines.append("\n" + theme.custom_css)

        return "\n".join(css_lines)
