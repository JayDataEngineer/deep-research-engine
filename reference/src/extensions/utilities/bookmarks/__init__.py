"""Bookmarks & Snapshots Extension

Manages chat state bookmarks for saving and restoring chat positions.
"""

from .manager import BookmarkManager, Bookmark

__all__ = ["BookmarkManager", "Bookmark"]
