"""Lorebooks Manager

Manages lorebooks/world info functionality using the lorebook_engine.
This is a thin wrapper that delegates to the advanced lorebook_engine.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from src.extensions.libraries.lorebook_engine.schema import Lorebook, LorebookEntry, DEFAULT_LOREBOOKS
from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


class LorebookManager:
    """Manages lorebooks using the lorebook_engine schema."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the lorebook manager.

        Args:
            base_dir: Base directory for lorebook storage
        """
        self.base_dir = StorageManager.get_project_subdir("lorebooks") if base_dir is None else Path(base_dir)
        self.base_dir = StorageManager.ensure_directory(self.base_dir)

        self._lorebooks: Dict[str, Lorebook] = {}
        self._load()

    def _load(self):
        """Load lorebooks from storage."""
        import json

        for file_path in self.base_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                lorebook = Lorebook(**data)
                self._lorebooks[lorebook.name] = lorebook
            except Exception as e:
                logger.warning(f"Failed to load lorebook {file_path}: {e}")

        # Load defaults if no lorebooks exist
        if not self._lorebooks:
            for name, lorebook in DEFAULT_LOREBOOKS.items():
                self._lorebooks[name] = lorebook
                self._save_lorebook(lorebook)

        logger.info(f"LorebookManager initialized: {len(self._lorebooks)} lorebooks")

    def _save_lorebook(self, lorebook: Lorebook) -> bool:
        """Save a lorebook to storage.

        Args:
            lorebook: Lorebook to save

        Returns:
            True if successful
        """
        safe_name = StorageManager.generate_id_from_name(lorebook.name)
        file_path = self.base_dir / f"{safe_name}.json"
        return StorageManager.save_json_file(file_path, lorebook.model_dump())

    # ========================================================================
    # Lorebook CRUD
    # ========================================================================

    def list_lorebooks(self) -> List[str]:
        """List all lorebook names.

        Returns:
            List of lorebook names
        """
        return list(self._lorebooks.keys())

    def get_lorebook(self, name: str) -> Optional[Lorebook]:
        """Get a lorebook by name.

        Args:
            name: Lorebook name

        Returns:
            Lorebook or None
        """
        return self._lorebooks.get(name)

    def create_lorebook(
        self,
        name: str,
        description: str = "",
        scan_depth: int = 10,
        token_budget: int = 2000,
        entries: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Lorebook]:
        """Create a new lorebook.

        Args:
            name: Name of the lorebook
            description: Description of the lorebook
            scan_depth: How many messages back to scan
            token_budget: Max tokens for activated entries
            entries: Initial entries (dicts that will be converted to LorebookEntry)

        Returns:
            Created Lorebook or None
        """
        if name in self._lorebooks:
            logger.warning(f"Lorebook '{name}' already exists")
            return None

        # Convert entry dicts to LorebookEntry objects
        lorebook_entries = []
        if entries:
            for entry_data in entries:
                lorebook_entries.append(LorebookEntry(**entry_data))

        lorebook = Lorebook(
            name=name,
            description=description,
            scan_depth=scan_depth,
            token_budget=token_budget,
            entries=lorebook_entries,
        )

        self._lorebooks[name] = lorebook
        self._save_lorebook(lorebook)
        logger.info(f"Created lorebook: {name}")
        return lorebook

    def delete_lorebook(self, name: str) -> bool:
        """Delete a lorebook.

        Args:
            name: Name of the lorebook to delete

        Returns:
            True if deleted
        """
        if name not in self._lorebooks:
            return False

        del self._lorebooks[name]

        # Delete file
        safe_name = StorageManager.generate_id_from_name(name)
        file_path = self.base_dir / f"{safe_name}.json"
        if file_path.exists():
            file_path.unlink()

        logger.info(f"Deleted lorebook: {name}")
        return True

    # ========================================================================
    # Entry Management
    # ========================================================================

    def add_entry(self, lorebook_name: str, entry: LorebookEntry) -> bool:
        """Add an entry to a lorebook.

        Args:
            lorebook_name: Name of the lorebook
            entry: Entry to add

        Returns:
            True if added
        """
        lorebook = self._lorebooks.get(lorebook_name)
        if not lorebook:
            return False

        lorebook.add_entry(entry)
        self._save_lorebook(lorebook)
        return True

    def remove_entry(self, lorebook_name: str, entry_id: str) -> bool:
        """Remove an entry from a lorebook.

        Args:
            lorebook_name: Name of the lorebook
            entry_id: ID of the entry to remove

        Returns:
            True if removed
        """
        lorebook = self._lorebooks.get(lorebook_name)
        if not lorebook:
            return False

        result = lorebook.remove_entry(entry_id)
        if result:
            self._save_lorebook(lorebook)
        return result

    def list_entries(self, lorebook_name: str) -> List[LorebookEntry]:
        """List entries in a lorebook.

        Args:
            lorebook_name: Name of the lorebook

        Returns:
            List of entries
        """
        lorebook = self._lorebooks.get(lorebook_name)
        if not lorebook:
            return []
        return lorebook.get_enabled_entries()

    # ========================================================================
    # Search
    # ========================================================================

    def search(self, query: str) -> List[Lorebook]:
        """Search lorebooks by name, description, or entry keywords.

        Args:
            query: Search query

        Returns:
            Matching lorebooks
        """
        query_lower = query.lower()
        results = []

        for lorebook in self._lorebooks.values():
            if query_lower in lorebook.name.lower():
                results.append(lorebook)
                continue

            if lorebook.description and query_lower in lorebook.description.lower():
                results.append(lorebook)
                continue

            for entry in lorebook.entries:
                if any(query_lower in kw.lower() for kw in entry.keys):
                    results.append(lorebook)
                    break

        return results

    # ========================================================================
    # Stats
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get lorebook statistics.

        Returns:
            Stats dict
        """
        total_entries = sum(len(lb.entries) for lb in self._lorebooks.values())
        enabled_entries = sum(len(lb.get_enabled_entries()) for lb in self._lorebooks.values())

        return {
            "total_lorebooks": len(self._lorebooks),
            "total_entries": total_entries,
            "enabled_entries": enabled_entries,
            "storage_path": str(self.base_dir),
        }