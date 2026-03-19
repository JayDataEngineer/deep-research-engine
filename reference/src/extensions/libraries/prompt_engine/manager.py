"""Prompt Stack Manager - Handles CRUD operations for YAML-based prompt stacks.

Provides:
- Loading stacks from YAML files
- Saving stacks to YAML files
- Syncing between registry and filesystem
- Hot-reload when files change
"""

import asyncio
import yaml
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

from .schema import PromptStack, StackRegistry, DEFAULT_STACKS
from src.shared.utils.api_helpers import get_prompts_dir, ensure_dir

logger = logging.getLogger(__name__)


class StackFileHandler(FileSystemEventHandler):
    """Handles file system events for hot-reloading stacks."""

    def __init__(self, manager: 'StackManager'):
        self.manager = manager

    def on_modified(self, event):
        """Handle file modification."""
        if event.is_directory:
            return

        if event.src_path.endswith('.yaml') or event.src_path.endswith('.yml'):
            logger.info(f"Stack file modified: {event.src_path}")
            # Trigger async reload
            asyncio.create_task(self.manager.reload_file(event.src_path))


class StackManager:
    """Manages prompt stacks stored as YAML files."""

    def __init__(
        self,
        stacks_dir: Path,
        registry_file: Path = None,
        auto_reload: bool = True,
    ):
        """
        Initialize the stack manager.

        Args:
            stacks_dir: Directory containing stack YAML files
            registry_file: Optional registry cache file
            auto_reload: Watch for file changes and auto-reload
        """
        self.stacks_dir = Path(stacks_dir)
        self.registry_file = registry_file or self.stacks_dir / "stack_registry.yaml"
        self.auto_reload = auto_reload

        self.registry = StackRegistry()
        self._observer = None
        self._watch_task = None

    async def initialize(self) -> None:
        """Initialize the stack manager.

        - Create directories if needed
        - Load existing stacks
        - Save default stacks if none exist
        - Start file watcher if enabled
        """
        # Create directories
        self.stacks_dir.mkdir(parents=True, exist_ok=True)

        # Create category subdirectories
        categories = ["system", "personality", "context", "style", "custom"]
        for category in categories:
            (self.stacks_dir / category).mkdir(exist_ok=True)

        # Try to load registry
        loaded = await self.load_registry()

        # If no stacks exist, save defaults
        if not loaded or not self.registry.stacks:
            await self.save_defaults()

        # Start file watcher
        if self.auto_reload:
            await self.start_watcher()

    async def load_registry(self) -> bool:
        """Load registry from cache or scan directory."""
        # Try loading from registry file
        if self.registry_file.exists():
            try:
                data = yaml.safe_load(self.registry_file.read_text())
                if data and "stacks" in data:
                    self.registry = StackRegistry(**data)
                    logger.info(f"Loaded registry with {len(self.registry.stacks)} stacks")
                    return True
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")

        # Scan directory instead
        await self.scan_directory()
        return False

    async def save_registry(self) -> None:
        """Save registry to cache file."""
        self.registry.last_sync = datetime.now()
        data = self.registry.model_dump(mode='json')
        with open(self.registry_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.debug(f"Saved registry with {len(self.registry.stacks)} stacks")

    async def scan_directory(self) -> int:
        """Scan stacks directory and load all YAML files.

        Returns:
            Number of stacks loaded
        """
        count = 0
        for yaml_file in self.stacks_dir.rglob("*.yaml"):
            if yaml_file == self.registry_file:
                continue

            try:
                stack = await self.load_stack_file(yaml_file)
                if stack:
                    self.registry.register(stack)
                    count += 1
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

        logger.info(f"Scanned directory: loaded {count} stacks")
        return count

    async def load_stack_file(self, path: Path) -> Optional[PromptStack]:
        """Load a single stack from YAML file."""
        try:
            data = yaml.safe_load(path.read_text())
            if not data:
                return None

            # Add name from filename if not present
            if "name" not in data:
                data["name"] = path.stem

            return PromptStack(**data)
        except Exception as e:
            logger.error(f"Failed to parse {path}: {e}")
            return None

    async def save_stack(self, stack: PromptStack, category: str = None) -> Path:
        """Save a stack to YAML file.

        Args:
            stack: The stack to save
            category: Optional category override (defaults to stack.category)

        Returns:
            Path to saved file
        """
        category = category or stack.category
        filename = f"{stack.name}.yaml"
        file_path = self.stacks_dir / category / filename

        # Update timestamp
        stack.updated_at = datetime.now()

        # Save to YAML
        data = stack.model_dump(mode='json', exclude_none=True)
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        # Register in registry
        self.registry.register(stack)
        await self.save_registry()

        logger.info(f"Saved stack: {file_path}")
        return file_path

    async def delete_stack(self, stack_id: str) -> bool:
        """Delete a stack by ID.

        Args:
            stack_id: Stack identifier (name)

        Returns:
            True if deleted, False if not found
        """
        stack = self.registry.get(stack_id)
        if not stack:
            return False

        # Find and delete file
        category = stack.category
        filename = f"{stack.name}.yaml"
        file_path = self.stacks_dir / category / filename

        if file_path.exists():
            file_path.unlink()

        # Remove from registry
        del self.registry.stacks[stack_id]
        await self.save_registry()

        logger.info(f"Deleted stack: {stack_id}")
        return True

    async def save_defaults(self) -> None:
        """Save default built-in stacks to YAML files."""
        for stack_id, stack in DEFAULT_STACKS.items():
            await self.save_stack(stack)
        logger.info(f"Saved {len(DEFAULT_STACKS)} default stacks")

    async def reload_file(self, file_path: str) -> None:
        """Reload a specific stack file (hot-reload)."""
        path = Path(file_path)
        try:
            stack = await self.load_stack_file(path)
            if stack:
                self.registry.register(stack)
                await self.save_registry()
                logger.info(f"Reloaded stack: {path.name}")
        except Exception as e:
            logger.error(f"Failed to reload {file_path}: {e}")

    def get_all(self) -> List[PromptStack]:
        """Get all registered stacks."""
        return list(self.registry.stacks.values())

    def get_by_category(self, category: str) -> List[PromptStack]:
        """Get stacks by category."""
        return self.registry.list_by_category(category)

    def search(self, query: str) -> List[PromptStack]:
        """Search stacks by query."""
        return self.registry.search(query)

    async def start_watcher(self) -> None:
        """Start the file system watcher for hot-reload."""
        if self._observer:
            return  # Already running

        handler = StackFileHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.stacks_dir), recursive=True)
        self._observer.start()
        logger.info("Started stack file watcher")

    async def stop_watcher(self) -> None:
        """Stop the file system watcher."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Stopped stack file watcher")

    async def shutdown(self) -> None:
        """Cleanup and shutdown."""
        await self.stop_watcher()
        await self.save_registry()


# Global stack manager instance
_stack_manager: Optional[StackManager] = None


def get_stack_manager() -> StackManager:
    """Get the global stack manager instance."""
    global _stack_manager
    if _stack_manager is None:
        raise RuntimeError("StackManager not initialized. Call init_stack_manager() first.")
    return _stack_manager


async def init_stack_manager(
    stacks_dir: Path = None,
    auto_reload: bool = True,
) -> StackManager:
    """Initialize the global stack manager.

    Args:
        stacks_dir: Directory for stack YAML files (default: data/inputs/prompts/stacks/)
        auto_reload: Enable hot-reload on file changes

    Returns:
        Initialized StackManager instance
    """
    global _stack_manager

    if stacks_dir is None:
        # Use unified data directory
        stacks_dir = get_prompts_dir() / "stacks"

    _stack_manager = StackManager(
        stacks_dir=stacks_dir,
        auto_reload=auto_reload,
    )
    await _stack_manager.initialize()

    return _stack_manager
