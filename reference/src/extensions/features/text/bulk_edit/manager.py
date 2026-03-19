"""Bulk Edit Manager

Provides bulk operations for editing messages and chat content.
Similar to SillyTavern's Bulk Edit Overlay.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class BulkEditOperation:
    """Represents a bulk edit operation."""
    operation: str  # replace, delete, prepend, append, regex
    target_pattern: str = ""
    replacement: str = ""
    regex_flags: int = 0  # re.IGNORECASE, etc.
    case_sensitive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BulkEditOperation":
        return cls(**data)


@dataclass
class BulkEditResult:
    """Result of a bulk edit operation."""
    operation: BulkEditOperation
    items_processed: int
    items_modified: int
    errors: List[str]
    preview_changes: List[Dict[str, str]]  # [{"id": "...", "before": "...", "after": "..."}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation.to_dict(),
            "items_processed": self.items_processed,
            "items_modified": self.items_modified,
            "errors": self.errors,
            "preview_changes": self.preview_changes,
        }


class BulkEditManager:
    """Manages bulk edit operations on messages and content."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the bulk edit manager.

        Args:
            base_dir: Base directory for storage
        """
        if base_dir is None:
            from src.constants import PROJECT_ROOT
            base_dir = PROJECT_ROOT / "data" / "inputs" / "bulk_edit"

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.base_dir / "history.json"
        self.templates_file = self.base_dir / "templates.json"

        self._history: List[Dict[str, Any]] = []
        self._templates: Dict[str, BulkEditOperation] = {}

        self._load()

    def _load(self):
        """Load history and templates from storage."""
        # Load history
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    self._history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load bulk edit history: {e}")

        # Load templates
        if self.templates_file.exists():
            try:
                with open(self.templates_file) as f:
                    data = json.load(f)
                for name, op_data in data.items():
                    self._templates[name] = BulkEditOperation.from_dict(op_data)
            except Exception as e:
                logger.warning(f"Failed to load bulk edit templates: {e}")

        logger.info(f"Bulk Edit Manager initialized: {len(self._templates)} templates")

    def _save(self):
        """Save history and templates to storage."""
        # Save history (keep last 100)
        with open(self.history_file, "w") as f:
            json.dump(self._history[-100:], f, indent=2)

        # Save templates
        with open(self.templates_file, "w") as f:
            json.dump({
                name: op.to_dict()
                for name, op in self._templates.items()
            }, f, indent=2)

    # Template Management

    def save_template(self, name: str, operation: BulkEditOperation) -> bool:
        """Save an operation as a reusable template.

        Args:
            name: Template name
            operation: Operation to save

        Returns:
            True if saved
        """
        self._templates[name] = operation
        self._save()
        return True

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all saved templates."""
        return [
            {"name": name, **op.to_dict()}
            for name, op in self._templates.items()
        ]

    def get_template(self, name: str) -> Optional[BulkEditOperation]:
        """Get a template by name."""
        return self._templates.get(name)

    def delete_template(self, name: str) -> bool:
        """Delete a template."""
        if name in self._templates:
            del self._templates[name]
            self._save()
            return True
        return False

    # Bulk Operations

    def bulk_replace(
        self,
        items: List[Dict[str, Any]],
        target: str,
        replacement: str,
        case_sensitive: bool = False,
        field: str = "content",
        preview: bool = False,
    ) -> BulkEditResult:
        """Bulk replace text in items.

        Args:
            items: List of items (each must have 'id' and specified field)
            target: Text to find
            replacement: Text to replace with
            case_sensitive: Whether search is case-sensitive
            field: Field name to operate on (default: "content")
            preview: If True, don't actually modify items

        Returns:
            BulkEditResult with changes
        """
        operation = BulkEditOperation(
            operation="replace",
            target_pattern=target,
            replacement=replacement,
            case_sensitive=case_sensitive,
        )

        modified = 0
        errors = []
        preview_changes = []

        for item in items:
            item_id = item.get("id", "unknown")
            content = item.get(field, "")

            if not isinstance(content, str):
                errors.append(f"Item {item_id}: field '{field}' is not a string")
                continue

            # Check if target exists
            search_content = content if case_sensitive else content.lower()
            search_target = target if case_sensitive else target.lower()

            if search_target not in search_content:
                continue

            # Apply replacement
            if case_sensitive:
                new_content = content.replace(target, replacement)
            else:
                # Case-insensitive replace (more complex)
                import re
                pattern = re.compile(re.escape(target), re.IGNORECASE)
                new_content = pattern.sub(replacement, content)

            modified += 1

            if preview:
                preview_changes.append({
                    "id": item_id,
                    "before": content[:200] + "..." if len(content) > 200 else content,
                    "after": new_content[:200] + "..." if len(new_content) > 200 else new_content,
                })
            else:
                item[field] = new_content

        result = BulkEditResult(
            operation=operation,
            items_processed=len(items),
            items_modified=modified,
            errors=errors,
            preview_changes=preview_changes,
        )

        if not preview:
            self._record_operation(result)

        return result

    def bulk_regex_replace(
        self,
        items: List[Dict[str, Any]],
        pattern: str,
        replacement: str,
        flags: int = 0,
        field: str = "content",
        preview: bool = False,
    ) -> BulkEditResult:
        """Bulk replace using regex.

        Args:
            items: List of items
            pattern: Regex pattern
            replacement: Replacement string (can use groups)
            flags: Regex flags (re.IGNORECASE, etc.)
            field: Field name to operate on
            preview: If True, don't modify

        Returns:
            BulkEditResult
        """
        import re

        operation = BulkEditOperation(
            operation="regex",
            target_pattern=pattern,
            replacement=replacement,
            regex_flags=flags,
        )

        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return BulkEditResult(
                operation=operation,
                items_processed=0,
                items_modified=0,
                errors=[f"Invalid regex pattern: {e}"],
                preview_changes=[],
            )

        modified = 0
        errors = []
        preview_changes = []

        for item in items:
            item_id = item.get("id", "unknown")
            content = item.get(field, "")

            if not isinstance(content, str):
                errors.append(f"Item {item_id}: field '{field}' is not a string")
                continue

            if not regex.search(content):
                continue

            new_content = regex.sub(replacement, content)
            modified += 1

            if preview:
                preview_changes.append({
                    "id": item_id,
                    "before": content[:200] + "..." if len(content) > 200 else content,
                    "after": new_content[:200] + "..." if len(new_content) > 200 else new_content,
                })
            else:
                item[field] = new_content

        result = BulkEditResult(
            operation=operation,
            items_processed=len(items),
            items_modified=modified,
            errors=errors,
            preview_changes=preview_changes,
        )

        if not preview:
            self._record_operation(result)

        return result

    def bulk_prepend(
        self,
        items: List[Dict[str, Any]],
        text: str,
        field: str = "content",
        preview: bool = False,
    ) -> BulkEditResult:
        """Prepend text to items.

        Args:
            items: List of items
            text: Text to prepend
            field: Field name
            preview: If True, don't modify

        Returns:
            BulkEditResult
        """
        operation = BulkEditOperation(
            operation="prepend",
            replacement=text,
        )

        modified = len(items)
        preview_changes = []

        for item in items:
            item_id = item.get("id", "unknown")
            content = item.get(field, "")

            if preview:
                preview_changes.append({
                    "id": item_id,
                    "before": content[:100] + "..." if len(content) > 100 else content,
                    "after": text + content[:100] + "..." if len(text + content) > 100 else text + content,
                })
            else:
                item[field] = text + content

        result = BulkEditResult(
            operation=operation,
            items_processed=len(items),
            items_modified=modified,
            errors=[],
            preview_changes=preview_changes,
        )

        if not preview:
            self._record_operation(result)

        return result

    def bulk_append(
        self,
        items: List[Dict[str, Any]],
        text: str,
        field: str = "content",
        preview: bool = False,
    ) -> BulkEditResult:
        """Append text to items.

        Args:
            items: List of items
            text: Text to append
            field: Field name
            preview: If True, don't modify

        Returns:
            BulkEditResult
        """
        operation = BulkEditOperation(
            operation="append",
            replacement=text,
        )

        modified = len(items)
        preview_changes = []

        for item in items:
            item_id = item.get("id", "unknown")
            content = item.get(field, "")

            if preview:
                preview_changes.append({
                    "id": item_id,
                    "before": (content[-100:] if len(content) > 100 else content),
                    "after": (content[-100:] if len(content) > 100 else content) + text[:100] + "...",
                })
            else:
                item[field] = content + text

        result = BulkEditResult(
            operation=operation,
            items_processed=len(items),
            items_modified=modified,
            errors=[],
            preview_changes=preview_changes,
        )

        if not preview:
            self._record_operation(result)

        return result

    def bulk_delete(
        self,
        items: List[Dict[str, Any]],
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        preview: bool = False,
    ) -> BulkEditResult:
        """Bulk delete items matching a filter.

        Args:
            items: List of items (will be modified in place)
            filter_fn: Function that returns True to delete item
            preview: If True, don't modify

        Returns:
            BulkEditResult with deleted items
        """
        operation = BulkEditOperation(
            operation="delete",
        )

        to_delete = []
        if filter_fn:
            to_delete = [item for item in items if filter_fn(item)]

        result = BulkEditResult(
            operation=operation,
            items_processed=len(items),
            items_modified=len(to_delete),
            errors=[],
            preview_changes=[{"id": item.get("id", "unknown")} for item in to_delete],
        )

        if not preview:
            # Remove items from original list
            for item in to_delete:
                if item in items:
                    items.remove(item)
            self._record_operation(result)

        return result

    # History

    def _record_operation(self, result: BulkEditResult):
        """Record an operation in history."""
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "operation": result.operation.to_dict(),
            "items_processed": result.items_processed,
            "items_modified": result.items_modified,
        })
        self._save()

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent bulk edit operations.

        Args:
            limit: Max number of history items

        Returns:
            List of history entries
        """
        return self._history[-limit:]

    def clear_history(self):
        """Clear operation history."""
        self._history = []
        self._save()
