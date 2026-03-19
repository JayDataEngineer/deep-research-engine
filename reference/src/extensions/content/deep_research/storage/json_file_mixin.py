"""JSONFileMixin - Mixin for JSON file operations in deepagents backends.

This mixin provides reusable methods for reading and writing JSON files
in deepagents backend classes, eliminating repeated inline imports
and DRY-ing up the backend code.

Usage:
    class MyBackend(StateBackend, JSONFileMixin):
        def write_config(self, config: dict) -> WriteResult:
            return self._write_json("/state/config.json", config)

        def read_config(self) -> Optional[dict]:
            return self._read_json("/state/config.json")
"""

import json
import re
from datetime import datetime
from typing import Any, Optional, Dict, List

from deepagents.backends.protocol import WriteResult
from deepagents.backends.utils import create_file_data, file_data_to_string


class JSONFileMixin:
    """Mixin class for JSON file operations in deepagents backends.

    Provides generic methods for reading and writing JSON files in the
    deepagents filesystem state.

    Methods:
        _write_json: Write data as JSON file
        _read_json: Read data from JSON file
        _write_json_as_markdown: Write JSON data wrapped in markdown
        _read_json_from_markdown: Read JSON data from markdown code block
    """

    def _get_files(self) -> Dict[str, Any]:
        """Get files dict from runtime state.

        Returns:
            Files dictionary from runtime state
        """
        if hasattr(self, 'runtime') and self.runtime:
            return self.runtime.state.get("files", {})
        return {}

    def _write_json(
        self,
        path: str,
        data: Any,
        indent: int = 2
    ) -> WriteResult:
        """Write data as JSON file.

        Args:
            path: File path (e.g., "/state/config.json")
            data: Data to serialize as JSON
            indent: JSON indentation level (default: 2)

        Returns:
            WriteResult with file update
        """
        content = json.dumps(data, indent=indent)
        file_data = create_file_data(content)
        return WriteResult(
            path=path,
            files_update={path: file_data}
        )

    def _read_json(
        self,
        path: str,
        default: Any = None
    ) -> Any:
        """Read data from JSON file.

        Args:
            path: File path (e.g., "/state/config.json")
            default: Default value if file doesn't exist or invalid JSON

        Returns:
            Parsed JSON data, or default if file doesn't exist
        """
        files = self._get_files()
        file_data = files.get(path)

        if not file_data:
            return default

        content = file_data_to_string(file_data)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return default

    def _write_json_as_markdown(
        self,
        path: str,
        data: Any,
        title: str = "JSON Data",
        indent: int = 2
    ) -> WriteResult:
        """Write JSON data wrapped in markdown format.

        Creates a markdown file with JSON data in a code block.

        Args:
            path: File path (e.g., "/state/brief.md")
            data: Data to serialize as JSON
            title: Title for the markdown file
            indent: JSON indentation level (default: 2)

        Returns:
            WriteResult with file update
        """

        content = f"""# {title}

Generated: {datetime.now().isoformat()}

```json
{json.dumps(data, indent=indent)}
```
"""
        file_data = create_file_data(content)
        return WriteResult(
            path=path,
            files_update={path: file_data}
        )

    def _read_json_from_markdown(
        self,
        path: str,
        default: Any = None
    ) -> Any:
        """Read JSON data from markdown code block.

        Extracts JSON from a markdown code block.

        Args:
            path: File path (e.g., "/state/brief.md")
            default: Default value if file doesn't exist or invalid JSON

        Returns:
            Parsed JSON data, or default if file doesn't exist or JSON invalid
        """
        files = self._get_files()
        file_data = files.get(path)

        if not file_data:
            return default

        content = file_data_to_string(file_data)

        # Extract JSON from markdown code block
        match = re.search(r'```json\n(.+?)\n```', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return default

    def _write_markdown(
        self,
        path: str,
        content: str
    ) -> WriteResult:
        """Write plain markdown content to file.

        Args:
            path: File path (e.g., "/state/context.md")
            content: Markdown content

        Returns:
            WriteResult with file update
        """
        file_data = create_file_data(content)
        return WriteResult(
            path=path,
            files_update={path: file_data}
        )

    def _read_markdown(
        self,
        path: str,
        default: str = ""
    ) -> str:
        """Read markdown content from file.

        Args:
            path: File path (e.g., "/state/context.md")
            default: Default value if file doesn't exist

        Returns:
            File content as string, or default if file doesn't exist
        """
        files = self._get_files()
        file_data = files.get(path)

        if not file_data:
            return default

        return file_data_to_string(file_data)
