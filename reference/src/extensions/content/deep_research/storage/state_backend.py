"""ResearchStateBackend: State backend for ephemeral research data.

Provides filesystem access to research brief, query, config, and other
ephemeral state data that doesn't need to persist across sessions.
"""

from typing import Any, Optional

from deepagents.backends.state import StateBackend
from deepagents.backends.protocol import FileInfo, WriteResult, EditResult
from deepagents.backends.utils import perform_string_replacement, create_file_data, file_data_to_string

from .json_file_mixin import JSONFileMixin


class ResearchStateBackend(StateBackend, JSONFileMixin):
    """Backend for ephemeral research state data.

    Path Structure:
        /state/
        ├── brief.md          → ResearchBrief (JSON as Markdown)
        ├── query.md          → Original query
        ├── config.json       → Current config
        ├── stats.json        → Session statistics
        └── context.md        → Additional context

    This extends StateBackend to provide convenient methods for reading/writing
    research-specific state as files.

    Example:
        >>> backend = ResearchStateBackend(runtime)
        >>> backend.write_brief({"complexity": 0.8, "strategy": "research"})
        >>> brief = backend.read_brief()
    """

    def __init__(self, runtime: Any):
        """Initialize ResearchStateBackend.

        Args:
            runtime: LangGraph ToolRuntime
        """
        super().__init__(runtime)
        self.runtime = runtime

    # ========================================================================
    # Convenience Methods for Research State
    # ========================================================================

    def write_brief(self, brief: dict) -> WriteResult:
        """Write research brief as a file.

        Args:
            brief: ResearchBrief dict

        Returns:
            WriteResult
        """
        return self._write_json_as_markdown(
            path="/state/brief.md",
            data=brief,
            title="Research Brief"
        )

    def read_brief(self) -> Optional[dict]:
        """Read research brief from file.

        Returns:
            ResearchBrief dict or None
        """
        return self._read_json_from_markdown("/state/brief.md")

    def write_query(self, query: str) -> WriteResult:
        """Write original query as a file.

        Args:
            query: Original research query

        Returns:
            WriteResult
        """
        return self._write_markdown(
            path="/state/query.md",
            content=f"# Research Query\n\n{query}"
        )

    def read_query(self) -> Optional[str]:
        """Read original query from file.

        Returns:
            Query string or None
        """
        content = self._read_markdown("/state/query.md")

        if not content:
            return None

        # Extract query after "# Research Query"
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == "# Research Query":
                return "\n".join(lines[i+1:]).strip()

        return content.strip()

    def write_config(self, config: dict) -> WriteResult:
        """Write config as JSON file.

        Args:
            config: Config dict

        Returns:
            WriteResult
        """
        return self._write_json("/state/config.json", config)

    def read_config(self) -> Optional[dict]:
        """Read config from file.

        Returns:
            Config dict or None
        """
        return self._read_json("/state/config.json")

    def write_stats(self, stats: dict) -> WriteResult:
        """Write session statistics as JSON file.

        Args:
            stats: Statistics dict

        Returns:
            WriteResult
        """
        return self._write_json("/state/stats.json", stats)

    def read_stats(self) -> Optional[dict]:
        """Read statistics from file.

        Returns:
            Statistics dict or None
        """
        return self._read_json("/state/stats.json")

    def update_phase(self, phase: str) -> WriteResult:
        """Write current phase to state.

        Args:
            phase: Current phase name

        Returns:
            WriteResult
        """
        content = f"# Current Phase\n\n{phase}"
        file_data = create_file_data(content)
        return WriteResult(
            path="/state/phase.md",
            files_update={"/state/phase.md": file_data}
        )

    def read_phase(self) -> Optional[str]:
        """Read current phase from file.

        Returns:
            Phase name or None
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get("/state/phase.md")

        if not file_data:
            return None

        content = file_data_to_string(file_data)
        lines = content.split("\n")

        for line in lines:
            if line and not line.startswith("#"):
                return line.strip()

        return None

    # ========================================================================
    # Standard ls_info Override
    # ========================================================================

    def ls_info(self, path: str) -> list[FileInfo]:
        """List state files.

        Args:
            path: Path to list

        Returns:
            List of FileInfo dicts
        """
        files = self.runtime.state.get("files", {})

        if path == "/" or path == "/state" or path == "/state/":
            # List all state files
            infos = []
            for file_path, file_data in files.items():
                if file_path.startswith("/state/"):
                    size = len("\n".join(file_data.get("content", [])))
                    infos.append({
                        "path": file_path,
                        "is_dir": False,
                        "size": size,
                        "modified_at": file_data.get("modified_at", "")
                    })
            return sorted(infos, key=lambda x: x.get("path", ""))

        # List specific directory
        normalized_path = path if path.endswith("/") else path + "/"

        infos = []
        for file_path, file_data in files.items():
            if file_path.startswith(normalized_path):
                # Check if it's a direct child (not in subdirectory)
                relative = file_path[len(normalized_path):]
                if "/" not in relative:
                    size = len("\n".join(file_data.get("content", [])))
                    infos.append({
                        "path": file_path,
                        "is_dir": False,
                        "size": size,
                        "modified_at": file_data.get("modified_at", "")
                    })

        return sorted(infos, key=lambda x: x.get("path", ""))


class OutputBackend(StateBackend):
    """Backend for final research output.

    Similar to ResearchStateBackend but specifically for the final synthesis
    output and intermediate drafts.

    Path Structure:
        /output/
        ├── draft.md          → Current draft
        ├── final.md          → Final output
        ├── citations.json    → Citation data
        └── metadata.json     → Output metadata
    """

    def __init__(self, runtime: Any):
        """Initialize OutputBackend.

        Args:
            runtime: LangGraph ToolRuntime
        """
        super().__init__(runtime)
        self.runtime = runtime

    def write_draft(self, content: str) -> WriteResult:
        """Write draft output.

        Args:
            content: Draft content

        Returns:
            WriteResult
        """
        file_data = create_file_data(content)
        return WriteResult(
            path="/output/draft.md",
            files_update={"/output/draft.md": file_data}
        )

    def read_draft(self) -> Optional[str]:
        """Read draft output.

        Returns:
            Draft content or None
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get("/output/draft.md")

        if not file_data:
            return None

        return file_data_to_string(file_data)

    def write_final(self, content: str) -> WriteResult:
        """Write final output.

        Args:
            content: Final content

        Returns:
            WriteResult
        """
        file_data = create_file_data(content)
        return WriteResult(
            path="/output/final.md",
            files_update={"/output/final.md": file_data}
        )

    def read_final(self) -> Optional[str]:
        """Read final output.

        Returns:
            Final content or None
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get("/output/final.md")

        if not file_data:
            return None

        return file_data_to_string(file_data)

    def write_citations(self, citations: list[dict]) -> WriteResult:
        """Write citation data as JSON.

        Args:
            citations: List of citation dicts

        Returns:
            WriteResult
        """
        return self._write_json("/output/citations.json", citations)

    def read_citations(self) -> list[dict]:
        """Read citation data.

        Returns:
            List of citation dicts
        """
        return self._read_json("/output/citations.json", default=[])

    def write_metadata(self, metadata: dict) -> WriteResult:
        """Write output metadata.

        Args:
            metadata: Metadata dict

        Returns:
            WriteResult
        """
        return self._write_json("/output/metadata.json", metadata)

    def read_metadata(self) -> dict:
        """Read output metadata.

        Returns:
            Metadata dict (empty dict if not found)
        """
        return self._read_json("/output/metadata.json", default={})
