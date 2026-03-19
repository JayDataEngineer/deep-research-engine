"""Extension Data Access Interface.

Provides a clean abstraction for extensions to access database operations
without directly coupling to the checkpointer implementation.

This interface allows extensions to:
- Read/write conversation state
- Manage threads and messages
- Access conversation history
- Perform database operations through a standardized API

Extensions should implement the DataAccessExtension interface to provide
data access capabilities.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass
from datetime import datetime


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Message:
    """Message data model."""
    id: str
    thread_id: str
    role: str  # "user", "assistant", "system", "tool"
    content: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    version: int = 1


@dataclass
class Thread:
    """Thread data model."""
    id: str
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    message_count: int = 0


@dataclass
class ThreadState:
    """Complete thread state with messages."""
    thread: Thread
    messages: List[Message]


# =============================================================================
# Data Access Interface
# =============================================================================

class DataAccessInterface(ABC):
    """Abstract interface for data access operations.

    Extensions can depend on this interface rather than concrete
    checkpointer implementations, enabling easier testing and
    flexibility in storage backends.
    """

    # =========================================================================
    # Thread Operations
    # =========================================================================

    @abstractmethod
    async def create_thread(
        self,
        thread_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Thread:
        """Create a new thread.

        Args:
            thread_id: Optional thread ID (auto-generated if None)
            title: Optional thread title
            metadata: Optional metadata dictionary

        Returns:
            Created Thread instance
        """
        pass

    @abstractmethod
    async def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get a thread by ID.

        Args:
            thread_id: Thread identifier

        Returns:
            Thread instance or None if not found
        """
        pass

    @abstractmethod
    async def update_thread(
        self,
        thread_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Thread]:
        """Update a thread.

        Args:
            thread_id: Thread identifier
            title: New title (optional)
            metadata: New metadata (optional)

        Returns:
            Updated Thread instance or None if not found
        """
        pass

    @abstractmethod
    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and all its messages.

        Args:
            thread_id: Thread identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def list_threads(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Thread]:
        """List all threads.

        Args:
            limit: Maximum number of threads to return
            offset: Number of threads to skip

        Returns:
            List of Thread instances
        """
        pass

    # =========================================================================
    # Message Operations
    # =========================================================================

    @abstractmethod
    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add a message to a thread.

        Args:
            thread_id: Thread identifier
            role: Message role ("user", "assistant", etc.)
            content: Message content
            parent_id: Optional parent message ID
            metadata: Optional metadata dictionary

        Returns:
            Created Message instance

        Raises:
            ValueError: If thread_id is invalid
        """
        pass

    @abstractmethod
    async def get_message(self, message_id: str) -> Optional[Message]:
        """Get a message by ID.

        Args:
            message_id: Message identifier

        Returns:
            Message instance or None if not found
        """
        pass

    @abstractmethod
    async def get_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Get all messages for a thread.

        Args:
            thread_id: Thread identifier
            limit: Optional maximum number of messages

        Returns:
            List of Message instances (chronological order)
        """
        pass

    @abstractmethod
    async def update_message(
        self,
        message_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        """Update a message.

        Args:
            message_id: Message identifier
            content: New content (optional)
            metadata: New metadata (optional)

        Returns:
            Updated Message instance or None if not found
        """
        pass

    @abstractmethod
    async def delete_message(self, message_id: str) -> bool:
        """Delete a message.

        Args:
            message_id: Message identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    # =========================================================================
    # Thread State Operations
    # =========================================================================

    @abstractmethod
    async def get_thread_state(self, thread_id: str) -> Optional[ThreadState]:
        """Get complete thread state including messages.

        Args:
            thread_id: Thread identifier

        Returns:
            ThreadState instance or None if not found
        """
        pass

    # =========================================================================
    # Search Operations
    # =========================================================================

    @abstractmethod
    async def search_threads(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Thread]:
        """Search threads by title or content.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching Thread instances
        """
        pass


# =============================================================================
# Checkpointer Adapter
# =============================================================================

class CheckpointerDataAccessAdapter(DataAccessInterface):
    """Adapts the checkpointer to the DataAccessInterface.

    This adapter wraps the existing checkpointer implementation
    to provide the standardized data access interface.
    """

    def __init__(self, checkpointer):
        """Initialize the adapter.

        Args:
            checkpointer: The shared checkpointer instance
        """
        self._checkpointer = checkpointer

    async def create_thread(
        self,
        thread_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Thread:
        """Create a new thread using the checkpointer."""
        import uuid
        from datetime import datetime

        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Initialize empty state in checkpointer
        config = {"configurable": {"thread_id": thread_id}}
        self._checkpointer.put(config, {})

        return Thread(
            id=thread_id,
            title=title,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {},
            message_count=0,
        )

    async def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get a thread from the checkpointer."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self._checkpointer.get(config, None)

        if state is None:
            return None

        # Extract thread metadata from state
        metadata = state.get("metadata", {})

        return Thread(
            id=thread_id,
            title=state.get("thread_title"),
            created_at=state.get("created_at"),
            updated_at=state.get("updated_at"),
            metadata=metadata,
            message_count=len(state.get("messages", [])),
        )

    async def update_thread(
        self,
        thread_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Thread]:
        """Update a thread in the checkpointer."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self._checkpointer.get(config, None)

        if state is None:
            return None

        # Update state
        if title is not None:
            state["thread_title"] = title
        if metadata is not None:
            state["metadata"] = {state.get("metadata", {}), **metadata}
        state["updated_at"] = datetime.now()

        self._checkpointer.put(config, state)

        return Thread(
            id=thread_id,
            title=state.get("thread_title"),
            created_at=state.get("created_at"),
            updated_at=state["updated_at"],
            metadata=state.get("metadata", {}),
            message_count=len(state.get("messages", [])),
        )

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread from the checkpointer."""
        config = {"configurable": {"thread_id": thread_id}}
        return self._checkpointer.delete(config)

    async def list_threads(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Thread]:
        """List threads from the checkpointer."""
        # This is a simplified implementation
        # The actual checkpointer may have a more efficient way to list threads
        config_keys = self._checkpointer.list_keys()
        threads = []

        for key in config_keys[offset:offset + limit]:
            # Extract thread_id from config key
            thread_id = key.split(":")[-1] if ":" in key else key
            thread = await self.get_thread(thread_id)
            if thread:
                threads.append(thread)

        return threads

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add a message using the checkpointer."""
        import uuid
        from datetime import datetime
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

        config = {"configurable": {"thread_id": thread_id}}
        state = self._checkpointer.get(config, None)

        if state is None:
            raise ValueError(f"Thread {thread_id} not found")

        # Create message
        message_id = str(uuid.uuid4())
        created_at = datetime.now()

        # Convert role to appropriate message type
        role_to_type = {
            "user": HumanMessage,
            "assistant": AIMessage,
            "system": SystemMessage,
            "tool": ToolMessage,
        }

        msg_class = role_to_type.get(role, HumanMessage)
        langchain_msg = msg_class(content=content)

        # Add to state messages
        if "messages" not in state:
            state["messages"] = []

        state["messages"].append(langchain_msg)
        state["updated_at"] = created_at

        # Save to checkpointer
        self._checkpointer.put(config, state)

        return Message(
            id=message_id,
            thread_id=thread_id,
            role=role,
            content=content,
            created_at=created_at,
            metadata=metadata,
            parent_id=parent_id,
        )

    async def get_message(self, message_id: str) -> Optional[Message]:
        """Get a message by ID."""
        # This would require indexing messages by ID
        # For now, return None as this requires checkpointer enhancements
        return None

    async def get_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Get messages for a thread from the checkpointer."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self._checkpointer.get(config, None)

        if state is None:
            return []

        messages = []
        for msg in state.get("messages", []):
            # Convert LangChain message to our Message model
            role_map = {
                "human": "user",
                "ai": "assistant",
                "system": "system",
                "tool": "tool",
            }

            messages.append(Message(
                id=str(hash(str(msg.content))),  # Temporary ID generation
                thread_id=thread_id,
                role=role_map.get(msg.type, msg.type),
                content=str(msg.content),
                created_at=datetime.now(),  # Would be stored in real implementation
            ))

        if limit:
            messages = messages[-limit:]

        return messages

    async def update_message(
        self,
        message_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        """Update a message.

        Note: This is not fully implemented as it requires
        message indexing in the checkpointer.
        """
        return None

    async def delete_message(self, message_id: str) -> bool:
        """Delete a message.

        Note: This is not fully implemented as it requires
        message indexing in the checkpointer.
        """
        return False

    async def get_thread_state(self, thread_id: str) -> Optional[ThreadState]:
        """Get complete thread state from the checkpointer."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self._checkpointer.get(config, None)

        if state is None:
            return None

        messages = await self.get_messages(thread_id)

        return ThreadState(
            thread=Thread(
                id=thread_id,
                title=state.get("thread_title"),
                created_at=state.get("created_at"),
                updated_at=state.get("updated_at"),
                metadata=state.get("metadata", {}),
                message_count=len(messages),
            ),
            messages=messages,
        )

    async def search_threads(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Thread]:
        """Search threads.

        Note: This is a simplified implementation.
        A full implementation would need text indexing.
        """
        threads = await self.list_threads(limit=limit * 2)  # Get more for filtering

        # Simple filter by title
        query_lower = query.lower()
        results = [
            t for t in threads
            if t.title and query_lower in t.title.lower()
        ]

        return results[:limit]


# =============================================================================
# Global Data Access Provider
# =============================================================================

_default_data_access: Optional[DataAccessInterface] = None


async def get_data_access(checkpointer=None) -> DataAccessInterface:
    """Get the global data access interface.

    Args:
        checkpointer: Optional checkpointer to wrap.
                     Uses shared checkpointer if not provided.

    Returns:
        DataAccessInterface instance
    """
    global _default_data_access

    if _default_data_access is None:
        if checkpointer is None:
            from src.shared.checkpointer import get_shared_checkpointer
            checkpointer = await get_shared_checkpointer()

        _default_data_access = CheckpointerDataAccessAdapter(checkpointer)

    return _default_data_access


def set_data_access(data_access: DataAccessInterface) -> None:
    """Set a custom data access implementation.

    Args:
        data_access: DataAccessInterface instance to use
    """
    global _default_data_access
    _default_data_access = data_access
