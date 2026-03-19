"""Group Chat Extension

Multi-character group conversations.
Provides comprehensive backend for managing multi-character chats.

CLI Usage:
    uv run backend/manage.py groups list
    uv run backend/manage.py groups create "My Group" --characters char1,char2,char3
    uv run backend/manage.py groups activate <group_id>
    uv run backend/manage.py groups members add <group_id> <char_id>
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class ActivationStrategy(Enum):
    """Group chat activation strategies."""
    NATURAL = 0  # Context-aware activation
    LIST = 1  # Sequential activation
    MANUAL = 2  # Manual selection


@dataclass
class GroupMember:
    """A member in a group chat."""

    character_id: str
    name: str
    enabled: bool = True
    talkativeness: float = 1.0  # How likely to respond (0.0-1.0)
    position: int = 0
    avatar: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupMember":
        return cls(**data)


@dataclass
class GroupChat:
    """A group chat configuration."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    members: List[GroupMember] = field(default_factory=list)
    activation_strategy: ActivationStrategy = ActivationStrategy.NATURAL
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["activation_strategy"] = self.activation_strategy.value
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["members"] = [m.to_dict() for m in self.members]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupChat":
        data = data.copy()
        data["activation_strategy"] = ActivationStrategy(data.get("activation_strategy", 0))
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        members_data = data.pop("members", [])
        group = cls(**data)
        group.members = [GroupMember(**m) for m in members_data]
        return group


class GroupChatExtension(LogicExtension):
    """Group Chat extension - Comprehensive multi-character chat backend.

    This extension provides full backend logic for managing multi-character
    conversations with various activation strategies.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage_dir = None
        self._groups: List[GroupChat] = []
        self._active_group_id: Optional[str] = None
        self._chat_histories: Dict[str, List[Dict[str, Any]]] = {}

    async def initialize(self, context):
        """Initialize the group chat extension."""
        await super().initialize(context)

        # Set up storage directory
        try:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            self._storage_dir = get_inputs_root_dir() / "groups"
        except (ImportError, AttributeError):
            self._storage_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "inputs"
                / "groups"
            )

        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Load groups
        self._load_groups()

        logger.info(f"Group Chat extension initialized with {len(self._groups)} groups")

    async def shutdown(self):
        """Shutdown the extension."""
        self._save_groups()

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            list                    - List all groups
            create <name>           - Create a new group
            info <group_id>          - Show group details
            activate <group_id>      - Activate a group
            deactivate              - Deactivate current group
            delete <group_id>        - Delete a group

            members list <group_id>             - List group members
            members add <group_id> <char_id>     - Add member
            members remove <group_id> <member_id> - Remove member
            members enable <group_id> <member_id> - Enable member
            members disable <group_id> <member_id> - Disable member
            members move <group_id> <member_id> <direction> - Move member

            chat list <group_id>               - List chat messages
            chat send <group_id> <message>      - Send message to group
            chat history <group_id>            - Get full chat history

            strategy set <group_id> <strategy>  - Set activation strategy
            strategy get <group_id>             - Get activation strategy

            active                  - Show active group
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("groups", "group", "gc", "groupchat") or args:
            subcommand = args[0] if args else "list"
            subargs = args[1:] if len(args) > 1 else []

            # Check for nested commands
            if subcommand in ("members", "chat", "strategy") and subargs:
                nested_cmd = subargs[0]
                nested_args = subargs[1:]
                async for result in self._handle_nested_command(subcommand, nested_cmd, nested_args):
                    yield result
            else:
                async for result in self._handle_cli_command(subcommand, subargs):
                    yield result
        else:
            # API mode
            operation = input_data.get("operation")
            async for result in self._handle_api_operation(operation, input_data):
                yield result

    async def _handle_cli_command(
        self,
        subcommand: str,
        args: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle CLI subcommands."""
        if subcommand == "list":
            yield await self._cmd_list()
        elif subcommand in ("create", "add", "new"):
            if not args:
                yield {"error": "Please provide a group name"}
                return
            yield await self._cmd_create(args[0], args[1:])
        elif subcommand in ("info", "show", "get"):
            if not args:
                yield await self._cmd_get_active()
            else:
                yield await self._cmd_info(args[0])
        elif subcommand == "activate":
            if not args:
                yield {"error": "Please provide a group ID"}
                return
            yield await self._cmd_activate(args[0])
        elif subcommand == "deactivate":
            yield await self._cmd_deactivate()
        elif subcommand in ("delete", "remove", "rm"):
            if not args:
                yield {"error": "Please provide a group ID"}
                return
            yield await self._cmd_delete(args[0])
        elif subcommand == "active":
            yield await self._cmd_get_active()
        elif subcommand == "export":
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_export(args[0])
        elif subcommand == "import":
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_import(args[0])
        else:
            yield {"error": f"Unknown subcommand: {subcommand}"}
            yield self._help_text()

    async def _handle_nested_command(
        self,
        category: str,
        subcommand: str,
        args: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle nested commands (members, chat, strategy)."""
        if category == "members":
            async for result in self._handle_members_command(subcommand, args):
                yield result
        elif category == "chat":
            async for result in self._handle_chat_command(subcommand, args):
                yield result
        elif category == "strategy":
            async for result in self._handle_strategy_command(subcommand, args):
                yield result

    async def _handle_members_command(
        self,
        subcommand: str,
        args: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle members subcommands."""
        if subcommand == "list":
            if not args:
                yield await self._cmd_members_list()
            else:
                yield await self._cmd_members_list(args[0])
        elif subcommand == "add":
            if len(args) < 2:
                yield {"error": "Usage: members add <group_id> <character_id>"}
                return
            yield await self._cmd_members_add(args[0], args[1])
        elif subcommand == "remove":
            if len(args) < 2:
                yield {"error": "Usage: members remove <group_id> <member_index>"}
                return
            yield await self._cmd_members_remove(args[0], args[1])
        elif subcommand == "enable":
            if len(args) < 2:
                yield {"error": "Usage: members enable <group_id> <member_index>"}
                return
            yield await self._cmd_members_enable(args[0], args[1])
        elif subcommand == "disable":
            if len(args) < 2:
                yield {"error": "Usage: members disable <group_id> <member_index>"}
                return
            yield await self._cmd_members_disable(args[0], args[1])
        elif subcommand == "move":
            if len(args) < 3:
                yield {"error": "Usage: members move <group_id> <member_index> <direction>"}
                return
            yield await self._cmd_members_move(args[0], args[1], args[2])
        elif subcommand == "talkativeness":
            if len(args) < 3:
                yield {"error": "Usage: members talkativeness <group_id> <member_index> <value>"}
                return
            yield await self._cmd_members_talkativeness(args[0], args[1], args[2])
        else:
            yield {"error": f"Unknown members subcommand: {subcommand}"}

    async def _handle_chat_command(
        self,
        subcommand: str,
        args: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle chat subcommands."""
        if subcommand == "list":
            if not args:
                yield await self._cmd_chat_list()
            else:
                yield await self._cmd_chat_list(args[0])
        elif subcommand in ("send", "write"):
            if len(args) < 2:
                yield {"error": "Usage: chat send <group_id> <message>"}
                return
            yield await self._cmd_chat_send(args[0], " ".join(args[1:]))
        elif subcommand == "history":
            if not args:
                yield await self._cmd_chat_history()
            else:
                yield await self._cmd_chat_history(args[0])
        elif subcommand == "clear":
            if not args:
                yield {"error": "Please provide a group ID"}
                return
            yield await self._cmd_chat_clear(args[0])
        else:
            yield {"error": f"Unknown chat subcommand: {subcommand}"}

    async def _handle_strategy_command(
        self,
        subcommand: str,
        args: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle strategy subcommands."""
        if subcommand == "set":
            if len(args) < 2:
                yield {"error": "Usage: strategy set <group_id> <strategy>"}
                return
            yield await self._cmd_strategy_set(args[0], args[1])
        elif subcommand == "get":
            if not args:
                yield await self._cmd_strategy_get_active()
            else:
                yield await self._cmd_strategy_get(args[0])
        else:
            yield {"error": f"Unknown strategy subcommand: {subcommand}"}

    async def _handle_api_operation(
        self,
        operation: Optional[str],
        input_data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle API operations."""
        if operation == "list_groups":
            yield {"groups": [g.to_dict() for g in self._groups]}
        elif operation == "create_group":
            name = input_data.get("name")
            if not name:
                yield {"error": "name required"}
                return
            group = await self._create_group(
                name=name,
                description=input_data.get("description", ""),
                character_ids=input_data.get("character_ids", []),
            )
            yield {"group": group.to_dict()}
        elif operation == "get_group":
            group_id = input_data.get("id")
            if not group_id:
                yield {"error": "id required"}
                return
            group = self._find_group(group_id)
            if group:
                yield {"group": group.to_dict()}
            else:
                yield {"error": "Group not found"}
        elif operation == "activate_group":
            group_id = input_data.get("id")
            if not group_id:
                yield {"error": "id required"}
                return
            success = self._activate_group(group_id)
            if success:
                yield {"success": True, "active_id": self._active_group_id}
            else:
                yield {"error": "Group not found"}
        elif operation == "deactivate_group":
            self._deactivate_group()
            yield {"success": True, "active_id": None}
        elif operation == "delete_group":
            group_id = input_data.get("id")
            if not group_id:
                yield {"error": "id required"}
                return
            success = self._delete_group(group_id)
            if success:
                self._save_groups()
                yield {"success": True}
            else:
                yield {"error": "Group not found"}
        elif operation == "add_member":
            group_id = input_data.get("group_id")
            character_id = input_data.get("character_id")
            if not group_id or not character_id:
                yield {"error": "group_id and character_id required"}
                return
            member = await self._add_member(group_id, character_id)
            if member:
                self._save_groups()
                yield {"member": member.to_dict()}
            else:
                yield {"error": "Group not found"}
        elif operation == "remove_member":
            group_id = input_data.get("group_id")
            member_index = input_data.get("member_index")
            if group_id is None or member_index is None:
                yield {"error": "group_id and member_index required"}
                return
            success = self._remove_member(group_id, member_index)
            if success:
                self._save_groups()
                yield {"success": True}
            else:
                yield {"error": "Failed to remove member"}
        elif operation == "send_message":
            group_id = input_data.get("group_id")
            message = input_data.get("message")
            if not group_id or not message:
                yield {"error": "group_id and message required"}
                return
            result = await self._send_message(group_id, message)
            yield {"result": result}
        elif operation == "get_chat_history":
            group_id = input_data.get("id")
            if not group_id:
                yield {"error": "id required"}
                return
            history = self._chat_histories.get(group_id, [])
            yield {"history": history}
        elif operation == "get_active":
            if self._active_group_id:
                group = self._find_group(self._active_group_id)
                if group:
                    yield {"group": group.to_dict()}
                else:
                    yield {"active": None}
            else:
                yield {"active": None}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Storage Methods
    # ========================================================================

    def _load_groups(self):
        """Load groups from storage."""
        groups_file = self._storage_dir / "groups.json"
        if groups_file.exists():
            try:
                with open(groups_file, "r") as f:
                    data = json.load(f)
                    self._groups = [GroupChat.from_dict(g) for g in data.get("groups", [])]
                    self._active_group_id = data.get("active_id")
            except Exception as e:
                logger.error(f"Error loading groups: {e}")

        # Load chat histories
        history_dir = self._storage_dir / "chats"
        if history_dir.exists():
            for history_file in history_dir.glob("*.json"):
                try:
                    with open(history_file, "r") as f:
                        self._chat_histories[history_file.stem] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading chat history: {e}")

        # Create default group if none exist
        if not self._groups:
            self._create_default_group()

    def _save_groups(self):
        """Save groups to storage."""
        groups_file = self._storage_dir / "groups.json"
        with open(groups_file, "w") as f:
            json.dump({
                "groups": [g.to_dict() for g in self._groups],
                "active_id": self._active_group_id,
            }, f, indent=2)

        # Save chat histories
        history_dir = self._storage_dir / "chats"
        history_dir.mkdir(parents=True, exist_ok=True)
        for group_id, history in self._chat_histories.items():
            history_file = history_dir / f"{group_id}.json"
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)

    def _create_default_group(self):
        """Create a default group chat."""
        group = GroupChat(
            name="Default Group",
            description="A default group chat",
            activation_strategy=ActivationStrategy.NATURAL,
        )
        self._groups.append(group)
        self._save_groups()

    # ========================================================================
    # Group Management
    # ========================================================================

    def _find_group(self, identifier: str) -> Optional[GroupChat]:
        """Find a group by ID or name."""
        for group in self._groups:
            if group.id == identifier or group.name.lower() == identifier.lower():
                return group
        return None

    async def _create_group(
        self,
        name: str,
        description: str,
        character_ids: List[str],
    ) -> GroupChat:
        """Create a new group chat."""
        group = GroupChat(
            name=name,
            description=description,
        )

        # Add members
        for i, char_id in enumerate(character_ids):
            member = GroupMember(
                character_id=char_id,
                name=f"Character {i+1}",
                position=i,
            )
            group.members.append(member)

        self._groups.append(group)
        self._save_groups()

        return group

    def _activate_group(self, identifier: str) -> bool:
        """Activate a group by ID or name."""
        group = self._find_group(identifier)
        if group:
            self._active_group_id = group.id
            return True
        return False

    def _deactivate_group(self):
        """Deactivate the current group."""
        self._active_group_id = None

    def _delete_group(self, identifier: str) -> bool:
        """Delete a group by ID or name."""
        group = self._find_group(identifier)
        if group:
            self._groups.remove(group)

            # Clear active if this was the active group
            if self._active_group_id == group.id:
                self._active_group_id = None

            # Clear chat history
            if group.id in self._chat_histories:
                del self._chat_histories[group.id]

            return True
        return False

    # ========================================================================
    # Member Management
    # ========================================================================

    async def _add_member(
        self,
        group_id: str,
        character_id: str,
        name: Optional[str] = None,
    ) -> Optional[GroupMember]:
        """Add a member to a group."""
        group = self._find_group(group_id)
        if not group:
            return None

        # Try to get character name from character_id if not provided
        if not name:
            name = character_id.split("/")[-1].replace("_", " ").title()

        member = GroupMember(
            character_id=character_id,
            name=name,
            position=len(group.members),
        )

        group.members.append(member)
        group.updated_at = datetime.now()
        self._save_groups()

        return member

    def _remove_member(self, group_id: str, member_index: int) -> bool:
        """Remove a member by index."""
        group = self._find_group(group_id)
        if not group:
            return False

        try:
            member_index = int(member_index)
            if 0 <= member_index < len(group.members):
                group.members.pop(member_index)
                # Update positions
                for i, member in enumerate(group.members):
                    member.position = i
                group.updated_at = datetime.now()
                return True
        except (ValueError, IndexError):
            pass

        return False

    def _set_member_enabled(self, group_id: str, member_index: int, enabled: bool) -> bool:
        """Set enabled state of a member."""
        group = self._find_group(group_id)
        if not group:
            return False

        try:
            member_index = int(member_index)
            if 0 <= member_index < len(group.members):
                group.members[member_index].enabled = enabled
                group.updated_at = datetime.now()
                return True
        except (ValueError, IndexError):
            pass

        return False

    def _move_member(self, group_id: str, member_index: int, direction: str) -> bool:
        """Move a member up or down."""
        group = self._find_group(group_id)
        if not group:
            return False

        try:
            member_index = int(member_index)
            if not (0 <= member_index < len(group.members)):
                return False

            if direction in ("up", "u", "-1"):
                new_index = member_index - 1
            elif direction in ("down", "d", "1"):
                new_index = member_index + 1
            else:
                return False

            if 0 <= new_index < len(group.members):
                # Swap
                group.members[member_index], group.members[new_index] = (
                    group.members[new_index],
                    group.members[member_index],
                )
                # Update positions
                for i, member in enumerate(group.members):
                    member.position = i
                group.updated_at = datetime.now()
                return True
        except (ValueError, IndexError):
            pass

        return False

    def _set_member_talkativeness(self, group_id: str, member_index: int, value: float) -> bool:
        """Set member talkativeness value."""
        group = self._find_group(group_id)
        if not group:
            return False

        try:
            member_index = int(member_index)
            if 0 <= member_index < len(group.members):
                group.members[member_index].talkativeness = max(0.0, min(1.0, value))
                group.updated_at = datetime.now()
                return True
        except (ValueError, IndexError):
            pass

        return False

    # ========================================================================
    # Chat Management
    # ========================================================================

    async def _send_message(self, group_id: str, message: str) -> Dict[str, Any]:
        """Send a message to the group and get response."""
        group = self._find_group(group_id)
        if not group:
            return {"error": "Group not found"}

        # Add message to history
        if group_id not in self._chat_histories:
            self._chat_histories[group_id] = []

        self._chat_histories[group_id].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })

        # Determine which member should respond based on strategy
        responder = self._select_responder(group)

        if responder:
            # Simulate response (in real implementation, this would call LLM)
            response = f"[{responder.name}]: (This is a simulated response to: {message})"
        else:
            response = "[System]: No enabled members to respond."

        self._chat_histories[group_id].append({
            "role": "assistant",
            "name": responder.name if responder else "System",
            "content": response,
            "timestamp": datetime.now().isoformat(),
        })

        self._save_groups()

        return {
            "response": response,
            "responder": responder.to_dict() if responder else None,
            "group_id": group_id,
        }

    def _select_responder(self, group: GroupChat) -> Optional[GroupMember]:
        """Select which member should respond based on strategy."""
        enabled_members = [m for m in group.members if m.enabled]

        if not enabled_members:
            return None

        if group.activation_strategy == ActivationStrategy.LIST:
            # Sequential: pick the first enabled member
            return enabled_members[0]
        elif group.activation_strategy == ActivationStrategy.MANUAL:
            # Manual: return first member (user would select)
            return enabled_members[0]
        else:  # NATURAL
            # Weighted random based on talkativeness
            import random

            total_talkativeness = sum(m.talkativeness for m in enabled_members)
            if total_talkativeness == 0:
                return random.choice(enabled_members)

            rand_val = random.uniform(0, total_talkativeness)
            cumulative = 0
            for member in enabled_members:
                cumulative += member.talkativeness
                if rand_val <= cumulative:
                    return member

        return enabled_members[-1]

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all groups."""
        try:
            if not self._groups:
                return {
                    "output": "No groups found. Use 'create' to make one.",
                    "data": {"groups": []}
                }

            lines = [f"\nGroup Chats ({len(self._groups)} total):"]

            for group in self._groups:
                active = " [ACTIVE]" if group.id == self._active_group_id else ""
                strategy = group.activation_strategy.name
                lines.append(f"\n{group.name}{active}")
                lines.append(f"  ID: {group.id}")
                lines.append(f"  Members: {len(group.members)}")
                lines.append(f"  Strategy: {strategy}")
                if group.description:
                    lines.append(f"  Description: {group.description}")

            return {
                "output": "\n".join(lines),
                "data": {"groups": [g.to_dict() for g in self._groups]},
            }
        except Exception as e:
            logger.error(f"Error listing groups: {e}")
            return {"error": str(e)}

    async def _cmd_create(self, name: str, args: List[str]) -> Dict[str, Any]:
        """Create a new group."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Create group")
            parser.add_argument("--description", default="", help="Group description")
            parser.add_argument("--characters", default="", help="Comma-separated character IDs")
            parser.add_argument("--strategy", choices=["natural", "list", "manual"],
                               default="natural", help="Activation strategy")

            parsed = parser.parse_args(args)

            character_ids = parsed.characters.split(",") if parsed.characters else []

            group = await self._create_group(
                name=name,
                description=parsed.description,
                character_ids=character_ids,
            )

            # Set strategy
            strategy_map = {
                "natural": ActivationStrategy.NATURAL,
                "list": ActivationStrategy.LIST,
                "manual": ActivationStrategy.MANUAL,
            }
            group.activation_strategy = strategy_map[parsed.strategy]
            self._save_groups()

            return {
                "output": f"Group '{name}' created (ID: {group.id})",
                "data": {"group": group.to_dict()},
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_info(self, identifier: str) -> Dict[str, Any]:
        """Show group details."""
        try:
            group = self._find_group(identifier)
            if not group:
                return {"error": f"Group '{identifier}' not found"}

            lines = [f"\nGroup: {group.name}"]
            lines.append(f"ID: {group.id}")
            lines.append(f"Description: {group.description or 'None'}")
            lines.append(f"Strategy: {group.activation_strategy.name}")
            lines.append(f"Created: {group.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"\nMembers ({len(group.members)}):")

            for member in group.members:
                status = "✓" if member.enabled else "✗"
                lines.append(f"  {status} [{member.position}] {member.name}")
                lines.append(f"      Character ID: {member.character_id}")
                lines.append(f"      Talkativeness: {member.talkativeness}")

            chat_count = len(self._chat_histories.get(group.id, []))
            lines.append(f"\nChat messages: {chat_count}")

            return {
                "output": "\n".join(lines),
                "data": {"group": group.to_dict()},
            }
        except Exception as e:
            logger.error(f"Error getting group info: {e}")
            return {"error": str(e)}

    async def _cmd_activate(self, identifier: str) -> Dict[str, Any]:
        """Activate a group."""
        try:
            group = self._find_group(identifier)
            if not group:
                return {"error": f"Group '{identifier}' not found"}

            self._activate_group(identifier)
            self._save_groups()

            return {
                "output": f"Activated group: {group.name}",
                "data": {"group": group.to_dict()},
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_deactivate(self) -> Dict[str, Any]:
        """Deactivate current group."""
        self._deactivate_group()
        self._save_groups()
        return {"output": "Group deactivated", "data": {"active": None}}

    async def _cmd_delete(self, identifier: str) -> Dict[str, Any]:
        """Delete a group."""
        try:
            group = self._find_group(identifier)
            if not group:
                return {"error": f"Group '{identifier}' not found"}

            name = group.name
            success = self._delete_group(identifier)
            if success:
                self._save_groups()
                return {"output": f"Group '{name}' deleted"}
            return {"error": f"Failed to delete group '{identifier}'"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_get_active(self) -> Dict[str, Any]:
        """Get active group."""
        if self._active_group_id:
            return await self._cmd_info(self._active_group_id)
        return {"output": "No active group", "data": {"active": None}}

    async def _cmd_members_list(self, group_id: Optional[str] = None) -> Dict[str, Any]:
        """List group members."""
        if not group_id:
            group_id = self._active_group_id

        if not group_id:
            return {"error": "No group specified or active"}

        group = self._find_group(group_id)
        if not group:
            return {"error": f"Group '{group_id}' not found"}

        lines = [f"\nMembers of {group.name}:"]
        for member in group.members:
            status = "✓" if member.enabled else "✗"
            lines.append(f"  [{member.position}] {status} {member.name}")
            lines.append(f"      Character: {member.character_id}")
            lines.append(f"      Talkativeness: {member.talkativeness}")

        return {
            "output": "\n".join(lines),
            "data": {"members": [m.to_dict() for m in group.members]},
        }

    async def _cmd_members_add(self, group_id: str, character_id: str) -> Dict[str, Any]:
        """Add a member to a group."""
        member = await self._add_member(group_id, character_id)
        if member:
            return {
                "output": f"Member '{member.name}' added to group",
                "data": {"member": member.to_dict()},
            }
        return {"error": f"Failed to add member to group '{group_id}'"}

    async def _cmd_members_remove(self, group_id: str, member_index: str) -> Dict[str, Any]:
        """Remove a member from a group."""
        success = self._remove_member(group_id, member_index)
        if success:
            self._save_groups()
            return {"output": f"Member at position {member_index} removed"}
        return {"error": f"Failed to remove member at position {member_index}"}

    async def _cmd_members_enable(self, group_id: str, member_index: str) -> Dict[str, Any]:
        """Enable a member."""
        success = self._set_member_enabled(group_id, member_index, True)
        if success:
            self._save_groups()
            return {"output": f"Member at position {member_index} enabled"}
        return {"error": f"Failed to enable member at position {member_index}"}

    async def _cmd_members_disable(self, group_id: str, member_index: str) -> Dict[str, Any]:
        """Disable a member."""
        success = self._set_member_enabled(group_id, member_index, False)
        if success:
            self._save_groups()
            return {"output": f"Member at position {member_index} disabled"}
        return {"error": f"Failed to disable member at position {member_index}"}

    async def _cmd_members_move(self, group_id: str, member_index: str, direction: str) -> Dict[str, Any]:
        """Move a member up or down."""
        success = self._move_member(group_id, member_index, direction)
        if success:
            self._save_groups()
            return {"output": f"Member moved {direction}"}
        return {"error": f"Failed to move member"}

    async def _cmd_members_talkativeness(self, group_id: str, member_index: str, value: str) -> Dict[str, Any]:
        """Set member talkativeness."""
        try:
            value_float = float(value)
            success = self._set_member_talkativeness(group_id, member_index, value_float)
            if success:
                self._save_groups()
                return {"output": f"Talkativeness set to {value_float}"}
            return {"error": f"Failed to set talkativeness"}
        except ValueError:
            return {"error": "Invalid value. Must be a float between 0.0 and 1.0"}

    async def _cmd_chat_list(self, group_id: Optional[str] = None) -> Dict[str, Any]:
        """List chat messages."""
        if not group_id:
            group_id = self._active_group_id

        if not group_id:
            return {"error": "No group specified or active"}

        history = self._chat_histories.get(group_id, [])

        lines = [f"\nChat history for group {group_id}:"]
        if not history:
            lines.append("  (empty)")
        else:
            for msg in history[-20:]:  # Last 20 messages
                role = msg.get("role", "unknown")
                name = msg.get("name", "")
                content = msg.get("content", "")[:60]
                lines.append(f"  [{role}] {name}: {content}...")

        return {
            "output": "\n".join(lines),
            "data": {"history": history, "count": len(history)},
        }

    async def _cmd_chat_send(self, group_id: str, message: str) -> Dict[str, Any]:
        """Send a message to the group."""
        result = await self._send_message(group_id, message)
        if "error" not in result:
            return {
                "output": f"You: {message}\n{result['response']}",
                "data": result,
            }
        return result

    async def _cmd_chat_history(self, group_id: Optional[str] = None) -> Dict[str, Any]:
        """Get full chat history."""
        if not group_id:
            group_id = self._active_group_id

        if not group_id:
            return {"error": "No group specified or active"}

        history = self._chat_histories.get(group_id, [])

        lines = [f"\nFull chat history for group {group_id}:"]
        for msg in history:
            role = msg.get("role", "unknown")
            name = msg.get("name", "")
            timestamp = msg.get("timestamp", "")[:19]
            lines.append(f"\n[{timestamp}] [{role}] {name}:")
            lines.append(f"  {msg.get('content', '')}")

        return {
            "output": "\n".join(lines),
            "data": {"history": history},
        }

    async def _cmd_chat_clear(self, group_id: str) -> Dict[str, Any]:
        """Clear chat history for a group."""
        if group_id in self._chat_histories:
            self._chat_histories[group_id].clear()
            self._save_groups()
            return {"output": f"Chat history cleared for group {group_id}"}
        return {"error": f"No chat history for group {group_id}"}

    async def _cmd_strategy_set(self, group_id: str, strategy: str) -> Dict[str, Any]:
        """Set activation strategy."""
        group = self._find_group(group_id)
        if not group:
            return {"error": f"Group '{group_id}' not found"}

        strategy_map = {
            "natural": ActivationStrategy.NATURAL,
            "list": ActivationStrategy.LIST,
            "manual": ActivationStrategy.MANUAL,
        }

        strategy_lower = strategy.lower()
        if strategy_lower not in strategy_map:
            return {
                "error": f"Invalid strategy. Options: {', '.join(strategy_map.keys())}"
            }

        group.activation_strategy = strategy_map[strategy_lower]
        group.updated_at = datetime.now()
        self._save_groups()

        return {
            "output": f"Strategy set to {strategy_lower} for group {group.name}",
            "data": {"strategy": strategy_lower},
        }

    async def _cmd_strategy_get(self, group_id: Optional[str] = None) -> Dict[str, Any]:
        """Get activation strategy."""
        if not group_id:
            group_id = self._active_group_id

        if not group_id:
            return {"error": "No group specified or active"}

        group = self._find_group(group_id)
        if not group:
            return {"error": f"Group '{group_id}' not found"}

        return {
            "output": f"Strategy for {group.name}: {group.activation_strategy.name}",
            "data": {"strategy": group.activation_strategy.name},
        }

    async def _cmd_strategy_get_active(self) -> Dict[str, Any]:
        """Get active group's strategy."""
        if self._active_group_id:
            return await self._cmd_strategy_get(self._active_group_id)
        return {"output": "No active group", "data": {"strategy": None}}

    async def _cmd_export(self, file_path: str) -> Dict[str, Any]:
        """Export groups to file."""
        try:
            path = Path(file_path)
            if not path.suffix:
                path = path.with_suffix(".json")

            data = {
                "groups": [g.to_dict() for g in self._groups],
                "active_id": self._active_group_id,
                "chat_histories": self._chat_histories,
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            return {
                "output": f"Groups exported to {path}",
                "data": {"filepath": str(path)},
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import groups from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                path = self._storage_dir / file_path
                if not path.exists():
                    return {"error": f"File not found: {file_path}"}

            with open(path, "r") as f:
                data = json.load(f)

            imported = 0
            if "groups" in data:
                for group_data in data["groups"]:
                    group = GroupChat.from_dict(group_data)
                    self._groups.append(group)
                    imported += 1

            if "chat_histories" in data:
                self._chat_histories.update(data["chat_histories"])

            self._save_groups()

            return {
                "output": f"Imported {imported} groups from {path}",
                "data": {"imported": imported},
            }
        except Exception as e:
            logger.error(f"Error importing groups: {e}")
            return {"error": str(e)}

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Group Chat CLI

Commands:
    list                    List all groups
    create <name>           Create a new group
    info <group_id>          Show group details
    activate <group_id>      Activate a group
    deactivate              Deactivate current group
    delete <group_id>        Delete a group
    active                  Show active group

    members list <group_id>             List group members
    members add <group_id> <char_id>     - Add member
    members remove <group_id> <index>     - Remove member
    members enable <group_id> <index>     - Enable member
    members disable <group_id> <index>    - Disable member
    members move <group_id> <index> <dir> - Move member (up/down)
    members talkativeness <group_id> <index> <value> - Set talkativeness

    chat list <group_id>               - List chat messages
    chat send <group_id> <message>       - Send message to group
    chat history <group_id>             - Get full chat history
    chat clear <group_id>                - Clear chat history

    strategy set <group_id> <strategy>   - Set activation strategy (natural/list/manual)
    strategy get <group_id>              - Get activation strategy

    export <file>           - Export groups to file
    import <file>           - Import groups from file

Create Options:
    --description <text>    Group description
    --characters <ids>      Comma-separated character IDs
    --strategy <type>       Activation strategy (natural/list/manual)

Examples:
    uv run backend/manage.py groups list
    uv run backend/manage.py groups create "My Group" --characters char1,char2
    uv run backend/manage.py groups activate <group_id>
    uv run backend/manage.py groups members add <group_id> character123
    uv run backend/manage.py groups chat send <group_id> "Hello everyone!"
    uv run backend/manage.py groups strategy set <group_id> natural
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/groups", tags=["groups"])

        @router.get("/")
        async def list_groups():
            """List all groups."""
            return {"groups": [g.to_dict() for g in self._groups]}

        @router.post("/")
        async def create_group(request: dict):
            """Create a new group."""
            name = request.get("name")
            if not name:
                return {"error": "name required"}, 400
            group = await self._create_group(
                name=name,
                description=request.get("description", ""),
                character_ids=request.get("character_ids", []),
            )
            return {"group": group.to_dict()}

        @router.get("/{group_id}")
        async def get_group(group_id: str):
            """Get a group by ID."""
            group = self._find_group(group_id)
            if group:
                return {"group": group.to_dict()}
            return {"error": "Group not found"}, 404

        @router.put("/{group_id}/activate")
        async def activate_group(group_id: str):
            """Activate a group."""
            success = self._activate_group(group_id)
            if success:
                self._save_groups()
                return {"success": True, "active_id": self._active_group_id}
            return {"error": "Group not found"}, 404

        @router.delete("/{group_id}")
        async def delete_group(group_id: str):
            """Delete a group."""
            success = self._delete_group(group_id)
            if success:
                self._save_groups()
                return {"success": True}
            return {"error": "Group not found"}, 404

        @router.post("/{group_id}/members")
        async def add_member(group_id: str, request: dict):
            """Add a member to the group."""
            character_id = request.get("character_id")
            if not character_id:
                return {"error": "character_id required"}, 400
            member = await self._add_member(group_id, character_id, request.get("name"))
            if member:
                self._save_groups()
                return {"member": member.to_dict()}
            return {"error": "Group not found"}, 404

        @router.delete("/{group_id}/members/{member_index}")
        async def remove_member(group_id: str, member_index: int):
            """Remove a member from the group."""
            success = self._remove_member(group_id, member_index)
            if success:
                self._save_groups()
                return {"success": True}
            return {"error": "Member not found or invalid index"}, 404

        @router.post("/{group_id}/chat")
        async def send_message(group_id: str, request: dict):
            """Send a message to the group."""
            message = request.get("message")
            if not message:
                return {"error": "message required"}, 400
            result = await self._send_message(group_id, message)
            return result

        @router.get("/{group_id}/chat")
        async def get_chat_history(group_id: str):
            """Get chat history for a group."""
            history = self._chat_histories.get(group_id, [])
            return {"history": history, "count": len(history)}

        @router.delete("/{group_id}/chat")
        async def clear_chat(group_id: str):
            """Clear chat history for a group."""
            if group_id in self._chat_histories:
                self._chat_histories[group_id].clear()
                self._save_groups()
                return {"success": True}
            return {"error": "Group not found"}, 404

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(GroupChatExtension)
