"""Probability Control Extension

Probability and random sampling management.
Provides CLI tools for managing probability settings and response variation.

CLI Usage:
    uv run backend/manage.py probability info
    uv run backend/manage.py probability set --temperature 0.8 --top-p 0.9
    uv run backend/manage.py probability profiles list
    uv run backend/manage.py probability profiles create creative --temperature 1.0
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from datetime import datetime

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class ProbabilityProfile:
    """A probability sampling profile."""

    def __init__(
        self,
        name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ):
        self.id = f"{name.lower().replace(' ', '_')}"
        self.name = name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbabilityProfile":
        profile = cls(
            name=data["name"],
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            top_k=data.get("top_k", 0),
            min_p=data.get("min_p", 0.0),
            repetition_penalty=data.get("repetition_penalty", 1.0),
            presence_penalty=data.get("presence_penalty", 0.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
        )
        if "created_at" in data:
            profile.created_at = datetime.fromisoformat(data["created_at"])
        return profile


class ProbabilityExtension(LogicExtension):
    """Probability extension - CLI wrapper for probability management.

    This extension provides CLI access to probability control tools
    for managing sampling parameters and response variation.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage_dir = None
        self._profiles: List[ProbabilityProfile] = []
        self._active_profile_id: Optional[str] = None
        self._current_settings = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }

    async def initialize(self, context):
        """Initialize the probability extension."""
        await super().initialize(context)

        # Set up storage directory
        try:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            self._storage_dir = get_inputs_root_dir() / "probability"
        except (ImportError, AttributeError):
            self._storage_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "inputs"
                / "probability"
            )

        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Load existing profiles
        self._load_profiles()

        logger.info("Probability extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        self._save_profiles()

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            info                    - Show current probability settings
            set --temperature N     - Set probability parameters
            reset                   - Reset to default settings
            profiles list           - List all probability profiles
            profiles create <name>  - Create a new profile
            profiles use <name>     - Use a profile
            profiles delete <name>  - Delete a profile
            randomize               - Randomize settings
            roll NdN                - Roll dice (e.g., 2d6)
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("probability", "prob", "random", "sampling", "rng") or args:
            subcommand = args[0] if args else "info"
            subargs = args[1:] if len(args) > 1 else []

            # Check if it's a profiles command
            if subcommand == "profiles" and subargs:
                profile_cmd = subargs[0]
                profile_args = subargs[1:]
                async for result in self._handle_profiles_command(profile_cmd, profile_args):
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
        if subcommand == "info":
            yield await self._cmd_info()
        elif subcommand == "set":
            yield await self._cmd_set(args)
        elif subcommand == "reset":
            yield await self._cmd_reset()
        elif subcommand in ("randomize", "random"):
            yield await self._cmd_randomize(args)
        elif subcommand in ("roll", "dice"):
            if not args:
                yield {"error": "Please provide dice notation (e.g., 2d6)"}
                return
            yield await self._cmd_roll(args[0])
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

    async def _handle_profiles_command(
        self,
        subcommand: str,
        args: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle profiles subcommands."""
        if subcommand == "list":
            yield await self._cmd_profiles_list()
        elif subcommand in ("create", "add", "new"):
            if not args:
                yield {"error": "Please provide a profile name"}
                return
            yield await self._cmd_profiles_create(args[0], args[1:])
        elif subcommand in ("use", "load", "activate"):
            if not args:
                yield {"error": "Please provide a profile name or ID"}
                return
            yield await self._cmd_profiles_use(args[0])
        elif subcommand in ("delete", "remove", "rm"):
            if not args:
                yield {"error": "Please provide a profile name or ID"}
                return
            yield await self._cmd_profiles_delete(args[0])
        else:
            yield {"error": f"Unknown profiles subcommand: {subcommand}"}

    async def _handle_api_operation(
        self,
        operation: Optional[str],
        input_data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle API operations."""
        if operation == "get_info":
            yield {"settings": self._current_settings, "active_profile": self._active_profile_id}
        elif operation == "set_settings":
            settings = input_data.get("settings", {})
            for key, value in settings.items():
                if key in self._current_settings:
                    self._current_settings[key] = value
            yield {"settings": self._current_settings}
        elif operation == "reset_settings":
            self._reset_settings()
            yield {"settings": self._current_settings}
        elif operation == "randomize_settings":
            self._randomize_settings()
            yield {"settings": self._current_settings}
        elif operation == "roll_dice":
            notation = input_data.get("notation", "1d6")
            result = self._roll_dice(notation)
            yield {"notation": notation, "result": result}
        elif operation == "list_profiles":
            yield {"profiles": [p.to_dict() for p in self._profiles]}
        elif operation == "create_profile":
            name = input_data.get("name")
            if not name:
                yield {"error": "name required"}
                return
            profile = self._add_profile(name, input_data.get("settings", {}))
            self._save_profiles()
            yield {"profile": profile.to_dict()}
        elif operation == "use_profile":
            profile_id = input_data.get("id") or input_data.get("name")
            if not profile_id:
                yield {"error": "id or name required"}
                return
            success = self._use_profile(profile_id)
            if success:
                yield {"success": True, "active_profile": self._active_profile_id}
            else:
                yield {"error": "Profile not found"}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Storage Methods
    # ========================================================================

    def _load_profiles(self):
        """Load profiles from storage."""
        profiles_file = self._storage_dir / "profiles.json"

        if not profiles_file.exists():
            # Create default profiles
            self._create_default_profiles()
            return

        try:
            with open(profiles_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._profiles = [ProbabilityProfile.from_dict(p) for p in data]

            # Load active profile
            active_file = self._storage_dir / "active.json"
            if active_file.exists():
                with open(active_file, "r") as f:
                    active_data = json.load(f)
                    self._active_profile_id = active_data.get("active_id")
                    if self._active_profile_id:
                        profile = self._find_profile(self._active_profile_id)
                        if profile:
                            self._current_settings = {
                                "temperature": profile.temperature,
                                "top_p": profile.top_p,
                                "top_k": profile.top_k,
                                "min_p": profile.min_p,
                                "repetition_penalty": profile.repetition_penalty,
                                "presence_penalty": profile.presence_penalty,
                                "frequency_penalty": profile.frequency_penalty,
                            }
        except Exception as e:
            logger.error(f"Error loading profiles: {e}")
            self._profiles = []

    def _save_profiles(self):
        """Save profiles to storage."""
        profiles_file = self._storage_dir / "profiles.json"
        with open(profiles_file, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in self._profiles], f, indent=2, ensure_ascii=False)

        # Save active profile
        active_file = self._storage_dir / "active.json"
        with open(active_file, "w") as f:
            json.dump({"active_id": self._active_profile_id}, f)

    def _create_default_profiles(self):
        """Create default probability profiles."""
        defaults = [
            {
                "name": "Balanced",
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 0,
                "repetition_penalty": 1.0,
            },
            {
                "name": "Creative",
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 0,
                "repetition_penalty": 1.0,
            },
            {
                "name": "Precise",
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.1,
            },
            {
                "name": "Chaotic",
                "temperature": 1.5,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 0.8,
            },
        ]

        self._profiles = [ProbabilityProfile(**d) for d in defaults]
        self._active_profile_id = "balanced"
        self._save_profiles()

    # ========================================================================
    # Profile Management
    # ========================================================================

    def _add_profile(
        self,
        name: str,
        settings: Dict[str, Any]
    ) -> ProbabilityProfile:
        """Add a new probability profile."""
        profile = ProbabilityProfile(
            name=name,
            temperature=settings.get("temperature", 0.7),
            top_p=settings.get("top_p", 0.9),
            top_k=settings.get("top_k", 0),
            min_p=settings.get("min_p", 0.0),
            repetition_penalty=settings.get("repetition_penalty", 1.0),
            presence_penalty=settings.get("presence_penalty", 0.0),
            frequency_penalty=settings.get("frequency_penalty", 0.0),
        )
        self._profiles.append(profile)
        return profile

    def _find_profile(self, identifier: str) -> Optional[ProbabilityProfile]:
        """Find a profile by ID or name."""
        for profile in self._profiles:
            if profile.id == identifier or profile.name.lower() == identifier.lower():
                return profile
        return None

    def _remove_profile(self, identifier: str) -> bool:
        """Remove a profile by ID or name."""
        profile = self._find_profile(identifier)
        if profile:
            self._profiles.remove(profile)
            if self._active_profile_id == profile.id:
                self._active_profile_id = None
            return True
        return False

    def _use_profile(self, identifier: str) -> bool:
        """Use a profile by ID or name."""
        profile = self._find_profile(identifier)
        if profile:
            self._active_profile_id = profile.id
            self._current_settings = {
                "temperature": profile.temperature,
                "top_p": profile.top_p,
                "top_k": profile.top_k,
                "min_p": profile.min_p,
                "repetition_penalty": profile.repetition_penalty,
                "presence_penalty": profile.presence_penalty,
                "frequency_penalty": profile.frequency_penalty,
            }
            return True
        return False

    def _reset_settings(self):
        """Reset to default settings."""
        self._current_settings = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }
        self._active_profile_id = None

    def _randomize_settings(self):
        """Randomize settings within reasonable ranges."""
        self._current_settings = {
            "temperature": random.uniform(0.1, 1.5),
            "top_p": random.uniform(0.7, 1.0),
            "top_k": random.choice([0, 20, 40, 50]),
            "min_p": random.uniform(0.0, 0.1),
            "repetition_penalty": random.uniform(0.8, 1.2),
            "presence_penalty": random.uniform(0.0, 0.5),
            "frequency_penalty": random.uniform(0.0, 0.5),
        }
        self._active_profile_id = None

    def _roll_dice(self, notation: str) -> List[int]:
        """Roll dice using NdN notation (e.g., 2d6)."""
        try:
            parts = notation.lower().split("d")
            if len(parts) != 2:
                return [0]

            count = int(parts[0])
            sides = int(parts[1])

            return [random.randint(1, sides) for _ in range(count)]
        except Exception:
            return [0]

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_info(self) -> Dict[str, Any]:
        """Show current probability settings."""
        try:
            lines = ["\nProbability Settings:"]
            if self._active_profile_id:
                profile = self._find_profile(self._active_profile_id)
                profile_name = profile.name if profile else self._active_profile_id
                lines.append(f"Active Profile: {profile_name}")
            else:
                lines.append("Active Profile: Custom")

            lines.append(f"Temperature: {self._current_settings['temperature']:.2f}")
            lines.append(f"Top P: {self._current_settings['top_p']:.2f}")
            lines.append(f"Top K: {self._current_settings['top_k']}")
            lines.append(f"Min P: {self._current_settings['min_p']:.2f}")
            lines.append(f"Repetition Penalty: {self._current_settings['repetition_penalty']:.2f}")
            lines.append(f"Presence Penalty: {self._current_settings['presence_penalty']:.2f}")
            lines.append(f"Frequency Penalty: {self._current_settings['frequency_penalty']:.2f}")

            return {
                "output": "\n".join(lines),
                "data": {"settings": self._current_settings, "active_profile": self._active_profile_id}
            }
        except Exception as e:
            logger.error(f"Error getting probability info: {e}")
            return {"error": str(e)}

    async def _cmd_set(self, args: List[str]) -> Dict[str, Any]:
        """Set probability parameters."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Set probability parameters")
            parser.add_argument("--temperature", type=float, help="Temperature (0.0-2.0)")
            parser.add_argument("--top-p", type=float, help="Top P (0.0-1.0)")
            parser.add_argument("--top-k", type=int, help="Top K")
            parser.add_argument("--min-p", type=float, help="Min P (0.0-1.0)")
            parser.add_argument("--repetition-penalty", type=float, help="Repetition penalty")
            parser.add_argument("--presence-penalty", type=float, help="Presence penalty")
            parser.add_argument("--frequency-penalty", type=float, help="Frequency penalty")

            parsed = parser.parse_args(args)

            if parsed.temperature is not None:
                self._current_settings["temperature"] = parsed.temperature
            if parsed.top_p is not None:
                self._current_settings["top_p"] = parsed.top_p
            if parsed.top_k is not None:
                self._current_settings["top_k"] = parsed.top_k
            if parsed.min_p is not None:
                self._current_settings["min_p"] = parsed.min_p
            if parsed.repetition_penalty is not None:
                self._current_settings["repetition_penalty"] = parsed.repetition_penalty
            if parsed.presence_penalty is not None:
                self._current_settings["presence_penalty"] = parsed.presence_penalty
            if parsed.frequency_penalty is not None:
                self._current_settings["frequency_penalty"] = parsed.frequency_penalty

            self._active_profile_id = None

            return {
                "output": "Probability settings updated",
                "data": {"settings": self._current_settings}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_reset(self) -> Dict[str, Any]:
        """Reset to default settings."""
        self._reset_settings()
        return {
            "output": "Probability settings reset to defaults",
            "data": {"settings": self._current_settings}
        }

    async def _cmd_randomize(self, args: List[str]) -> Dict[str, Any]:
        """Randomize settings."""
        self._randomize_settings()
        lines = ["Randomized probability settings:"]
        lines.append(f"Temperature: {self._current_settings['temperature']:.2f}")
        lines.append(f"Top P: {self._current_settings['top_p']:.2f}")
        lines.append(f"Top K: {self._current_settings['top_k']}")
        lines.append(f"Repetition Penalty: {self._current_settings['repetition_penalty']:.2f}")

        return {
            "output": "\n".join(lines),
            "data": {"settings": self._current_settings}
        }

    async def _cmd_roll(self, notation: str) -> Dict[str, Any]:
        """Roll dice."""
        try:
            results = self._roll_dice(notation)
            total = sum(results)

            lines = [f"Rolling {notation}:"]
            lines.append(f"Results: {', '.join(map(str, results))}")
            lines.append(f"Total: {total}")

            return {
                "output": "\n".join(lines),
                "data": {"notation": notation, "results": results, "total": total}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_profiles_list(self) -> Dict[str, Any]:
        """List all probability profiles."""
        try:
            if not self._profiles:
                return {"output": "No probability profiles found.", "data": {"profiles": []}}

            lines = ["\nProbability Profiles:"]
            for profile in self._profiles:
                active = " [ACTIVE]" if profile.id == self._active_profile_id else ""
                lines.append(f"  [{profile.id}]{active} {profile.name}")
                lines.append(f"      Temperature: {profile.temperature}, Top P: {profile.top_p}")

            return {
                "output": "\n".join(lines),
                "data": {"profiles": [p.to_dict() for p in self._profiles]}
            }
        except Exception as e:
            logger.error(f"Error listing profiles: {e}")
            return {"error": str(e)}

    async def _cmd_profiles_create(self, name: str, args: List[str]) -> Dict[str, Any]:
        """Create a new probability profile."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Create probability profile")
            parser.add_argument("--temperature", type=float, default=0.7)
            parser.add_argument("--top-p", type=float, default=0.9)
            parser.add_argument("--top-k", type=int, default=0)
            parser.add_argument("--repetition-penalty", type=float, default=1.0)

            parsed = parser.parse_args(args)

            settings = {
                "temperature": parsed.temperature,
                "top_p": parsed.top_p,
                "top_k": parsed.top_k,
                "repetition_penalty": parsed.repetition_penalty,
            }

            profile = self._add_profile(name, settings)
            self._save_profiles()

            return {
                "output": f"Probability profile '{name}' created",
                "data": {"profile": profile.to_dict()}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_profiles_use(self, identifier: str) -> Dict[str, Any]:
        """Use a probability profile."""
        try:
            profile = self._find_profile(identifier)
            if not profile:
                return {"error": f"Probability profile '{identifier}' not found"}

            self._use_profile(identifier)
            self._save_profiles()

            return {
                "output": f"Using probability profile '{profile.name}'",
                "data": {"profile": profile.to_dict()}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_profiles_delete(self, identifier: str) -> Dict[str, Any]:
        """Delete a probability profile."""
        try:
            profile = self._find_profile(identifier)
            if not profile:
                return {"error": f"Probability profile '{identifier}' not found"}

            name = profile.name
            success = self._remove_profile(identifier)
            if success:
                self._save_profiles()
                return {"output": f"Probability profile '{name}' deleted"}
            return {"error": f"Failed to delete profile '{identifier}'"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_export(self, file_path: str) -> Dict[str, Any]:
        """Export profiles to file."""
        try:
            path = Path(file_path)
            if not path.suffix:
                path = path.with_suffix(".json")

            data = {
                "profiles": [p.to_dict() for p in self._profiles],
                "active_id": self._active_profile_id,
                "current_settings": self._current_settings,
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return {
                "output": f"Probability profiles exported to {path}",
                "data": {"filepath": str(path)}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import profiles from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                path = self._storage_dir / file_path
                if not path.exists():
                    return {"error": f"File not found: {file_path}"}

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            imported = 0
            if "profiles" in data:
                for profile_data in data["profiles"]:
                    profile = ProbabilityProfile.from_dict(profile_data)
                    self._profiles.append(profile)
                    imported += 1

            self._save_profiles()

            return {
                "output": f"Imported {imported} probability profiles from {path}",
                "data": {"imported": imported}
            }
        except Exception as e:
            logger.error(f"Error importing profiles: {e}")
            return {"error": str(e)}

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Probability Control CLI

Commands:
    info                    Show current probability settings
    set --temperature N     Set probability parameters
    reset                   Reset to default settings
    randomize               Randomize settings
    roll NdN                Roll dice (e.g., 2d6)
    profiles list           List all probability profiles
    profiles create <name>  Create a new profile
    profiles use <name>     Use a profile
    profiles delete <name>  Delete a profile
    export <file>           Export profiles to file
    import <file>           Import profiles from file

Set Options:
    --temperature <float>   Temperature (0.0-2.0)
    --top-p <float>         Top P (0.0-1.0)
    --top-k <int>           Top K
    --min-p <float>         Min P (0.0-1.0)
    --repetition-penalty <float>  Repetition penalty
    --presence-penalty <float>    Presence penalty
    --frequency-penalty <float>   Frequency penalty

Examples:
    uv run backend/manage.py probability info
    uv run backend/manage.py probability set --temperature 0.8 --top-p 0.9
    uv run backend/manage.py probability profiles list
    uv run backend/manage.py probability profiles create creative --temperature 1.0
    uv run backend/manage.py probability profiles use creative
    uv run backend/manage.py probability roll 2d6
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/probability", tags=["probability"])

        @router.get("/info")
        async def get_info():
            """Get probability settings."""
            return {"settings": self._current_settings, "active_profile": self._active_profile_id}

        @router.post("/settings")
        async def set_settings(request: dict):
            """Set probability settings."""
            settings = request.get("settings", {})
            for key, value in settings.items():
                if key in self._current_settings:
                    self._current_settings[key] = value
            return {"settings": self._current_settings}

        @router.get("/profiles")
        async def list_profiles():
            """List all profiles."""
            return {"profiles": [p.to_dict() for p in self._profiles]}

        @router.post("/profiles")
        async def create_profile(request: dict):
            """Create a profile."""
            name = request.get("name")
            if not name:
                return {"error": "name required"}, 400
            profile = self._add_profile(name, request.get("settings", {}))
            self._save_profiles()
            return {"profile": profile.to_dict()}

        @router.put("/profiles/{profile_id}/use")
        async def use_profile(profile_id: str):
            """Use a profile."""
            success = self._use_profile(profile_id)
            if success:
                self._save_profiles()
                return {"success": True, "active_profile": self._active_profile_id}
            return {"error": "Profile not found"}, 404

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(ProbabilityExtension)
