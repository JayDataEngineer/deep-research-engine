"""
Personality Layer

Applies consistent personality to responses.
Integrates semantic-rp-system components (prompt stack, author's note, lorebook).
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys
from enum import Enum

logger = logging.getLogger(__name__)

from src.constants import PROJECT_ROOT
from ..config.personality_config import PersonalityConfig, CommunicationStyle, MoodMatching
from ..state.brain_state import ConversationState, ProcessedInput, UserMood
from ..state.intent_types import IntentType

# Add src to path for imports
src_path = PROJECT_ROOT / "backend" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import semantic-rp-system components
# Note: Classes have been renamed in the new prompt_engine structure
from src.extensions.libraries.prompt_engine.schema import PromptStack, StackLayer, StackVariable

# Import prompt loader for YAML-based prompts
from src.extensions.libraries.prompt_loader import get_prompt_loader


# ============================================================================
# STUB CLASSES for missing prompt_engine functionality
# These should be properly implemented in prompt_engine later
# ============================================================================

class MarkerType(Enum):
    """Marker types for prompt layers."""
    KNOWLEDGE = "knowledge"
    DIALOGUE_EXAMPLES = "dialogue_examples"
    CHAT_HISTORY = "chat_history"
    AUTHOR_NOTE = "author_note"
    SYSTEM = "system"


class AuthorsNote:
    """Stub for AuthorsNote functionality."""

    def __init__(self):
        self._default_note = ""
        self._depth = 4
        self._interval = 1
        self._notes = {}

    def set_default(self, content: str, depth: int = 4, interval: int = 1):
        """Set default author's note."""
        self._default_note = content
        self._depth = depth
        self._interval = interval

    def get_note_for_turn(self, turn_number: int) -> str:
        """Get author's note for a specific turn."""
        # Check if there's a specific note for this turn
        if turn_number in self._notes:
            return self._notes[turn_number]
        # Use interval-based notes
        if self._interval > 0 and turn_number % self._interval == 0:
            return self._default_note
        return ""

    def set_note_for_turn(self, turn_number: int, content: str):
        """Set a note for a specific turn."""
        self._notes[turn_number] = content


class VariableSubstitutor:
    """Stub for variable substitution functionality."""

    def __init__(self):
        self._vars = {}

    def set(self, key: str, value: str):
        """Set a variable."""
        self._vars[key] = value

    def get(self, key: str, default: str = "") -> str:
        """Get a variable."""
        return self._vars.get(key, default)

    def substitute(self, template: str, variables: Dict[str, str]) -> str:
        """Substitute variables in template."""
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{{ {key} }}}}", value)
            result = result.replace(f"{{{{{key}}}}}", value)
        return result


class PromptLayer:
    """
    Wrapper around StackLayer with additional marker functionality.

    This bridges the old semantic-rp-system PromptLayer with the new StackLayer.
    """

    def __init__(
        self,
        content: str,
        role: str = "system",
        injection_depth: int = 0,
        injection_position: str = "before",
        identifier: str = None,
        forbid_overrides: bool = False,
        is_marker: bool = False,
        marker_type: MarkerType = None,
    ):
        self.content = content
        self.role = role
        self.injection_depth = injection_depth
        self.injection_position = injection_position
        self.identifier = identifier or ""
        self.forbid_overrides = forbid_overrides
        self.is_marker = is_marker
        self.marker_type = marker_type

    def to_stack_layer(self) -> StackLayer:
        """Convert to StackLayer schema."""
        return StackLayer(
            id=self.identifier,
            content=self.content,
            role=self.role,
        )


class EnhancedPromptStack(PromptStack):
    """Extended PromptStack with Jinja2 rendering and marker support."""

    def __init__(self, use_jinja: bool = True):
        """Initialize the enhanced prompt stack.

        Args:
            use_jinja: Whether to use Jinja2 rendering (True) or legacy string replacement (False)
        """
        super().__init__(name="personality_stack", layers=[])
        self._marker_content = {}
        self._use_jinja = use_jinja
        self._jinja_renderer = None

        # Lazy-load Jinja renderer if enabled
        if use_jinja:
            try:
                from jinja2 import Environment, BaseLoader
                # Create a simple Jinja2 environment for rendering
                self._jinja_env = Environment(loader=BaseLoader())
                logger.info("EnhancedPromptStack using Jinja2 rendering")
            except ImportError:
                logger.warning("Jinja2 not available, falling back to legacy rendering")
                self._use_jinja = False
                self._jinja_env = None

    def add_layer(self, layer: PromptLayer):
        """Add a PromptLayer to the stack."""
        # Convert to StackLayer and add
        stack_layer = layer.to_stack_layer()
        # We need to append to the layers list
        if not hasattr(self, 'layers'):
            self.layers = []
        self.layers.append(stack_layer)

    def set_marker_content(self, marker_type: MarkerType, content: str, role: str = "system"):
        """Set content for a marker type."""
        self._marker_content[marker_type] = (content, role)

    def clear_marker_content(self):
        """Clear all marker content."""
        self._marker_content.clear()

    def render(self, variables: Dict[str, str], max_depth: int = 4) -> List[Dict[str, str]]:
        """Render prompts up to max_depth using Jinja2 or legacy replacement."""
        messages = []

        # Prepare marker content for template variables
        marker_vars = {}
        for marker_type, (marker_content, _role) in self._marker_content.items():
            marker_vars[f"{marker_type.value}_marker"] = marker_content

        # Merge with regular variables
        all_vars = {**variables, **marker_vars}

        # Get layers up to max_depth
        if hasattr(self, 'layers'):
            for layer in self.layers:
                if layer.injection_depth <= max_depth:
                    content = layer.content

                    # Use Jinja rendering if enabled
                    if self._use_jinja and self._jinja_env:
                        try:
                            from jinja2 import Template
                            template = self._jinja_env.from_string(content)
                            content = template.render(**all_vars)
                        except Exception as e:
                            logger.warning(f"Jinja rendering failed for layer {layer.id}: {e}, falling back to legacy")
                            # Fallback to legacy replacement
                            content = self._legacy_replace(content, all_vars)
                    else:
                        # Legacy string replacement
                        content = self._legacy_replace(content, all_vars)

                    messages.append({"role": layer.role, "content": content})

        return messages

    def _legacy_replace(self, content: str, variables: Dict[str, str]) -> str:
        """Legacy string replacement for backward compatibility."""
        # First replace marker placeholders
        for key, value in variables.items():
            if "_marker" in key.lower():
                placeholder = f"[{key.upper().replace('_MARKER', '_MARKER')}]"
                content = content.replace(placeholder, value)

        # Then regular variables
        for key, value in variables.items():
            content = content.replace(f"{{{{ {key} }}}}", value)
            content = content.replace(f"{{{{{key}}}}}", value)

        return content


# Alias for compatibility
PromptStack = EnhancedPromptStack


class PersonalityLayer:
    """
    Apply consistent personality to responses.

    When personality_enabled=False, uses minimal prompts (Pure Assistant mode).
    When personality_enabled=True, adds personality layers to the stack.

    Reuses semantic-rp-system components:
    - PromptStack for depth-based prompts
    - AuthorsNote for per-turn guidance
    - VariableSubstitutor for template variables
    """

    def __init__(
        self,
        config: PersonalityConfig,
        mode: str = "companion",  # "companion" or "rp"
        personality_enabled: bool = True,  # Toggle for personality prompts
    ):
        """
        Initialize personality layer.

        Args:
            config: Personality configuration
            mode: Operating mode
            personality_enabled: Whether to include personality prompts (False = Pure Assistant)
        """
        self.config = config
        self.mode = mode
        self.personality_enabled = personality_enabled

        # Initialize prompt loader for YAML-based prompts
        self.prompt_loader = get_prompt_loader()

        # Initialize semantic-rp-system components
        self.prompt_stack = PromptStack()
        self.authors_note = AuthorsNote()
        self.variables = VariableSubstitutor()

        # Build initial prompt stack
        self._build_prompt_stack()

    def _build_prompt_stack(self) -> None:
        """Build the prompt stack from personality config."""
        # Depth 0: System prompt (always included)
        self.prompt_stack.add_layer(PromptLayer(
            content=self._get_system_prompt(),
            role="system",
            injection_depth=0,
            injection_position="before",
            identifier="system",
            forbid_overrides=True,
        ))

        # Only add personality layers if enabled
        if not self.personality_enabled:
            return

        # Depth 1: Personality
        self.prompt_stack.add_layer(PromptLayer(
            content=self._get_personality_prompt(),
            role="system",
            injection_depth=1,
            injection_position="before",
            identifier="personality",
        ))

        # Depth 2: Context marker
        self.prompt_stack.add_layer(PromptLayer(
            content="# Context\n\n{{scenario}}\n\n{{summary}}",
            role="system",
            injection_depth=2,
            injection_position="before",
            is_marker=True,
            marker_type=MarkerType.KNOWLEDGE,
            identifier="context",
        ))

        # Depth 3: Examples marker
        self.prompt_stack.add_layer(PromptLayer(
            content="# Examples\n\n{{examples}}",
            role="system",
            injection_depth=3,
            injection_position="before",
            is_marker=True,
            marker_type=MarkerType.DIALOGUE_EXAMPLES,
            identifier="examples",
        ))

        # Depth 4: Chat history marker
        self.prompt_stack.add_layer(PromptLayer(
            content="[CHAT_HISTORY_MARKER]",
            role="system",
            injection_depth=3,
            injection_position="before",
            is_marker=True,
            marker_type=MarkerType.CHAT_HISTORY,
            identifier="chat_history",
        ))

        # Depth 4: Author's note marker
        self.prompt_stack.add_layer(PromptLayer(
            content="[AUTHOR_NOTE_MARKER]",
            role="system",
            injection_depth=4,
            injection_position="after",
            is_marker=True,
            marker_type=MarkerType.AUTHOR_NOTE,
            identifier="author_note",
        ))

    def _get_system_prompt(self) -> str:
        """Get system prompt based on mode and personality_enabled."""
        if not self.personality_enabled:
            # Pure Assistant mode - minimal prompt from YAML
            return self.prompt_loader.get_system_prompt("assistant")

        # Load from YAML with template variable substitution
        template = self.prompt_loader.get_system_prompt(self.mode)
        return template.format(
            name=self.config.name,
            role=self.config.role,
            description=self.config.description.strip(),
        )

    def _get_personality_prompt(self) -> str:
        """Get personality prompt from YAML configuration."""
        parts = []

        # Communication style from YAML
        style_key = self.config.communication_style.value if hasattr(self.config.communication_style, 'value') else str(self.config.communication_style).lower()
        style_prompt = self.prompt_loader.get_personality_style(style_key)
        if style_prompt:
            parts.append(style_prompt)

        # Response voice
        parts.append(f"Voice: {self.config.response_voice}")

        # Formality from YAML
        if self.config.formality_level > 0.7:
            brain = self.prompt_loader.load_brain_prompts()
            parts.append(brain.get("formality", {}).get("high", "Maintain a formal tone."))
        elif self.config.formality_level < 0.3:
            brain = self.prompt_loader.load_brain_prompts()
            parts.append(brain.get("formality", {}).get("low", "Be very casual and informal."))

        # Humor from YAML
        if self.config.humor_level > 0.5:
            brain = self.prompt_loader.load_brain_prompts()
            parts.append(brain.get("humor", {}).get("high", "Use humor and wit when appropriate."))
        elif self.config.humor_level < 0.2:
            brain = self.prompt_loader.load_brain_prompts()
            parts.append(brain.get("humor", {}).get("low", "Be serious and focused."))

        return "\n".join(parts)

    def enhance_input(
        self,
        input_text: str,
        context: ConversationState,
    ) -> str:
        """Add personality context to input before LLM (if needed)."""
        # For companion mode, we don't modify input
        # The personality is applied via prompt stack
        return input_text

    def apply_to_response(
        self,
        response: str,
        context: ConversationState,
        intent: IntentType,
    ) -> str:
        """Post-process response with personality."""
        # Apply mood matching
        if self.config.mood_matching != MoodMatching.NONE and context.user_mood:
            response = self._apply_mood_matching(response, context.user_mood)

        # Add personality flourishes based on style
        response = self._apply_style_flourishes(response, intent)

        return response

    def _apply_mood_matching(self, response: str, user_mood: UserMood) -> str:
        """Apply mood matching to response."""
        if self.config.mood_matching == MoodMatching.MIRROR:
            # Match user's emotional tone
            if user_mood == UserMood.HAPPY:
                return response  # Keep positive tone
            elif user_mood == UserMood.SAD:
                # Add gentle/empathetic prefix if not present
                if not any(word in response.lower() for word in ["sorry", "understand", "hear"]):
                    return "I understand. " + response
            elif user_mood == UserMood.ANGRY:
                # Add calming prefix
                return "I hear you. " + response

        elif self.config.mood_matching == MoodMatching.EMPATHIC:
            # Acknowledge emotions but stay grounded
            if user_mood in [UserMood.SAD, UserMood.ANGRY, UserMood.ANXIOUS]:
                if not any(word in response.lower() for word in ["understand", "hear", "here"]):
                    return "I'm here for you. " + response

        return response

    def _apply_style_flourishes(self, response: str, intent: IntentType) -> str:
        """Apply style-specific flourishes."""
        # Wit style
        if self.config.communication_style == CommunicationStyle.WITTY:
            if intent == IntentType.GREETING and self.config.humor_level > 0.5:
                # Add a playful element to greetings
                if not response.endswith(("!", "?", "...")):
                    response = response + "!"

        # Supportive style
        if self.config.communication_style == CommunicationStyle.SUPPORTIVE:
            if intent == IntentType.EMOTIONAL:
                # Add encouraging closing
                if not any(word in response.lower() for word in ["there for you", "here", "support"]):
                    response = response + " I'm here if you want to talk more."

        return response

    def build_prompt_variables(
        self,
        user_input: str,
        context: ConversationState,
        additional_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Build template variables for prompt rendering."""
        vars = {
            "char": self.config.name,
            "user": "User",
            "input": user_input,
            "scenario": f"Currently discussing: {context.current_topic or 'general conversation'}",
            "summary": context.summary or "",
            "examples": self._get_examples(),
        }

        # Add conversation context
        recent_context = context.get_recent_context(5)
        if recent_context:
            chat_history = self._format_chat_history(recent_context)
            vars["chat_history"] = chat_history

        # Add author's note
        note_content = self.authors_note.get_note_for_turn(
            turn_number=context.turn_count,
        )
        vars["author_note"] = note_content or ""

        # Add additional variables
        if additional_vars:
            vars.update(additional_vars)

        return vars

    def _get_examples(self) -> str:
        """Get dialogue examples from YAML based on mode and style."""
        examples = self.prompt_loader.get_examples(
            mode=self.mode,
            style=self.config.communication_style.value if hasattr(self.config.communication_style, 'value') else str(self.config.communication_style).lower(),
        )

        if not examples:
            # Fallback to simple default
            if self.mode == "rp":
                return """Example roleplay dialogue:
<START>
{{user}}: Hello!
{{char}}: *smiles warmly* Hey there! Good to see you."""
            else:
                return f"""Example conversation with {self.config.name}:
<START>
User: Hi {self.config.name}!
{self.config.name}: Hello! How can I help you today?"""

        # Format examples from YAML
        lines = ["Example dialogue:"]
        for ex in examples[:3]:  # Limit to 3 examples
            user_msg = ex.get("user", "")
            asst_msg = ex.get("assistant", "")
            note = ex.get("note", "")
            lines.append(f"User: {user_msg}")
            lines.append(f"{self.config.name}: {asst_msg}")
            if note:
                lines.append(f"({note})")
        return "\n".join(lines)

    def _format_chat_history(self, context: List[Dict[str, str]]) -> str:
        """Format chat history for prompt."""
        lines = []
        for msg in context:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"User: {content}")
            else:
                lines.append(f"{self.config.name}: {content}")
        return "\n".join(lines)

    def render_prompts(
        self,
        user_input: str,
        context: ConversationState,
        additional_vars: Optional[Dict[str, str]] = None,
        max_depth: int = 4,
    ) -> List[Dict[str, str]]:
        """
        Render prompts for LLM.

        Args:
            user_input: Current user input
            context: Conversation context
            additional_vars: Additional template variables
            max_depth: Maximum injection depth

        Returns:
            List of message dicts for LLM API
        """
        # Build variables
        variables = self.build_prompt_variables(user_input, context, additional_vars)

        # Set marker content
        self.prompt_stack.set_marker_content(
            MarkerType.CHAT_HISTORY,
            variables.get("chat_history", ""),
            role="system",
        )

        self.prompt_stack.set_marker_content(
            MarkerType.AUTHOR_NOTE,
            variables.get("author_note", ""),
            role="system",
        )

        # Render
        messages = self.prompt_stack.render(
            variables=variables,
            max_depth=max_depth,
        )

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Clear markers
        self.prompt_stack.clear_marker_content()

        return messages

    def set_author_note(
        self,
        content: str,
        depth: int = 4,
        interval: int = 1,
    ) -> None:
        """Set author's note for per-turn guidance."""
        self.authors_note.set_default(
            content=content,
            depth=depth,
            interval=interval,
        )

    def set_variable(self, key: str, value: str) -> None:
        """Set a template variable."""
        self.variables.set(key, value)

    def get_variable(self, key: str) -> Optional[str]:
        """Get a template variable."""
        return self.variables.get(key)

    def adapt_to_user(
        self,
        context: ConversationState,
        user_profile: Any,
    ) -> None:
        """Adapt personality based on learned user preferences."""
        if not self.config.adaptation_enabled:
            return

        # Could adjust style based on user preferences
        # For now, this is a placeholder for future learning


    def set_personality_enabled(self, enabled: bool) -> None:
        """Enable or disable personality prompts at runtime.

        Args:
            enabled: True to enable personality prompts, False for Pure Assistant mode
        """
        if self.personality_enabled == enabled:
            return  # No change

        self.personality_enabled = enabled

        # Rebuild the prompt stack with new setting
        self.prompt_stack = PromptStack()
        self._build_prompt_stack()


def create_personality_layer(
    config: PersonalityConfig = None,
    mode: str = "companion",
    personality_enabled: bool = True,
) -> PersonalityLayer:
    """Factory function to create PersonalityLayer.

    Args:
        config: Personality configuration
        mode: Operating mode ("companion" or "rp")
        personality_enabled: Whether to include personality prompts (False = Pure Assistant)

    Returns:
        PersonalityLayer instance
    """
    if config is None:
        from ..config.personality_config import get_preset
        config = get_preset("friendly")

    return PersonalityLayer(config=config, mode=mode, personality_enabled=personality_enabled)
