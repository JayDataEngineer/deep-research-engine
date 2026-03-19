"""Prompt Stack Schema - Pydantic models for YAML-based prompt stacks.

Compatible with SillyTavern-style prompt management with YAML export/import.
Enhanced with Jinja2 rendering and LangChain message conversion.
"""

from typing import Optional, List, Dict, Any, Literal, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import re

# Jinja2 for advanced template rendering
from jinja2 import Template, TemplateError, StrictUndefined

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


class StackVariable(BaseModel):
    """A variable that can be substituted in prompt content."""

    name: str = Field(..., description="Variable name (e.g., 'char_name')")
    default: str = Field(..., description="Default value if not provided")
    description: Optional[str] = Field(None, description="Variable description")


class StackLayer(BaseModel):
    """A single layer in the prompt stack."""

    id: str = Field(..., description="Unique layer identifier")
    content: str = Field(..., description="Prompt content (can include variables)")
    role: Literal["system", "user", "assistant"] = Field("system", description="Message role")

    # Injection control
    injection_depth: int = Field(0, ge=0, description="Depth in conversation to inject (0=top)")
    injection_position: Literal["before", "after"] = Field("before", description="Before or after content at depth")
    enabled: bool = Field(True, description="Whether this layer is active")

    # Priority for sorting
    priority: int = Field(50, ge=0, le=100, description="Higher priority = injected first")

    # Metadata
    name: Optional[str] = Field(None, description="Human-readable layer name")
    comment: Optional[str] = Field(None, description="Notes about this layer")

    # Variable substitution
    variables: Dict[str, str] = Field(default_factory=dict, description="Custom variable values")

    # Extensions for custom behavior
    extensions: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")

    def render(self, **kwargs) -> str:
        """Render the layer content with Jinja2 template rendering.

        Supports:
        - Simple variable substitution: {{variable_name}}
        - Conditionals: {% if variable %}...{% endif %}
        - Loops: {% for item in items %}...{% endfor %}
        - Filters: {{variable|upper}}
        - Expressions: {{1 + 1}}

        Args:
            **kwargs: Variable values for template rendering

        Returns:
            Rendered content string
        """
        # Merge default variables with custom variables
        all_vars = {
            **self.extensions.get("default_variables", {}),
            **self.variables,
            **kwargs
        }

        content = self.content

        try:
            # Use Jinja2 for advanced template rendering
            template = Template(
                content,
                undefined=StrictUndefined,  # Fail loudly on missing variables
                trim_blocks=True,
                lstrip_blocks=True,
            )
            return template.render(**all_vars)
        except TemplateError:
            # Fallback to simple substitution for backward compatibility
            # This handles cases where content is not a valid Jinja2 template
            result = content
            for var_name, var_value in all_vars.items():
                result = result.replace(f"{{{{{var_name}}}}}", str(var_value))
                result = result.replace(f"{{{var_name}}}", str(var_value))
            return result


class PromptStack(BaseModel):
    """A complete prompt stack that can be saved to YAML.

    Compatible with SillyTavern's quick replies format but enhanced
    for layer-based prompt management.
    """

    # Identification
    name: str = Field(..., min_length=1, max_length=100, description="Stack name")
    description: Optional[str] = Field(None, description="What this stack does")
    version: str = Field("1.0", description="Stack version for compatibility")
    author: Optional[str] = Field(None, description="Creator name")

    # Categorization
    category: Literal["system", "personality", "context", "style", "custom"] = Field(
        "custom", description="Stack category for organization"
    )
    tags: List[str] = Field(default_factory=list, description="Searchable tags")

    # Core content
    layers: List[StackLayer] = Field(default_factory=list, description="Prompt layers in order")

    # Global variables available to all layers
    variables: List[StackVariable] = Field(default_factory=list, description="Available variables")

    # Default extensions
    extensions: Dict[str, Any] = Field(default_factory=dict, description="Global extensions")

    # Metadata
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation time")
    updated_at: Optional[datetime] = Field(default_factory=datetime.now, description="Last update")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is filename-safe."""
        # Remove invalid characters
        clean = re.sub(r'[<>:"/\\|?*]', '', v)
        # Replace spaces with underscores
        clean = re.sub(r'\s+', '_', clean)
        if not clean:
            raise ValueError("Name must contain valid characters")
        return clean

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Normalize tags to lowercase."""
        return [tag.lower().strip() for tag in v if tag.strip()]

    def get_enabled_layers(self) -> List[StackLayer]:
        """Get only enabled layers sorted by injection_depth (ascending), then priority (descending).

        Lower injection_depth = earlier in conversation.
        Higher priority = earlier within the same depth.
        """
        enabled = [layer for layer in self.layers if layer.enabled]
        return sorted(enabled, key=lambda l: (l.injection_depth, -l.priority))

    def render(self, **kwargs) -> str:
        """Render all enabled layers into a single prompt.

        Args:
            **kwargs: Variable values for substitution

        Returns:
            Concatenated prompt content
        """
        layers = self.get_enabled_layers()
        rendered = []

        for layer in layers:
            content = layer.render(**kwargs)
            if content.strip():
                rendered.append(f"[{layer.role}]\n{content}")

        return "\n\n".join(rendered)

    def add_layer(self, layer: StackLayer) -> None:
        """Add a layer to the stack."""
        # Check for duplicate ID
        existing_ids = {l.id for l in self.layers}
        if layer.id in existing_ids:
            raise ValueError(f"Layer ID '{layer.id}' already exists")
        self.layers.append(layer)
        self.updated_at = datetime.now()

    def update_layer(self, layer_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing layer."""
        for layer in self.layers:
            if layer.id == layer_id:
                for key, value in updates.items():
                    if hasattr(layer, key):
                        setattr(layer, key, value)
                self.updated_at = datetime.now()
                return True
        return False

    def remove_layer(self, layer_id: str) -> bool:
        """Remove a layer from the stack."""
        for i, layer in enumerate(self.layers):
            if layer.id == layer_id:
                self.layers.pop(i)
                self.updated_at = datetime.now()
                return True
        return False

    def to_langchain_messages(self, **kwargs) -> List["BaseMessage"]:
        """Convert the stack to LangChain messages for LLM invocation.

        This is the primary method for using prompt stacks with LLMs.
        It renders all enabled layers and converts them to proper LangChain
        message objects (SystemMessage, HumanMessage, AIMessage).

        Args:
            **kwargs: Variable values for template rendering

        Returns:
            List of LangChain BaseMessage objects

        Example:
            >>> stack = PromptStack(...)
            >>> messages = stack.to_langchain_messages(
            ...     char_name="Elena",
            ...     scenario="a mysterious forest",
            ...     user_name="Alex"
            ... )
            >>> llm.invoke(messages)
        """
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        messages = []
        layers = self.get_enabled_layers()

        for layer in layers:
            # Render the layer content
            content = layer.render(**kwargs)

            if not content.strip():
                # Skip empty layers
                continue

            # Convert to appropriate LangChain message type
            if layer.role == "system":
                messages.append(SystemMessage(content=content))
            elif layer.role == "user":
                messages.append(HumanMessage(content=content))
            elif layer.role == "assistant":
                messages.append(AIMessage(content=content))

        return messages

    def to_dict_messages(self, **kwargs) -> List[Dict[str, str]]:
        """Convert the stack to dict messages for LangGraph state.

        Returns messages in the format used by LangGraph state:
        [{"role": "system", "content": "..."}, ...]

        Args:
            **kwargs: Variable values for template rendering

        Returns:
            List of dict messages with role and content
        """
        messages = []
        layers = self.get_enabled_layers()

        for layer in layers:
            content = layer.render(**kwargs)

            if not content.strip():
                continue

            messages.append({
                "role": layer.role,
                "content": content
            })

        return messages


class StackRegistry(BaseModel):
    """Registry of all available prompt stacks.

    This is the master list that gets synced with YAML files.
    """

    stacks: Dict[str, PromptStack] = Field(default_factory=dict, description="Registered stacks by ID")
    last_sync: Optional[datetime] = Field(None, description="Last sync with filesystem")

    def register(self, stack: PromptStack) -> None:
        """Register a prompt stack."""
        stack_id = stack.name.lower().replace(" ", "_")
        self.stacks[stack_id] = stack

    def get(self, stack_id: str) -> Optional[PromptStack]:
        """Get a stack by ID."""
        return self.stacks.get(stack_id)

    def list_by_category(self, category: str) -> List[PromptStack]:
        """List all stacks in a category."""
        return [s for s in self.stacks.values() if s.category == category]

    def search(self, query: str) -> List[PromptStack]:
        """Search stacks by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for stack in self.stacks.values():
            # Search name
            if query_lower in stack.name.lower():
                results.append(stack)
                continue

            # Search description
            if stack.description and query_lower in stack.description.lower():
                results.append(stack)
                continue

            # Search tags
            if any(query_lower in tag for tag in stack.tags):
                results.append(stack)

        return results


# Default built-in stacks
DEFAULT_STACKS = {
    "system_default": PromptStack(
        name="system_default",
        description="Default system prompt for assistant behavior",
        category="system",
        tags=["system", "default", "assistant"],
        layers=[
            StackLayer(
                id="core_directive",
                name="Core Directive",
                content="You are a helpful AI assistant. Be concise, accurate, and friendly.",
                role="system",
                injection_depth=0,
                priority=100,
                enabled=True,
            ),
            StackLayer(
                id="safety_guidelines",
                name="Safety Guidelines",
                content="""- Do not generate harmful content
- Respect user preferences
- Ask for clarification when needed""",
                role="system",
                injection_depth=0,
                priority=90,
                enabled=True,
            ),
        ],
        extensions={
            "mood_matching": False,
            "style_flourish": "neutral",
        },
    ),

    "personality_friendly": PromptStack(
        name="personality_friendly",
        description="Warm and approachable companion personality",
        category="personality",
        tags=["personality", "friendly", "casual"],
        layers=[
            StackLayer(
                id="personality_core",
                name="Friendly Personality",
                content="""You have a warm, approachable personality. You:
- Use casual language and occasional humor
- Show genuine interest in the conversation
- Are supportive and encouraging
- Use emoticons sparingly to convey warmth 😊""",
                role="system",
                injection_depth=1,
                priority=80,
                enabled=True,
            ),
        ],
        extensions={
            "mood_matching": True,
            "style_flourish": "casual",
            "formality": 0.3,
            "humor": 0.7,
        },
    ),

    "roleplay_default": PromptStack(
        name="roleplay_default",
        description="Default roleplay mode with character immersion",
        category="system",
        tags=["roleplay", "character", "immersion"],
        layers=[
            StackLayer(
                id="rp_instructions",
                name="Roleplay Instructions",
                content="""You are roleplaying as {{char_name}}. Stay in character at all times.
- Respond as the character would, not as an AI
- Use the character's speech patterns and mannerisms
- Reference the scenario: {{scenario}}
- Maintain character consistency throughout""",
                role="system",
                injection_depth=0,
                priority=100,
                enabled=True,
            ),
        ],
        variables=[
            StackVariable(name="char_name", default="Character", description="Character name"),
            StackVariable(name="scenario", default="the current situation", description="Current scenario"),
        ],
    ),

    "deep_research": PromptStack(
        name="deep_research",
        description="Deep research workers with true prefills for character support",
        category="system",
        tags=["research", "worker", "prefill"],
        layers=[
            StackLayer(
                id="worker_role",
                name="Worker Role",
                content="""You are {{worker_id}}, a specialized research worker for the deep research system.
Your task is to perform thorough research and provide accurate findings with full URL citations.""",
                role="system",
                injection_depth=0,
                priority=100,
                enabled=True,
            ),
            StackLayer(
                id="personality_injection",
                name="Personality",
                content="""{{personality_text}}""",
                role="system",
                injection_depth=1,
                priority=90,
                enabled=True,
            ),
            StackLayer(
                id="worker_directive",
                name="Worker Directive",
                content="""{{worker_directive}}""",
                role="system",
                injection_depth=2,
                priority=80,
                enabled=True,
            ),
            StackLayer(
                id="prefill",
                name="Prefill",
                content="""{{prefill_prompt}}""",
                role="assistant",
                injection_depth=3,
                priority=100,
                enabled=True,
            ),
        ],
        variables=[
            StackVariable(name="worker_id", default="researcher_0", description="Worker identifier"),
            StackVariable(name="personality_text", default="", description="Full personality description"),
            StackVariable(name="worker_directive", default="", description="Worker behavior directive"),
            StackVariable(name="prefill_prompt", default="", description="Prefill prompt (assistant's acceptance)"),
        ],
    ),
}
