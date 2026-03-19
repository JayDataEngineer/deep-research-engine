"""Extension Config Schema - Pydantic validation for config.yaml files.

This module provides Pydantic models for validating extension config.yaml files,
ensuring all required fields are present and correctly typed.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class VoiceConfig(BaseModel):
    """Voice configuration for TTS extensions."""
    id: str = Field(..., description="Unique voice identifier")
    name: str = Field(..., description="Human-readable voice name")
    gender: Optional[str] = Field(None, description="Voice gender (male/female/neutral)")
    language: Optional[str] = Field("en", description="Voice language code")
    requires_reference_audio: bool = Field(False, description="Whether voice cloning requires reference audio")


class ModelConfig(BaseModel):
    """Model configuration for extensions that require models."""
    local_path: str = Field(..., description="Local path to the model")
    size: Optional[str] = Field(None, description="Model size description (e.g., '2GB')")
    required: bool = Field(True, description="Whether the model is required")
    description: Optional[str] = Field(None, description="Model description")


class ParameterConfig(BaseModel):
    """Parameter configuration for extension UI."""
    name: str = Field(..., description="Parameter name")
    type: Literal["text", "range", "select", "file", "boolean", "number"] = Field(
        ..., description="Parameter type"
    )
    label: str = Field(..., description="Human-readable label")
    description: Optional[str] = Field(None, description="Parameter description")
    required: bool = Field(False, description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value")
    min_value: Optional[float] = Field(None, description="Minimum value for range/number types")
    max_value: Optional[float] = Field(None, description="Maximum value for range/number types")
    options: Optional[List[str]] = Field(None, description="Options for select type")


class ExtensionConfigModel(BaseModel):
    """Main extension configuration model.

    Corresponds to the 'extension' section in config.yaml.
    """
    name: str = Field(..., min_length=1, max_length=100, description="Extension identifier (filename-safe)")
    type: Optional[Literal["mode", "feature", "service", "tool", "library"]] = Field(
        None, description="Extension type (alias for category, for backward compatibility)"
    )
    category: Optional[Literal["mode", "feature", "service", "tool", "library", "text_modes"]] = Field(
        None, description="Extension category"
    )
    display_name: str = Field(..., min_length=1, max_length=100, description="Human-readable name")
    description: str = Field("", min_length=0, max_length=500, description="What this extension does")
    version: str = Field("1.0.0", description="Semantic version")
    author: str = Field("", description="Author name or handle")
    api_prefix: Optional[str] = Field(None, description="API prefix for this extension")
    enabled: bool = Field(True, description="Whether extension is enabled by default")
    subprocess_only: bool = Field(False, description="If True, extension runs in isolated subprocess only")
    priority: int = Field(50, ge=0, le=100, description="Loading priority (higher loads first)")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is filename-safe."""
        import re
        # Remove invalid characters
        clean = re.sub(r'[<>:"/\\|?*]', '', v)
        # Replace spaces with underscores
        clean = re.sub(r'\s+', '_', clean)
        if not clean:
            raise ValueError("Name must contain valid characters")
        return clean.lower()

    @model_validator(mode="after")
    def normalize_category(self) -> "ExtensionConfigModel":
        """Normalize type to category for backward compatibility."""
        # If type is set but category is not, use type as category
        if self.type is not None and self.category is None:
            self.category = self.type
        # If both are None, default to "feature"
        elif self.category is None:
            self.category = "feature"
        return self


class PythonConfig(BaseModel):
    """Python environment configuration for extensions."""
    use_host_env: bool = Field(False, description="Use host Python environment instead of isolated venv")


class SystemConfig(BaseModel):
    """System-level configuration for extensions."""
    lazy_load: bool = Field(True, description="Defer loading until first use for faster startup")


class CapabilitiesConfig(BaseModel):
    """Extension capabilities declaration."""
    input_type: List[Literal["text", "audio", "image", "video", "file"]] = Field(
        default_factory=lambda: ["text"], description="Supported input types"
    )
    output_type: List[Literal["text", "audio", "image", "video", "file", "none"]] = Field(
        default_factory=lambda: ["text"], description="Output types"
    )
    streaming: bool = Field(False, description="Supports streaming responses")
    supports_tools: bool = Field(False, description="Can use LangChain tools")
    supports_agents: bool = Field(False, description="Can act as an agent")


class UIConfig(BaseModel):
    """UI configuration for extension."""
    icon: Optional[str] = Field(None, description="Icon name or path")
    color: Optional[str] = Field(None, description="Theme color (hex)")
    sidebar_component: Optional[str] = Field(None, description="Sidebar component name")
    settings_component: Optional[str] = Field(None, description="Settings component name")
    input_component: Optional[str] = Field(None, description="Input component name")
    message_style: Optional[str] = Field(None, description="Message style name")


class DependencyRequirement(BaseModel):
    """Single dependency requirement."""
    name: str = Field(..., description="Extension name that is required")
    critical: bool = Field(True, description="If True, extension fails without this dependency")
    version: Optional[str] = Field(None, description="Minimum version required")


class RelationshipsConfig(BaseModel):
    """Extension relationships and dependencies."""
    requires: List[DependencyRequirement] = Field(
        default_factory=list, description="Required extensions"
    )
    conflicts: List[str] = Field(
        default_factory=list, description="Conflicting extension names"
    )
    suggests: List[str] = Field(
        default_factory=list, description="Suggested but optional extensions"
    )


class DependenciesConfig(BaseModel):
    """Extension dependencies."""
    python_version: str = Field(">=3.10", description="Minimum Python version")
    pip_packages: List[str] = Field(default_factory=list, description="Required pip packages")
    extensions: List[str] = Field(default_factory=list, description="Required extensions by name (deprecated, use relationships)")
    use_venv: bool = Field(False, description="Use isolated virtual environment")
    git_repo: Optional[str] = Field(None, description="Git repository to clone if needed")
    models: List[ModelConfig] = Field(default_factory=list, description="Required models to download")
    relationships: RelationshipsConfig = Field(
        default_factory=RelationshipsConfig, description="Extension relationships"
    )


class ExtensionConfigSchema(BaseModel):
    """Complete extension config.yaml schema.

    This is the top-level model for validating extension config files.
    """
    extension: ExtensionConfigModel = Field(..., description="Extension metadata")
    python: Optional[PythonConfig] = Field(None, description="Python environment configuration")
    system: Optional[SystemConfig] = Field(None, description="System-level configuration")
    capabilities: Optional[CapabilitiesConfig] = Field(None, description="Extension capabilities")
    ui: Optional[UIConfig] = Field(None, description="UI configuration")
    dependencies: Optional[DependenciesConfig] = Field(None, description="Dependencies")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extension settings schema")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    enabled_tools: Optional[List[str]] = Field(
        default_factory=list, description="Tools enabled for this extension"
    )
    models: Optional[List[ModelConfig]] = Field(
        default_factory=list, description="Models required by this extension"
    )
    voices: Optional[List[VoiceConfig]] = Field(
        default_factory=list, description="Voices available for TTS extensions"
    )
    parameters: Optional[List[ParameterConfig]] = Field(
        default_factory=list, description="UI parameters for this extension"
    )

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: List[str]) -> List[str]:
        """Normalize tags to lowercase."""
        return [tag.lower().strip() for tag in v if tag.strip()]


def validate_config(config_path: Path) -> tuple[bool, Optional[ExtensionConfigSchema], List[str]]:
    """Validate an extension config.yaml file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Tuple of (is_valid, validated_schema, errors)
    """
    import yaml

    errors = []

    # Load YAML
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        return False, None, [f"Failed to load YAML: {e}"]

    if not config_data:
        return False, None, ["Config file is empty"]

    # Validate against schema
    try:
        validated = ExtensionConfigSchema(**config_data)
        return True, validated, []
    except Exception as e:
        # Collect validation errors
        from pydantic import ValidationError
        if isinstance(e, ValidationError):
            error_list = []
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                error_list.append(f"{loc}: {error['msg']}")
            return False, None, error_list
        else:
            return False, None, [str(e)]


def get_missing_fields(config_path: Path) -> List[str]:
    """Check for missing recommended fields in config.yaml.

    Args:
        config_path: Path to config.yaml file

    Returns:
        List of missing recommended field names
    """
    import yaml

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f) or {}

    missing = []

    # Recommended fields
    recommended = {
        "extension": ["version", "author", "description"],
        "capabilities": None,  # Entire section is recommended
        "tags": None,  # At least one tag recommended
    }

    # Check extension section
    ext = config_data.get("extension", {})
    for field in recommended.get("extension", []):
        if not ext.get(field):
            missing.append(f"extension.{field}")

    # Check capabilities section
    if not config_data.get("capabilities"):
        missing.append("capabilities")

    # Check tags
    tags = config_data.get("tags", [])
    if not tags or not any(tags):
        missing.append("tags (at least one)")

    return missing
