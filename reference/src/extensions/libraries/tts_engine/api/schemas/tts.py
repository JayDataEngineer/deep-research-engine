"""
Pydantic models for TTS (Text-to-Speech) operations.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Extension categories
ExtensionCategory = Literal["tts", "tool", "analysis", "helper"]


class TTSRequest(BaseModel):
    """Request model for text-to-speech generation.

    NOTE: Voice selection is now extension-specific, not a global parameter.
    Each extension declares its own voice/audio input needs in its config.
    - Extensions with preset voices (Kokoro): declare 'voice' parameter in their config
    - Voice cloning extensions (IndexTTS, GPT-SoVITS): declare 'ref_audio_path' parameter
    - Internal voice selection (espeak_ng): no voice parameter needed

    The extension_name parameter allows the frontend to specify which extension to use
    for generation. This is important when the frontend knows which extension a voice
    belongs to, ensuring the backend loads the correct extension.
    """

    text: str = Field(
        ..., min_length=1, max_length=5000, description="The text to convert to speech"
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speech speed multiplier (0.25 to 4.0)",
    )
    pitch: float = Field(
        default=1.0, ge=0.5, le=2.0, description="Pitch adjustment (0.5 to 2.0)"
    )
    format: str = Field(
        default="wav", pattern="^(wav|mp3|ogg)$", description="Audio output format"
    )
    extension_name: str | None = Field(
        default=None,
        description="Optional: Name of the extension to use for generation. "
                    "If provided, the backend will ensure this extension is loaded before generation.",
    )

    model_config = {"extra": "allow"}  # Allow extension-specific parameters (voice, ref_audio_path, etc.)

    @field_validator("text")
    def validate_text(cls, v):
        """Validate that text is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class VoiceInfo(BaseModel):
    """Model for voice information."""

    id: str = Field(..., description="Unique identifier for the voice")
    name: str = Field(..., description="Human-readable name of the voice")
    language: str = Field(..., description="Language code (e.g., 'en-US')")
    gender: str | None = Field(None, description="Voice gender (male/female/neutral)")
    description: str | None = Field(None, description="Voice description")


class VoicesResponse(BaseModel):
    """Response model for available voices."""

    voices: list[VoiceInfo] = Field(..., description="List of available voices")


class TTSResponse(BaseModel):
    """Response model for TTS generation."""

    success: bool = Field(..., description="Whether the generation was successful")
    message: str = Field(..., description="Response message")
    audio_url: str | None = Field(None, description="URL to generated audio file")
    duration: float | None = Field(None, description="Audio duration in seconds")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: bool = Field(default=True, description="Indicates this is an error response")
    message: str = Field(..., description="Error message")
    details: dict | None = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Response timestamp")


class ExtensionParameter(BaseModel):
    """Model for extension-specific parameters."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(
        ...,
        description="Parameter type (string, number, boolean, file, select, range, etc.)",
    )
    label: str = Field(..., description="Human-readable parameter label")
    description: str | None = Field(None, description="Parameter description")
    required: bool = Field(
        default=False, description="Whether this parameter is required"
    )
    default: Any | None = Field(None, description="Default value")
    min_value: float | None = Field(
        None, description="Minimum value for numeric parameters"
    )
    max_value: float | None = Field(
        None, description="Maximum value for numeric parameters"
    )
    options: list[str] | None = Field(
        None, description="Available options for select parameters"
    )
    depends_on: str | None = Field(
        None, description="Conditional visibility: 'param_name:value' - only show if param equals value"
    )


class ExtensionConfig(BaseModel):
    """Model for extension configuration."""

    extension_name: str = Field(..., description="Extension name")
    display_name: str = Field(..., description="Human-readable extension name")
    description: str | None = Field(None, description="Extension description")
    category: ExtensionCategory = Field(
        default="tts",
        description="Extension category: 'tts' for text-to-speech engines, 'tool' for helper utilities, 'analysis' for audio analysis, 'helper' for misc helpers"
    )
    parameters: list[ExtensionParameter] = Field(
        default_factory=list, description="Extension-specific parameters"
    )
    requires_audio_sample: bool = Field(
        default=False, description="Whether extension requires audio sample"
    )
    supports_voice_cloning: bool = Field(
        default=False, description="Whether extension supports voice cloning"
    )
    supports_emotion_tagging: bool = Field(
        default=False, description="Whether extension supports emotion tagging (IndexTTS only)"
    )


class ExtensionConfigResponse(BaseModel):
    """Response model for extension configuration."""

    extensions: list[ExtensionConfig] = Field(
        ..., description="List of extension configurations"
    )
