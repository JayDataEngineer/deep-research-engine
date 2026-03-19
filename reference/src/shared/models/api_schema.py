"""Pydantic models for API requests and responses."""

from typing import Optional, Union, Any, Literal
from typing import List, Dict
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import EmailStr, HttpUrl, constr, confloat, conint, conlist

# Import shared types, fields, and validators
from src.shared.types import (
    ContentType,
    ApiRole,
    ChatMode,
    Mode,
    Personality,
    UseCase,
    StreamProtocol,
    VoiceType,
    AudioFormat,
    MetadataDict,
)
from src.shared.fields import (
    optional_temperature_field,
    max_tokens_field,
    created_at_field,
    metadata_field,
)
from src.shared.validators import create_url_dict_validator
from src.shared._base_models import (
    SystemHealthResponse,
    MediaUploadResponse as BaseMediaUploadResponse,
)


class MessageContent(BaseModel):
    """Content within a message."""

    type: ContentType
    text: Optional[constr(min_length=1, max_length=50000)] = Field(None, description="Text content (1-50000 chars)")
    image_url: Optional[dict] = Field(None, description="Image URL with 'url' key")
    input_audio: Optional[dict] = Field(None, description="Audio data with format and data/base64")
    audio: Optional[dict] = Field(None, description="Audio data with format and transcript")
    video_url: Optional[dict] = Field(None, description="Video URL with 'url' key")
    video: Optional[list[str] | str] = Field(None, description="Video data as list of frames or URL")

    # Use shared validators for URL fields
    validate_image_url = create_url_dict_validator('image_url')
    validate_video_url = create_url_dict_validator('video_url')


class Message(BaseModel):
    """A chat message."""

    role: ApiRole
    content: Optional[Union[str, List[MessageContent]]] = Field(None, description="Message content - text string or structured content list")
    parts: Optional[List[Dict[str, Any]]] = Field(None, description="Vercel SDK parts format - included for historical message loading")

    @field_validator('content')
    @classmethod
    def validate_content_not_empty(cls, v: Optional[Union[str, List[MessageContent]]]) -> Optional[Union[str, List[MessageContent]]]:
        # Log what we're receiving for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[VALIDATION] content type: {type(v)}, value: {repr(v)[:200] if v else 'None'}")

        if v is None:
            return v
        if isinstance(v, str):
            if v.strip() == "":
                logger.warning("[VALIDATION] Empty string content received")
                raise ValueError("Message content cannot be empty or whitespace-only")
            if len(v) > 100000:
                raise ValueError("Message content too long (max 100000 chars)")
        if isinstance(v, list):
            if len(v) == 0:
                logger.warning("[VALIDATION] Empty list content received")
                raise ValueError("Message content list cannot be empty")
            if len(v) > 50:
                raise ValueError("Message content cannot contain more than 50 content items")
        return v

    @model_validator(mode='before')
    @classmethod
    def convert_vercel_parts_to_content(cls, data: Any) -> Any:
        """
        Adapts Vercel AI SDK 'parts' format to standard OpenAI 'content' format.
        Vercel sends: {"parts": [{"type": "text", "text": "..."}]}
        OpenAI expects: {"content": "..."} or {"content": [{"type": "text", "text": "..."}]}
        """
        import logging
        import json
        logger = logging.getLogger(__name__)

        if isinstance(data, dict):
            # CRITICAL: Simple logging without JSON truncation
            content_val = data.get('content')
            parts_val = data.get('parts')
            role_val = data.get('role')

            # Use print to ensure we see the full data
            print(f"[VALIDATION INCOMING] role={role_val}, content={content_val}, parts={parts_val}")
            logger.info(f"[VALIDATION INCOMING] role={role_val}")

            logger.debug(f"[MESSAGE VALIDATION] Received message data keys: {list(data.keys())}")
            logger.debug(f"[MESSAGE VALIDATION] Has content: {'content' in data}, Has parts: {'parts' in data}")

            # If 'content' is missing but 'parts' exists
            if "content" not in data and "parts" in data:
                parts = data["parts"]

                # CRITICAL: Skip empty parts arrays - don't create empty content
                # The Vercel AI SDK sometimes sends empty assistant messages
                if not parts or len(parts) == 0:
                    print(f"[VALIDATION] Skipping message with empty parts array, role={role_val}")
                    # Return None to signal this message should be skipped
                    # But we can't return None from model_validator, so mark it somehow
                    # Actually, we'll leave content as None and let validator handle it
                    pass
                else:
                    # 1. Simple Text Extraction (most common)
                    text_parts = [p.get("text", "") for p in parts if p.get("type") == "text"]
                    text_content = "".join(text_parts)

                    # 2. Check for Images (to keep multimodal support)
                    media_parts = [p for p in parts if p.get("type") != "text"]

                    if not media_parts:
                        # If it's just text, return simple string
                        data["content"] = text_content if text_content else None
                    else:
                        # If mixed media, reconstruct standard OpenAI content list
                        new_content = [{"type": "text", "text": text_content}]
                        for p in media_parts:
                            if p.get("type") == "image_url":
                                new_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": p.get("image", p.get("image_url", {}).get("url"))}
                                })
                        data["content"] = new_content

            # Handle JSON string content (from ChatInput sending JSON.stringify)
            if "content" in data:
                content = data["content"]
                if isinstance(content, str) and content.startswith("["):
                    try:
                        # Try to parse as JSON array
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            data["content"] = parsed
                            logger.debug(f"[MESSAGE VALIDATION] Parsed JSON string content to list")
                    except (json.JSONDecodeError, ValueError):
                        # Not valid JSON, keep as string
                        pass

            # CRITICAL: Print final content value before returning
            final_content = data.get('content')
            print(f"[VALIDATION OUTGOING] role={data.get('role')}, final_content={repr(final_content)[:100]}")

        return data


# =============================================================================
# MESSAGE BRANCHING SYSTEM - Pydantic Models
# =============================================================================

class MessageVersion(BaseModel):
    """A single version/branch of a message.

    This represents one possible AI response to a user message.
    Multiple versions can be generated for the same message position.

    Follows Vercel AI SDK pattern:
    - reasoning_content: Chain of thought (separate from text)
    - content: Clean response text (no metadata blobs)
    - parts: Structured array with typed parts (text, reasoning, tool-*, etc.)
    """

    version_id: str = Field(..., description="Unique identifier for this version")
    version_number: int = Field(..., ge=1, description="Version number (1=original, 2=first regeneration, etc.)")
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content text (MUST NOT contain unparsed LLM metadata)")
    parts: Optional[List[Dict[str, Any]]] = Field(None, description="Vercel SDK parts (text, reasoning, tool-*, etc.)")
    reasoning_content: Optional[str] = Field(None, description="Chain of thought reasoning content (separate from text)")
    model: Optional[str] = Field(None, description="Model that generated this version")
    temperature: Optional[float] = optional_temperature_field(None)
    created_at: datetime = created_at_field("When this version was created")
    is_active: bool = Field(False, description="Whether this is the currently active version")
    metadata: Optional[MetadataDict] = metadata_field("Additional metadata including provider, tool_calls, etc.")

    @field_validator('content')
    @classmethod
    def validate_content_no_llm_metadata(cls, v: str) -> str:
        """Validate that content doesn't contain unparsed LLM metadata.

        Based on Vercel AI SDK pattern - reasoning should be in separate parts
        with type='reasoning', NOT mixed into text content as JSON blobs.

        The LLM sometimes returns metadata like {"tools": [], "reasoning": "..."}
        at the start of responses. This MUST be parsed BEFORE saving.

        Legitimate JSON code examples in responses are allowed - we only
        reject content that STARTS with LLM metadata format.
        """
        import logging
        logger = logging.getLogger(__name__)

        stripped = v.strip()

        # LLM metadata format: starts with { or [ and has both "tools" and "reasoning" keys
        # This pattern indicates unparsed metadata that should be extracted
        if (stripped.startswith('{') or stripped.startswith('[')):
            # Check first 500 chars for LLM metadata pattern
            prefix = stripped[:500]
            if '"tools"' in prefix and '"reasoning"' in prefix:
                error_msg = (
                    "Content starts with unparsed LLM metadata (tools/reasoning). "
                    "Parse the metadata BEFORE saving: extract reasoning to reasoning_content, "
                    "remove JSON blob from content. Content should only contain actual response text."
                )
                logger.error(f"[PYDANTIC VALIDATION] {error_msg}")
                logger.error(f"[PYDANTIC VALIDATION] Content preview: {stripped[:200]}")
                raise ValueError(error_msg)

        return v

    @field_validator('parts')
    @classmethod
    def validate_parts_structure(cls, v: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Validate parts structure matches Vercel AI SDK pattern.

        Parts must have proper types and structure:
        - 'text' parts: must have 'text' field
        - 'reasoning' parts: must have 'text' field
        - 'tool-*' parts: must have toolCallId, toolName, etc.
        """
        if v is None:
            return v

        import logging
        logger = logging.getLogger(__name__)

        for i, part in enumerate(v):
            part_type = part.get('type')

            # Every part MUST have a type
            if not part_type:
                raise ValueError(f"Part at index {i} missing required 'type' field")

            # Validate text parts
            if part_type == 'text':
                if 'text' not in part:
                    raise ValueError(f"Text part at index {i} missing required 'text' field")

                # Check for unparsed LLM metadata in text parts
                text_content = part.get('text', '')
                stripped = text_content.strip()
                if (stripped.startswith('{') or stripped.startswith('[')):
                    prefix = stripped[:500]
                    if '"tools"' in prefix and '"reasoning"' in prefix:
                        raise ValueError(
                            f"Text part at index {i} starts with unparsed LLM metadata. "
                            f"Parse reasoning into reasoning_content field and remove JSON blob."
                        )

            # Validate reasoning parts
            elif part_type == 'reasoning':
                if 'text' not in part:
                    raise ValueError(f"Reasoning part at index {i} missing required 'text' field")

            # Validate tool parts
            elif part_type and part_type.startswith('tool-'):
                if 'toolCallId' not in part:
                    raise ValueError(f"Tool part '{part_type}' at index {i} missing required 'toolCallId' field")

            # Log unknown part types for debugging
            else:
                logger.debug(f"[PYDANTIC VALIDATION] Unknown part type '{part_type}' at index {i}")

        return v

    @model_validator(mode='after')
    def validate_reasoning_consistency(self) -> 'MessageVersion':
        """Ensure reasoning_content and parts are consistent.

        STRICT VALIDATION - If reasoning_content exists, parts MUST include a reasoning part.
        This follows the Vercel AI SDK pattern where reasoning is in separate parts.

        RAISES ValueError if inconsistent - no warnings, FAIL HARD.
        """
        import logging
        logger = logging.getLogger(__name__)

        # If reasoning_content exists, parts MUST have reasoning part
        if self.reasoning_content and self.parts:
            has_reasoning_part = any(
                p.get('type') == 'reasoning' for p in self.parts
            )
            if not has_reasoning_part:
                error_msg = (
                    f"reasoning_content exists but no 'reasoning' part in parts array. "
                    f"STRICT VALIDATION FAILED: Add reasoning part for consistency with Vercel AI SDK pattern. "
                    f"Parts types: {[p.get('type') for p in self.parts]}"
                )
                logger.error(f"[PYDANTIC VALIDATION] {error_msg}")
                raise ValueError(error_msg)

        # Conversely, if parts has reasoning part, reasoning_content should exist
        if self.parts:
            has_reasoning_part = any(
                p.get('type') == 'reasoning' for p in self.parts
            )
            if has_reasoning_part and not self.reasoning_content:
                error_msg = (
                    f"Parts contains 'reasoning' part but reasoning_content field is empty. "
                    f"STRICT VALIDATION FAILED: Extract reasoning text into reasoning_content field."
                )
                logger.error(f"[PYDANTIC VALIDATION] {error_msg}")
                raise ValueError(error_msg)

        return self


class MessageWithVersions(BaseModel):
    """A message with multiple version/branch support.

    This wraps a message and provides access to all its versions,
    allowing users to switch between different AI generations.
    """

    message_id: str = Field(..., description="Unique identifier for this message position")
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    versions: List[MessageVersion] = Field(..., min_length=1, description="All available versions")
    active_version_id: str = Field(..., description="Currently active version ID")

    @field_validator('versions')
    @classmethod
    def validate_has_active_version(cls, versions: List[MessageVersion]) -> List[MessageVersion]:
        """Ensure exactly one version is marked as active."""
        active_count = sum(1 for v in versions if v.is_active)
        if active_count == 0:
            raise ValueError("At least one version must be marked as active")
        if active_count > 1:
            raise ValueError("Only one version can be marked as active")
        return versions

    @model_validator(mode='after')
    def validate_active_version_exists(self) -> 'MessageWithVersions':
        """Ensure active_version_id points to an existing version."""
        version_ids = [v.version_id for v in self.versions]
        if self.active_version_id not in version_ids:
            raise ValueError(f"active_version_id '{self.active_version_id}' not found in versions")
        return self

    @property
    def active_version(self) -> MessageVersion:
        """Get the currently active version."""
        for version in self.versions:
            if version.version_id == self.active_version_id:
                return version
        raise ValueError("Active version not found (this should not happen after validation)")


class RegenerateRequest(BaseModel):
    """Request to regenerate a message with a new version."""

    message_id: str = Field(..., description="Message position to regenerate")
    thread_id: str = Field(..., description="Thread/conversation ID")
    model: Optional[str] = Field(None, description="Model to use for regeneration (defaults to original)")
    temperature: Optional[float] = optional_temperature_field(None)


class SwitchVersionRequest(BaseModel):
    """Request to switch to a different message version."""

    message_id: str = Field(..., description="Message position")
    version_id: str = Field(..., description="Version ID to switch to")


class MessageVersionsResponse(BaseModel):
    """Response containing all versions of a message."""

    message_id: str
    versions: List[MessageVersion]
    active_version_id: str
    total_count: int = Field(..., description="Total number of versions")


# =============================================================================
# REFINEMENT MODELS (Experimental)
# =============================================================================


class RefineCustomFlow(BaseModel):
    """Custom boss flow for refinement (experimental).

    Allows user to specify exactly how workers should be allocated,
    bypassing the LLM-based boss decision.
    """
    worker_tasks: Dict[str, str] = Field(
        ...,
        description="Map of worker_id to task description. Example: {'worker_0': 'Research X', 'worker_1': 'Analyze Y'}"
    )
    worker_dependencies: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Dependency graph for dynamic_graph mode: {worker_id: [deps]}. Example: {'worker_1': ['worker_0']}"
    )
    worker_types: Optional[Dict[str, str]] = Field(
        None,
        description="Optional worker specialization: {worker_id: worker_type}. Example: {'worker_0': 'technical'}"
    )
    flow_mode: Optional[Literal["dynamic", "dynamic_graph", "parallel"]] = Field(
        "dynamic_graph",
        description="Execution mode for the custom flow"
    )


# =============================================================================
# ORIGINAL MODELS (unchanged below this point)
# =============================================================================


class ChatRequest(BaseModel):
    """Request for chat completion."""

    messages: conlist(Message, min_length=1) = Field(
        ..., description="List of messages"
    )
    model: Optional[constr(min_length=1, max_length=100)] = Field(None, description="Main model identifier (for final response)")
    tool_model: Optional[constr(min_length=1, max_length=100)] = Field(None, description="Tool model identifier (for tool routing/context gathering). Defaults to main model if not specified.")
    modalities: Optional[conlist(Literal["text", "audio", "image"], min_length=1, max_length=3)] = Field(
        None, description="Output modalities: text, audio, image"
    )
    audio: Optional[dict] = Field(None, description="Audio configuration (voice, format)")
    stream: bool = Field(True, description="Enable streaming response")
    temperature: Optional[float] = optional_temperature_field(None)
    max_tokens: Optional[int] = max_tokens_field()
    session_id: Optional[constr(min_length=1, max_length=100)] = Field(
        None, description="Session ID for character reference storage"
    )
    stream_protocol: StreamProtocol = Field(
        "data", description="Stream protocol: 'data' for SSE events, 'text'/'raw' for plain text"
    )

    # File attachments (for image uploads, etc.)
    files: Optional[conlist(dict, min_length=1, max_length=10)] = Field(
        None,
        description="File attachments. For images: [{name: string, data: base64_string, type: 'image'}]"
    )

    # Companion Brain options
    mode: Optional[Mode] = Field(None, description="Operating mode: companion, rp, or deep_research")
    chatMode: ChatMode = Field(
        "standard", description="Chat mode: standard (basic chat), companion (personality + memory), roleplay (+character cards), deep_research (multi-agent research)"
    )
    brainMode: Optional[Literal["direct", "tool_call", "deep_agent", "research"]] = Field(
        None, description="Brain mode: direct (no tools), tool_call (tools enabled), deep_agent (multi-agent), research (deep research)"
    )
    personality: Optional[Personality] = Field(
        None, description="Personality preset for companion brain"
    )
    personality_enabled: Optional[bool] = Field(True, description="Whether personality features are enabled")

    # Deep Research options (for deep_research mode)
    use_case: UseCase = Field(
        "rag", description="Research domain for deep_research mode"
    )
    variant: Optional[constr(min_length=1, max_length=50)] = Field(
        "default", description="Configuration variant for the use case"
    )

    # Refinement options (experimental - for deep_research mode)
    is_refinement: Optional[bool] = Field(
        False, description="Enable refinement mode: re-run research with user feedback as manual critic"
    )
    refine_feedback: Optional[constr(min_length=1, max_length=5000)] = Field(
        None,
        description="User's refinement instructions (acts as manual critic feedback). Example: 'Focus more on X', 'Add Y details'"
    )
    refine_custom_flow: Optional[RefineCustomFlow] = Field(
        None,
        description="Optional custom boss flow (bypasses LLM boss decision). Specify worker_tasks, worker_dependencies, and flow_mode directly."
    )

    # Roleplay options (for roleplay mode)
    character_name: Optional[constr(min_length=1, max_length=100)] = Field(
        None, description="Character name for roleplay mode"
    )
    character_backstory: Optional[constr(min_length=1, max_length=5000)] = Field(
        None, description="Character backstory and description for roleplay mode"
    )

    @field_validator('audio')
    @classmethod
    def validate_audio(cls, v: Optional[dict]) -> Optional[dict]:
        """Validate audio configuration."""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("audio must be a dictionary")
            # Validate voice if provided
            if "voice" in v:
                valid_voices = list([VoiceType.__values__]) if hasattr(VoiceType, "__values__") else ["Cherry", "Serena", "Ethan", "Chelsie"]
                if v["voice"] not in valid_voices:
                    raise ValueError(f"voice must be one of {valid_voices}")
            # Validate format if provided
            if "format" in v:
                valid_formats = ["wav", "mp3", "ogg"]
                if v["format"] not in valid_formats:
                    raise ValueError(f"audio format must be one of {valid_formats}")
        return v

    @field_validator('files')
    @classmethod
    def validate_files(cls, v: Optional[list[dict]]) -> Optional[list[dict]]:
        """Validate file attachments."""
        if v is not None:
            for i, file_obj in enumerate(v):
                if not isinstance(file_obj, dict):
                    raise ValueError(f"files[{i}] must be a dictionary")
                if "name" not in file_obj:
                    raise ValueError(f"files[{i}] must contain 'name' key")
                if "type" not in file_obj:
                    raise ValueError(f"files[{i}] must contain 'type' key")
                if file_obj.get("type") not in ["image", "audio", "video"]:
                    raise ValueError(f"files[{i}].type must be 'image', 'audio', or 'video'")
                if "data" not in file_obj and "url" not in file_obj:
                    raise ValueError(f"files[{i}] must contain either 'data' or 'url' key")
        return v

    @model_validator(mode='before')
    @classmethod
    def filter_empty_messages(cls, data: Any) -> Any:
        """Filter out messages with None or empty content before validation.

        The Vercel AI SDK sometimes sends empty assistant messages with parts=[],
        which get converted to content=None. Filter these out before validation.
        """
        import logging
        logger = logging.getLogger(__name__)

        if isinstance(data, dict) and "messages" in data:
            messages = data["messages"]
            if isinstance(messages, list):
                # Filter out messages with None/empty content
                filtered_messages = []
                for msg in messages:
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        # Keep message if content is not None and not empty string
                        if content is not None and (not isinstance(content, str) or content.strip() != ""):
                            filtered_messages.append(msg)
                        elif msg.get("parts") and len(msg.get("parts", [])) > 0:
                            # Keep message with non-empty parts (will be converted to content)
                            filtered_messages.append(msg)
                        else:
                            # Skip empty message
                            print(f"[VALIDATION] Filtering out empty message: role={msg.get('role')}, content={content}")
                            logger.info(f"[VALIDATION] Filtering out empty message: role={msg.get('role')}")

                # Ensure we still have at least one message
                if len(filtered_messages) == 0:
                    logger.warning("[VALIDATION] All messages were filtered out, keeping first message")
                    filtered_messages = [messages[0]] if messages else []

                data["messages"] = filtered_messages

        return data

    @model_validator(mode='after')
    def validate_modalities_not_empty(self) -> 'ChatRequest':
        """Ensure modalities is not empty when provided."""
        if self.modalities is not None and len(self.modalities) == 0:
            raise ValueError("modalities cannot be an empty list")
        return self


class ChatResponse(BaseModel):
    """Chat completion response chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict]
    usage: Optional[dict] = None


class MediaUploadResponse(BaseMediaUploadResponse):
    """Response for media upload."""
    pass


class HealthResponse(SystemHealthResponse):
    """Health check response."""
    pass
