"""
Application-wide constants.

Centralized location for all hardcoded values to improve maintainability
and avoid magic numbers throughout the codebase.
"""

import os
from pathlib import Path

# =============================================================================
# API & Server Configuration
# =============================================================================

# Default LLM API configuration (shared infrastructure via Traefik)
# All LLM inference goes through LiteLLM proxy (http://litellm.local/v1)
DEFAULT_LLM_BASE_URL = "http://litellm.local/v1"
DEFAULT_LLM_API_KEY = "sk-litellm-master"
DEFAULT_LLM_MODEL = "qwen-35-27"  # Default model via LiteLLM

# Llama.cpp server port (deprecated - using shared infrastructure)
# LLAMACPP_PORT = 18734  # Use llama.local instead

# Default Embedding model configuration
# NOTE: This should be a valid embedding model, NOT an LLM model
# LLM models like glm-4.7, qwen-omni-turbo, etc. CANNOT be used for embeddings
# With shared infrastructure, use LiteLLM's embedding models
DEFAULT_EMBEDDING_MODEL = "jina-v5-embed"  # Available via LiteLLM
DEFAULT_EMBEDDING_PROVIDER = "litellm"

# ComfyUI configuration
DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_COMFYUI_ADDRESS = DEFAULT_COMFYUI_URL
COMFYUI_PORT = 8188
COMFYUI_ALT_PORT = 18188  # Alternative ComfyUI port

# Backend server
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8787

# Frontend server
FRONTEND_PORT = 31245
FRONTEND_ALT_PORT = 31246

# =============================================================================
# Shared Infrastructure (.local domains via Traefik)
# =============================================================================

# Shared infrastructure hosts (via Traefik reverse proxy)
SHARED_POSTGRES_HOST = "postgres.local"
SHARED_NEO4J_HOST = "neo4j.local"
SHARED_LANGFUSE_HOST = "langfuse.local"
SHARED_MINIO_HOST = "s3.local"
SHARED_LLM_HOST = "llama.local"
SHARED_LITELLM_HOST = "litellm.local"
SHARED_ORCHESTRATOR_HOST = "orchestrator.local"

# Shared infrastructure URLs
SHARED_LANGFUSE_URL = f"http://{SHARED_LANGFUSE_HOST}"
SHARED_MINIO_URL = f"http://{SHARED_MINIO_HOST}"
SHARED_LLM_URL = f"http://{SHARED_LLM_HOST}/v1"
SHARED_LITELLM_URL = f"http://{SHARED_LITELLM_HOST}/v1"
SHARED_ORCHESTRATOR_URL = f"http://{SHARED_ORCHESTRATOR_HOST}"

# =============================================================================
# Database & Storage Ports
# =============================================================================

# PostgreSQL
POSTGRES_PORT = 5432
POSTGRES_ALT_PORT = 5433

# Neo4j
NEO4J_PORT = 7687

# MinIO/S3
MINIO_PORT = 9000
MINIO_INTERNAL_PORT = 9000

# =============================================================================
# External Service Ports
# =============================================================================

# MCP Web Server (replaces SearXNG + Firecrawl)
MCP_WEB_PORT = 8000
MCP_WEB_HOST = "100.102.244.97"  # Tailscale IP

# Ollama
OLLAMA_PORT = 11434

# =============================================================================
# File Size Limits
# =============================================================================

# Upload limits (in bytes)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_SIZE = MAX_UPLOAD_SIZE
MAX_AUDIO_SIZE = MAX_UPLOAD_SIZE
MAX_VIDEO_SIZE = MAX_UPLOAD_SIZE

# =============================================================================
# Model Configuration
# =============================================================================

# Title generation model
TITLE_MODEL_MAX_LENGTH = 50
TITLE_MODEL_MAX_TOKENS = 20
TITLE_MODEL_TEMPERATURE = 0.8
TITLE_MODEL_TOP_P = 0.95
TITLE_MODEL_CONTEXT_SIZE = 512

# =============================================================================
# Unified Directory Structure (Root Level)
# =============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Unified root-level directories
ROOT_DATA_DIR = PROJECT_ROOT / "data"
ROOT_MODELS_DIR = PROJECT_ROOT / "models"
ROOT_TESTS_DIR = PROJECT_ROOT / "tests"
# Extensions are in backend/src/extensions with category subdirectories
ROOT_EXTENSIONS_DIR = PROJECT_ROOT / "backend" / "src" / "extensions"
# TTS services (isolated venv extensions)
TTS_EXTENSIONS_DIR = ROOT_EXTENSIONS_DIR / "integrations" / "tts"
ROOT_DOCS_DIR = PROJECT_ROOT / "docs"

# Vendor directories (third-party services)
VENDOR_DIR = PROJECT_ROOT / "backend" / "vendor"
COMFYUI_VENDOR_DIR = VENDOR_DIR / "ComfyUI"
# LLAMACPP_VENDOR_DIR = VENDOR_DIR / "llama.cpp"  # Deprecated: using shared infrastructure

# ComfyUI workflow extensions (project-specific workflow configs)
# NOTE: Different from ComfyUI's custom_nodes (which are actual plugins)
COMFYUI_EXTENSIONS_DIR = ROOT_EXTENSIONS_DIR / "services" / "comfyui"
COMFYUI_WORKFLOWS_DIR = ROOT_DATA_DIR / "inputs" / "workflows"
COMFYUI_USER_WORKFLOWS_DIR = COMFYUI_VENDOR_DIR / "user" / "default" / "workflows"

# =============================================================================
# Character & Content Configuration
# =============================================================================

# Character repository - central source of truth for character profiles
# Used by deep_research, content_formatter, and any character-dependent features
CHARACTER_REPOSITORY_DIR = ROOT_EXTENSIONS_DIR / "integrations" / "research" / "podcast"
CHARACTER_TEXT_INJECTIONS_FILE = CHARACTER_REPOSITORY_DIR / "text_injections.yaml"

# Data subdirectories
DATA_CONFIG_DIR = ROOT_DATA_DIR / "config"
DATA_DB_DIR = ROOT_DATA_DIR / "db"
DATA_INPUTS_DIR = ROOT_DATA_DIR / "inputs"
DATA_OUTPUTS_DIR = ROOT_DATA_DIR / "outputs"
DATA_RP_DIR = ROOT_DATA_DIR / "rp"

# Config subdirectories (under data/config/)
PROMPTS_CONFIG_DIR = DATA_CONFIG_DIR / "prompts"
EMOTION_EXPERIMENTS_DIR = DATA_CONFIG_DIR / "emotion-experiments"

# Voice samples directory
VOICE_SAMPLES_DIR = DATA_INPUTS_DIR / "audio" / "voice_samples"

# Backend directories (re-exported for backward compatibility)
BACKEND_ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT_DATA_DIR
MODELS_DIR = ROOT_MODELS_DIR
COMFYUI_MODELS_DIR = ROOT_MODELS_DIR / "comfyui"

# Output directories (under data/outputs/)
CHECKPOINTS_DIR = DATA_OUTPUTS_DIR / "checkpoints"
UPLOADS_DIR = DATA_OUTPUTS_DIR / "uploads"
GENERATED_DIR = DATA_OUTPUTS_DIR / "generated"
IMAGES_DIR = DATA_OUTPUTS_DIR / "generated_images"
LOGS_DIR = DATA_OUTPUTS_DIR / "logs"
GRAPH_VISUALIZATIONS_DIR = DATA_OUTPUTS_DIR / "graph_visualizations"

# Input directories (under data/inputs/)
PROMPTS_INPUTS_DIR = DATA_INPUTS_DIR / "prompts"
AUDIO_SAMPLES_DIR = DATA_INPUTS_DIR / "audio_samples"

# Roleplay directories (under data/rp/)
LOREBOOKS_DIR = DATA_RP_DIR / "lorebooks"
ROLEPLAY_DIR = DATA_RP_DIR / "roleplay"
CHARACTERS_DIR = DATA_RP_DIR / "characters"

# Legacy aliases (deprecated, use specific dirs above)
ROOT_INPUTS_DIR = DATA_INPUTS_DIR
ROOT_OUTPUTS_DIR = DATA_OUTPUTS_DIR
OUTPUT_DIR = GENERATED_DIR

# Companion (animation) directories
Assistant_OUTPUT_DIR = DATA_OUTPUTS_DIR / "companion"
Assistant_CHARACTERS_DIR = CHARACTERS_DIR

# Research state directory (for deep research navigation resume)
RESEARCH_STATE_DIR = DATA_OUTPUTS_DIR / "research_state"

# Podcast directories
PODCAST_AUDIO_DIR = DATA_OUTPUTS_DIR / "audio" / "podcasts"
PODCAST_ANALYSIS_DIR = DATA_OUTPUTS_DIR / "podcast_analysis"

# Default character assets
Assistant_DEFAULT_CHARACTER_IMAGE = "assets/characters/default.png"

# =============================================================================
# Timeouts & Limits
# =============================================================================

# Request timeouts (seconds)
DEFAULT_REQUEST_TIMEOUT = 30
IMAGE_GENERATION_TIMEOUT = 180
STREAM_TIMEOUT = 300

# Rate limiting
DEFAULT_RATE_LIMIT = 60  # requests per minute

# =============================================================================
# Logging
# =============================================================================

# Log prefixes for consistent formatting
LOG_PREFIX_BACKEND = "[BACKEND]"
LOG_PREFIX_API = "[API]"
LOG_PREFIX_COMPANION = "[COMPANION]"
LOG_PREFIX_TITLE_GEN = "[TITLE GEN]"
LOG_PREFIX_COMFYUI = "[COMFYUI]"
LOG_PREFIX_MODEL = "[MODEL]"
LOG_PREFIX_CACHE = "[CACHE]"
LOG_PREFIX_DOWNLOAD = "[DOWNLOAD]"

# =============================================================================
# Chat & Conversation
# =============================================================================

# Default chat settings
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.9

# Message limits
MAX_CONTEXT_MESSAGES = 50
MAX_MESSAGE_LENGTH = 10000

# Title fallback
FALLBACK_TITLE_MAX_LENGTH = 40
FALLBACK_TITLE_MIN_WORD_BOUNDARY = 20

# =============================================================================
# SSE (Server-Sent Events) Configuration
# =============================================================================

# SSE event types
SSE_EVENT_START = "start"
SSE_EVENT_TEXT_DELTA = "text-delta"
SSE_EVENT_TEXT_END = "text-end"
SSE_EVENT_DATA_COT = "data-cot"
SSE_EVENT_FINISH = "finish"
SSE_EVENT_ERROR = "error"

# =============================================================================
# Database & Persistence
# =============================================================================

# Checkpointing
DEFAULT_CHECKPOINT_DIR = str(CHECKPOINTS_DIR)
CHECKPOINT_SAVE_INTERVAL = 60  # seconds
MAX_CHECKPOINT_SIZE = 100 * 1024 * 1024  # 100MB
CONVERSATIONS_DB_PATH = CHECKPOINTS_DIR / "conversations.db"
RUNS_DB_PATH = DATA_DB_DIR / "runs.db"
RESEARCH_SOURCES_DB_PATH = ROOT_DATA_DIR / "research" / "sources.db"

# =============================================================================
# Audio Configuration (ASR + TTS)
# =============================================================================

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = ["wav", "mp3", "ogg", "flac", "aac"]
DEFAULT_AUDIO_VOICE = "Cherry"
DEFAULT_AUDIO_FORMAT = "wav"

# ASR via Shared Infrastructure (Orchestrator + LiteLLM)
# Models available: distil-whisper (fast), qwen3-asr (accurate)
# Access via LiteLLM audio API: client.audio.transcriptions.create(model="distil-whisper", ...)
ASR_VIA_LITELLM = True
ASR_LITELLM_MODEL = "distil-whisper"  # Default ASR model via LiteLLM
ASR_LITELLM_URL = f"http://{SHARED_LITELLM_HOST}/v1"

# =============================================================================
# Image Configuration
# =============================================================================

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "gif", "webp"]

# Default generation settings
DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024
DEFAULT_IMAGE_STEPS = 9

# =============================================================================
# Video Configuration
# =============================================================================

# Supported video formats
SUPPORTED_VIDEO_FORMATS = ["mp4", "webm", "mov"]

# =============================================================================
# Environment Variable Keys (for type safety)
# =============================================================================

# LLM Configuration
ENV_LLM_API_KEY = "LLM_API_KEY"
ENV_LLM_BASE_URL = "LLM_BASE_URL"
ENV_LLM_MODEL = "LLM_MODEL"

# Embedding Configuration
ENV_EMBEDDING_MODEL = "EMBEDDING_MODEL"
ENV_EMBEDDING_PROVIDER = "EMBEDDING_PROVIDER"

# ComfyUI Configuration
ENV_COMFYUI_ADDRESS = "COMFYUI_ADDRESS"

# Server Configuration
ENV_HOST = "HOST"
ENV_PORT = "PORT"
ENV_CORS_ORIGINS = "CORS_ORIGINS"

# Default CORS origins (docs: 3001-3002, main frontend: 31245, alt frontend: 31246)
DEFAULT_CORS_ORIGINS = '["http://localhost:3001", "http://localhost:3002", "http://localhost:31245", "http://localhost:31246"]'

# Model Paths
ENV_Z_IMAGE_UNET_PATH = "Z_IMAGE_UNET_PATH"
ENV_Z_IMAGE_CLIP_PATH = "Z_IMAGE_CLIP_PATH"
ENV_Z_IMAGE_VAE_PATH = "Z_IMAGE_VAE_PATH"
ENV_Z_IMAGE_UNET_URL = "Z_IMAGE_UNET_URL"
ENV_Z_IMAGE_CLIP_URL = "Z_IMAGE_CLIP_URL"
ENV_Z_IMAGE_VAE_URL = "Z_IMAGE_VAE_URL"

# Directory Configuration
ENV_UPLOAD_DIR = "UPLOAD_DIR"
ENV_MODEL_DIR = "MODEL_DIR"
ENV_COMFYUI_MODEL_DIR = "COMFYUI_MODEL_DIR"
ENV_OUTPUT_DIR = "COMFY_OUTPUT_DIR"

# =============================================================================
# TTS (Text-to-Speech) Configuration
# =============================================================================

# Main API URL (for extension subprocess callbacks)
# Extensions need to call back to the main API for things like emotion analysis
MAIN_API_PORT = int(os.getenv(ENV_PORT, DEFAULT_PORT))
MAIN_API_HOST = os.getenv(ENV_HOST, DEFAULT_HOST)
MAIN_API_URL = f"http://{MAIN_API_HOST}:{MAIN_API_PORT}"

# TTS via Shared Infrastructure (Orchestrator + LiteLLM)
# Models available: kokoro (fast), indextts2 (high quality)
# Access via LiteLLM audio API: client.audio.speech.create(model="kokoro", ...)
TTS_VIA_LITELLM = True
TTS_LITELLM_MODEL = "kokoro"  # Default TTS model via LiteLLM
TTS_LITELLM_URL = f"http://{SHARED_LITELLM_HOST}/v1"

# TTS extension service (legacy - for local TTS fallback)
TTS_EXTENSION_PORT = 9000
TTS_EXTENSION_HOST = "127.0.0.1"
TTS_EXTENSION_URL = f"http://{TTS_EXTENSION_HOST}:{TTS_EXTENSION_PORT}"

# TTS timeouts (seconds)
TTS_READY_TIMEOUT = 30.0
TTS_GENERATE_TIMEOUT = 120.0
TTS_DEFAULT_TIMEOUT = 5.0

# TTS text chunking
TTS_CHUNK_SIZE_DEFAULT = 500
TTS_CHUNK_SIZE_INDEX = 200
TTS_CHUNK_SIZE_OPENAI = 300
TTS_CHUNK_SIZE_ELEVEN = 400
TTS_CHUNK_SIZE_MMS = 250
TTS_CHUNK_SIZE_ESPEAK = 5000

# TTS emotion tagging
TTS_EMOTION_MAX_TOKENS = 500

# TTS batch processing
TTS_BATCH_MAX_ITEMS = 100
TTS_REF_AUDIO_MAX_SIZE_MB = 50

# =============================================================================
# IndexTTS Configuration
# =============================================================================

INDEXTTS_DEFAULT_URL = "http://localhost:9000"
INDEXTTS_DEFAULT_SPEAKER = "default"
INDEXTTS_TIMEOUT = 30.0

# IndexTTS vLLM external project path (for advanced vLLM-accelerated TTS)
# Configurable via INDEXTTS_VLLM_PATH environment variable
# Falls back to local model directory if external path not found
INDEXTTS_VLLM_EXTERNAL_PATH = Path(os.getenv("INDEXTTS_VLLM_PATH", "/opt/index-tts-vllm"))
INDEXTTS_VLLM_MODEL_DIR = INDEXTTS_VLLM_EXTERNAL_PATH / "checkpoints" / "IndexTTS-2-vLLM"

# =============================================================================
# ZAI / GLM API Configuration (DEPRECATED)
# =============================================================================
# Deprecated: Use LiteLLM proxy for all LLM inference.
# These constants are kept for backward compatibility only.
# New code should use LITELLM_BASE_URL from shared infrastructure.

ZAI_DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"  # Deprecated
GLM_DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"  # Deprecated
GLM_IMAGE_MODEL = "glm-4v-flash"  # Deprecated - use VISION_MODEL instead

# =============================================================================
# Companion / Animation Configuration
# =============================================================================

# Companion timeouts (seconds)
Assistant_REQUEST_TIMEOUT = 30
Assistant_PROCESS_TIMEOUT = 300

# Companion model paths
Assistant_CODEFORMER_MODEL_PATH = "assets/models/codeformer.pth"
Assistant_GFPGAN_MODEL_PATH = "assets/models/GFPGANv1.4.pth"
Assistant_REALESRGAN_MODEL_PATH = "assets/models/RealESRGAN_x4.pth"
Assistant_ULTRASHARP_MODEL_PATH = "assets/models/4x-UltraSharp.pth"

# =============================================================================
# Gradio UI Configuration
# =============================================================================

GRADIO_DEFAULT_HOST = "0.0.0.0"
GRADIO_DEFAULT_PORT = 7860

# =============================================================================
# Image Generation Configuration
# =============================================================================

# Z-Image Turbo paths
Z_IMAGE_DEFAULT_MODEL = "z-image-turbo"
Z_IMAGE_DEFAULT_VAE = "ae.safetensors"
Z_IMAGE_DEFAULT_LLM = "qwen"

# =============================================================================
# Download & Network Configuration
# =============================================================================

# Download settings
DOWNLOAD_MAX_RETRIES = 10
DOWNLOAD_RETRY_WAIT = 5
DOWNLOAD_MIN_SPLIT_SIZE = "1M"
DOWNLOAD_DEFAULT_TIMEOUT = 300
DOWNLOAD_MAX_TIMEOUT = 21600  # 6 hours

# aria2c settings
ARIAC_MAX_CONCURRENT = 5
ARIAC_MAX_KEEPALIVE = 2

# =============================================================================
# HuggingFace Configuration
# =============================================================================

HUGGINGFACE_BASE_URL = "https://huggingface.co"

# =============================================================================
# Database Configuration
# =============================================================================

# SQLite databases
MEMORY_DB_PATH = DATA_DB_DIR / "memory.db"
USER_DATA_DB_PATH = DATA_DB_DIR / "user_data.db"
API_KEYS_DB_PATH = DATA_DB_DIR / "api_keys.db"

# Security config
ENCRYPTION_KEY_PATH = DATA_CONFIG_DIR / "secure" / ".encryption_key"

# Vector store defaults
VECTOR_STORE_DEFAULT_DIR = ROOT_DATA_DIR / "vectorstores" / "chroma"

# =============================================================================
# Pydantic Model Limits
# =============================================================================

# Message content limits
PYDANTIC_MAX_TEXT_LENGTH = 50000
PYDANTIC_MAX_CONTENT_LENGTH = 100000
PYDANTIC_MAX_MESSAGES = 100
PYDANTIC_MAX_FILES = 10
PYDANTIC_MAX_MODEL_LENGTH = 100
PYDANTIC_MAX_MODALITIES = 3
PYDANTIC_MAX_SESSION_ID_LENGTH = 100
PYDANTIC_MIN_TEXT_LENGTH = 1
PYDANTIC_MAX_PROMPT_LENGTH = 5000
PYDANTIC_MAX_INPUT_LENGTH = 1000

# =============================================================================
# File Extension Constants
# =============================================================================

# Config files
EXT_YAML = ".yaml"
EXT_YML = ".yml"
EXT_JSON = ".json"

# Media files
EXT_WAV = ".wav"
EXT_MP3 = ".mp3"
EXT_OGG = ".ogg"
EXT_FLAC = ".flac"
EXT_AAC = ".aac"
EXT_PNG = ".png"
EXT_JPG = ".jpg"
EXT_JPEG = ".jpeg"
EXT_GIF = ".gif"
EXT_WEBP = ".webp"
EXT_MP4 = ".mp4"
EXT_WEBM = ".webm"
EXT_MOV = ".mov"

# Model files
EXT_SAFETENSORS = ".safetensors"
EXT_GGUF = ".gguf"
EXT_PT = ".pt"
EXT_PTH = ".pth"
EXT_CKPT = ".ckpt"
EXT_BIN = ".bin"
EXT_MODEL = ".model"

# =============================================================================
# Companion Brain Configuration
# =============================================================================

# Deep research graph (parent directory containing base/ and use_cases/)
DEEP_RESEARCH_CONFIG_DIR = ROOT_DATA_DIR / "config" / "deep_research"

# Research extensions directory (deep research configs)
RESEARCH_EXTENSIONS_DIR = ROOT_EXTENSIONS_DIR / "content" / "deep_research"

# Research use cases directory (soc_media_gen, longform_gen, etc.)
RESEARCH_USE_CASES_DIR = ROOT_EXTENSIONS_DIR / "integrations" / "research"

# Content formatter directory (for post-synthesis formatting)
CONTENT_FORMATTER_DIR = ROOT_EXTENSIONS_DIR / "content" / "content_formatter"
CONTENT_FORMATTER_FORMATS_DIR = CONTENT_FORMATTER_DIR / "formats"

# User configuration directory (for frontend-editable user variants)
USER_CONFIG_DIR = ROOT_DATA_DIR / "config" / "user"

# =============================================================================
# Model Filenames
# =============================================================================

# ComfyUI workflow files
WORKFLOW_Z_IMAGE_TURBO = "z_image_turbo.json"
WORKFLOW_Z_IMAGE_TURBO_IMG2IMG = "z_image_turbo_img2img.json"

# Config filenames
# CONFIG_BACKENDS removed - LiteLLM handles model routing now
CONFIG_OPENAPI = "openapi.json"
CONFIG_SPEAKER_MAPPINGS = DATA_CONFIG_DIR / "speaker_mappings.json"

# =============================================================================
# Vector Store / RAG Configuration
# =============================================================================

ENV_SPAWNER_RAG_DIR = "SPAWNER_RAG_DIR"
ENV_SPAWNER_RAG_COLLECTION = "SPAWNER_RAG_COLLECTION"

# =============================================================================
# OpenRouter Configuration
# =============================================================================

ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
