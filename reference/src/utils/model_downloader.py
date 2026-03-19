"""
Model Download Utility

Provides utilities for downloading and caching models from HuggingFace.
Ensures models are only downloaded once and reused across sessions.

Supports:
- GGUF models for title generation
- ComfyUI models (UNET, VAE, CLIP, LoRA, etc.)
- General model downloads with progress tracking
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from urllib.request import urlretrieve
from urllib.error import URLError
import asyncio

from src.constants import (
    PROJECT_ROOT, ROOT_MODELS_DIR, COMFYUI_MODELS_DIR, ENV_COMFYUI_MODEL_DIR,
    HUGGINGFACE_BASE_URL, DOWNLOAD_DEFAULT_TIMEOUT, DOWNLOAD_MAX_RETRIES,
    DOWNLOAD_MAX_TIMEOUT, BACKEND_ROOT,
)

logger = logging.getLogger(__name__)

# Default model directories (using constants from constants.py)
DEFAULT_MODEL_DIR = ROOT_MODELS_DIR
TITLE_MODEL_DIR = DEFAULT_MODEL_DIR / "llm" / "title_generator"
COMFYUI_MODEL_DIR = Path(os.getenv(ENV_COMFYUI_MODEL_DIR, COMFYUI_MODELS_DIR))

# ComfyUI model subdirectories (matching extra_model_paths.yaml structure)
# These are the subdirectories under models/comfyui/
COMFYUI_SUBDIRS = {
    "unet": "unet",                      # UNET/diffusion models
    "diffusion_models": "diffusion_models",  # Alternative diffusion models location
    "clip": "clip",                      # CLIP text encoders
    "text_encoders": "text_encoders",    # Alternative text encoders location
    "vae": "vae",                        # VAE models
    "loras": "loras",                    # LoRA models
    "embeddings": "embeddings",          # Textual inversion embeddings
    "controlnet": "controlnet",          # ControlNet models
    "upscale": "upscale_models",         # Upscaling models (ESRGAN, etc)
    "upscale_models": "upscale_models",  # Alternative upscale location
    "clip_vision": "clip_vision",        # CLIP vision models
    "checkpoints": "checkpoints",        # Full model checkpoints
    "configs": "configs",                # Model configs
    "style_models": "style_models",      # Style models
    "gligen": "gligen",                  # GLIGEN models
    "hypernetworks": "hypernetworks",    # Hypernetworks
    "photomaker": "photomaker",          # PhotoMaker models
    "classifiers": "classifiers",        # T2I adapters/classifiers
    "audio_encoders": "audio_encoders",  # Audio encoder models
    "model_patches": "model_patches",    # Model patches
}


def get_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_with_progress(
    url: str,
    dest: Path,
    description: str = "Downloading"
) -> bool:
    """
    Download a file with progress reporting.

    Args:
        url: URL to download from
        dest: Destination path
        description: Description for progress logs

    Returns:
        True if successful, False otherwise
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if dest.exists():
        logger.info(f"[DOWNLOAD] {description} already exists at {dest}")
        return True

    # Download with progress
    logger.info(f"[DOWNLOAD] {description} from {url}")
    logger.info(f"[DOWNLOAD] Destination: {dest}")

    try:
        # Download to temp file first
        temp_dest = dest.with_suffix(dest.suffix + ".tmp")

        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = (downloaded / total_size * 100) if total_size > 0 else 0
            if total_size > 0:
                logger.debug(f"[DOWNLOAD] Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)")

        urlretrieve(url, temp_dest, reporthook=report_progress)

        # Rename to final destination
        temp_dest.rename(dest)

        logger.info(f"[DOWNLOAD] {description} completed successfully")
        logger.info(f"[DOWNLOAD] File size: {dest.stat().st_size / (1024*1024):.1f} MB")

        return True

    except URLError as e:
        logger.error(f"[DOWNLOAD] Failed to download {description}: {e}")
        # Clean up temp file if it exists
        if temp_dest.exists():
            temp_dest.unlink()
        return False
    except Exception as e:
        logger.error(f"[DOWNLOAD] Error during download: {e}")
        if temp_dest.exists():
            temp_dest.unlink()
        return False


def ensure_gguf_model(
    model_name: str,
    url: str,
    model_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Ensure a GGUF model is downloaded and available.

    Args:
        model_name: Name of the model (for logging)
        url: Direct download URL from HuggingFace
        model_dir: Directory to store the model (default: ./models/title_generator)

    Returns:
        Path to the downloaded model, or None if download failed
    """
    if model_dir is None:
        model_dir = TITLE_MODEL_DIR

    # Extract filename from URL
    filename = url.split("/")[-1].split("?")[0]
    model_path = model_dir / filename

    if model_path.exists():
        logger.info(f"[MODEL] {model_name} already available at {model_path}")
        return model_path

    logger.info(f"[MODEL] Downloading {model_name}...")
    success = download_with_progress(url, model_path, model_name)

    if success:
        return model_path

    return None


def get_qwen_title_model() -> Optional[Path]:
    """
    Get the path to the Qwen title generation model, downloading if necessary.

    Uses: unsloth/Qwen3-1.7B-Q4_K_M.gguf (1.7GB, 4-bit quantized)

    Returns:
        Path to the model file, or None if unavailable
    """
    # Qwen3 1.7B GGUF model (Q4_K_M quantization)
    # This is a fast, efficient model perfect for title generation
    model_url = "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf"

    return ensure_gguf_model(
        model_name="Qwen3 1.7B GGUF (Q4_K_M)",
        url=model_url,
    )


def clear_model_cache(
    model_dir: Optional[Path] = None,
    keep_gguf: bool = True
) -> None:
    """
    Clear the model cache.

    Args:
        model_dir: Directory to clear (default: ./models)
        keep_gguf: If True, keep GGUF models (they're large, avoid re-downloading)
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    if not model_dir.exists():
        logger.info("[CACHE] Model directory does not exist")
        return

    total_size = 0
    deleted_files = 0

    for item in model_dir.rglob("*"):
        if item.is_file():
            # Skip GGUF files if keep_gguf is True
            if keep_gguf and item.suffix == ".gguf":
                continue

            size = item.stat().st_size
            total_size += size
            item.unlink()
            deleted_files += 1

    logger.info(f"[CACHE] Deleted {deleted_files} files, freed {total_size / (1024*1024):.1f} MB")


# =============================================================================
# ComfyUI Model Downloads
# =============================================================================

def get_comfyui_model_dir(model_type: str) -> Path:
    """
    Get the ComfyUI model directory for a specific model type.

    Args:
        model_type: Type of model (unet, clip, vae, loras, embeddings, etc.)

    Returns:
        Path to the ComfyUI model subdirectory
    """
    subdir = COMFYUI_SUBDIRS.get(model_type.lower(), model_type.lower())
    return COMFYUI_MODEL_DIR / subdir


def ensure_comfyui_model(
    model_name: str,
    url: str,
    model_type: str,
    subdir: Optional[str] = None
) -> Optional[Path]:
    """
    Ensure a ComfyUI model is downloaded and available.

    Args:
        model_name: Name of the model (for logging)
        url: Direct download URL from HuggingFace or other source
        model_type: Type of model (unet, clip, vae, loras, etc.)
        subdir: Optional subdirectory override

    Returns:
        Path to the downloaded model, or None if download failed
    """
    # Determine target directory
    if subdir:
        target_dir = COMFYUI_MODEL_DIR / subdir
    else:
        target_dir = get_comfyui_model_dir(model_type)

    # Use the general download function
    return ensure_gguf_model(
        model_name=model_name,
        url=url,
        model_dir=target_dir,
    )


def get_z_image_unet() -> Optional[Path]:
    """
    Get the z-image-turbo UNET model, downloading if necessary.

    Returns:
        Path to the UNET model, or None if unavailable
    """
    # Z-Image-Turbo UNET model
    # Update this URL with the actual model URL
    model_url = os.getenv("Z_IMAGE_UNET_URL", "")
    if not model_url:
        logger.warning("[COMFYUI] Z_IMAGE_UNET_URL not set, skipping UNET download")
        return None

    return ensure_comfyui_model(
        model_name="z-image-turbo UNET",
        url=model_url,
        model_type="unet",
    )


def get_z_image_clip() -> Optional[Path]:
    """
    Get the z-image-turbo CLIP model, downloading if necessary.

    Returns:
        Path to the CLIP model, or None if unavailable
    """
    model_url = os.getenv("Z_IMAGE_CLIP_URL", "")
    if not model_url:
        logger.warning("[COMFYUI] Z_IMAGE_CLIP_URL not set, skipping CLIP download")
        return None

    return ensure_comfyui_model(
        model_name="z-image-turbo CLIP",
        url=model_url,
        model_type="clip",
    )


def get_z_image_vae() -> Optional[Path]:
    """
    Get the z-image-turbo VAE model, downloading if necessary.

    Returns:
        Path to the VAE model, or None if unavailable
    """
    model_url = os.getenv("Z_IMAGE_VAE_URL", "")
    if not model_url:
        logger.warning("[COMFYUI] Z_IMAGE_VAE_URL not set, skipping VAE download")
        return None

    return ensure_comfyui_model(
        model_name="z-image-turbo VAE",
        url=model_url,
        model_type="vae",
    )


def setup_comfyui_models() -> Dict[str, Optional[Path]]:
    """
    Setup all ComfyUI models by downloading if necessary.

    Returns:
        Dict mapping model types to their paths
    """
    logger.info("[COMFYUI] Setting up ComfyUI models...")

    models = {
        "unet": get_z_image_unet(),
        "clip": get_z_image_clip(),
        "vae": get_z_image_vae(),
    }

    available_count = sum(1 for p in models.values() if p is not None)
    logger.info(f"[COMFYUI] {available_count}/{len(models)} models available")

    return models


def create_comfyui_symlinks(model_paths: Dict[str, Optional[Path]]) -> List[Path]:
    """
    Create symlinks from ComfyUI's expected model locations to our data directory.

    NOTE: This function is DEPRECATED. With extra_model_paths.yaml, ComfyUI
    directly reads from models/comfyui/, so symlinks are no longer needed.
    This function is kept for backward compatibility only.

    Args:
        model_paths: Dict mapping model types to their paths

    Returns:
        List of created symlink paths (empty list in new implementation)
    """
    logger.info("[COMFYUI] Using extra_model_paths.yaml - symlinks no longer needed")
    return []


# =============================================================================
# TTS Extension Model Downloads
# =============================================================================

TTS_EXTENSIONS_DIR = BACKEND_ROOT / "src" / "extensions" / "tts" / "inference"


def read_extension_configs(extension_names: Optional[List[str]] = None) -> Dict[str, dict]:
    """
    Read all TTS extension config.yaml files.

    Args:
        extension_names: Optional list of extension names to read. If None, reads all.

    Returns:
        Dict mapping extension names to their parsed config dictionaries
    """
    import yaml

    configs = {}

    if not TTS_EXTENSIONS_DIR.exists():
        logger.warning(f"[TTS MODELS] TTS extensions directory not found: {TTS_EXTENSIONS_DIR}")
        return configs

    # Find all extension directories
    if extension_names:
        ext_dirs = [TTS_EXTENSIONS_DIR / name for name in extension_names]
    else:
        ext_dirs = [d for d in TTS_EXTENSIONS_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]

    for ext_dir in ext_dirs:
        config_file = ext_dir / "config.yaml"
        if not config_file.exists():
            continue

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                ext_name = config.get('extension', {}).get('name', ext_dir.name)
                configs[ext_name] = config
                logger.debug(f"[TTS MODELS] Loaded config for {ext_name}")
        except Exception as e:
            logger.warning(f"[TTS MODELS] Failed to load config from {config_file}: {e}")

    return configs


def build_huggingface_url(repo: str, filename: str) -> str:
    """
    Build a HuggingFace download URL.

    Args:
        repo: Repo ID (e.g., "canzie/GPT-SoVITS")
        filename: Filename in the repo

    Returns:
        Full HuggingFace URL
    """
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def download_tts_models(
    extension_names: Optional[List[str]] = None,
    force: bool = False,
    retries: int = DOWNLOAD_MAX_RETRIES,
    timeout: int = DOWNLOAD_MAX_TIMEOUT
) -> Dict[str, List[Path]]:
    """
    Download TTS extension models from config.yaml files.

    This function reads extension config.yaml files and downloads required models
    using the stable download system with retries and proper error handling.

    Args:
        extension_names: Optional list of extension names. If None, processes all extensions.
        force: If True, re-download even if files exist
        retries: Number of download retries (default from constants)
        timeout: Download timeout in seconds (default from constants)

    Returns:
        Dict mapping extension names to lists of downloaded model paths
    """
    import time

    configs = read_extension_configs(extension_names)
    downloaded = {}

    for ext_name, config in configs.items():
        logger.info(f"[TTS MODELS] Processing extension: {ext_name}")

        models_config = config.get('models', [])
        if not models_config:
            logger.debug(f"[TTS MODELS] No models defined for {ext_name}")
            continue

        ext_dir = TTS_EXTENSIONS_DIR / ext_name
        downloaded[ext_name] = []

        for model in models_config:
            model_name = model.get('name', 'Unknown')
            source = model.get('source', '')
            repo = model.get('repo', '')
            destination = model.get('destination', 'models')
            is_local = model.get('local', False)
            required = model.get('required', True)
            files = model.get('files', [])

            # Skip non-HF sources for now
            if source != 'hf':
                logger.debug(f"[TTS MODELS] Skipping non-HF model: {model_name} (source: {source})")
                continue

            # Determine download destination
            if is_local:
                base_dir = ext_dir
            else:
                base_dir = ROOT_MODELS_DIR

            target_dir = base_dir / destination
            target_dir.mkdir(parents=True, exist_ok=True)

            # Download files
            if files:
                # Specific files listed
                for filename in files:
                    url = build_huggingface_url(repo, filename)
                    dest_path = target_dir / filename

                    if dest_path.exists() and not force:
                        logger.info(f"[TTS MODELS] {filename} already exists for {ext_name}")
                        downloaded[ext_name].append(dest_path)
                        continue

                    # Download with retries
                    success = False
                    for attempt in range(retries):
                        if attempt > 0:
                            wait_time = 2 ** attempt  # Exponential backoff
                            logger.info(f"[TTS MODELS] Retry {attempt + 1}/{retries} for {filename} (waiting {wait_time}s)")
                            time.sleep(wait_time)

                        logger.info(f"[TTS MODELS] Downloading {filename} for {ext_name} (attempt {attempt + 1}/{retries})")

                        if download_with_progress(url, dest_path, f"{ext_name}/{filename}"):
                            downloaded[ext_name].append(dest_path)
                            success = True
                            break

                    if not success and required:
                        logger.error(f"[TTS MODELS] Failed to download required model: {filename} for {ext_name}")
                    elif not success:
                        logger.warning(f"[TTS MODELS] Failed to download optional model: {filename} for {ext_name}")
            else:
                # No specific files - download entire repo using git-lfs or huggingface-cli
                logger.info(f"[TTS MODELS] No specific files listed for {model_name}, using git/hf-cli approach")
                # This would require additional logic - for now, skip
                logger.warning(f"[TTS MODELS] Skipping {model_name} - need to implement repo-level download")

    return downloaded


def setup_tts_extension_models(extension_name: str) -> List[Path]:
    """
    Setup models for a single TTS extension.

    This is a convenience function for downloading models for a specific extension.

    Args:
        extension_name: Name of the extension (e.g., "gpt_sovits", "index_tts")

    Returns:
        List of downloaded model paths
    """
    logger.info(f"[TTS MODELS] Setting up models for {extension_name}...")
    result = download_tts_models(extension_names=[extension_name])

    if extension_name in result:
        paths = result[extension_name]
        logger.info(f"[TTS MODELS] Downloaded {len(paths)} model(s) for {extension_name}")
        return paths

    logger.warning(f"[TTS MODELS] No models downloaded for {extension_name}")
    return []


def setup_all_tts_models() -> Dict[str, List[Path]]:
    """
    Setup models for all TTS extensions.

    Returns:
        Dict mapping extension names to lists of downloaded model paths
    """
    logger.info("[TTS MODELS] Setting up models for all TTS extensions...")
    return download_tts_models()


if __name__ == "__main__":
    # Test download
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing model download...")
    model_path = get_qwen_title_model()

    if model_path:
        logger.info(f"✓ Model available at: {model_path}")
        logger.info(f"  Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    else:
        logger.error("✗ Failed to download model")
