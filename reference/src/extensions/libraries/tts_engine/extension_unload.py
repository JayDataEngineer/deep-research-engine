"""
Generic unloading functions for TTS extensions.

Provides utilities for safely unloading TTS extensions and freeing resources.
"""

import logging

from src.extensions.core.base import BaseExtension, TTSExtensionInterface

logger = logging.getLogger(__name__)

# Try to import torch, but handle the case where it's not available
try:
    import torch
    torch_available = True
except (ImportError, OSError, ValueError) as e:
    logger.warning(f"PyTorch not available: {e}. GPU memory management will be disabled.")
    torch = None
    torch_available = False


def generic_unload_extension(extension: BaseExtension) -> bool:
    """
    Unload a TTS extension safely and free up resources.

    Args:
        extension: The TTS extension to unload

    Returns:
        bool: True if unloading was successful, False otherwise
    """
    try:
        logger.info(f"Starting generic unloading for {extension.name} extension...")

        # First, call the extension's specific unload method
        if hasattr(extension, 'unload') and callable(extension.unload):
            success = extension.unload()
            if not success:
                logger.error(f"Extension's unload method returned False for {extension.name}")
                return False

        # Clean up any model or processor attributes that might be holding onto memory
        if hasattr(extension, 'model') and extension.model is not None:
            del extension.model
            extension.model = None
            logger.debug(f"Deleted model attribute for {extension.name}")

        if hasattr(extension, 'processor') and extension.processor is not None:
            del extension.processor
            extension.processor = None
            logger.debug(f"Deleted processor attribute for {extension.name}")

        if hasattr(extension, 'tts_model') and extension.tts_model is not None:
            del extension.tts_model
            extension.tts_model = None
            logger.debug(f"Deleted tts_model attribute for {extension.name}")

        if hasattr(extension, 'model_wrapper') and extension.model_wrapper is not None:
            del extension.model_wrapper
            extension.model_wrapper = None
            logger.debug(f"Deleted model_wrapper attribute for {extension.name}")

        # Clear PyTorch CUDA cache if using CUDA and torch is available
        if torch_available and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"Cleared CUDA cache for {extension.name}")

        # Reset loading state
        extension.is_loaded = False
        if hasattr(extension, 'model_loaded'):
            extension.model_loaded = False

        logger.info(f"{extension.name} extension unloaded successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to unload {extension.name} extension: {str(e)}", exc_info=True)
        return False


def hotswap_extension(current_extension_name: str, new_extension_name: str, extension_manager) -> bool:
    """
    Safely hotswap from one extension to another.

    Args:
        current_extension_name: Name of the extension to unload
        new_extension_name: Name of the extension to load
        extension_manager: The extension manager instance

    Returns:
        bool: True if hotswap was successful, False otherwise
    """
    try:
        logger.info(f"Starting hotswap from {current_extension_name} to {new_extension_name}")

        # Check if the current extension is loaded
        current_loaded = current_extension_name in extension_manager.loaded_extensions
        new_loaded = new_extension_name in extension_manager.loaded_extensions

        # If the new extension is already loaded, we don't need to do anything
        if new_loaded:
            logger.info(f"{new_extension_name} is already loaded, no need to hotswap")
            return True

        # If the current extension is loaded, unload it first
        if current_loaded:
            logger.info(f"Unloading current extension: {current_extension_name}")
            unload_success = extension_manager.unload_extension(current_extension_name)
            if not unload_success:
                logger.error(f"Failed to unload {current_extension_name}, aborting hotswap")
                return False

        # Now load the new extension
        logger.info(f"Loading new extension: {new_extension_name}")
        load_success = extension_manager.load_extension(new_extension_name)
        if not load_success:
            logger.error(f"Failed to load {new_extension_name} during hotswap")
            # Try to reload the original extension if we unloaded it
            if current_loaded:
                logger.info(f"Attempting to reload original extension: {current_extension_name}")
                extension_manager.load_extension(current_extension_name)
            return False

        logger.info(f"Successfully hotswapped from {current_extension_name} to {new_extension_name}")
        return True

    except Exception as e:
        logger.error(f"Error during hotswap from {current_extension_name} to {new_extension_name}: {str(e)}", exc_info=True)
        return False
