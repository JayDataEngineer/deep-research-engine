"""
Model validation service for automatic background validation of models.
"""

import logging
import threading
import time
from datetime import datetime, timedelta

from src.extensions.libraries.tts_engine.api.utils.model_validator import model_validator

logger = logging.getLogger(__name__)


class ModelValidationService:
    """Service for automatic background validation of models."""

    def __init__(self):
        self.validation_results: dict[str, dict] = {}
        self.last_validation_time: dict[str, datetime] = {}
        self.validation_in_progress: dict[str, bool] = {}
        self.validation_interval = timedelta(hours=1)  # Validate every hour by default
        self._running = False
        self._thread = None

    def start_background_validation(self):
        """Start the background validation service."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._validation_loop, daemon=True)
            self._thread.start()
            logger.info("Model validation service started")

    def stop_background_validation(self):
        """Stop the background validation service."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
            logger.info("Model validation service stopped")

    def _validation_loop(self):
        """Main validation loop that runs in background thread."""
        while self._running:
            try:
                # Check if any models need validation
                for model_name in [
                    "kokoro",
                    "vibevoice",
                    "vibevoice-7b",
                    "indextts",
                    "gptsovits",
                    "maya1",
                ]:
                    if self._should_validate_model(model_name):
                        self._validate_model_background(model_name)

                # Sleep for 10 minutes between checks
                for _ in range(60):  # 60 * 10 seconds = 10 minutes
                    if not self._running:
                        break
                    time.sleep(10)
            except Exception as e:
                logger.error(f"Error in validation loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying

    def _should_validate_model(self, model_name: str) -> bool:
        """Check if a model should be validated."""
        if self.validation_in_progress.get(model_name, False):
            return False

        last_validated = self.last_validation_time.get(model_name)
        if not last_validated:
            return True  # First validation

        return datetime.now() - last_validated > self.validation_interval

    def _validate_model_background(self, model_name: str):
        """Validate a single model in the background."""
        self.validation_in_progress[model_name] = True
        try:
            result = model_validator.validate_model(model_name)
            self.validation_results[model_name] = result
            self.last_validation_time[model_name] = datetime.now()

            if not result.get("valid", False):
                logger.warning(
                    f"Model {model_name} failed validation: {result.get('error', 'Unknown error')}"
                )
                # Try Hugging Face checksum validation for more detailed analysis
                self._perform_hf_checksum_validation(model_name)
            else:
                logger.info(f"Model {model_name} validated successfully")

        except Exception as e:
            logger.error(f"Error validating model {model_name}: {str(e)}")
            self.validation_results[model_name] = {
                "valid": False,
                "error": str(e),
                "checksum": None,
            }
        finally:
            self.validation_in_progress[model_name] = False

    def _perform_hf_checksum_validation(self, model_name: str):
        """Perform Hugging Face checksum validation and handle failures."""
        try:
            logger.info(f"Performing Hugging Face checksum validation for {model_name}...")
            hf_validation_result = model_validator.validate_model_with_hf_checksums(model_name)

            if not hf_validation_result["valid"]:
                logger.error(f"Hugging Face checksum validation failed for {model_name}")
                logger.error(f"Details: {hf_validation_result.get('error', 'Unknown error')}")

                # Note: Do NOT trigger automatic re-download - let user decide when to download
                logger.warning(f"Model {model_name} failed validation, but automatic download is disabled. "
                              f"User must manually trigger download when needed.")
            else:
                logger.info(f"Hugging Face checksum validation passed for {model_name}")

        except Exception as e:
            logger.error(f"Error during Hugging Face checksum validation for {model_name}: {str(e)}")

    def _trigger_model_redownload(self, model_name: str):
        """Trigger re-download of a corrupted model."""
        try:
            from src.extensions.libraries.tts_engine.api.utils.download_models import (
                delete_model_directory,
                download_model_by_name,
            )

            logger.info(f"Deleting corrupted model {model_name}...")
            if delete_model_directory(model_name):
                logger.info(f"Re-downloading model {model_name}...")
                # Start download in a separate thread to avoid blocking
                def redownload():
                    try:
                        success = download_model_by_name(model_name)
                        if success:
                            logger.info(f"Successfully re-downloaded and validated model {model_name}")
                            # Clear validation cache to force re-validation
                            if model_name in self.validation_results:
                                del self.validation_results[model_name]
                            if model_name in self.last_validation_time:
                                del self.last_validation_time[model_name]
                        else:
                            logger.error(f"Failed to re-download model {model_name}")
                    except Exception as e:
                        logger.error(f"Error during model re-download: {str(e)}")

                thread = threading.Thread(target=redownload, daemon=True)
                thread.start()
            else:
                logger.error(f"Failed to delete corrupted model {model_name}")

        except Exception as e:
            logger.error(f"Error triggering model re-download for {model_name}: {str(e)}")

    def validate_model_now(self, model_name: str) -> dict:
        """Validate a model immediately (not in background)."""
        result = model_validator.validate_model(model_name)
        self.validation_results[model_name] = result
        self.last_validation_time[model_name] = datetime.now()

        # If validation failed, trigger alert mechanism
        if not result.get("valid", False):
            self._handle_validation_failure(model_name, result)

        return result

    def validate_model_with_hf_checksums(self, model_name: str) -> dict:
        """Validate a model using Hugging Face checksums immediately."""
        try:
            result = model_validator.validate_model_with_hf_checksums(model_name)
            self.validation_results[model_name] = result
            self.last_validation_time[model_name] = datetime.now()

            # If validation failed, DO NOT automatically trigger re-download
            if not result.get("valid", False):
                logger.warning(f"Model {model_name} failed Hugging Face checksum validation, "
                              f"but automatic download is disabled. User must manually trigger download when needed.")

            return result
        except Exception as e:
            logger.error(f"Error validating model {model_name} with Hugging Face checksums: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "checksum": None,
            }

    def _handle_validation_failure(self, model_name: str, result: dict):
        """Handle model validation failure by logging and potentially triggering re-download."""
        error_msg = result.get("error", "Unknown validation error")
        logger.warning(f"Model validation failed for {model_name}: {error_msg}")

        # Try Hugging Face checksum validation for more detailed analysis
        self._perform_hf_checksum_validation(model_name)

        return result

    def get_validation_result(self, model_name: str) -> dict | None:
        """Get the validation result for a model."""
        return self.validation_results.get(model_name)

    def is_model_valid(self, model_name: str) -> bool:
        """Check if a model is valid."""
        result = self.get_validation_result(model_name)
        return result is not None and result.get("valid", False)

    def has_validation_error(self, model_name: str) -> str | None:
        """Get the validation error for a model, if any."""
        result = self.get_validation_result(model_name)
        if result is None:
            # No validation result yet, trigger validation
            self.validate_model_now(model_name)
            result = self.get_validation_result(model_name)

        if result is None:
            return None

        if not result.get("valid", False):
            return result.get("error")

        return None

    def is_model_healthy(self, model_name: str) -> bool:
        """Check if a model is healthy (exists and passes validation)."""
        from app.api.v1.models import check_model_exists

        # First check if model exists
        if not check_model_exists(model_name):
            return False

        # Then check validation
        result = self.get_validation_result(model_name)
        if result is None:
            # No validation result yet, trigger validation
            self.validate_model_now(model_name)
            result = self.get_validation_result(model_name)

        if result is None:
            return False

        return result.get("valid", False)

    def force_model_redownload(self, model_name: str) -> bool:
        """Force re-download of a model, even if it appears valid."""
        try:
            logger.info(f"Force re-downloading model {model_name}...")
            self._trigger_model_redownload(model_name)
            return True
        except Exception as e:
            logger.error(f"Error forcing model re-download for {model_name}: {str(e)}")
            return False


# Global instance
model_validation_service = ModelValidationService()
