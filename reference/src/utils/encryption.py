"""Encryption utilities for secure API key storage.

Uses Fernet symmetric encryption from the cryptography library.
Keys are encrypted at rest using a master key from the environment.
"""

import base64
import os
import logging
from cryptography.fernet import Fernet

from src.constants import ENCRYPTION_KEY_PATH

logger = logging.getLogger(__name__)

# Master key for encryption (from environment or generated)
_MASTER_KEY: bytes | None = None
_FERNET: Fernet | None = None


def _get_master_key() -> bytes:
    """Get or generate the master encryption key.

    Returns:
        32-byte encryption key
    """
    global _MASTER_KEY

    if _MASTER_KEY is not None:
        return _MASTER_KEY

    # Try to get from environment
    key_str = os.getenv("API_KEY_ENCRYPTION_KEY")
    if key_str:
        # Ensure key is 32 bytes (256 bits) for Fernet
        _MASTER_KEY = base64.urlsafe_b64encode(key_str.encode()[:32].ljust(32, b'\0'))
    else:
        # Generate a persistent key from a file
        key_file = os.getenv("API_KEY_ENCRYPTION_KEY_FILE", str(ENCRYPTION_KEY_PATH))
        try:
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            if os.path.exists(key_file):
                with open(key_file, "rb") as f:
                    _MASTER_KEY = f.read()
            else:
                # Generate new key and save it
                _MASTER_KEY = base64.urlsafe_b64encode(os.urandom(32))
                with open(key_file, "wb") as f:
                    f.write(_MASTER_KEY)
                # Set restrictive permissions
                os.chmod(key_file, 0o600)
                logger.warning(f"Generated new encryption key at {key_file}")
        except Exception as e:
            logger.error(f"Failed to load/create encryption key: {e}")
            # Fallback to session key (not recommended for production)
            _MASTER_KEY = base64.urlsafe_b64encode(os.urandom(32))
            logger.warning("Using ephemeral encryption key - keys will be lost on restart!")

    return _MASTER_KEY


def get_fernet() -> Fernet:
    """Get the Fernet cipher instance.

    Returns:
        Fernet cipher instance
    """
    global _FERNET

    if _FERNET is None:
        key = _get_master_key()
        _FERNET = Fernet(key)

    return _FERNET


def encrypt_value(value: str) -> str:
    """Encrypt a string value.

    Args:
        value: Plain text value to encrypt

    Returns:
        Encrypted value (base64 encoded string)
    """
    try:
        fernet = get_fernet()
        encrypted = fernet.encrypt(value.encode())
        # Return as base64 string for safe storage
        return base64.urlsafe_b64encode(encrypted).decode()
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise


def decrypt_value(encrypted_value: str) -> str:
    """Decrypt an encrypted string value.

    Args:
        encrypted_value: Encrypted value (base64 encoded string)

    Returns:
        Decrypted plain text value
    """
    try:
        fernet = get_fernet()
        # Decode from base64
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise


def is_encrypted(value: str) -> bool:
    """Check if a value appears to be encrypted.

    Args:
        value: Value to check

    Returns:
        True if value appears to be encrypted
    """
    try:
        # Try to decrypt - if it succeeds, it's encrypted
        decrypt_value(value)
        return True
    except Exception:
        return False
