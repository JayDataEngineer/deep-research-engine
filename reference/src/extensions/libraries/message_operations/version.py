"""Message version operations.

Switch between different versions of a message.
"""

import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def switch_message_version(
    checkpointer,
    message_id: str,
    version_id: str,
) -> dict:
    """Switch to a different version of a message.

    Changes which version of a message is currently active/displayed.

    Args:
        checkpointer: The shared checkpointer instance
        message_id: The ID of the message
        version_id: The ID of the version to switch to

    Returns:
        Dict with success status, message_id, and active_version_id

    Raises:
        HTTPException: If validation fails or version not found
    """
    # Validate inputs
    if not message_id:
        raise HTTPException(status_code=400, detail="message_id is required")
    if not version_id:
        raise HTTPException(status_code=400, detail="version_id is required")

    success = checkpointer.switch_active_version(message_id, version_id)

    if not success:
        raise HTTPException(status_code=404, detail="Version not found")

    logger.info(f"[MESSAGE VERSION] Switched to version {version_id} for message {message_id}")

    return {
        "success": True,
        "message_id": message_id,
        "active_version_id": version_id
    }
