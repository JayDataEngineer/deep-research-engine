"""Attachments Manager Extension

Manages file attachments and web scraping history.
"""

from .extension import AttachmentsExtension

# Export as Extension for the loader to find
Extension = AttachmentsExtension

__all__ = ["AttachmentsExtension", "Extension"]
