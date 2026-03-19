"""
Backward compatibility module.

Re-exports BaseTTSExtension and SimpleModelExtension from base.py
for extensions that import from base_extension.
"""

from .base import BaseTTSExtension, SimpleModelExtension

__all__ = ['BaseTTSExtension', 'SimpleModelExtension']
