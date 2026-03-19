"""
Services package.

Provides various service modules for the application.
"""

# Make title_generator import optional (requires llama_cpp which may not be installed)
try:
    from .title_generator import TitleGenerator, get_title_generator
    _title_generator_available = True
except ImportError:
    _title_generator_available = False

__all__ = []

if _title_generator_available:
    __all__.extend(["TitleGenerator", "get_title_generator"])
