"""Audio Producer Workflows - Specific audio production strategies.

Each workflow orchestrates the core components to achieve a specific goal:
- podcast_director: Multi-speaker podcast production
- simple_tagger: Basic text-to-speech (future)
- audiobook_reader: Audiobook creation (future)
"""

from .podcast_director import PodcastDirector, create_podcast_director

__all__ = [
    "PodcastDirector",
    "create_podcast_director",
]
