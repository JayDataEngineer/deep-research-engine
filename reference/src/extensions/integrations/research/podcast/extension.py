"""Podcast Research Strategy Extension.

Generates podcast scripts with character-driven dialogue using
contrasting personalities for engaging content.
"""

from src.extensions.core.base_research import ResearchExtension


class PodcastExtension(ResearchExtension):
    """Podcast research strategy extension.

    Provides:
    - Profile: PodcastProfile with character personality management
    - Output Strategy: PodcastFormatter for script generation
    - Variants: 5worker, 2worker, accelerated (different worker configs)

    Config Hierarchy:
    1. System base (deep_research/base/config.yaml)
    2. Podcast extension base (config.yaml)
    3. Variant (variants/2worker.yaml, etc.)
    4. User variant (user_variants/{user}/podcast/{variant}.yaml)
    """

    def get_profile(self):
        """Return the Podcast profile for worker behavior."""
        from .profile import PodcastProfile
        return PodcastProfile(self.config)

    def get_output_strategy(self, mode=None):
        """Return the Podcast output formatter."""
        from .output_formatting.output import PodcastFormatter
        return PodcastFormatter
