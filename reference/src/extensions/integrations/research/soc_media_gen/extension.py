"""Social Media Content Generator Extension."""

from src.extensions.core.base_research import ResearchExtension


class SocMediaGenExtension(ResearchExtension):
    """Social media content generator for viral content across platforms."""

    def get_profile(self):
        from .profile import SocMediaGenProfile
        return SocMediaGenProfile(self.config)

    def get_output_strategy(self, mode=None):
        # Twitter variant uses TwitterFormatter, others may use different formatters
        if mode == "twitter":
            from .output_formatting.output import TwitterFormatter
            return TwitterFormatter
        # Default to text formatter
        from src.extensions.content.deep_research.graph.output import TextFormatter
        return TextFormatter
