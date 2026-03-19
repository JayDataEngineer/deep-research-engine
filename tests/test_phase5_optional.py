"""Tests for Phase 5: Optional Modules (Graph Builder + Synthesis).

Note: Some tests require postgres libraries to be installed on the system.
These tests are marked with @pytest.mark.skipif when dependencies are missing.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Check if postgres dependencies are available
try:
    from langgraph.store.postgres import PostgresStore

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


# =============================================================================
# GRAPH BUILDER TESTS
# =============================================================================


@pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="Postgres dependencies not available")
class TestNeo4jBuilder:
    """Tests for Neo4j Graph Builder."""

    def test_entity_creation(self):
        """Test Entity dataclass."""
        from context_engine.graph_builder import Entity, EntityType

        entity = Entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            description="A programming language",
            source_ids=["doc1.txt"],
            confidence=0.95,
        )

        assert entity.name == "Python"
        assert entity.entity_type == EntityType.TECHNOLOGY
        assert entity.confidence == 0.95

        # Test to_neo4j_dict
        neo4j_dict = entity.to_neo4j_dict()
        assert neo4j_dict["name"] == "Python"
        assert neo4j_dict["type"] == "Technology"
        assert "created_at" in neo4j_dict

    def test_relationship_creation(self):
        """Test Relationship dataclass."""
        from context_engine.graph_builder import Relationship, RelationType

        rel = Relationship(
            source_entity="Python",
            target_entity="Django",
            relation_type=RelationType.USES,
            context="Django is built with Python",
            confidence=0.9,
        )

        assert rel.source_entity == "Python"
        assert rel.target_entity == "Django"
        assert rel.relation_type == RelationType.USES

        # Test to_neo4j_dict
        neo4j_dict = rel.to_neo4j_dict()
        assert neo4j_dict["source"] == "Python"
        assert neo4j_dict["target"] == "Django"
        assert neo4j_dict["type"] == "USES"

    def test_entity_types_enum(self):
        """Test EntityType enum values."""
        from context_engine.graph_builder import EntityType

        assert EntityType.PERSON.value == "Person"
        assert EntityType.ORGANIZATION.value == "Organization"
        assert EntityType.TECHNOLOGY.value == "Technology"
        assert EntityType.CONCEPT.value == "Concept"

    def test_relation_types_enum(self):
        """Test RelationType enum values."""
        from context_engine.graph_builder import RelationType

        assert RelationType.RELATED_TO.value == "RELATED_TO"
        assert RelationType.PART_OF.value == "PART_OF"
        assert RelationType.CREATED_BY.value == "CREATED_BY"
        assert RelationType.USES.value == "USES"

    @pytest.mark.asyncio
    async def test_build_knowledge_graph_no_neo4j(self):
        """Test build_knowledge_graph when Neo4j is unavailable."""
        from context_engine.graph_builder import build_knowledge_graph

        with patch(
            "context_engine.graph_builder.neo4j_builder.Neo4jBuilder"
        ) as mock_builder_class:
            mock_builder = MagicMock()
            mock_builder.is_available.return_value = False
            mock_builder_class.return_value = mock_builder

            result = await build_knowledge_graph("test-session")

            assert result["entities_created"] == 0
            assert result["relationships_created"] == 0
            assert "Neo4j connection not available" in result["errors"]


# =============================================================================
# TWITTER SYNTHESIS TESTS (Unit tests without postgres dependency)
# =============================================================================


class TestTweetSchemas:
    """Test Tweet-related Pydantic schemas without postgres dependency."""

    def test_tweet_schema_basic(self):
        """Test basic Tweet schema creation."""
        from context_engine.synthesis.twitter import Tweet

        tweet = Tweet(
            content="This is a test tweet with #hashtags",
            hashtags=["test", "pytest"],
            media_hint="image",
        )

        assert tweet.content == "This is a test tweet with #hashtags"
        assert len(tweet.hashtags) == 2
        assert tweet.media_hint == "image"

    def test_tweet_thread_schema(self):
        """Test TweetThread schema creation."""
        from context_engine.synthesis.twitter import Tweet, TweetThread

        thread = TweetThread(
            tweets=[
                Tweet(content="First tweet"),
                Tweet(content="Second tweet"),
            ],
            hook_strength="strong",
            call_to_action="Follow for more!",
            total_char_count=30,
        )

        assert len(thread.tweets) == 2
        assert thread.hook_strength == "strong"
        assert thread.call_to_action == "Follow for more!"

    def test_format_thread_output(self):
        """Test format_thread_output function."""
        from context_engine.synthesis.twitter import (
            Tweet,
            TweetThread,
            format_thread_output,
        )

        thread = TweetThread(
            tweets=[
                Tweet(content="First tweet", hashtags=["test"]),
                Tweet(content="Second tweet"),
            ],
            hook_strength="strong",
            call_to_action="Follow for more!",
            total_char_count=30,
        )

        output = format_thread_output(thread)

        assert "TWITTER THREAD" in output
        assert "First tweet" in output
        assert "Second tweet" in output
        assert "TWEET 1/2" in output
        assert "#test" in output


# =============================================================================
# PODCAST SYNTHESIS TESTS
# =============================================================================


class TestPodcastSchemas:
    """Test Podcast-related Pydantic schemas."""

    def test_dialogue_line_schema(self):
        """Test DialogueLine schema creation."""
        from context_engine.synthesis.podcast import DialogueLine

        line = DialogueLine(
            speaker="Alex",
            text="Welcome to the show!",
            emotion="happy",
            stage_direction="Opens with enthusiasm",
        )

        assert line.speaker == "Alex"
        assert line.text == "Welcome to the show!"
        assert line.emotion == "happy"
        assert line.stage_direction == "Opens with enthusiasm"

    def test_podcast_segment_schema(self):
        """Test PodcastSegment schema creation."""
        from context_engine.synthesis.podcast import DialogueLine, PodcastSegment

        segment = PodcastSegment(
            segment_type="intro",
            title="Welcome",
            dialogue=[
                DialogueLine(speaker="Alex", text="Hello!"),
                DialogueLine(speaker="Sam", text="Hi there!"),
            ],
        )

        assert segment.segment_type == "intro"
        assert segment.title == "Welcome"
        assert len(segment.dialogue) == 2

    def test_podcast_script_schema(self):
        """Test PodcastScript schema creation."""
        from context_engine.synthesis.podcast import (
            DialogueLine,
            PodcastSegment,
            PodcastScript,
        )

        script = PodcastScript(
            title="AI Revolution",
            summary="Exploring the latest in AI",
            segments=[
                PodcastSegment(
                    segment_type="intro",
                    dialogue=[DialogueLine(speaker="Alex", text="Welcome!")],
                )
            ],
            total_duration_estimate="15-20 minutes",
            notes="Use upbeat background music",
        )

        assert script.title == "AI Revolution"
        assert len(script.segments) == 1
        assert script.total_duration_estimate == "15-20 minutes"

    def test_format_podcast_output(self):
        """Test format_podcast_output function."""
        from context_engine.synthesis.podcast import (
            DialogueLine,
            PodcastSegment,
            PodcastScript,
            format_podcast_output,
        )

        script = PodcastScript(
            title="Test Podcast",
            summary="A test podcast",
            segments=[
                PodcastSegment(
                    segment_type="intro",
                    dialogue=[
                        DialogueLine(speaker="Alex", text="Hello!"),
                        DialogueLine(
                            speaker="Sam", text="Hi!", stage_direction="Waves"
                        ),
                    ],
                )
            ],
            notes="Production notes",
        )

        output = format_podcast_output(script)

        assert "PODCAST: Test Podcast" in output
        assert "ALEX: Hello!" in output
        assert "SAM: Hi!" in output
        assert "[Waves]" in output
        assert "Production notes" in output


# =============================================================================
# REPORT SYNTHESIS TESTS
# =============================================================================


class TestReportSchemas:
    """Test Report-related Pydantic schemas."""

    def test_report_section_schema(self):
        """Test ReportSection schema creation."""
        from context_engine.synthesis.report import ReportSection

        section = ReportSection(
            title="Introduction",
            content="This is the intro.",
            subsections=[
                ReportSection(title="Background", content="Background info.")
            ],
        )

        assert section.title == "Introduction"
        assert section.content == "This is the intro."
        assert len(section.subsections) == 1

    def test_report_schema(self):
        """Test Report schema creation."""
        from context_engine.synthesis.report import Report, ReportSection

        report = Report(
            title="Research Report",
            executive_summary="This report covers X, Y, and Z.",
            sections=[
                ReportSection(title="Introduction", content="Intro content."),
                ReportSection(title="Analysis", content="Analysis content."),
            ],
            key_findings=["Finding 1", "Finding 2"],
            recommendations=["Do this", "Do that"],
            sources=["Source 1", "Source 2"],
            generated_at="2024-01-01T00:00:00",
        )

        assert report.title == "Research Report"
        assert len(report.sections) == 2
        assert len(report.key_findings) == 2
        assert len(report.recommendations) == 2

    def test_format_report_output(self):
        """Test format_report_output function."""
        from context_engine.synthesis.report import Report, ReportSection, format_report_output

        report = Report(
            title="Test Report",
            executive_summary="Summary here.",
            sections=[
                ReportSection(title="Intro", content="Intro content.")
            ],
            key_findings=["Key finding 1"],
            recommendations=["Recommendation 1"],
            generated_at="2024-01-01T00:00:00",
        )

        output = format_report_output(report)

        assert "# Test Report" in output
        assert "## Executive Summary" in output
        assert "Summary here." in output
        assert "## Intro" in output
        assert "## Key Findings" in output
        assert "## Recommendations" in output
        assert "- Key finding 1" in output


# =============================================================================
# CLI TESTS
# =============================================================================


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_import(self):
        """Test that CLI module can be imported."""
        from context_engine.cli import cli

        assert cli is not None

    def test_cli_has_build_command(self):
        """Test that build command is registered."""
        from context_engine.cli import cli

        # Check that build command exists
        commands = list(cli.commands.keys())
        assert "build" in commands

    def test_cli_has_build_graph_command(self):
        """Test that build-graph command is registered."""
        from context_engine.cli import cli

        commands = list(cli.commands.keys())
        assert "build-graph" in commands

    def test_cli_has_synthesize_command(self):
        """Test that synthesize command is registered."""
        from context_engine.cli import cli

        commands = list(cli.commands.keys())
        assert "synthesize" in commands

    def test_cli_has_query_command(self):
        """Test that query command is registered."""
        from context_engine.cli import cli

        commands = list(cli.commands.keys())
        assert "query" in commands


# =============================================================================
# INTEGRATION TESTS (Require mocking)
# =============================================================================


class TestSynthesisIntegration:
    """Integration tests for Synthesis module."""

    @pytest.mark.asyncio
    async def test_synthesize_twitter_no_content(self):
        """Test synthesize_twitter_thread when no content found."""
        from context_engine.synthesis.twitter import synthesize_twitter_thread

        with patch(
            "context_engine.synthesis.twitter._get_store"
        ) as mock_get_store:
            # Setup mock store
            mock_store = AsyncMock()
            mock_store.asearch.return_value = []
            mock_get_store.return_value = mock_store

            with pytest.raises(ValueError, match="No content found"):
                await synthesize_twitter_thread("nonexistent-session")

    @pytest.mark.asyncio
    async def test_synthesize_podcast_no_content(self):
        """Test synthesize_podcast when no content found."""
        from context_engine.synthesis.podcast import synthesize_podcast

        with patch(
            "context_engine.synthesis.podcast._get_store"
        ) as mock_get_store:
            # Setup mock store
            mock_store = AsyncMock()
            mock_store.asearch.return_value = []
            mock_get_store.return_value = mock_store

            with pytest.raises(ValueError, match="No content found"):
                await synthesize_podcast("nonexistent-session")

    @pytest.mark.asyncio
    async def test_synthesize_report_no_content(self):
        """Test synthesize_report when no content found."""
        from context_engine.synthesis.report import synthesize_report

        with patch(
            "context_engine.synthesis.report._get_store"
        ) as mock_get_store:
            # Setup mock store
            mock_store = AsyncMock()
            mock_store.asearch.return_value = []
            mock_get_store.return_value = mock_store

            with pytest.raises(ValueError, match="No content found"):
                await synthesize_report("nonexistent-session")


# =============================================================================
# ACCEPTANCE TESTS
# =============================================================================


class TestPhase5Acceptance:
    """Acceptance tests for Phase 5 deliverables."""

    @pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="Postgres dependencies not available")
    def test_graph_builder_module_exists(self):
        """Test that graph_builder module exists with required components."""
        from context_engine import graph_builder

        assert hasattr(graph_builder, "__init__")

    def test_synthesis_module_exists(self):
        """Test that synthesis module exists with required components."""
        from context_engine import synthesis

        assert hasattr(synthesis, "__init__")

    def test_synthesis_has_twitter(self):
        """Test that synthesis has Twitter module."""
        from context_engine.synthesis import twitter

        assert hasattr(twitter, "synthesize_twitter_thread")
        assert hasattr(twitter, "Tweet")
        assert hasattr(twitter, "TweetThread")
        assert hasattr(twitter, "format_thread_output")

    def test_synthesis_has_podcast(self):
        """Test that synthesis has Podcast module."""
        from context_engine.synthesis import podcast

        assert hasattr(podcast, "synthesize_podcast")
        assert hasattr(podcast, "PodcastScript")
        assert hasattr(podcast, "PodcastSegment")
        assert hasattr(podcast, "DialogueLine")
        assert hasattr(podcast, "format_podcast_output")

    def test_synthesis_has_report(self):
        """Test that synthesis has Report module."""
        from context_engine.synthesis import report

        assert hasattr(report, "synthesize_report")
        assert hasattr(report, "Report")
        assert hasattr(report, "ReportSection")
        assert hasattr(report, "format_report_output")

    def test_cli_has_all_commands(self):
        """Test that CLI has all required Phase 5 commands."""
        from context_engine.cli import cli

        commands = list(cli.commands.keys())

        # Main pipeline
        assert "build" in commands

        # Optional Phase 5 commands
        assert "build-graph" in commands
        assert "synthesize" in commands
        assert "query" in commands

    def test_cli_synthesize_has_format_option(self):
        """Test that synthesize command has format choices."""
        from context_engine.cli import cli

        # Get synthesize command
        synthesize_cmd = cli.commands.get("synthesize")
        assert synthesize_cmd is not None

        # Check that --format option has the expected choices
        params = {p.name: p for p in synthesize_cmd.params}
        assert "output_format" in params

        format_param = params["output_format"]
        # Check that it's a Choice type with the expected values
        assert hasattr(format_param, "type")
        choices = ["twitter", "podcast", "report"]
        for choice in choices:
            assert choice in format_param.type.choices
