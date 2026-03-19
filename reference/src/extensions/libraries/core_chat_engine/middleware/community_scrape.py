"""Middleware for automatically scraping community sentiment and injecting into system prompt.

This middleware runs before the agent starts, scrapes content from Reddit or 4chan,
and injects the results into the system message as context.

Usage:
    from src.extensions.libraries.core_chat_engine.middleware import CommunityScrapeMiddleware

    # For left character (Reddit)
    middleware = CommunityScrapeMiddleware(
        platform="reddit",
        community="r/politics",
        query="progressive",
        limit=20
    )

    # For right character (4chan)
    middleware = CommunityScrapeMiddleware(
        platform="4chan",
        board="pol",
        limit=20
    )

    agent = create_deep_agent(middleware=[middleware])
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Annotated, NotRequired, TypedDict
from collections.abc import Awaitable, Callable

from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langgraph.runtime import Runtime

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CommunityScrapeState(AgentState):
    """State schema for CommunityScrapeMiddleware.

    Attributes:
        community_content: Scraped community content.
            Marked as private so it's not included in the final agent state.
    """
    community_content: NotRequired[Annotated[str, PrivateStateAttr]]


class CommunityScrapeStateUpdate(TypedDict):
    """State update for CommunityScrapeMiddleware."""
    community_content: str


class CommunityScrapeMiddleware(AgentMiddleware):
    """Middleware for automatically scraping community sentiment before agent execution.

    Scrapes content from Reddit or 4chan before the agent starts and injects
    the results into the system message as context.

    Args:
        platform: Platform to scrape ("reddit" or "4chan").
        community: For Reddit - subreddit name (e.g., "r/politics").
                   For 4chan - board name (e.g., "pol").
        board: Alternative way to specify 4chan board.
        query: Optional search query for filtering posts.
        limit: Number of posts to retrieve (default: 20).
    """

    state_schema = CommunityScrapeState

    def __init__(
        self,
        *,
        platform: str = "reddit",
        community: str | None = None,
        board: str | None = None,
        query: str = "",
        limit: int = 20,
    ):
        """Initialize the community scrape middleware.

        Args:
            platform: Platform to scrape ("reddit" or "4chan").
            community: For Reddit - subreddit name (e.g., "r/politics").
            board: For 4chan - board name (e.g., "pol").
            query: Optional search query for filtering posts.
            limit: Number of posts to retrieve.
        """
        self.platform = platform.lower()
        self.community = community
        self.board = board
        self.query = query
        self.limit = limit

        # Validate platform
        if self.platform not in ("reddit", "4chan"):
            raise ValueError(f"Unsupported platform: {platform}. Use 'reddit' or '4chan'.")

        # Set target based on platform
        if self.platform == "reddit":
            self.target = self.community or "r/politics"
        else:  # 4chan
            self.target = self.board or "pol"

    def _run_async_in_thread(self, coro) -> str:
        """Run an async coroutine in a separate thread to avoid event loop conflicts.

        This is needed when before_agent runs in an async context where the
        event loop is already running.
        """
        async def run_and_get_result():
            return await coro

        with ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, run_and_get_result())
            return future.result()

    def _scrape_reddit_sync(self) -> str:
        """Scrape Reddit synchronously."""
        try:
            from src.shared.tools.logic.web.community import scrape_reddit

            # Try to run in current event loop, or fall back to thread pool
            try:
                loop = asyncio.get_running_loop()
                # Event loop is already running, use thread pool approach
                logger.debug("[CommunityScrape] Event loop running, using thread pool for Reddit scraping")
                return self._run_async_in_thread(scrape_reddit(
                    community=self.target,
                    query=self.query,
                    limit=self.limit,
                ))
            except RuntimeError:
                # No running loop, we can use asyncio.run directly
                return asyncio.run(scrape_reddit(
                    community=self.target,
                    query=self.query,
                    limit=self.limit,
                ))
        except Exception as e:
            logger.error(f"Error scraping Reddit: {e}")
            return f"[Error scraping Reddit: {e}]"

    def _scrape_4chan_sync(self) -> str:
        """Scrape 4chan synchronously."""
        try:
            from src.shared.tools.logic.web.community import scrape_4chan

            # Try to run in current event loop, or fall back to thread pool
            try:
                loop = asyncio.get_running_loop()
                # Event loop is already running, use thread pool approach
                logger.debug("[CommunityScrape] Event loop running, using thread pool for 4chan scraping")
                return self._run_async_in_thread(scrape_4chan(
                    board=self.target,
                    limit=self.limit,
                ))
            except RuntimeError:
                # No running loop, we can use asyncio.run directly
                return asyncio.run(scrape_4chan(
                    board=self.target,
                    limit=self.limit,
                ))
        except Exception as e:
            logger.error(f"Error scraping 4chan: {e}")
            return f"[Error scraping 4chan: {e}]"

    def before_agent(self, state: CommunityScrapeState, runtime: Runtime, config: RunnableConfig) -> CommunityScrapeStateUpdate | None:
        """Scrape community content before agent execution.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with community_content populated.
        """
        # Skip if already scraped
        if "community_content" in state:
            return None

        logger.info(f"[CommunityScrape] Scraping {self.platform} from {self.target}...")

        # Scrape based on platform
        if self.platform == "reddit":
            content = self._scrape_reddit_sync()
        else:  # 4chan
            content = self._scrape_4chan_sync()

        logger.info(f"[CommunityScrape] Scraped {len(content)} chars from {self.platform}")

        return CommunityScrapeStateUpdate(community_content=content)

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Inject scraped community content into the system message.

        Args:
            request: Model request to modify.

        Returns:
            Modified request with community content injected into system message.
        """
        content = request.state.get("community_content", "")
        if not content:
            return request

        # Format the community content for the system prompt
        formatted_content = f"""

<community_sentiment platform="{self.platform}" source="{self.target}">
The following is authentic community sentiment scraped from {self.platform} ({self.target}):

{content}
</community_sentiment>
"""

        # Append to system message
        from deepagents.middleware._utils import append_to_system_message
        new_system_message = append_to_system_message(request.system_message, formatted_content)

        return request.override(system_message=new_system_message)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Wrap model call to inject community content into system prompt.

        Args:
            request: Model request being processed.
            handler: Handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async wrap model call to inject community content into system prompt.

        Args:
            request: Model request being processed.
            handler: Async handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)
