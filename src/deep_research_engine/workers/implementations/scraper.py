"""Scraper Worker - Scrape documentation websites and generate LLMs.txt.

This worker scrapes documentation websites to:
1. Crawl site structure and discover pages
2. Scrape content from discovered URLs
3. Clean and format content as markdown
4. Generate LLMs.txt index file

Key Features:
- MCP-based scraping integration
- Configurable depth and max pages
- Content cleaning and formatting
- LLMs.txt generation for AI ingestion

Usage:
    worker = ScraperWorker(config)
    result = await worker.execute(state, config)
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from ...state import EngineState, WorkerResult
from ..base import BaseWorker, WorkerConfig, WorkerRegistry


@WorkerRegistry.register("scraper")
class ScraperWorker(BaseWorker):
    """Documentation scraper worker.

    This worker:
    1. Receives a documentation URL
    2. Crawls site to discover pages
    3. Scrapes content from each page
    4. Cleans and formats as markdown
    5. Optionally generates LLMs.txt index

    Scraping Strategy:
    - Uses MCP (Model Context Protocol) for scraping
    - Respects robots.txt and rate limits
    - Configurable depth and page limits
    """

    worker_type = "scraper"
    description = "Scrapes documentation websites and generates LLMs.txt"
    required_tools = ["mcp_scrape", "search_docs", "get_doc_by_url"]

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.url = config.custom.get("url", "")
        self.max_pages = config.custom.get("max_pages", 50)
        self.max_depth = config.custom.get("max_depth", 3)
        self.include_patterns = config.custom.get("include_patterns", [])
        self.exclude_patterns = config.custom.get("exclude_patterns", [])
        self.generate_llms_txt = config.custom.get("generate_llms_txt", True)
        self.output_dir = config.custom.get("output_dir", "./scraped")
        self.rate_limit = config.custom.get("rate_limit", 1.0)  # seconds between requests

    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Scrape documentation website.

        Args:
            state: Current engine state
            config: Worker configuration

        Returns:
            WorkerResult with scraped content
        """
        self._start_timing()

        try:
            # Get URL from config or state
            url = self.url or state.get("input_data", {}).get("url", "")
            if not url:
                return self._create_result(
                    success=False,
                    output=None,
                    error="No URL provided",
                )

            # Scrape the documentation site
            scraped_content = await self._scrape_site(
                base_url=url,
                max_pages=self.max_pages,
                max_depth=self.max_depth,
            )

            # Generate LLMs.txt if enabled
            llms_txt_path = None
            if self.generate_llms_txt:
                llms_txt_path = await self._generate_llms_txt(
                    scraped_content=scraped_content,
                    output_dir=self.output_dir,
                )

            self._log(f"Scraping complete: {len(scraped_content)} pages, LLMs.txt: {llms_txt_path}")

            return self._create_result(
                success=True,
                output={
                    "pages": scraped_content,
                    "page_count": len(scraped_content),
                    "llms_txt_path": llms_txt_path,
                    "base_url": url,
                    "scraped_at": datetime.now().isoformat(),
                },
                output_type="scraped_documentation",
                confidence=0.9,
                sources=[{"url": url, "type": "documentation"}],
            )

        except Exception as e:
            self._log(f"Scraping failed: {e}", level="error")
            return self._create_result(
                success=False,
                output=None,
                error=str(e),
            )

    async def _scrape_site(
        self,
        base_url: str,
        max_pages: int = 50,
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        """Scrape a documentation site.

        Args:
            base_url: Base URL to scrape
            max_pages: Maximum pages to scrape
            max_depth: Maximum crawl depth

        Returns:
            List of scraped page dicts
        """
        scraped_pages: list[dict[str, Any]] = []
        visited_urls = set()
        url_queue: list[tuple[str, int]] = [(base_url, 0)]  # (url, depth)

        self._log(f"Starting scrape: {base_url}, max_pages={max_pages}, max_depth={max_depth}")

        while url_queue and len(scraped_pages) < max_pages:
            url, depth = url_queue.pop(0)

            # Skip if already visited or too deep
            if url in visited_urls or depth > max_depth:
                continue

            visited_urls.add(url)

            try:
                # Scrape the page
                page_content = await self._scrape_page(url)
                if page_content:
                    scraped_pages.append(page_content)

                    # Extract links and add to queue
                    if depth < max_depth:
                        links = self._extract_links(page_content.get("content", ""))
                        new_urls = [
                            link for link in links if self._should_follow_link(link, base_url)
                        ]
                        url_queue.extend((link, depth + 1) for link in new_urls)

                # Rate limiting
                await asyncio.sleep(self.rate_limit)

            except Exception as e:
                self._log(f"Failed to scrape {url}: {e}", level="warning")

        return scraped_pages

    async def _scrape_page(
        self,
        url: str,
    ) -> dict[str, Any] | None:
        """Scrape a single page.

        Args:
            url: Page URL

        Returns:
            Scraped page dict or None
        """
        try:
            # Use MCP scraping tool
            from src.shared.tools.mcp.scraper import scrape_with_mcp

            result = await scrape_with_mcp(url=url)

            return {
                "url": url,
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "markdown": result.get("markdown", ""),
                "metadata": {
                    "description": result.get("description", ""),
                    "keywords": result.get("keywords", []),
                    "scraped_at": datetime.now().isoformat(),
                },
            }

        except ImportError:
            # Fallback: use httpx to fetch
            return await self._scrape_with_httpx(url)

        except Exception as e:
            self._log(f"Page scrape failed for {url}: {e}", level="warning")
            return None

    async def _scrape_with_httpx(
        self,
        url: str,
    ) -> dict[str, Any] | None:
        """Fallback scraping with httpx.

        Args:
            url: Page URL

        Returns:
            Scraped page dict or None
        """
        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                # Extract title
                title = soup.title.string if soup.title else ""

                # Extract main content (simplified)
                main_content = soup.find("main") or soup.find("article") or soup.body
                content = main_content.get_text(separator="\n", strip=True) if main_content else ""

                return {
                    "url": url,
                    "title": title,
                    "content": content,
                    "markdown": content,
                    "metadata": {
                        "scraped_at": datetime.now().isoformat(),
                    },
                }

        except Exception as e:
            self._log(f"HTTPX scrape failed for {url}: {e}", level="warning")
            return None

    def _extract_links(
        self,
        content: str,
    ) -> list[str]:
        """Extract links from content.

        Args:
            content: Page content

        Returns:
            List of URLs
        """
        import re

        # Simple regex to extract URLs
        url_pattern = r'href=["\'](https?://[^"\']+)["\']'
        matches = re.findall(url_pattern, content)

        return list(set(matches))

    def _should_follow_link(
        self,
        url: str,
        base_url: str,
    ) -> bool:
        """Check if link should be followed.

        Args:
            url: Link URL
            base_url: Base URL of site

        Returns:
            True if should follow
        """
        from urllib.parse import urlparse

        # Must be same domain
        base_parsed = urlparse(base_url)
        url_parsed = urlparse(url)

        if url_parsed.netloc != base_parsed.netloc:
            return False

        # Check include patterns
        if self.include_patterns and not any(pattern in url for pattern in self.include_patterns):
            return False

        # Check exclude patterns
        if self.exclude_patterns and any(pattern in url for pattern in self.exclude_patterns):
            return False

        # Skip common non-content URLs
        skip_patterns = [
            "/login",
            "/logout",
            "/signup",
            "/register",
            "/admin",
            "/api/",
            ".pdf",
            ".zip",
            ".exe",
        ]

        return not any(pattern in url for pattern in skip_patterns)

    async def _generate_llms_txt(
        self,
        scraped_content: list[dict[str, Any]],
        output_dir: str,
    ) -> str:
        """Generate LLMs.txt index file.

        Args:
            scraped_content: List of scraped page dicts
            output_dir: Output directory

        Returns:
            Path to generated LLMs.txt
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate LLMs.txt content
        llms_txt_content = self._build_llms_txt(scraped_content)

        # Write file
        llms_txt_path = output_path / "LLMs.txt"
        with llms_txt_path.open("w", encoding="utf-8") as f:
            f.write(llms_txt_content)

        # Also write individual page files
        pages_dir = output_path / "pages"
        pages_dir.mkdir(exist_ok=True)

        for page in scraped_content:
            page_path = pages_dir / f"{self._slugify(page['title'])}.md"
            with page_path.open("w", encoding="utf-8") as f:
                f.write(f"# {page['title']}\n\n")
                f.write(f"URL: {page['url']}\n\n")
                f.write(page.get("markdown", page.get("content", "")))

        return str(llms_txt_path)

    def _build_llms_txt(
        self,
        scraped_content: list[dict[str, Any]],
    ) -> str:
        """Build LLMs.txt content.

        Args:
            scraped_content: List of scraped page dicts

        Returns:
            LLMs.txt content string
        """
        lines = [
            "# LLMs.txt",
            "# Generated by deep-research-engine",
            f"# Generated at: {datetime.now().isoformat()}",
            "",
            "## Documentation Index",
            "",
        ]

        for page in scraped_content:
            title = page.get("title", "Untitled")
            url = page.get("url", "")
            description = page.get("metadata", {}).get("description", "")

            lines.append(f"### {title}")
            lines.append(f"URL: {url}")
            if description:
                lines.append(f"Description: {description}")
            lines.append("")

        return "\n".join(lines)

    def _slugify(
        self,
        text: str,
    ) -> str:
        """Convert text to filename-safe slug.

        Args:
            text: Text to slugify

        Returns:
            Slug string
        """
        import re

        # Lowercase and replace spaces with underscores
        slug = text.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[-\s]+", "-", slug)

        return slug[:50]  # Limit length
