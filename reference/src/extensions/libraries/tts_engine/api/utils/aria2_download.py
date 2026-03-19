"""
Reliable file download using aria2c with retries.

This module provides a robust download utility that uses aria2c
for downloading files with resume support, retries, and parallel connections.
"""

import logging
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from src.constants import HUGGINGFACE_BASE_URL, DOWNLOAD_MIN_SPLIT_SIZE, DOWNLOAD_MAX_RETRIES

logger = logging.getLogger(__name__)


def check_aria2c() -> bool:
    """Check if aria2c is available."""
    try:
        result = subprocess.run(
            ["aria2c", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_aria2c() -> bool:
    """Install aria2c if not available."""
    if check_aria2c():
        return True

    logger.info("aria2c not found, attempting to install...")

    # Try different installation methods
    install_commands = [
        ["sudo", "apt-get", "install", "-y", "aria2"],
        ["brew", "install", "aria2"],
        ["sudo", "dnf", "install", "-y", "aria2"],
        ["sudo", "yum", "install", "-y", "aria2"],
    ]

    for cmd in install_commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and check_aria2c():
                logger.info("aria2c installed successfully")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    logger.error("Failed to install aria2c")
    return False


def download_with_aria2c(
    url: str,
    output_dir: Path,
    filename: str | None = None,
    max_retries: int = None,
    timeout: int = None,
    connections: int = 16,
    user_agent: str | None = None,
    headers: dict | None = None,
) -> Path:
    """
    Download a file using aria2c with robust retry logic.

    Args:
        url: URL to download
        output_dir: Directory to save the file
        filename: Optional filename (auto-detected from URL if not provided)
        max_retries: Maximum number of retry attempts
        timeout: Timeout per download attempt in seconds
        connections: Number of parallel connections
        user_agent: Custom user agent string
        headers: Additional HTTP headers

    Returns:
        Path to the downloaded file

    Raises:
        RuntimeError: If download fails after all retries
    """
    # Use constants for default values
    if max_retries is None:
        max_retries = DOWNLOAD_MAX_RETRIES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = Path(urlparse(url).path).name
    output_path = output_dir / filename

    # Ensure aria2c is available
    if not check_aria2c():
        if not install_aria2c():
            raise RuntimeError("aria2c is not available and could not be installed")

    # Build aria2c command
    cmd = [
        "aria2c",
        "--dir=str(output_dir)",  # Output directory
        "--out=" + filename,  # Output filename
        "--max-tries=" + str(max_retries),  # Retry within aria2c
        "--retry-wait=5",  # Wait 5 seconds between retries
        "--timeout=" + str(timeout),  # Timeout
        "--split=" + str(connections),  # Split into connections
        "--min-split-size=" + DOWNLOAD_MIN_SPLIT_SIZE,  # Only split if file is > threshold
        "--continue=true",  # Resume partial downloads
        "--check-certificate=true",  # Verify SSL
        "--auto-file-renaming=true",  # Rename if file exists
        "--allow-overwrite=false",  # Don't overwrite existing
        "--max-file-not-found=" + str(max_retries),  # Retry 404s
        "--stderr=true",  # Show progress
        "--summary-interval=60",  # Print summary every 60s
        "--stop-with-process=$$",
        "--rpc-enable=false",
    ]

    # Add user agent
    if user_agent:
        cmd.append("--user-agent=" + user_agent)
    else:
        # Default user agent for HuggingFace
        cmd.append("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    # Add headers
    if headers:
        for key, value in headers.items():
            cmd.append(f"--header={key}: {value}")

    cmd.append(url)

    # Attempt download with retries
    last_error = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}: {url}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 60,  # Extra time for overhead
                cwd=str(output_dir)
            )

            if result.returncode == 0:
                # Verify file was created
                if output_path.exists() and output_path.stat().st_size > 0:
                    logger.info(f"Successfully downloaded: {output_path} ({output_path.stat().st_size} bytes)")
                    return output_path
                else:
                    last_error = f"Download completed but file not found: {output_path}"
            else:
                last_error = f"aria2c returned code {result.returncode}: {result.stderr}"

            # Clean up partial download if it failed
            if output_path.exists():
                if output_path.stat().st_size == 0:
                    output_path.unlink()
                    aria2c_files = list(output_dir.glob("*.aria2c"))
                    for f in aria2c_files:
                        f.unlink()

        except subprocess.TimeoutExpired:
            last_error = f"Download timed out after {timeout} seconds"
        except Exception as e:
            last_error = f"Download error: {e}"

        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 5  # Increasing wait times
            logger.warning(f"Download failed, retrying in {wait_time}s: {last_error}")
            time.sleep(wait_time)

    raise RuntimeError(f"Failed to download after {max_retries} attempts: {last_error}")


def download_huggingface_model(
    repo_id: str,
    filename: str,
    output_dir: Path,
    revision: str = "main",
    token: str | None = None,
) -> Path:
    """
    Download a file from HuggingFace using aria2c.

    Args:
        repo_id: HuggingFace repository ID (e.g., "microsoft/vibevoice-1.5b")
        filename: File name in the repository
        output_dir: Directory to save the file
        revision: Git revision (branch, tag, or commit)
        token: Optional HuggingFace access token

    Returns:
        Path to the downloaded file
    """
    # Construct HuggingFace CDN URL
    url = f"{HUGGINGFACE_BASE_URL}/{repo_id}/resolve/{revision}/{filename}"

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return download_with_aria2c(
        url=url,
        output_dir=output_dir,
        filename=filename,
        headers=headers if headers else None,
    )


def batch_download(
    urls_files: list[tuple[str, Path]],
    max_concurrent: int = 3,
) -> list[Path]:
    """
    Download multiple files with controlled concurrency.

    Args:
        urls_files: List of (url, output_path) tuples
        max_concurrent: Maximum number of concurrent downloads

    Returns:
        List of successfully downloaded file paths
    """
    import concurrent.futures

    successful = []
    failed = []

    def download_wrapper(url_and_path):
        url, path = url_and_path
        try:
            result = download_with_aria2c(url, path.parent, path.name)
            return result
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(download_wrapper, item): item for item in urls_files}

        for future in concurrent.futures.as_completed(futures, timeout=None):
            result = future.result()
            if result:
                successful.append(result)
            else:
                failed.append(futures[future])

    logger.info(f"Batch download complete: {len(successful)} succeeded, {len(failed)} failed")

    return successful


if __name__ == "__main__":
    # Test aria2c download
    logging.basicConfig(level=logging.INFO)

    test_url = "https://huggingface.co/espeak-ng/espeak-ng/resolve/main/espeak-ng-data.tar.gz"
    test_dir = Path("/tmp/aria2c_test")

    try:
        result = download_with_aria2c(
            url=test_url,
            output_dir=test_dir,
            max_retries=3,
            timeout=300,
        )
        print(f"Downloaded to: {result}")
    except Exception as e:
        print(f"Failed: {e}")
