"""Provider availability checking utilities for LiteLLM.

Simplified utilities for checking LiteLLM proxy and local service availability.
With LiteLLM handling all provider routing, we only need to check:
1. LiteLLM proxy connectivity
2. Optional: Local services for custom endpoints

Usage:
    from src.services.provider_checks import (
        check_litellm_proxy,
        check_local_service,
    )

    # Check if LiteLLM proxy is available
    is_available, message = await check_litellm_proxy()

    # Check if a local service is running
    is_running = await check_local_service("http://localhost:8188")
"""

import logging
from typing import Tuple, Optional

from src.shared.litellm_config import LITELLM_BASE_URL, LITELLM_API_KEY

logger = logging.getLogger(__name__)


async def check_litellm_proxy(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 5.0,
) -> Tuple[bool, str]:
    """Check if LiteLLM proxy is available and responding.

    Args:
        base_url: LiteLLM base URL (default: from LITELLM_BASE_URL env var)
        api_key: LiteLLM API key (default: from LITELLM_API_KEY env var)
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_available, message)

    Example:
        >>> is_available, msg = await check_litellm_proxy()
        >>> if is_available:
        ...     print(f"LiteLLM is available: {msg}")
    """
    import httpx

    base_url = base_url or LITELLM_BASE_URL
    api_key = api_key or LITELLM_API_KEY

    # Remove /v1 suffix for health check endpoints
    check_url = base_url.replace("/v1", "").rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Try health endpoint first
            try:
                response = await client.get(f"{check_url}/health")
                if response.status_code == 200:
                    return True, "LiteLLM proxy is healthy"
            except Exception:
                pass

            # Fallback: try models endpoint
            headers = {"Authorization": f"Bearer {api_key}"}
            response = await client.get(f"{base_url}/models", headers=headers)
            if response.status_code == 200:
                return True, "LiteLLM proxy is responding"
            elif response.status_code == 401:
                return False, "LiteLLM proxy returned 401 (invalid API key)"
            else:
                return False, f"LiteLLM proxy returned {response.status_code}"

    except httpx.ConnectError as e:
        return False, f"Cannot connect to LiteLLM proxy at {base_url}"
    except httpx.TimeoutException:
        return False, f"Timeout connecting to LiteLLM proxy at {base_url}"
    except Exception as e:
        logger.debug(f"LiteLLM proxy check failed: {e}")
        return False, f"Error checking LiteLLM proxy: {str(e)}"


def check_litellm_proxy_sync(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 5.0,
) -> Tuple[bool, str]:
    """Synchronous version of check_litellm_proxy.

    Args:
        base_url: LiteLLM base URL (default: from LITELLM_BASE_URL env var)
        api_key: LiteLLM API key (default: from LITELLM_API_KEY env var)
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_available, message)
    """
    import urllib.request
    import socket

    socket.setdefaulttimeout(timeout)

    base_url = base_url or os.getenv("LITELLM_BASE_URL", "http://litellm.local/v1")
    api_key = api_key or os.getenv("LITELLM_API_KEY", "sk-litellm-master")

    check_url = base_url.replace("/v1", "").rstrip("/")

    try:
        # Try health endpoint
        try:
            req = urllib.request.urlopen(f"{check_url}/health", timeout=timeout)
            if req.status == 200:
                return True, "LiteLLM proxy is healthy"
        except Exception:
            pass

        # Fallback: try models endpoint
        headers = {"Authorization": f"Bearer {api_key}"}
        request = urllib.request.Request(f"{base_url}/models", headers=headers)
        req = urllib.request.urlopen(request, timeout=timeout)
        if req.status == 200:
            return True, "LiteLLM proxy is responding"

        return False, "LiteLLM proxy returned error"

    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "LiteLLM proxy returned 401 (invalid API key)"
        return False, f"LiteLLM proxy returned {e.code}"
    except urllib.error.URLError as e:
        return False, f"Cannot connect to LiteLLM proxy: {e.reason}"
    except socket.timeout:
        return False, f"Timeout connecting to LiteLLM proxy"
    except Exception as e:
        return False, f"Error checking LiteLLM proxy: {str(e)}"


async def check_local_service(
    base_url: str,
    timeout: float = 2.0,
    endpoints: Optional[list] = None,
) -> bool:
    """Check if a local service is running and responding.

    Args:
        base_url: The base URL of the service
        timeout: Request timeout in seconds
        endpoints: List of endpoints to try (default: ["/health", "/models"])

    Returns:
        True if service is responding, False otherwise

    Example:
        >>> is_running = await check_local_service("http://localhost:8188")
        >>> if is_running:
        ...     print("ComfyUI is running")
    """
    import httpx

    if endpoints is None:
        endpoints = ["/health", "/models", "/"]

    check_url = base_url.rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for endpoint in endpoints:
                try:
                    response = await client.get(f"{check_url}{endpoint}")
                    if response.status_code in (200, 404):  # 404 means server is up
                        return True
                except Exception:
                    continue
            return False
    except Exception as e:
        logger.debug(f"Local service check failed for {base_url}: {e}")
        return False


def check_local_service_sync(
    base_url: str,
    timeout: float = 2.0,
) -> bool:
    """Synchronous version of check_local_service.

    Args:
        base_url: The base URL of the service
        timeout: Request timeout in seconds

    Returns:
        True if service is responding, False otherwise
    """
    import urllib.request
    import socket

    socket.setdefaulttimeout(timeout)
    check_url = base_url.rstrip("/")

    for endpoint in ["/health", "/models", "/"]:
        try:
            req = urllib.request.urlopen(f"{check_url}{endpoint}", timeout=timeout)
            if req.status in (200, 404):
                return True
        except Exception:
            continue

    return False


def get_available_services() -> dict:
    """Get dict of services and their availability status.

    With LiteLLM handling all provider routing, we only check:
    1. LiteLLM proxy (required)
    2. Optional local services (ComfyUI, etc.)

    Returns:
        Dict mapping service name to (is_available, message)

    Example:
        >>> services = get_available_services()
        >>> for name, (available, msg) in services.items():
        ...     print(f"{name}: {msg}")
    """
    available = {}

    # Check LiteLLM proxy (required)
    is_available, message = check_litellm_proxy_sync()
    available["litellm"] = (is_available, message)

    # Check ComfyUI (optional, for image generation)
    comfyui_url = os.getenv("COMFYUI_ADDRESS", "http://127.0.0.1:8188")
    is_running = check_local_service_sync(comfyui_url)
    available["comfyui"] = (
        is_running,
        f"ComfyUI {'running' if is_running else 'not running'} at {comfyui_url}"
    )

    return available


__all__ = [
    "check_litellm_proxy",
    "check_litellm_proxy_sync",
    "check_local_service",
    "check_local_service_sync",
    "get_available_services",
]
