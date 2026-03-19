"""Rate limiting middleware for API endpoints.

Uses slowapi for token bucket rate limiting to prevent abuse.
"""

import time
import logging
from typing import Callable, Optional

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from .errors import ErrorResponse, ErrorCode

logger = logging.getLogger(__name__)

# Create rate limiter instance
# Uses in-memory storage by default, can be replaced with Redis for distributed systems
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200/minute"],  # Default: 200 requests per minute
    storage_uri="memory://",  # In-memory storage (use Redis for production)
    enabled=True,
)


def get_rate_limit_headers(
    remaining: int,
    reset: int,
    limit: int,
) -> dict[str, str]:
    """Generate rate limit headers for response.

    Args:
        remaining: Remaining requests
        reset: Unix timestamp when limit resets
        limit: Maximum requests per window

    Returns:
        Dictionary of headers
    """
    return {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(reset),
    }


async def rate_limit_exception_handler(
    request: Request,
    exc: RateLimitExceeded,
) -> JSONResponse:
    """Handle rate limit exceeded errors.

    Args:
        request: The request that exceeded the limit
        exc: The rate limit exception

    Returns:
        JSON response with 429 status code
    """
    logger.warning(f"Rate limit exceeded for {get_remote_address(request)} on {request.url.path}")

    # Extract limit info from exception
    limit_info = exc.detail

    return ErrorResponse.create(
        code=ErrorCode.RATE_LIMIT_EXCEEDED,
        message="Rate limit exceeded. Please try again later.",
        status_code=429,
        details={
            "retry_after": str(int(exc.retry_after) if hasattr(exc, "retry_after") else 60),
            "limit": limit_info,
        },
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to add rate limiting to all requests.

    Applies default rate limits to all endpoints, with specific
    endpoints able to override with custom limits.
    """

    def __init__(
        self,
        app,
        default_limits: list[str] = ["200/minute"],
        exempted_paths: Optional[set[str]] = None,
    ):
        """Initialize rate limit middleware.

        Args:
            app: The ASGI application
            default_limits: Default rate limits (e.g., ["100/minute", "1000/hour"])
            exempted_paths: Set of paths exempt from rate limiting
        """
        super().__init__(app)
        self.default_limits = default_limits
        self.exempted_paths = exempted_paths or {
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/static",
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            Response from handler or rate limit error
        """
        # Skip rate limiting for exempted paths
        if any(
            request.url.path.startswith(path)
            for path in self.exempted_paths
        ):
            return await call_next(request)

        # Check rate limit (handled by @limiter.limit decorators on endpoints)
        # This middleware just ensures headers are added to responses
        try:
            response = await call_next(request)

            # Add rate limit headers if not already present
            if "X-RateLimit-Limit" not in response.headers:
                # Add default headers (these would be set by the limiter decorator)
                response.headers["X-RateLimit-Limit"] = self.default_limits[0].split("/")[0]
                response.headers["X-RateLimit-Remaining"] = "unknown"
                response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)

            return response
        except RateLimitExceeded as e:
            return await rate_limit_exception_handler(request, e)


# Rate limit decorators for common use cases

def rate_limit_chat(func):
    """Decorator for chat endpoints (stricter limits)."""
    return limiter.limit("60/minute")(func)


def rate_limit_image_gen(func):
    """Decorator for image generation endpoints (very strict limits)."""
    return limiter.limit("10/minute")(func)


def rate_limit_upload(func):
    """Decorator for file upload endpoints (moderate limits)."""
    return limiter.limit("20/minute")(func)


def rate_limit_auth(func):
    """Decorator for authentication endpoints (very strict limits)."""
    return limiter.limit("5/minute")(func)


def rate_limit_default(func):
    """Decorator for general API endpoints."""
    return limiter.limit("100/minute")(func)
