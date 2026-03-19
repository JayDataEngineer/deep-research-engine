"""API Error Handling Utilities.

Reduces boilerplate in route handlers by providing:
- Decorator for automatic error handling
- Standardized error responses
- Consistent logging

Usage:
    from src.api.core.error_utils import api_handler, APIError

    # Decorator approach
    @router.get("/items/{item_id}")
    @api_handler("items")
    async def get_item(item_id: str):
        item = await fetch_item(item_id)  # Raises FileNotFoundError if not found
        return item

    # Manual error handling
    @router.post("/items")
    async def create_item(item: ItemCreate):
        with api_error_context("items"):
            result = await create_item_internal(item)
            return result

    # Custom errors
    raise APIError.not_found("Item", item_id)
    raise APIError.bad_request("Invalid input", details={"field": "name"})
"""

import functools
import logging
from typing import Callable, Optional, Any, Dict, TypeVar, ParamSpec
from fastapi import HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


class APIError(Exception):
    """Structured API error that converts to HTTPException.

    Provides consistent error responses with:
    - Error type/code
    - User-friendly message
    - Optional details for debugging

    Example:
        raise APIError.not_found("User", user_id)
        raise APIError.bad_request("Invalid email", details={"email": email})
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.error_code = error_code or f"ERROR_{status_code}"
        self.details = details
        super().__init__(message)

    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException."""
        detail = {
            "message": self.message,
            "error_code": self.error_code,
        }
        if self.details:
            detail["details"] = self.details
        return HTTPException(status_code=self.status_code, detail=detail)

    @classmethod
    def not_found(cls, resource_type: str, resource_id: Optional[str] = None) -> "APIError":
        """Create 404 Not Found error."""
        message = f"{resource_type} not found"
        if resource_id:
            message = f"{resource_type} '{resource_id}' not found"
        return cls(status_code=404, message=message, error_code="NOT_FOUND")

    @classmethod
    def bad_request(cls, message: str, details: Optional[Dict[str, Any]] = None) -> "APIError":
        """Create 400 Bad Request error."""
        return cls(status_code=400, message=message, error_code="BAD_REQUEST", details=details)

    @classmethod
    def unauthorized(cls, message: str = "Unauthorized") -> "APIError":
        """Create 401 Unauthorized error."""
        return cls(status_code=401, message=message, error_code="UNAUTHORIZED")

    @classmethod
    def forbidden(cls, message: str = "Access denied") -> "APIError":
        """Create 403 Forbidden error."""
        return cls(status_code=403, message=message, error_code="FORBIDDEN")

    @classmethod
    def conflict(cls, message: str, details: Optional[Dict[str, Any]] = None) -> "APIError":
        """Create 409 Conflict error."""
        return cls(status_code=409, message=message, error_code="CONFLICT", details=details)

    @classmethod
    def internal(cls, message: str = "Internal server error", error: Optional[Exception] = None) -> "APIError":
        """Create 500 Internal Server Error."""
        details = {"original_error": str(error)} if error else None
        return cls(status_code=500, message=message, error_code="INTERNAL_ERROR", details=details)


class APIErrorContext:
    """Context manager for API error handling.

    Automatically converts common exceptions to HTTPException.

    Example:
        with api_error_context("users"):
            user = await fetch_user(user_id)  # FileNotFoundError -> 404
            return user
    """

    def __init__(
        self,
        tag: str = "api",
        *,
        not_found_message: Optional[str] = None,
        include_traceback: bool = False,
    ):
        self.tag = tag
        self.not_found_message = not_found_message
        self.include_traceback = include_traceback

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False

        # Handle APIError
        if isinstance(exc_val, APIError):
            logger.error(f"[{self.tag}] {exc_val.error_code}: {exc_val.message}")
            raise exc_val.to_http_exception()

        # Handle FileNotFoundError -> 404
        if issubclass(exc_type, FileNotFoundError):
            message = self.not_found_message or str(exc_val)
            logger.warning(f"[{self.tag}] Not found: {message}")
            raise HTTPException(status_code=404, detail=message)

        # Handle ValueError -> 400
        if issubclass(exc_type, ValueError):
            logger.warning(f"[{self.tag}] Bad request: {exc_val}")
            raise HTTPException(status_code=400, detail=str(exc_val))

        # Handle PermissionError -> 403
        if issubclass(exc_type, PermissionError):
            logger.warning(f"[{self.tag}] Permission denied: {exc_val}")
            raise HTTPException(status_code=403, detail=str(exc_val))

        # Handle other exceptions -> 500
        logger.error(f"[{self.tag}] Internal error: {exc_type.__name__}: {exc_val}")
        if self.include_traceback:
            import traceback
            logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=f"Internal error in {self.tag}: {str(exc_val)}"
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


def api_error_context(
    tag: str = "api",
    *,
    not_found_message: Optional[str] = None,
    include_traceback: bool = False,
) -> APIErrorContext:
    """Create an API error context manager.

    Args:
        tag: Tag for logging (e.g., "users", "items")
        not_found_message: Custom message for FileNotFoundError
        include_traceback: Include traceback in logs

    Returns:
        APIErrorContext instance

    Example:
        with api_error_context("users"):
            user = await fetch_user(user_id)
            return user
    """
    return APIErrorContext(
        tag=tag,
        not_found_message=not_found_message,
        include_traceback=include_traceback,
    )


def api_handler(
    tag: str = "api",
    *,
    not_found_message: Optional[str] = None,
    log_success: bool = False,
):
    """Decorator for automatic API error handling.

    Wraps route handlers with consistent error handling:
    - FileNotFoundError -> 404
    - ValueError -> 400
    - PermissionError -> 403
    - Other exceptions -> 500

    Args:
        tag: Tag for logging (e.g., "users", "items")
        not_found_message: Custom message for FileNotFoundError
        log_success: Log successful requests

    Returns:
        Decorated function

    Example:
        @router.get("/users/{user_id}")
        @api_handler("users")
        async def get_user(user_id: str):
            user = await fetch_user(user_id)
            return user
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                result = await func(*args, **kwargs)
                if log_success:
                    logger.debug(f"[{tag}] Success: {func.__name__}")
                return result

            except APIError as e:
                logger.error(f"[{tag}] {e.error_code}: {e.message}")
                raise e.to_http_exception()

            except FileNotFoundError as e:
                message = not_found_message or str(e)
                logger.warning(f"[{tag}] Not found: {message}")
                raise HTTPException(status_code=404, detail=message)

            except ValueError as e:
                logger.warning(f"[{tag}] Bad request: {e}")
                raise HTTPException(status_code=400, detail=str(e))

            except PermissionError as e:
                logger.warning(f"[{tag}] Permission denied: {e}")
                raise HTTPException(status_code=403, detail=str(e))

            except HTTPException:
                raise  # Re-raise HTTP exceptions as-is

            except Exception as e:
                logger.error(f"[{tag}] Internal error in {func.__name__}: {type(e).__name__}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal error: {str(e)}"
                )

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                result = func(*args, **kwargs)
                if log_success:
                    logger.debug(f"[{tag}] Success: {func.__name__}")
                return result

            except APIError as e:
                logger.error(f"[{tag}] {e.error_code}: {e.message}")
                raise e.to_http_exception()

            except FileNotFoundError as e:
                message = not_found_message or str(e)
                logger.warning(f"[{tag}] Not found: {message}")
                raise HTTPException(status_code=404, detail=message)

            except ValueError as e:
                logger.warning(f"[{tag}] Bad request: {e}")
                raise HTTPException(status_code=400, detail=str(e))

            except PermissionError as e:
                logger.warning(f"[{tag}] Permission denied: {e}")
                raise HTTPException(status_code=403, detail=str(e))

            except HTTPException:
                raise

            except Exception as e:
                logger.error(f"[{tag}] Internal error in {func.__name__}: {type(e).__name__}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal error: {str(e)}"
                )

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Convenience aliases
not_found = APIError.not_found
bad_request = APIError.bad_request
unauthorized = APIError.unauthorized
forbidden = APIError.forbidden
conflict = APIError.conflict
internal_error = APIError.internal


__all__ = [
    "APIError",
    "APIErrorContext",
    "api_error_context",
    "api_handler",
    "not_found",
    "bad_request",
    "unauthorized",
    "forbidden",
    "conflict",
    "internal_error",
]
