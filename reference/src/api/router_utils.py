"""Shared Router Configuration Utilities.

This module provides common utilities for configuring API routers,
reducing duplication across multiple router files.

Common patterns consolidated:
- Router creation with prefix and tags
- Error handling wrapper decorators
- Response formatting helpers
- Startup event handlers
- Logging helpers
"""

import logging
from functools import wraps
from typing import Callable, Optional, Any, TypeVar

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Router Creation Helpers
# =============================================================================

def create_router(
    prefix: str,
    tag: str,
    *,
    include_in_schema: bool = True,
) -> APIRouter:
    """Create an APIRouter with standard configuration.

    Args:
        prefix: URL prefix for the router (e.g., "/api/v1/feature")
        tag: OpenAPI tag for documentation
        include_in_schema: Whether to include in OpenAPI schema

    Returns:
        Configured APIRouter instance
    """
    return APIRouter(
        prefix=prefix,
        tags=[tag],
        include_in_schema=include_in_schema,
    )


def create_routers(
    prefixes: list[tuple[str, str]],
    **kwargs: Any,
) -> list[APIRouter]:
    """Create multiple APIRouters at once.

    Args:
        prefixes: List of (prefix, tag) tuples
        **kwargs: Additional arguments passed to create_router

    Returns:
        List of configured APIRouter instances
    """
    return [create_router(prefix, tag, **kwargs) for prefix, tag in prefixes]


# =============================================================================
# Error Handling Decorators
# =============================================================================

def handle_errors(
    *,
    reraise: bool = False,
    default_status: int = 500,
    log_errors: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to standardize error handling in endpoints.

    Args:
        reraise: Whether to reraise exceptions after logging
        default_status: Default HTTP status code for errors
        log_errors: Whether to log errors

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Let FastAPI HTTP exceptions propagate
                raise
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                if reraise:
                    raise
                raise HTTPException(status_code=default_status, detail=str(e))

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except HTTPException:
                raise
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                if reraise:
                    raise
                raise HTTPException(status_code=default_status, detail=str(e))

        # Return appropriate wrapper based on whether function is async
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def validate_request(
    *fields: tuple[str, Any],
    log_errors: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to validate request fields before processing.

    Args:
        *fields: Tuples of (field_name, validator_func)
        log_errors: Whether to log validation errors

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            for field_name, validator in fields:
                if field_name in kwargs:
                    value = kwargs[field_name]
                    try:
                        result = validator(value)
                        kwargs[field_name] = result
                    except (ValueError, TypeError) as e:
                        if log_errors:
                            logger.warning(f"Validation failed for {field_name}: {e}")
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid {field_name}: {str(e)}"
                        )
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            for field_name, validator in fields:
                if field_name in kwargs:
                    value = kwargs[field_name]
                    try:
                        result = validator(value)
                        kwargs[field_name] = result
                    except (ValueError, TypeError) as e:
                        if log_errors:
                            logger.warning(f"Validation failed for {field_name}: {e}")
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid {field_name}: {str(e)}"
                        )
            return func(*args, **kwargs)

        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Response Builders
# =============================================================================

def success_response(
    data: Any = None,
    message: Optional[str] = None,
    meta: Optional[dict] = None,
) -> dict:
    """Build a standard success response.

    Args:
        data: Response data
        message: Optional success message
        meta: Optional metadata

    Returns:
        Standardized success response dict
    """
    response = {"success": True}
    if data is not None:
        response["data"] = data
    if message:
        response["message"] = message
    if meta:
        response["meta"] = meta
    return response


def error_response(
    detail: str,
    *,
    status: int = 500,
    code: Optional[str] = None,
) -> JSONResponse:
    """Build a standard error response.

    Args:
        detail: Error message
        status: HTTP status code
        code: Optional error code

    Returns:
        JSONResponse with error details
    """
    content = {"success": False, "detail": detail}
    if code:
        content["code"] = code
    return JSONResponse(status_code=status, content=content)


# =============================================================================
# Startup Event Handler
# =============================================================================

def on_startup(
    message: str,
    *,
    log_level: int = logging.INFO,
) -> Callable[[], None]:
    """Create a startup event handler for a router.

    Args:
        message: Message to log on startup
        log_level: Logging level for the message

    Returns:
        Startup event handler function

    Example:
        @router.on_event("startup")
        async def startup():
            router_startup("My router initialized")
    """
    def handler() -> None:
        logger.log(log_level, message)

    return handler


# =============================================================================
# Logging Helpers
# =============================================================================

def log_request(
    router_name: str,
    endpoint: str,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to log incoming requests.

    Args:
        router_name: Name of the router/module
        endpoint: Endpoint path

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            # Extract request from args if available
            request: Optional[Request] = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request:
                logger.info(f"[{router_name}] {endpoint} - {request.method}")
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            request: Optional[Request] = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request:
                logger.info(f"[{router_name}] {endpoint} - {request.method}")
            return func(*args, **kwargs)

        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Extension Router Helpers
# =============================================================================

def get_extension_manager(request: Request):
    """Safely get the extension manager from app state.

    Args:
        request: FastAPI Request object

    Returns:
        ExtensionManager instance

    Raises:
        HTTPException: If extension manager is not initialized
    """
    manager = getattr(request.app.state, "extension_manager", None)
    if not manager:
        raise HTTPException(
            status_code=500,
            detail="Extension manager not initialized"
        )
    return manager


def get_extension(request: Request, extension_id: str):
    """Get a specific extension by ID from app state.

    Args:
        request: FastAPI Request object
        extension_id: Extension identifier

    Returns:
        Extension instance

    Raises:
        HTTPException: If extension not found
    """
    manager = get_extension_manager(request)
    extension = manager.get_extension(extension_id)
    if not extension:
        raise HTTPException(
            status_code=404,
            detail=f"Extension '{extension_id}' not found"
        )
    return extension
