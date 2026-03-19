"""Standard error response models and exception handlers for the API.

This module provides a consistent error response format across all endpoints,
making error handling more predictable for clients.
"""

from typing import Any
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import json


# =============================================================================
# CORS-enabled JSONResponse for error responses
# =============================================================================

class CORSJSONResponse(JSONResponse):
    """JSONResponse that includes CORS headers.

    This is needed because error responses from exception handlers
    don't go through FastAPI's CORSMiddleware.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add CORS headers after initialization
        self.headers["Access-Control-Allow-Origin"] = "*"
        self.headers["Access-Control-Allow-Credentials"] = "true"
        self.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        self.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Request-ID"


# =============================================================================
# Standard Error Response Format
# =============================================================================

class ErrorResponse:
    """Standard error response format.

    All API errors should return JSON in this format:
    {
        "error": {
            "code": "ERROR_CODE",
            "message": "Human-readable error message",
            "details": {...}  // Optional additional context
        },
        "request_id": "uuid"  // Optional request tracking
    }
    """

    @staticmethod
    def create(
        code: str,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> CORSJSONResponse:
        """Create a standardized error response with CORS headers.

        Args:
            code: Error code (e.g., "VALIDATION_ERROR", "NOT_FOUND")
            message: Human-readable error message
            status_code: HTTP status code
            details: Optional additional error context
            request_id: Optional request tracking ID

        Returns:
            CORSJSONResponse with standardized error format
        """
        error_body: dict[str, Any] = {
            "error": {
                "code": code,
                "message": message,
            }
        }

        if details:
            error_body["error"]["details"] = details

        if request_id:
            error_body["request_id"] = request_id

        return CORSJSONResponse(status_code=status_code, content=error_body)


# =============================================================================
# Custom Exceptions
# =============================================================================

class APIError(Exception):
    """Base exception for API errors.

    Usage:
        raise APIError("NOT_FOUND", "Thread not found", status_code=404)
    """

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


# Common error codes
class ErrorCode:
    """Standard error codes used across the API."""

    # Validation errors (400)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"

    # Not found errors (404)
    NOT_FOUND = "NOT_FOUND"
    THREAD_NOT_FOUND = "THREAD_NOT_FOUND"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"

    # Authentication/Authorization errors (401, 403)
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"

    # Rate limiting (429)
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Server errors (500)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"


# =============================================================================
# Exception Handlers
# =============================================================================

async def api_error_handler(request: Request, exc: APIError) -> CORSJSONResponse:
    """Handle custom APIError exceptions.

    Args:
        request: The request that caused the error
        exc: The APIError exception

    Returns:
        CORSJSONResponse with standardized error format
    """
    return ErrorResponse.create(
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
        request_id=_get_request_id(request),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> CORSJSONResponse:
    """Handle FastAPI HTTPException.

    Args:
        request: The request that caused the error
        exc: The HTTPException

    Returns:
        CORSJSONResponse with standardized error format
    """
    # Map HTTP status codes to error codes
    error_code_map = {
        400: ErrorCode.INVALID_INPUT,
        401: ErrorCode.UNAUTHORIZED,
        403: ErrorCode.FORBIDDEN,
        404: ErrorCode.NOT_FOUND,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_ERROR,
        503: ErrorCode.SERVICE_UNAVAILABLE,
    }

    error_code = error_code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR)

    return ErrorResponse.create(
        code=error_code,
        message=str(exc.detail),
        status_code=exc.status_code,
        request_id=_get_request_id(request),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> CORSJSONResponse:
    """Handle Pydantic validation errors.

    Args:
        request: The request that caused the error
        exc: The validation error

    Returns:
        CORSJSONResponse with standardized error format including field details
    """
    import logging
    logger = logging.getLogger(__name__)

    # Format validation errors
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        error_detail = {
            "field": field_path,
            "message": error["msg"],
            "type": error["type"],
        }
        errors.append(error_detail)

    # Log the validation errors for debugging
    logger.error(f"[VALIDATION ERROR] Request validation failed for {request.url.path}:")
    for error in errors:
        logger.error(f"  Field: {error['field']}, Message: {error['message']}, Type: {error['type']}")

    return ErrorResponse.create(
        code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        status_code=422,
        details={"errors": errors},
        request_id=_get_request_id(request),
    )


async def general_exception_handler(request: Request, exc: Exception) -> CORSJSONResponse:
    """Handle all unhandled exceptions.

    Args:
        request: The request that caused the error
        exc: The unhandled exception

    Returns:
        CORSJSONResponse with standardized error format
    """
    # Log the full exception for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=exc)

    return ErrorResponse.create(
        code=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred",
        status_code=500,
        request_id=_get_request_id(request),
    )


def _get_request_id(request: Request) -> str | None:
    """Extract or generate a request ID for tracking.

    Args:
        request: The request object

    Returns:
        Request ID string or None
    """
    # Try to get from header
    request_id = request.headers.get("X-Request-ID")
    if request_id:
        return request_id

    # Try to get from state (might be set by middleware)
    if hasattr(request.state, "request_id"):
        return request.state.request_id

    return None


# =============================================================================
# Convenience Functions for Common Error Patterns
# =============================================================================

def raise_not_found(resource: str, identifier: str | None = None) -> None:
    """Raise a 404 NOT_FOUND HTTPException.

    Args:
        resource: Type of resource (e.g., "Thread", "User", "Message")
        identifier: Optional identifier for the resource

    Raises:
        HTTPException: 404 error with standardized message
    """
    if identifier:
        message = f"{resource} {identifier} not found"
    else:
        message = f"{resource} not found"
    raise HTTPException(status_code=404, detail=message)


def raise_bad_request(message: str) -> None:
    """Raise a 400 BAD_REQUEST HTTPException.

    Args:
        message: Error message describing what was wrong

    Raises:
        HTTPException: 400 error with the message
    """
    raise HTTPException(status_code=400, detail=message)


def raise_internal_error(operation: str, original_error: Exception) -> None:
    """Raise a 500 INTERNAL_ERROR HTTPException.

    Args:
        operation: Description of the operation that failed
        original_error: The original exception

    Raises:
        HTTPException: 500 error with standardized message
    """
    message = f"Failed to {operation}: {str(original_error)}"
    raise HTTPException(status_code=500, detail=message)


def raise_validation_error(field: str, message: str) -> None:
    """Raise a 400 VALIDATION_ERROR HTTPException.

    Args:
        field: Field name that failed validation
        message: Validation error message

    Raises:
        HTTPException: 400 error with validation details
    """
    detail = f"{field} is required: {message}" if message else f"{field} is required"
    raise HTTPException(status_code=400, detail=detail)


def raise_unauthorized(message: str = "Authentication required") -> None:
    """Raise a 401 UNAUTHORIZED HTTPException.

    Args:
        message: Optional custom message

    Raises:
        HTTPException: 401 error
    """
    raise HTTPException(status_code=401, detail=message)


def raise_forbidden(message: str = "Access denied") -> None:
    """Raise a 403 FORBIDDEN HTTPException.

    Args:
        message: Optional custom message

    Raises:
        HTTPException: 403 error
    """
    raise HTTPException(status_code=403, detail=message)


# =============================================================================
# Middleware for Request ID Tracking
# =============================================================================

class RequestIDMiddleware:
    """Middleware to add request ID tracking for all requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Import uuid here to avoid issues with ASGI
        import uuid

        # Generate or extract request ID
        from starlette.datastructures import Headers

        # Get headers from scope
        headers = Headers(scope=scope)
        request_id = headers.get("X-Request-ID") or str(uuid.uuid4())

        # Add to state for access in endpoints
        from starlette.types import ASGIApp, Receive, Send

        class AppState(dict):
            pass

        scope_state = scope.get("state", {})
        scope_state["request_id"] = request_id
        scope["state"] = scope_state

        # Add request ID to response header
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-request-id"] = request_id.encode()
                message["headers"] = [(k, v) for k, v in headers.items()]
            await send(message)

        await self.app(scope, receive, send_wrapper)
