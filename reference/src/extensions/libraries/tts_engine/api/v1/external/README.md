# External API Compatibility Layer

This directory contains compatibility endpoints for third-party integrations with the Cozy TTS backend.

## Purpose

The External API provides translation layers between different API contracts used by external applications and the internal Cozy TTS API. This allows external middleware, applications, and services to integrate with Cozy TTS without needing to adapt to our internal API format.

## Available Integrations

### IndexTTS2 Middleware

**Module**: `indextts.py`

**Documentation**: [EXTERNAL_API_INDEXTTS.md](../../../../docs/EXTERNAL_API_INDEXTTS.md)

**Endpoints**:
- `POST /api/v1/external/indextts/synthesize` - Generate speech with emotion control
- `GET /api/v1/external/indextts/speakers` - List available speaker mappings

**Features**:
- 8-dimensional emotion vector support
- Speaker to reference audio mapping
- Base64 audio encoding
- Format translation between middleware and internal API

## Adding a New Integration

To add compatibility for a new external API or middleware:

1. **Create a new module** in this directory
   ```bash
   touch backend/app/api/v1/external/your_integration.py
   ```

2. **Implement the compatibility endpoint**
   ```python
   from fastapi import APIRouter

   router = APIRouter(tags=["External API - Your Integration"])

   @router.post("/yourintegration/synthesize")
   async def your_synthesize(request: YourRequest):
       # Translate request format to internal API
       # Call tts_service.generate_audio()
       # Translate response format back
       pass
   ```

3. **Register the router** in `backend/app/main.py`
   ```python
   from app.api.v1 import external

   app.include_router(
       external.your_integration.router,
       prefix=f"{settings.api_v1_prefix}/external",
       tags=["External API"]
   )
   ```

4. **Add documentation** in `backend/docs/EXTERNAL_API_YOUR_INTEGRATION.md`
   - API endpoint specifications
   - Request/response formats
   - Configuration requirements
   - Usage examples
   - Error handling

## Design Principles

### Namespaced Endpoints

Each integration gets its own namespace under `/api/v1/external/`:

```
/api/v1/external/indextts/...
/api/v1/external/yourintegration/...
```

This avoids:
- **God endpoint anti-pattern**: Single endpoint that changes behavior based on parameters
- **Unclear contracts**: Mixed API formats in one endpoint
- **Maintenance burden**: Complex conditional logic

### Translation Layer Pattern

Each integration module:
1. **Validates** incoming requests against the external API contract
2. **Translates** the request to internal Cozy TTS format
3. **Calls** `tts_service.generate_audio()` with translated parameters
4. **Translates** the response back to external API format
5. **Handles** errors with meaningful messages for the external client

### Separation of Concerns

- **External API** (`this directory`): Translation, validation, external contract
- **TTS Service** (`app/services/tts_service.py`): Internal orchestration
- **Extensions** (`backend/extensions/`): Model-specific implementations

## Configuration

Most integrations require configuration. Common patterns:

### Mapping Dictionaries

```python
# Map external IDs to internal resources
SPEAKER_AUDIO_MAP = {
    "external_name": "/path/to/audio.wav",
}
```

### Default Values

```python
# Fallback when no specific mapping exists
DEFAULT_REF_AUDIO_PATH = "/path/to/default.wav"
```

### Environment Variables

```python
import os

API_KEY = os.getenv("EXTERNAL_API_KEY")
MAX_DURATION = int(os.getenv("MAX_DURATION_SECONDS", "60"))
```

## Testing

Each integration should include tests for:

1. **Happy path**: Valid request → successful response
2. **Validation**: Invalid input → proper error message
3. **Translation**: Correct format conversion
4. **Edge cases**: Empty text, extreme values, missing parameters

Example test structure:

```python
def test_indextts_synthesize_happy_path(client):
    response = client.post("/api/v1/external/indextts/synthesize", json={
        "text": "Hello world!",
        "emotion": [0, 0.3, 0.5, 0.1, 0.1, 0, 0, 0.1],
    })
    assert response.status_code == 200
    assert "audio" in response.json()
    assert response.json()["format"] == "wav"
```

## Security Considerations

### Input Validation

Always validate external inputs:
- Check data types and ranges
- Sanitize file paths
- Limit text length to prevent DoS

### File Path Safety

```python
from pathlib import Path

# Prevent directory traversal
speaker_path = Path(request.speaker).resolve()
if not str(speaker_path).startswith("/allowed/base/path"):
    raise HTTPException(403, "Invalid file path")
```

### Rate Limiting

Consider adding rate limits for external endpoints to prevent abuse:

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@router.post("/synthesize")
@limiter.limit("10/minute")
async def synthesize(...):
    ...
```

## Troubleshooting

### CORS Issues

If external applications can't access the API:

1. Check CORS configuration in `backend/app/main.py`
2. Add the external application's origin to `allow_origins`
3. Verify `allow_credentials` and `allow_headers` settings

### Import Errors

If you get import errors for new integrations:

1. Ensure the module is in this directory
2. Check `__init__.py` exists (can be empty)
3. Verify router is registered in `main.py`
4. Restart the backend server

### Extension Not Found

If the integration can't find the required TTS extension:

1. Verify extension is installed in `backend/extensions/`
2. Check extension is loaded: `GET /api/v1/tts/extensions`
3. Ensure `extension_name` parameter matches extension directory name

## Resources

- [IndexTTS2 Compatibility Documentation](../../../../docs/EXTERNAL_API_INDEXTTS.md)
- [Internal TTS API Documentation](../../README.md)
- [Extension System Documentation](../../../../extensions/README.md)
- [Main Project README](../../../../README.md)
