"""Voice Chat API Routes - Simplified for Shared Infrastructure.

Voice chat using shared infrastructure:
- STT: Via LiteLLM (distil-whisper, qwen3-asr)
- TTS: Via LiteLLM (kokoro, indextts2)
- No local TTS extensions needed

Usage:
    POST /api/v1/voice/chat
    - audio_file: Audio upload (optional)
    - transcript: Pre-transcribed text (optional)
    - Returns: SSE stream with text and audio URLs
"""

import json
import logging
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from src.shared.sse_events import SSEvent
from src.shared.litellm_tts import generate_speech

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/voice", tags=["voice-chat"])


@router.post("/chat")
async def voice_chat_endpoint(
    request: Request,
    audio_file: UploadFile | None = File(None),
    transcript: str | None = Form(None),
    chatMode: str = Form("direct"),
    session_id: str | None = Form(None),
    enable_tts: bool = Form(True),
    tts_voice: str = Form("Cherry"),
    tts_model: str = Form("kokoro"),
):
    """
    Simplified voice chat using shared infrastructure.
    
    Flow:
    1. Audio → STT via LiteLLM (if audio_file provided)
    2. Text → LLM via shared infrastructure
    3. Response → TTS via LiteLLM (if enable_tts)
    
    Args:
        audio_file: Audio file for STT (optional)
        transcript: Pre-transcribed text (skips STT)
        chatMode: Chat mode (direct, tool_call, etc.)
        session_id: Session ID for conversation history
        enable_tts: Generate TTS audio
        tts_voice: Voice ID (Cherry, Serena, etc.)
        tts_model: TTS model (kokoro, indextts2)
    """
    import uuid
    from src.shared.litellm_factory import get_llm
    
    thread_id = session_id or f"voice_{uuid.uuid4().hex[:16]}"
    
    async def voice_chat_stream():
        """Stream voice chat with text and audio."""
        try:
            # Step 1: Get user input (STT or transcript)
            user_input = transcript
            
            if audio_file and not user_input:
                # Use LiteLLM for STT
                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    base_url="http://litellm.local/v1",
                    api_key="sk-litellm-master"
                )
                
                audio_content = await audio_file.read()
                from io import BytesIO
                audio_file_obj = BytesIO(audio_content)
                audio_file_obj.name = "audio.wav"
                
                transcription = await client.audio.transcriptions.create(
                    model="distil-whisper",
                    file=audio_file_obj
                )
                user_input = transcription.text
                logger.info(f"[VOICE] STT result: {user_input[:100]}...")
            
            if not user_input:
                yield SSEvent.error("No input provided", recoverable=False)
                return
            
            # Step 2: Get LLM response via shared infrastructure
            yield SSEvent.thinking("Processing...", label="Thinking")
            
            llm = await get_llm(model="qwen-35-27-fast", enable_thinking=False)
            response = await llm.ainvoke([
                {"role": "user", "content": user_input}
            ])
            
            full_response = response.content if hasattr(response, 'content') else str(response)
            
            # Stream text
            yield SSEvent.text(full_response)
            
            # Step 3: Generate TTS via LiteLLM (if enabled)
            if enable_tts and full_response.strip():
                yield SSEvent.custom("audio-start")
                
                try:
                    # Generate audio via LiteLLM
                    audio_bytes = await generate_speech(
                        text=full_response,
                        model=tts_model,
                        voice=tts_voice
                    )
                    
                    # Save audio and get URL
                    from src.shared.tools.audio_utils import AudioService
                    import io
                    
                    audio_buffer = io.BytesIO(audio_bytes)
                    audio_url, duration = await AudioService.save_and_get_url(
                        audio_buffer,
                        format="wav",
                        return_duration=True
                    )
                    
                    yield SSEvent.custom("audio-complete", audioUrl=audio_url)
                    logger.info(f"[VOICE] TTS generated: {duration:.2f}s")
                    
                except Exception as tts_error:
                    logger.warning(f"[VOICE] TTS failed: {tts_error}")
                    yield SSEvent.custom("audio-error", error=str(tts_error))
            
            yield SSEvent.end()
            
        except Exception as e:
            logger.error(f"[VOICE] Error: {e}", exc_info=True)
            yield SSEvent.error(str(e))
    
    return StreamingResponse(
        voice_chat_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/health")
async def health_check():
    """Health check for voice chat service."""
    return {
        "status": "healthy",
        "service": "voice-chat",
        "infrastructure": "shared (LiteLLM)"
    }
