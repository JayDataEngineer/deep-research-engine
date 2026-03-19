"""Memory Monitor API Routes.

Provides the /api/v1/system/memory and /api/v1/monitoring endpoints for the extension.
"""

import asyncio
import httpx
import os
import psutil
import subprocess
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .extension import MemoryMonitorExtension

router = APIRouter(prefix="/api/v1/system/memory", tags=["memory"])

# Also include the monitoring routes
monitoring_router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# Global extension instance - will be set by the extension loader
_extension: MemoryMonitorExtension = None


def set_extension(ext: MemoryMonitorExtension):
    """Set the global extension instance."""
    global _extension
    _extension = ext


@router.get("")
async def get_memory():
    """Get current RAM and GPU memory usage."""
    if _extension is None:
        return {
            "ram": {"total": 0, "used": 0, "free": 0, "percent": 0},
            "gpu": {"count": 0, "memory": {"total": 0, "used": 0, "free": 0, "percent": 0}, "utilization": 0, "info": {}},
            "timestamp": time.time()
        }
    return await _extension.get_memory()


# ============================================================================
# Monitoring endpoints (from monitoring.py)
# ============================================================================

class UnloadRequest(BaseModel):
    model_type: str = Field(..., description="Model type (tts, comfyui, ollama, all)")
    model_id: str | None = Field(None, description="Optional specific model ID")


class UnloadResponse(BaseModel):
    success: bool
    message: str
    unloaded_count: int = 0


async def get_loaded_models_async():
    """Get information about loaded AI models - async version with parallel API calls."""
    models = {
        "ollama": [],
        "comfyui": [],
        "tts": [],
        "llamacpp": [],
        "gpu_processes": []
    }

    # First, collect GPU processes to correlate actual memory usage
    llamacpp_gpu_memory = {}  # Maps PID to memory usage
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        pid = int(parts[0])
                        name = parts[1]
                        memory = int(parts[2]) * 1024 * 1024

                        process_name = name
                        process_type = "other"
                        try:
                            proc = psutil.Process(pid)
                            process_name = proc.name()
                            cmdline = proc.cmdline()
                            if cmdline:
                                cmdline_str = ' '.join(cmdline)

                                if 'ollama' in cmdline_str.lower():
                                    process_name = "Ollama"
                                    process_type = "ollama"
                                elif 'ComfyUI' in cmdline_str or 'comfyui' in cmdline_str:
                                    process_name = "ComfyUI"
                                    process_type = "comfyui"
                                elif any(tts in cmdline_str for tts in ['gpt_sovits', 'gpt-sovits', 'GPT-SoVITS']):
                                    process_name = "GPT-SoVITS"
                                    process_type = "tts"
                                elif 'index_tts' in cmdline_str or 'indexts' in cmdline_str:
                                    process_name = "IndexTTS"
                                    process_type = "tts"
                                elif 'kokoro' in cmdline_str.lower():
                                    process_name = "Kokoro"
                                    process_type = "tts"
                                elif 'vibe_voice' in cmdline_str or 'vibevoice' in cmdline_str:
                                    process_name = "VibeVoice"
                                    process_type = "tts"
                                elif 'maya' in cmdline_str.lower():
                                    process_name = "MayaTTS"
                                    process_type = "tts"
                                elif 'llama' in cmdline_str.lower():
                                    process_name = "llama.cpp"
                                    process_type = "llamacpp"
                                    # Store llama.cpp GPU memory for correlation
                                    llamacpp_gpu_memory[pid] = memory

                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                        models["gpu_processes"].append({
                            "pid": pid,
                            "name": process_name,
                            "memory": memory,
                            "original_name": name,
                            "type": process_type
                        })
    except Exception:
        pass

    async def fetch_ollama():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:11434/api/ps", timeout=1.0)
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("models", []):
                        models["ollama"].append({
                            "name": model.get("name", "unknown"),
                            "size": model.get("size", 0),
                            "vram": model.get("vram", {}).get("total", 0),
                            "pid": model.get("pid", 0)
                        })
        except Exception:
            pass

    async def fetch_comfyui():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:18188/system_stats", timeout=1.0)
                if response.status_code != 200:
                    return
        except Exception:
            return

        # Use nvidia-smi to get ComfyUI's actual process memory
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 2:
                            pid = int(parts[0])
                            memory_mb = int(parts[1])

                            # Check if this process is ComfyUI
                            try:
                                proc = psutil.Process(pid)
                                cmdline = proc.cmdline()
                                if cmdline:
                                    cmdline_str = ' '.join(cmdline)
                                    if 'ComfyUI' in cmdline_str or 'comfyui' in cmdline_str:
                                        models["comfyui"].append({
                                            "name": "ComfyUI (active)",
                                            "vram_used": memory_mb * 1024 * 1024,
                                            "device": "cuda:0"
                                        })
                                        break
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
        except Exception:
            pass

    async def fetch_tts():
        try:
            # Use MAIN_API_URL environment variable for production compatibility
            main_api_url = os.getenv("MAIN_API_URL", "http://localhost:8787")
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{main_api_url}/api/v1/tts/extensions", timeout=1.0)
                if response.status_code == 200:
                    data = response.json()
                    loaded_extensions = data.get("loaded", [])
                    available_extensions = data.get("extensions", [])

                    for ext_name in loaded_extensions:
                        ext_info = next((e for e in available_extensions if e.get("id") == ext_name), {})
                        models["tts"].append({
                            "name": ext_info.get("display_name", ext_name),
                            "id": ext_name,
                            "type": "tts"
                        })
        except Exception:
            pass

    async def fetch_llamacpp():
        """Fetch llama.cpp loaded models from Router Mode API."""
        try:
            llamacpp_port = os.getenv("LLAMACPP_PORT", "18734")
            llamacpp_host = os.getenv("LLAMACPP_HOST", "127.0.0.1")
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{llamacpp_host}:{llamacpp_port}/v1/models", timeout=1.0)
                if response.status_code == 200:
                    data = response.json()

                    # Get actual GPU memory used by llama.cpp processes from nvidia-smi
                    # If there are multiple llama.cpp processes, sum their memory
                    total_llamacpp_vram = sum(llamacpp_gpu_memory.values())

                    for model in data.get("data", []):
                        model_id = model.get("id", "unknown")
                        status = model.get("status", {})
                        status_value = status.get("value", "")

                        if status_value != "unloaded":
                            args = status.get("args", [])
                            real_model_name = model_id

                            for i, arg in enumerate(args):
                                if arg == "--model" and i + 1 < len(args):
                                    model_path = args[i + 1]
                                    real_model_name = os.path.splitext(os.path.basename(model_path))[0]
                                    break

                            # Use actual GPU memory from nvidia-smi, divided by number of loaded models
                            # If we have GPU memory data for llama.cpp, use it proportionally
                            loaded_model_count = len([m for m in data.get("data", []) if m.get("status", {}).get("value") != "unloaded"])
                            if total_llamacpp_vram > 0 and loaded_model_count > 0:
                                vram = total_llamacpp_vram // loaded_model_count
                            else:
                                vram = 0

                            models["llamacpp"].append({
                                "name": real_model_name,
                                "id": model_id,
                                "type": "llamacpp",
                                "vram": vram,
                                "status": status_value
                            })
        except Exception:
            pass

    # Run all API calls in parallel
    await asyncio.gather(fetch_ollama(), fetch_comfyui(), fetch_tts(), fetch_llamacpp())

    return models


@monitoring_router.get("/models")
async def get_models():
    """Get information about loaded AI models."""
    return {
        "models": await get_loaded_models_async(),
        "timestamp": time.time()
    }


@monitoring_router.post("/models/unload", response_model=UnloadResponse)
async def unload_models(request: UnloadRequest):
    """Unload AI models to free GPU memory."""
    model_type = request.model_type.lower()
    unloaded_count = 0
    messages = []

    try:
        # Unload TTS extensions
        if model_type in ["tts", "all"]:
            try:
                from src.extensions.libraries.tts_engine.extension_process_manager import extension_process_manager
                await extension_process_manager.cleanup_leftover_processes()
                result = await extension_process_manager.stop_current()
                if result.get("status") in ["stopped", "no_extension"]:
                    unloaded_count += 1
                    messages.append("TTS extension unloaded")
                else:
                    messages.append("TTS extension unload failed")
            except Exception as e:
                messages.append(f"TTS extension error: {str(e)}")

        # Unload ComfyUI models
        if model_type in ["comfyui", "all"]:
            try:
                response = httpx.post(
                    "http://localhost:18188/free",
                    json={"unload_models": True, "free_memory": True},
                    timeout=5.0
                )
                if response.status_code == 200:
                    unloaded_count += 1
                    messages.append("ComfyUI models freed")
                else:
                    messages.append(f"ComfyUI unload failed: {response.status_code}")
            except Exception as e:
                messages.append(f"ComfyUI error: {str(e)}")

        # Unload Ollama models
        if model_type in ["ollama", "all"]:
            try:
                response = httpx.get("http://localhost:11434/api/ps", timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])

                    for model in models:
                        model_name = model.get("name", "")
                        if request.model_id is None or request.model_id in model_name:
                            httpx.post(
                                "http://localhost:11434/api/generate",
                                json={"model": model_name, "keep_alive": 0},
                                timeout=5.0
                            )
                            unloaded_count += 1
                            messages.append(f"Ollama model {model_name} unloaded")
            except Exception as e:
                messages.append(f"Ollama error: {str(e)}")

        # Unload llama.cpp models
        if model_type in ["llamacpp", "all"]:
            try:
                llamacpp_port = os.getenv("LLAMACPP_PORT", "18734")
                response = httpx.get(f"http://localhost:{llamacpp_port}/v1/models", timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    models_list = data.get("data", [])

                    for model in models_list:
                        model_id = model.get("id", "")
                        status = model.get("status", {})
                        status_value = status.get("value", "")

                        if status_value not in ["unloaded", ""] and (request.model_id is None or request.model_id == model_id):
                            messages.append(f"llama.cpp model {model_id} will auto-unload (Router Mode)")
                            unloaded_count += 1

                    if unloaded_count == 0:
                        messages.append("llama.cpp: No models currently loaded (Router Mode auto-unloads)")
                else:
                    messages.append("llama.cpp: No response from server")
            except Exception as e:
                messages.append(f"llama.cpp error: {str(e)}")

        # Clear GPU cache
        if model_type == "all":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    messages.append("GPU cache cleared")
            except ImportError:
                pass

        return UnloadResponse(
            success=unloaded_count > 0,
            message="; ".join(messages),
            unloaded_count=unloaded_count
        )

    except Exception as e:
        return UnloadResponse(
            success=False,
            message=f"Unload failed: {str(e)}",
            unloaded_count=unloaded_count
        )


# Export both routers
__all__ = ["router", "monitoring_router"]
