"""Memory Monitor Extension.

Provides GPU and system memory monitoring as an optional feature.
"""

import logging
from pathlib import Path
from typing import Any
from fastapi import APIRouter

import psutil

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class MemoryMonitorExtension(LogicExtension):
    """Memory monitoring extension.

    Provides endpoints for monitoring GPU and RAM usage.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)

    async def initialize(self, context):
        """Initialize the memory monitor extension."""
        await super().initialize(context)
        logger.info("Memory monitor extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        pass

    async def process(self, input_data: dict, **kwargs):
        """Process memory monitoring request."""
        action = input_data.get('action', 'memory')

        if action == 'memory':
            return await self.get_memory()
        elif action == 'gpu':
            return await self.get_gpu_info()
        elif action == 'ram':
            return await self.get_ram_info()
        else:
            return {"error": f"Unknown action: {action}"}

    def register_routes(self, app) -> None:
        """Register API routes with the FastAPI application."""
        from .api import router, monitoring_router, set_extension

        # Set extension instance for API handlers
        set_extension(self)

        # Include both routers - they have their own prefixes defined
        app.include_router(router)
        app.include_router(monitoring_router)

    def get_router_info(self) -> dict:
        """Get router information for frontend discovery."""
        return {
            "prefix": "/api/v1/system/memory",
            "name": "memory_monitor",
            "tags": ["monitoring", "memory"],
            "description": "GPU and system memory monitoring endpoints"
        }

    async def get_gpu_info(self) -> dict[str, Any]:
        """Get GPU information if available."""
        gpu_info = {
            "count": 0,
            "memory": {"total": 0, "used": 0, "free": 0, "percent": 0.0},
            "utilization": 0.0,
            "info": {}
        }

        # Try pynvml first (most efficient)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_info["count"] = device_count
            total_memory = 0
            used_memory = 0
            total_utilization = 0

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                total_memory += mem_info.total
                used_memory += mem_info.used

                # Get utilization
                utilization = 0
                try:
                    util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util_rates.gpu
                    total_utilization += utilization
                except Exception:
                    pass

                gpu_info["info"][f"gpu_{i}"] = {
                    "name": name,
                    "total_memory": mem_info.total,
                    "used_memory": mem_info.used,
                    "free_memory": mem_info.free,
                    "utilization": utilization
                }

            gpu_info["memory"]["total"] = total_memory
            gpu_info["memory"]["used"] = used_memory
            gpu_info["memory"]["free"] = total_memory - used_memory
            gpu_info["memory"]["percent"] = (used_memory / total_memory * 100) if total_memory > 0 else 0
            gpu_info["utilization"] = total_utilization / device_count if device_count > 0 else 0
            return gpu_info
        except ImportError:
            pass  # pynvml not installed, try fallbacks
        except Exception:
            pass  # NVML error, try fallbacks

        # Fallback: Try PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["count"] = torch.cuda.device_count()
                total_utilization = 0

                for i in range(gpu_info["count"]):
                    props = torch.cuda.get_device_properties(i)

                    gpu_info["info"][f"gpu_{i}"] = {
                        "name": props.name,
                        "total_memory": props.total_memory,
                        "allocated_memory": torch.cuda.memory_allocated(i),
                        "utilization": torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
                    }
                    total_utilization += gpu_info["info"][f"gpu_{i}"]["utilization"]

                gpu_info["memory"]["total"] = sum(
                    gpu_info["info"][f"gpu_{i}"]["total_memory"] for i in range(gpu_info["count"])
                )
                gpu_info["memory"]["used"] = sum(
                    gpu_info["info"][f"gpu_{i}"]["allocated_memory"] for i in range(gpu_info["count"])
                )
                gpu_info["memory"]["free"] = gpu_info["memory"]["total"] - gpu_info["memory"]["used"]
                gpu_info["memory"]["percent"] = (
                    gpu_info["memory"]["used"] / gpu_info["memory"]["total"] * 100
                    if gpu_info["memory"]["total"] > 0 else 0
                )
                gpu_info["utilization"] = total_utilization / gpu_info["count"] if gpu_info["count"] > 0 else 0
        except (ImportError, AttributeError):
            pass

        return gpu_info

    async def get_ram_info(self) -> dict[str, Any]:
        """Get RAM information."""
        ram_usage = psutil.virtual_memory()
        return {
            "total": ram_usage.total,
            "used": ram_usage.used,
            "free": ram_usage.available,
            "percent": ram_usage.percent
        }

    async def get_memory(self) -> dict[str, Any]:
        """Get current RAM and GPU memory usage."""
        import time
        return {
            "ram": await self.get_ram_info(),
            "gpu": await self.get_gpu_info(),
            "timestamp": time.time()
        }

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(MemoryMonitorExtension)
