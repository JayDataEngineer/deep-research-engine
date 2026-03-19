"""
Memory monitoring service to track RAM and GPU usage.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryUsage:
    """Data class to represent memory usage information."""
    ram_total: int = 0
    ram_used: int = 0
    ram_free: int = 0
    ram_percent: float = 0.0
    gpu_count: int = 0
    gpu_memory_total: int = 0
    gpu_memory_used: int = 0
    gpu_memory_free: int = 0
    gpu_memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    gpu_info: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())


class MemoryMonitor:
    """Service to monitor RAM and GPU memory usage."""

    def __init__(self, update_interval: float = 1.0):
        """Initialize the memory monitor."""
        self.update_interval = update_interval
        self._stop_event = threading.Event()
        self._monitoring_thread: threading.Thread | None = None
        self._current_usage = MemoryUsage()
        self._lock = threading.Lock()
        self._observers = []

        # Initialize CUDA availability as False by default
        self.cuda_available = False
        self.torch_available = False

        # Check if CUDA is available by trying to import torch
        try:
            import torch
            self.torch_available = True
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                logger.info(f"CUDA available with {torch.cuda.device_count()} devices")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                logger.info("CUDA not available - GPU monitoring will be limited")
        except (AttributeError, ImportError, OSError, ValueError) as e:
            # torch might not be installed properly or CUDA libraries missing
            self.cuda_available = False
            self.torch_available = False
            logger.info(f"PyTorch/CUDA not available - GPU monitoring will be limited: {e}")

    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available by trying to import torch."""
        try:
            import torch
            return torch.cuda.is_available()
        except (AttributeError, ImportError, OSError, ValueError):
            # torch might not have cuda module in some installations
            # or CUDA libraries might not be available
            return False

    def start_monitoring(self):
        """Start the memory monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Memory monitoring is already running")
            return

        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Started memory monitoring")

    def stop_monitoring(self):
        """Stop the memory monitoring thread."""
        self._stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
        logger.info("Stopped memory monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop that updates memory usage."""
        while not self._stop_event.is_set():
            try:
                usage = self._get_memory_usage()
                with self._lock:
                    self._current_usage = usage
                # Notify observers of the updated memory usage
                self._notify_observers()
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}", exc_info=True)

            # Wait for the update interval or until stop event
            if self._stop_event.wait(timeout=self.update_interval):
                break

    def _get_memory_usage(self) -> MemoryUsage:
        """Get current memory usage."""
        usage = MemoryUsage()

        # Get RAM usage
        ram_usage = psutil.virtual_memory()
        usage.ram_total = ram_usage.total
        usage.ram_used = ram_usage.used
        usage.ram_free = ram_usage.available
        usage.ram_percent = ram_usage.percent

        # Get GPU usage if available
        if self.cuda_available and self.torch_available:
            try:
                import torch
                usage.gpu_count = torch.cuda.device_count()

                # Total GPU memory across all devices
                total_gpu_memory = 0
                used_gpu_memory = 0

                for i in range(usage.gpu_count):
                    try:
                        # For CUDA devices, we can get memory stats
                        device_props = torch.cuda.get_device_properties(i)
                        device_memory = torch.cuda.get_device_properties(i).total_memory
                        allocated_memory = torch.cuda.memory_allocated(i)

                        total_gpu_memory += device_memory
                        used_gpu_memory += allocated_memory

                        # Store device-specific info
                        usage.gpu_info[f"gpu_{i}"] = {
                            "name": device_props.name,
                            "total_memory": device_memory,
                            "allocated_memory": allocated_memory,
                            "utilization": torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
                        }
                    except Exception as e:
                        logger.warning(f"Could not get GPU {i} memory info: {e}")

                usage.gpu_memory_total = total_gpu_memory
                usage.gpu_memory_used = used_gpu_memory
                usage.gpu_memory_free = total_gpu_memory - used_gpu_memory
                usage.gpu_memory_percent = (used_gpu_memory / total_gpu_memory) * 100 if total_gpu_memory > 0 else 0

                # Calculate average GPU utilization across all devices
                total_utilization = 0
                valid_utilization_count = 0
                for i in range(usage.gpu_count):
                    gpu_key = f"gpu_{i}"
                    if gpu_key in usage.gpu_info and "utilization" in usage.gpu_info[gpu_key]:
                        total_utilization += usage.gpu_info[gpu_key]["utilization"]
                        valid_utilization_count += 1

                usage.gpu_utilization = total_utilization / valid_utilization_count if valid_utilization_count > 0 else 0
            except Exception as e:
                logger.warning(f"Error getting PyTorch GPU info: {e}")
                self.cuda_available = False
        else:
            # Try to get GPU info via pynvml if available
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()

                usage.gpu_count = device_count
                total_gpu_memory = 0
                used_gpu_memory = 0

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    total_gpu_memory += mem_info.total
                    used_gpu_memory += mem_info.used

                    # Store device-specific info
                    # pynvml.nvmlDeviceGetName() may return bytes or str depending on version
                    device_name_raw = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(device_name_raw, bytes):
                        device_name = device_name_raw.decode('utf-8')
                    else:
                        device_name = device_name_raw
                    usage.gpu_info[f"gpu_{i}"] = {
                        "name": device_name,
                        "total_memory": mem_info.total,
                        "used_memory": mem_info.used,
                        "free_memory": mem_info.free
                    }

                usage.gpu_memory_total = total_gpu_memory
                usage.gpu_memory_used = used_gpu_memory
                usage.gpu_memory_free = total_gpu_memory - used_gpu_memory
                usage.gpu_memory_percent = (used_gpu_memory / total_gpu_memory) * 100 if total_gpu_memory > 0 else 0

                # Average utilization
                total_utilization = 0
                valid_utilization_count = 0
                for i in range(device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        total_utilization += utilization.gpu
                        valid_utilization_count += 1
                    except Exception:
                        # Some GPUs might not support utilization monitoring
                        pass

                usage.gpu_utilization = total_utilization / valid_utilization_count if valid_utilization_count > 0 else 0
            except ImportError:
                logger.info("pynvml not available - GPU monitoring is limited")
            except Exception as e:
                logger.warning(f"Error getting GPU info: {e}")

        usage.timestamp = time.time()
        return usage

    def get_current_usage(self) -> MemoryUsage:
        """Get the current memory usage."""
        with self._lock:
            return self._current_usage

    def get_formatted_usage(self) -> dict[str, Any]:
        """Get memory usage formatted for API responses."""
        usage = self.get_current_usage()

        return {
            "ram": {
                "total": usage.ram_total,
                "used": usage.ram_used,
                "free": usage.ram_free,
                "percent": usage.ram_percent
            },
            "gpu": {
                "count": usage.gpu_count,
                "memory": {
                    "total": usage.gpu_memory_total,
                    "used": usage.gpu_memory_used,
                    "free": usage.gpu_memory_free,
                    "percent": usage.gpu_memory_percent
                },
                "utilization": usage.gpu_utilization,
                "info": usage.gpu_info
            },
            "timestamp": usage.timestamp
        }

    def register_observer(self, callback: Callable):
        """Register a callback to be notified of memory usage changes."""
        if callback not in self._observers:
            self._observers.append(callback)
            logger.debug("Registered observer for memory monitor")

    def unregister_observer(self, callback: Callable):
        """Unregister a callback."""
        if callback in self._observers:
            self._observers.remove(callback)
            logger.debug("Unregistered observer for memory monitor")

    def _notify_observers(self):
        """Notify all registered observers of memory usage changes."""
        for observer in self._observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    # If it's an async function, run it in a new event loop (not ideal but works for threading)
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, observer())
                        future.result()
                elif callable(observer):
                    # Call the observer function, awaiting if it returns a coroutine
                    result = observer()
                    if result and asyncio.iscoroutine(result):
                        # If it returns a coroutine, we can't await it from a thread
                        # So we run it in a new event loop
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, result)
                            future.result()
            except Exception as e:
                logger.error(f"Error notifying memory monitor observer: {e}", exc_info=True)


# Global instance of the memory monitor
memory_monitor = MemoryMonitor()
