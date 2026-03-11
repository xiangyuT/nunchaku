"""
Nunchaku runtime device abstraction layer.

Provides device-agnostic utilities for backend detection, device capability
queries, and stream/event management to support multiple hardware backends
(CUDA, Intel XPU, and pure-PyTorch fallback).
"""

from .device_utils import (
    DeviceCapability,
    DeviceStream,
    create_event,
    create_stream,
    current_stream,
    empty_cache,
    get_device_capability,
    get_device_memory,
    get_supported_backends,
    is_cuda_available,
    is_xpu_available,
    stream_context,
    synchronize_device,
)

__all__ = [
    "DeviceCapability",
    "DeviceStream",
    "create_event",
    "create_stream",
    "current_stream",
    "empty_cache",
    "get_device_capability",
    "get_device_memory",
    "get_supported_backends",
    "is_cuda_available",
    "is_xpu_available",
    "stream_context",
    "synchronize_device",
]
