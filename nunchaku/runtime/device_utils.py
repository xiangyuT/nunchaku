"""
Device-agnostic utilities for stream, event, and device operations.

This module abstracts hardware-specific APIs (CUDA, Intel XPU) behind a
unified interface so that higher-level code can work on any supported backend
without ``if device.type == ...`` scattered everywhere.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch

# ── backend availability ──────────────────────────────────────────────

SUPPORTED_DEVICE_TYPES = ("cuda", "xpu")


def is_cuda_available() -> bool:
    """Return ``True`` if CUDA is available."""
    return torch.cuda.is_available()


def is_xpu_available() -> bool:
    """Return ``True`` if Intel XPU is available (requires ``torch>=2.4`` with XPU support)."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def get_supported_backends() -> list[str]:
    """Return a list of currently available backend names."""
    backends: list[str] = []
    if is_cuda_available():
        backends.append("cuda")
    if is_xpu_available():
        backends.append("xpu")
    return backends


# ── device capability ─────────────────────────────────────────────────


@dataclass
class DeviceCapability:
    """Minimal representation of compute capability / device info."""

    major: int
    minor: int

    @property
    def sm(self) -> str:
        """NVIDIA-style SM string, e.g. ``"89"``."""
        return f"{self.major}{self.minor}"


def get_device_capability(device: str | torch.device = "cuda") -> DeviceCapability:
    """
    Query the device compute capability.

    For CUDA devices this returns the SM version.  For Intel XPU devices
    this currently returns ``DeviceCapability(0, 0)`` as a placeholder
    until Intel exposes a comparable API.

    Parameters
    ----------
    device : str or torch.device
        Target device.

    Returns
    -------
    DeviceCapability
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        idx = 0 if device.index is None else device.index
        major, minor = torch.cuda.get_device_capability(idx)
        return DeviceCapability(major=major, minor=minor)
    elif device.type == "xpu":
        # Intel XPU does not currently expose an SM-style capability.
        return DeviceCapability(major=0, minor=0)
    else:
        return DeviceCapability(major=0, minor=0)


def get_device_memory(device: str | torch.device = "cuda", unit: str = "GiB") -> int:
    """
    Return total device memory in the requested unit.

    Parameters
    ----------
    device : str or torch.device
    unit : ``"GiB"``, ``"MiB"`` or ``"B"``

    Returns
    -------
    int
    """
    if isinstance(device, str):
        device = torch.device(device)
    assert unit in ("GiB", "MiB", "B")

    if device.type == "cuda":
        memory = torch.cuda.get_device_properties(device).total_memory
    elif device.type == "xpu":
        memory = torch.xpu.get_device_properties(device).total_memory
    else:
        raise ValueError(f"get_device_memory not supported for device type '{device.type}'")

    if unit == "GiB":
        return memory // (1024**3)
    elif unit == "MiB":
        return memory // (1024**2)
    return memory


# ── stream / event abstractions ──────────────────────────────────────

class DeviceStream:
    """
    Thin wrapper around device-specific stream objects.

    Exposes ``wait_event`` and ``record_event`` so that callers do not need
    to know the underlying backend.
    """

    def __init__(self, device: torch.device | str | None = None):
        if device is not None and isinstance(device, str):
            device = torch.device(device)
        self._device = device
        device_type = device.type if device is not None else "cuda"
        if device_type == "cuda":
            self._stream = torch.cuda.Stream(device=device)
        elif device_type == "xpu":
            self._stream = torch.xpu.Stream(device=device)
        else:
            self._stream = None  # no-op on unsupported backends

    @property
    def raw(self):
        """Access the underlying backend stream (may be ``None``)."""
        return self._stream

    def wait_event(self, event) -> None:
        if self._stream is not None and event is not None:
            raw_event = event.raw if isinstance(event, DeviceEvent) else event
            self._stream.wait_event(raw_event)

    def record_event(self, event=None):
        if self._stream is None:
            return create_event(self._device)
        if event is None:
            event = create_event(self._device)
        raw_event = event.raw if isinstance(event, DeviceEvent) else event
        raw_event.record(self._stream)
        return event


class DeviceEvent:
    """
    Thin wrapper around device-specific event objects.
    """

    def __init__(self, device: torch.device | str | None = None, blocking: bool = False):
        if device is not None and isinstance(device, str):
            device = torch.device(device)
        self._device = device
        device_type = device.type if device is not None else "cuda"
        if device_type == "cuda":
            self._event = torch.cuda.Event(blocking=blocking)
        elif device_type == "xpu":
            self._event = torch.xpu.Event()
        else:
            self._event = None

    @property
    def raw(self):
        return self._event

    def record(self, stream=None) -> None:
        if self._event is None:
            return
        if stream is not None:
            raw_stream = stream.raw if isinstance(stream, DeviceStream) else stream
            self._event.record(raw_stream)
        else:
            self._event.record()


def create_stream(device: torch.device | str | None = None) -> DeviceStream:
    """Create a :class:`DeviceStream` for the given device."""
    return DeviceStream(device)


def create_event(device: torch.device | str | None = None, blocking: bool = False) -> DeviceEvent:
    """Create a :class:`DeviceEvent` for the given device."""
    return DeviceEvent(device, blocking=blocking)


def current_stream(device: torch.device | str | None = None):
    """Return the current backend stream for *device*."""
    if device is not None and isinstance(device, str):
        device = torch.device(device)
    device_type = device.type if device is not None else "cuda"
    if device_type == "cuda":
        return torch.cuda.current_stream(device)
    elif device_type == "xpu":
        return torch.xpu.current_stream(device)
    return None


@contextmanager
def stream_context(stream) -> Generator:
    """Context-manager that sets *stream* as the current stream."""
    if stream is None:
        yield
        return
    raw = stream.raw if isinstance(stream, DeviceStream) else stream
    if isinstance(raw, torch.cuda.Stream):
        with torch.cuda.stream(raw):
            yield
    elif hasattr(torch, "xpu") and isinstance(raw, torch.xpu.Stream):
        with torch.xpu.stream(raw):
            yield
    else:
        yield


def empty_cache(device: torch.device | str | None = None) -> None:
    """Release unused cached memory on the given device."""
    if device is not None and isinstance(device, str):
        device = torch.device(device)
    device_type = device.type if device is not None else "cuda"
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "xpu":
        torch.xpu.empty_cache()


def synchronize_device(device: torch.device | str | None = None) -> None:
    """Block until all pending work on *device* has finished."""
    if device is not None and isinstance(device, str):
        device = torch.device(device)
    device_type = device.type if device is not None else "cuda"
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "xpu":
        torch.xpu.synchronize(device)
