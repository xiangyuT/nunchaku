"""
Python wrapper for Nunchaku's high-performance GEMV (General Matrix-Vector Multiplication) kernels.

On CUDA devices the wrapper delegates to the native C++/CUDA extension.
On other devices (e.g. Intel XPU) a pure-PyTorch fallback is used automatically.

Backend selection priority (configurable via ``NUNCHAKU_BACKEND`` env var):
    1. ``cuda``  – native C++/CUDA extension (best performance on NVIDIA GPUs)
    2. ``torch``  – pure-PyTorch fallback (any device)
"""

import os

import torch


def _get_backend(device_type: str) -> str:
    """Determine which backend to use for the AWQ GEMV."""
    forced = os.environ.get("NUNCHAKU_BACKEND", "").lower().strip()
    if forced in ("cuda", "triton", "torch"):
        # Triton path falls through to torch for AWQ
        return "cuda" if forced == "cuda" else "torch"

    if device_type == "cuda":
        try:
            from .._C import ops as _ops  # noqa: F401

            return "cuda"
        except ImportError:
            pass

    return "torch"


def awq_gemv_w4a16_cuda(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> torch.Tensor:
    """
    Performs quantized GEMV using the AWQ W4A16 format.

    Parameters
    ----------
    in_feats : torch.Tensor, shape (k,) or (m, k), dtype float16 or bfloat16
        Input feature vector or batch of vectors.
    kernel : torch.Tensor, shape (n // 4, k // 2), dtype int32
        Packed quantized weight matrix.
    scaling_factors : torch.Tensor, shape (k // group_size, n), dtype float16 or bfloat16
        Per-group scaling factors.
    zeros : torch.Tensor, shape (k // group_size, n), dtype float16 or bfloat16
        Per-group zero points.
    m : int
        Batch size (number of input vectors).
    n : int
        Output feature dimension.
    k : int
        Input feature dimension.
    group_size : int, optional
        Number of input channels per quantization group. Default is 64.

    Returns
    -------
    torch.Tensor, shape (m, n), dtype float16 or bfloat16
        Output tensor.

    Notes
    -----
    Notations:

    - m: batch size
    - n: output features
    - k: input features
    - group_size: quantization group size
    """
    _device_type = in_feats.device.type if in_feats is not None else "cuda"
    _backend = _get_backend(_device_type)

    if _backend == "cuda":
        try:
            from .._C import ops

            return ops.gemv_awq(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size)
        except ImportError:
            pass

    from .torch_fallback import awq_gemv_w4a16_fallback

    return awq_gemv_w4a16_fallback(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size)
