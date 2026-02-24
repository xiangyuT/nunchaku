"""
Pure-PyTorch fallback implementations for Nunchaku quantized operations.

These fallbacks allow the inference pipeline to run on devices that lack
the custom CUDA kernels (e.g. Intel XPU or CPU) at the cost of lower
performance and slightly different numerical behaviour.  They are intended
as *functional* stand-ins so that the rest of the model code can remain
device-agnostic.

.. note::
    The fallback kernels operate on the *packed* INT4/UINT8 representation
    that is already stored in the model weights.  They unpack on-the-fly
    using standard PyTorch ops so that no custom C++ code is needed.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# ── helpers ───────────────────────────────────────────────────────────


def _unpack_int4(packed: Tensor) -> Tensor:
    """Unpack ``int8``-packed INT4 pairs into two ``int8`` values per element.

    Parameters
    ----------
    packed : Tensor, dtype int8, shape (..., K // 2)

    Returns
    -------
    Tensor, dtype int8, shape (..., K)
    """
    low = (packed << 4) >> 4  # sign-extend lower nibble
    high = packed >> 4  # sign-extend upper nibble
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _dequantize_int4(packed_weight: Tensor, scales: Tensor, group_size: int = 64) -> Tensor:
    """Dequantize INT4 packed weights to fp16/bf16.

    Parameters
    ----------
    packed_weight : Tensor, shape (N, K // 2), dtype int8
    scales : Tensor, shape (K // group_size, N)
    group_size : int

    Returns
    -------
    Tensor, shape (N, K), dtype of *scales*
    """
    weight_i8 = _unpack_int4(packed_weight)  # (N, K)
    N, K = weight_i8.shape
    num_groups = K // group_size
    weight_grouped = weight_i8.reshape(N, num_groups, group_size).to(scales.dtype)
    scales_t = scales.T.unsqueeze(-1)  # (N, num_groups, 1)
    return (weight_grouped * scales_t).reshape(N, K)


# ── public fallback ops ──────────────────────────────────────────────


def svdq_gemm_w4a4_fallback(
    act: Tensor,
    wgt: Tensor,
    out: Tensor | None = None,
    qout: Tensor | None = None,
    ascales: Tensor | None = None,
    wscales: Tensor | None = None,
    oscales: Tensor | None = None,
    poolout: Tensor | None = None,
    lora_act_in: Tensor | None = None,
    lora_up: Tensor | None = None,
    lora_down: Tensor | None = None,
    lora_act_out: Tensor | None = None,
    norm_q: Tensor | None = None,
    norm_k: Tensor | None = None,
    rotary_emb: Tensor | None = None,
    bias: Tensor | None = None,
    smooth_factor: Tensor | None = None,
    out_vk: Tensor | None = None,
    out_linearattn: Tensor | None = None,
    act_unsigned: bool = False,
    lora_scales: list[float] | None = None,
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: float | None = 1.0,
    wcscales: Tensor | None = None,
    out_q: Tensor | None = None,
    out_k: Tensor | None = None,
    out_v: Tensor | None = None,
    attn_tokens: int = 0,
) -> None:
    """Fallback W4A4 GEMM using pure PyTorch.

    Dequantizes both activations and weights, then performs a standard matmul.
    Results are written in-place to the provided *out* tensor when it is given.
    """
    group_size = 16 if fp4 else 64
    compute_dtype = wscales.dtype if wscales is not None else torch.bfloat16

    # dequantize activations
    act_unpacked = _unpack_int4(act)  # (M, K)
    M, K = act_unpacked.shape
    if ascales is not None:
        num_groups_a = K // group_size
        act_grouped = act_unpacked.reshape(M, num_groups_a, group_size).to(compute_dtype)
        ascales_expanded = ascales[:, :M].unsqueeze(-1)  # (num_groups, M, 1)
        act_deq = (act_grouped * ascales_expanded.permute(1, 0, 2)).reshape(M, K)
    else:
        act_deq = act_unpacked.to(compute_dtype)

    # dequantize weights
    weight_deq = _dequantize_int4(wgt, wscales, group_size) if wscales is not None else _unpack_int4(wgt).to(compute_dtype)

    if alpha is not None and alpha != 1.0:
        weight_deq = weight_deq * alpha

    # matmul
    result = act_deq @ weight_deq.T

    # LoRA residual
    if lora_act_in is not None and lora_up is not None:
        lora_result = lora_act_in[:M].to(compute_dtype) @ lora_up.T.to(compute_dtype)
        result = result + lora_result

    if bias is not None:
        result = result + bias.unsqueeze(0)

    if fuse_silu:
        result = torch.nn.functional.silu(result)

    if out is not None:
        out_M = min(M, out.shape[0])
        out[:out_M].copy_(result[:out_M])


def svdq_quantize_w4a4_act_fuse_lora_fallback(
    input: Tensor,
    output: Tensor | None = None,
    oscales: Tensor | None = None,
    lora_down: Tensor | None = None,
    lora_act_out: Tensor | None = None,
    smooth: Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fallback activation quantization + LoRA down-projection.

    Computes per-group absmax quantization to INT4 and the low-rank
    projection using standard PyTorch operations.
    """
    batch_size, channels = input.shape
    group_size = 16 if fp4 else 64
    rank = lora_down.shape[1] if lora_down is not None else 0
    batch_size_pad = ((batch_size + pad_size - 1) // pad_size) * pad_size

    x = input.clone()
    if smooth is not None:
        x = x * smooth.unsqueeze(0)

    # per-group absmax quantization
    num_groups = channels // group_size
    x_grouped = x.reshape(batch_size, num_groups, group_size)
    absmax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax.squeeze(-1) / 7.0  # INT4 range [-8, 7]
    x_quant = torch.clamp(torch.round(x_grouped / absmax * 7.0), -8, 7).to(torch.int8)
    x_quant_flat = x_quant.reshape(batch_size, channels)

    # pack pairs of int4 into int8
    packed = (x_quant_flat[:, 1::2] << 4) | (x_quant_flat[:, ::2] & 0xF)
    packed = packed.to(torch.uint8)

    if output is None:
        output = torch.zeros(batch_size_pad, channels // 2, dtype=torch.uint8, device=input.device)
    output[:batch_size].copy_(packed)

    if oscales is None:
        if fp4:
            oscales = torch.zeros(channels // group_size, batch_size_pad, dtype=torch.float8_e4m3fn, device=input.device)
        else:
            oscales = torch.zeros(channels // group_size, batch_size_pad, dtype=input.dtype, device=input.device)
    oscales[:, :batch_size] = scales.permute(1, 0).to(oscales.dtype)

    if lora_act_out is None:
        lora_act_out = torch.zeros(batch_size_pad, max(rank, 1), dtype=torch.float32, device=input.device)
    if lora_down is not None and rank > 0:
        lora_act_out[:batch_size] = (x[:batch_size].float() @ lora_down.float())

    return output, oscales, lora_act_out


def _unpack_awq_int32(kernel: Tensor, n: int, k: int) -> Tensor:
    """Unpack AWQ TinyChat-format int32 weights to a ``(n, k)`` uint4 tensor.

    This reverses the ``pack_w4`` packing from TinyChat format, which:
      1. Reshapes weight ``(n, k)`` to ``(-1, 4, 8)``
      2. Packs 4 × uint4 into int16: ``int16[j] = val0[j] | (val1[j] << 4) | …``
      3. Permutes ``(n//4, 4, k//64, 16)`` → ``(n//4, k//64, 4, 16)``
      4. Stores as ``(n//4, k)`` int16 (viewed as ``(n//4, k//2)`` int32)

    Parameters
    ----------
    kernel : Tensor, shape (n // 4, k // 2), dtype int32
        Packed quantized weights.
    n : int
        Number of output features.
    k : int
        Number of input features.

    Returns
    -------
    Tensor, shape (n, k), dtype int32, values in [0, 15]
    """
    # View the int32 data as int16 to recover the original packed int16 values.
    packed_i16 = kernel.view(torch.int16).reshape(n // 4, k)

    # Reverse the permute(0,2,1,3) and view that was applied during packing.
    packed_i16 = packed_i16.view(n // 4, k // 64, 4, 16)
    packed_i16 = packed_i16.permute(0, 2, 1, 3).contiguous()
    packed_i16 = packed_i16.reshape(-1, 8)  # (n*k//32, 8) int16

    # Extract 4 × uint4 from each int16.
    # int16[j] was: ic[j] | (ic[8+j] << 4) | (ic[16+j] << 8) | (ic[24+j] << 12)
    val0 = (packed_i16) & 0xF
    val1 = (packed_i16 >> 4) & 0xF
    val2 = (packed_i16 >> 8) & 0xF
    val3 = (packed_i16 >> 12) & 0xF

    # Stack and reshape to recover the original (n, k) weight matrix.
    weight = torch.stack([val0, val1, val2, val3], dim=1)  # (-1, 4, 8)
    return weight.reshape(n, k).to(torch.int32)


def awq_gemv_w4a16_fallback(
    in_feats: Tensor,
    kernel: Tensor,
    scaling_factors: Tensor,
    zeros: Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> Tensor:
    """Fallback AWQ W4A16 GEMV/GEMM using pure PyTorch.

    Unpacks the TinyChat-format int32-packed weights, dequantizes with
    per-group scales and zero-points, and performs a standard matmul.

    Parameters
    ----------
    in_feats : Tensor, shape (m, k)
        Input activations.
    kernel : Tensor, shape (n // 4, k // 2), dtype int32
        Packed quantized weights (TinyChat format).
    scaling_factors : Tensor, shape (k // group_size, n)
        Per-group weight scales.
    zeros : Tensor, shape (k // group_size, n)
        Per-group zero-points (pre-scaled/negated as stored by TinyChat).
    m, n, k : int
        Matrix dimensions.
    group_size : int
        Quantization group size (default 64).

    Returns
    -------
    Tensor, shape (m, n)
    """
    compute_dtype = in_feats.dtype

    # Unpack packed int32 → (n, k) uint4 values in [0, 15].
    weight_uint4 = _unpack_awq_int32(kernel, n, k)

    # Dequantize: w_float = w_uint4 * scale + zeros
    # scaling_factors: (k // group_size, n), zeros: (k // group_size, n)
    num_groups = k // group_size
    weight_fp = weight_uint4.to(compute_dtype).reshape(n, num_groups, group_size)
    scales_expanded = scaling_factors.T.unsqueeze(-1)   # (n, num_groups, 1)
    zeros_expanded = zeros.T.unsqueeze(-1)              # (n, num_groups, 1)

    weight_deq = weight_fp * scales_expanded + zeros_expanded
    weight_deq = weight_deq.reshape(n, k)

    # GEMV/GEMM: output = input @ weight^T
    x = in_feats if in_feats.dim() == 2 else in_feats.unsqueeze(0)
    output = x.to(compute_dtype) @ weight_deq.T.to(compute_dtype)

    return output
