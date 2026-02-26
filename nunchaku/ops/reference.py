"""
Torch-native reference implementations of all Nunchaku custom CUDA operators.

These implementations are functionally equivalent to the CUDA kernels but use
standard PyTorch operations for readability, testing, and debugging. They operate
on **logical (unpacked) tensor formats** to clearly describe each operator's
mathematical semantics.

.. note::
    The actual CUDA kernels in Nunchaku use hardware-specific packed data layouts
    optimized for NVIDIA Tensor Core MMA instructions. These reference
    implementations use standard dense tensors and are not optimized for
    performance.

Operators covered:

- **Quantization primitives**: INT4 and NVFP4 (E2M1) quantize / dequantize
- **AWQ primitives**: AWQ dequantize (W4A16)
- **GEMM**: W4A4 quantized matrix multiplication with LoRA, bias, SiLU, RMSNorm, and rotary embeddings
- **Activation quantization**: fused quantization with LoRA down-projection
- **Attention**: FP16 scaled dot-product attention
- **AWQ linear**: GEMV / GEMM for AWQ W4A16 format
- **Building blocks**: RMS normalization, rotary embedding, SiLU, GELU
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHIFT_GELU: float = 0.171875
"""GELU shift constant used for unsigned INT4 quantization after GELU activation."""

INT4_GROUP_SIZE: int = 64
"""Default group size for INT4 quantization."""

NVFP4_GROUP_SIZE: int = 16
"""Default group size for NVFP4 (E2M1) quantization."""

# Lookup table for E2M1 (NVFP4) dequantization.
# 4-bit E2M1 encoding (unsigned, no sign bit in the nibble):
#   0b0000 -> 0.0,   0b0001 -> 0.5,  0b0010 -> 1.0,  0b0011 -> 1.5,
#   0b0100 -> 2.0,   0b0101 -> 3.0,  0b0110 -> 4.0,  0b0111 -> 6.0,
#   0b1000 -> -0.0,  0b1001 -> -0.5, 0b1010 -> -1.0, 0b1011 -> -1.5,
#   0b1100 -> -2.0,  0b1101 -> -3.0, 0b1110 -> -4.0, 0b1111 -> -6.0,
_E2M1_LUT: list[float] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


# ===================================================================
# 1. Quantization / Dequantization Primitives
# ===================================================================


def dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = INT4_GROUP_SIZE,
    unsigned: bool = False,
) -> torch.Tensor:
    """Dequantize INT4 packed tensor to floating-point.

    Each ``int8`` element stores two 4-bit values (lower nibble first).

    Parameters
    ----------
    packed : torch.Tensor, shape ``(..., K // 2)``, dtype ``int8``
        Packed INT4 values.
    scales : torch.Tensor, shape ``(K // group_size, N)``, dtype ``float16 / bfloat16``
        Per-group quantization scales, where *N* is the leading dimension of
        ``packed`` and *K* is the unpacked feature dimension.
    group_size : int
        Number of elements per quantization group (default 64).
    unsigned : bool
        If ``True``, treat the 4-bit values as unsigned ``[0, 15]``;
        otherwise signed ``[-8, 7]``.

    Returns
    -------
    torch.Tensor, shape ``(..., K)``, dtype matching *scales*
        Dequantized tensor.
    """
    packed_uint8 = packed.view(torch.uint8)
    low = (packed_uint8 & 0xF).to(torch.int32)
    high = ((packed_uint8 >> 4) & 0xF).to(torch.int32)
    if not unsigned:
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
    # Interleave: low_0, high_0, low_1, high_1, ...
    unpacked = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    unpacked = unpacked.to(scales.dtype)

    # Apply per-group scales.
    # scales shape: (num_groups, M) where M is the first dimension of packed
    # unpacked shape: (M, K)
    M = unpacked.shape[0]
    K = unpacked.shape[-1]
    num_groups = K // group_size
    unpacked = unpacked.reshape(M, num_groups, group_size)
    # scales: (num_groups, M) → (M, num_groups, 1)
    scales_expanded = scales[:, :M].permute(1, 0).unsqueeze(-1).to(unpacked.dtype)
    unpacked = unpacked * scales_expanded
    return unpacked.reshape(M, K)


def dequantize_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = NVFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Dequantize NVFP4 (E2M1) packed tensor to floating-point.

    Each ``uint8`` element stores two E2M1 4-bit values (lower nibble first).

    Parameters
    ----------
    packed : torch.Tensor, shape ``(..., K // 2)``, dtype ``uint8``
        Packed FP4 values.
    scales : torch.Tensor, shape ``(K // group_size, N)``, dtype ``float8_e4m3fn``
        Per-group quantization scales.
    group_size : int
        Number of elements per quantization group (default 16).

    Returns
    -------
    torch.Tensor, shape ``(..., K)``, dtype ``float32``
        Dequantized tensor.
    """
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=packed.device)
    packed_uint8 = packed.view(torch.uint8)
    low_idx = (packed_uint8 & 0xF).to(torch.long)
    high_idx = ((packed_uint8 >> 4) & 0xF).to(torch.long)
    low = lut[low_idx]
    high = lut[high_idx]
    unpacked = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)

    # Apply per-group scales (convert float8 scales to float32 for computation).
    M = unpacked.shape[0]
    K = unpacked.shape[-1]
    num_groups = K // group_size
    unpacked = unpacked.reshape(M, num_groups, group_size)
    # scales: (num_groups, M) → (M, num_groups, 1)
    scales_f32 = scales[:, :M].to(torch.float32).permute(1, 0).unsqueeze(-1)
    unpacked = unpacked * scales_f32
    return unpacked.reshape(M, K)


def quantize_to_int4(
    x: torch.Tensor,
    group_size: int = INT4_GROUP_SIZE,
    unsigned: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating-point tensor to packed INT4.

    Parameters
    ----------
    x : torch.Tensor, shape ``(M, K)``, dtype ``float16 / bfloat16``
        Input tensor.
    group_size : int
        Number of elements per quantization group.
    unsigned : bool
        If ``True``, quantize to unsigned ``[0, 15]``; otherwise signed ``[-8, 7]``.

    Returns
    -------
    packed : torch.Tensor, shape ``(M, K // 2)``, dtype ``uint8``
        Packed INT4 values (lower nibble first).
    scales : torch.Tensor, shape ``(K // group_size, M)``, dtype matching *x*
        Per-group quantization scales.
    """
    M, K = x.shape
    assert K % group_size == 0
    num_groups = K // group_size
    x_grouped = x.reshape(M, num_groups, group_size)

    if unsigned:
        qmin, qmax = 0, 15
    else:
        qmin, qmax = -8, 7

    amax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    if unsigned:
        scales = amax / qmax
    else:
        scales = amax / max(abs(qmin), abs(qmax))

    x_scaled = (x_grouped / scales).clamp(qmin, qmax).round().to(torch.int32)
    x_scaled = x_scaled.reshape(M, K)

    if not unsigned:
        x_scaled = x_scaled & 0xF  # Store as unsigned nibble bits

    # Pack two values into one byte: low nibble first
    even = x_scaled[:, 0::2] & 0xF
    odd = (x_scaled[:, 1::2] & 0xF) << 4
    packed = (even | odd).to(torch.uint8)

    # scales: (M, num_groups) → (num_groups, M) to match CUDA layout
    scales = scales.squeeze(-1).permute(1, 0).to(x.dtype)
    return packed, scales


def quantize_to_nvfp4(
    x: torch.Tensor,
    group_size: int = NVFP4_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating-point tensor to packed NVFP4 (E2M1).

    Parameters
    ----------
    x : torch.Tensor, shape ``(M, K)``
        Input tensor.
    group_size : int
        Number of elements per quantization group (default 16).

    Returns
    -------
    packed : torch.Tensor, shape ``(M, K // 2)``, dtype ``uint8``
        Packed E2M1 values (lower nibble first).
    scales : torch.Tensor, shape ``(K // group_size, M)``, dtype ``float8_e4m3fn``
        Per-group quantization scales.
    """
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=x.device)
    abs_lut = lut.abs()

    M, K = x.shape
    assert K % group_size == 0
    num_groups = K // group_size
    x_grouped = x.reshape(M, num_groups, group_size)

    amax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    max_fp4 = 6.0  # Maximum representable value in E2M1
    scales = amax / max_fp4

    x_scaled = x_grouped / scales
    x_flat = x_scaled.reshape(M, K)

    # Find nearest E2M1 value for each element
    x_abs = x_flat.abs().unsqueeze(-1)
    distances = (x_abs - abs_lut[:8].unsqueeze(0).unsqueeze(0)).abs()
    nearest_idx = distances.argmin(dim=-1)
    # Apply sign
    nearest_idx = torch.where(x_flat < 0, nearest_idx + 8, nearest_idx)

    # Pack two values into one byte
    even = nearest_idx[:, 0::2] & 0xF
    odd = (nearest_idx[:, 1::2] & 0xF) << 4
    packed = (even | odd).to(torch.uint8)

    # Scales: (M, num_groups) → (num_groups, M), convert to float8_e4m3fn
    scales = scales.squeeze(-1).permute(1, 0).to(torch.float32)
    scales = scales.to(torch.float8_e4m3fn)
    return packed, scales


# ===================================================================
# 2. AWQ Dequantization
# ===================================================================


def dequantize_awq(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    """Dequantize AWQ W4A16 packed weights.

    Each ``int32`` element stores 8 consecutive 4-bit unsigned weight values.

    Parameters
    ----------
    qweight : torch.Tensor, shape ``(N // 4, K // 2)``, dtype ``int32``
        Packed quantized weights.  *N* = output features, *K* = input features.
    scales : torch.Tensor, shape ``(K // group_size, N)``, dtype ``float16 / bfloat16``
        Per-group scaling factors.
    zeros : torch.Tensor, shape ``(K // group_size, N)``, dtype ``float16 / bfloat16``
        Per-group zero points.
    group_size : int
        Quantization group size (default 64).

    Returns
    -------
    torch.Tensor, shape ``(N, K)``, dtype matching *scales*
        Dequantized weight matrix.
    """
    # qweight: (N // 4, K // 2) int32 → each int32 has 8 × 4-bit values
    # Total elements = (N // 4) * (K // 2) * 8 = N * K
    rows, cols = qweight.shape  # (N // 4, K // 2)
    N = rows * 4
    K = cols * 2

    # Unpack 8 × 4-bit values from each int32
    qw = qweight.to(torch.int32)
    unpacked_values = []
    for i in range(8):
        val = (qw >> (i * 4)) & 0xF
        unpacked_values.append(val)
    # Stack: (N // 4, K // 2, 8)
    unpacked = torch.stack(unpacked_values, dim=-1)
    # Reshape to (N, K): 4 output channels per row, 2 × 4 = 8 values per int32 along K
    unpacked = unpacked.reshape(rows, cols * 8)
    # Map to (N, K): interleave rows (4 per group) and columns (8 per int32)
    weight_int = unpacked.reshape(N, K).to(scales.dtype)

    # Apply (weight_int - zeros) * scales
    num_groups = K // group_size
    weight_int = weight_int.reshape(N, num_groups, group_size)
    # scales, zeros: (num_groups, N) → (N, num_groups, 1)
    scales_exp = scales.permute(1, 0).unsqueeze(-1)  # (N, num_groups, 1)
    zeros_exp = zeros.permute(1, 0).unsqueeze(-1)  # (N, num_groups, 1)
    weight_float = (weight_int - zeros_exp) * scales_exp
    return weight_float.reshape(N, K)


# ===================================================================
# 3. Building Blocks
# ===================================================================


def rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMS (Root Mean Square) normalization.

    .. math::
        \\text{out} = \\frac{x}{\\text{RMS}(x)} \\cdot \\text{weight}

    Parameters
    ----------
    x : torch.Tensor, shape ``(..., D)``
        Input tensor.
    weight : torch.Tensor, shape ``(D,)``
        Learnable scale parameter.
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    torch.Tensor, shape ``(..., D)``
        Normalized tensor.
    """
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


def apply_rotary_emb_reference(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embedding.

    Parameters
    ----------
    x : torch.Tensor, shape ``(..., D)``
        Input tensor (typically Q or K).
    sin : torch.Tensor, shape ``(..., D // 2)``
        Sine component of rotary embedding.
    cos : torch.Tensor, shape ``(..., D // 2)``
        Cosine component of rotary embedding.

    Returns
    -------
    torch.Tensor, shape ``(..., D)``
        Tensor with rotary embedding applied.
    """
    D = x.shape[-1]
    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 :]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


def silu_reference(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Sigmoid Linear Unit) activation.

    .. math::
        \\text{SiLU}(x) = x \\cdot \\sigma(x)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Activated tensor.
    """
    return F.silu(x)


def gelu_reference(x: torch.Tensor) -> torch.Tensor:
    """GELU (Gaussian Error Linear Unit) activation.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Activated tensor.
    """
    return F.gelu(x)


# ===================================================================
# 4. Core Operators — Reference Implementations
# ===================================================================


def gemm_w4a4_reference(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    lora_act_in: Optional[torch.Tensor] = None,
    lora_up: Optional[torch.Tensor] = None,
    lora_scales: Optional[list[float]] = None,
    norm_q: Optional[torch.Tensor] = None,
    norm_k: Optional[torch.Tensor] = None,
    rotary_emb_sin: Optional[torch.Tensor] = None,
    rotary_emb_cos: Optional[torch.Tensor] = None,
    smooth_factor: Optional[torch.Tensor] = None,
    lora_down: Optional[torch.Tensor] = None,
    act_unsigned: bool = False,
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: float = 1.0,
    wcscales: Optional[torch.Tensor] = None,
    head_dim: int = 128,
) -> dict[str, Optional[torch.Tensor]]:
    """Reference implementation of the W4A4 quantized GEMM.

    This function computes:

    .. math::
        \\text{out} = \\text{dequant}(\\text{act}, \\text{ascales})
                      \\;@\\; \\text{dequant}(\\text{wgt}, \\text{wscales})^T

    with optional fused LoRA, bias, activation, normalization, rotary
    embeddings, and output quantization.

    Parameters
    ----------
    act : torch.Tensor, shape ``(M, K)``
        Dequantized (or float) input activations.
    wgt : torch.Tensor, shape ``(N, K)``
        Dequantized (or float) weight matrix.
    ascales : torch.Tensor, shape ``(K // G, M)``
        Activation per-group scales.  Pass ones if *act* is already in float.
    wscales : torch.Tensor, shape ``(K // G, N)``
        Weight per-group scales.  Pass ones if *wgt* is already in float.
    bias : torch.Tensor or None, shape ``(N,)``
        Optional bias.
    lora_act_in : torch.Tensor or None, shape ``(M, R)``
        LoRA down-projection activations (from previous quantization step).
    lora_up : torch.Tensor or None, shape ``(N, R)``
        LoRA up-projection weight.
    lora_scales : list[float] or None
        Per-group (groups of 16 ranks) LoRA scaling factors.
    norm_q : torch.Tensor or None, shape ``(head_dim,)``
        RMSNorm weight for query heads.
    norm_k : torch.Tensor or None, shape ``(head_dim,)``
        RMSNorm weight for key heads.
    rotary_emb_sin : torch.Tensor or None, shape ``(M, head_dim // 2)``
        Sine of rotary embeddings.
    rotary_emb_cos : torch.Tensor or None, shape ``(M, head_dim // 2)``
        Cosine of rotary embeddings.
    smooth_factor : torch.Tensor or None, shape ``(N,)``
        Smooth factor for output quantization (divide before quantize).
    lora_down : torch.Tensor or None, shape ``(N, R)``
        LoRA down-projection weight for the *next* layer.
    act_unsigned : bool
        ``True`` if activations are unsigned INT4.
    fuse_silu : bool
        ``True`` to apply SiLU on the output.
    fp4 : bool
        ``True`` for NVFP4 mode (group_size = 16); else INT4 (group_size = 64).
    alpha : float
        Per-tensor weight scale (used with NVFP4).
    wcscales : torch.Tensor or None, shape ``(N,)``
        Per-channel weight scale (NVFP4).
    head_dim : int
        Head dimension for QKV-norm-rotary mode.

    Returns
    -------
    dict with keys:
        ``"out"`` : torch.Tensor, shape ``(M, N)``
            Main output (before optional SiLU or quantization).
        ``"out_silu"`` : torch.Tensor or None
            Output after SiLU (if *fuse_silu*).
        ``"out_quantized"`` : torch.Tensor or None
            Quantized output for the next layer (if *smooth_factor* provided).
        ``"out_oscales"`` : torch.Tensor or None
            Output scales (if quantized).
        ``"lora_act_out"`` : torch.Tensor or None
            LoRA down-projection output for the next layer.
        ``"out_normed_rotary"`` : torch.Tensor or None
            Output with RMSNorm + rotary applied (if *norm_q* provided).

    Notes
    -----
    Notations:

    - M : batch (token) dimension
    - K : input feature dimension
    - N : output feature dimension
    - G : group size (64 for INT4, 16 for NVFP4)
    - R : LoRA rank
    """
    group_size = NVFP4_GROUP_SIZE if fp4 else INT4_GROUP_SIZE
    M, K = act.shape
    N = wgt.shape[0]

    # --- Step 1: Grouped dequantized matmul ---
    # Conceptually: out[m, n] = Σ_g Σ_k act[m,g*G+k] * wgt[n,g*G+k] * ascales[g,m] * wscales[g,n]
    num_groups = K // group_size
    act_grouped = act.reshape(M, num_groups, group_size).float()
    wgt_grouped = wgt.reshape(N, num_groups, group_size).float()
    ascales_f = ascales.float()  # (num_groups, M)
    wscales_f = wscales.float()  # (num_groups, N)

    # Scale each group
    act_scaled = act_grouped * ascales_f.permute(1, 0).unsqueeze(-1)  # (M, num_groups, G)
    wgt_scaled = wgt_grouped * wscales_f.permute(1, 0).unsqueeze(-1)  # (N, num_groups, G)

    # Matmul: flatten groups back
    act_full = act_scaled.reshape(M, K)
    wgt_full = wgt_scaled.reshape(N, K)
    out = act_full @ wgt_full.T  # (M, N)

    # --- Step 2: Per-channel weight scale (NVFP4) ---
    if wcscales is not None:
        out = out * wcscales.float().unsqueeze(0) * alpha

    # --- Step 3: LoRA ---
    if lora_act_in is not None and lora_up is not None:
        R = lora_up.shape[1]
        if lora_scales is None:
            lora_scales = [1.0] * math.ceil(R / 16)
        # Apply per-group LoRA scales to lora_act_in
        lora_act_scaled = lora_act_in.float().clone()
        for g in range(len(lora_scales)):
            start = g * 16
            end = min(start + 16, R)
            lora_act_scaled[:, start:end] *= lora_scales[g]
        lora_out = lora_act_scaled @ lora_up.float().T  # (M, N)
        out = out + lora_out

    # --- Step 4: Bias ---
    if bias is not None:
        out = out + bias.float().unsqueeze(0)

    result: dict[str, Optional[torch.Tensor]] = {
        "out": out,
        "out_silu": None,
        "out_quantized": None,
        "out_oscales": None,
        "lora_act_out": None,
        "out_normed_rotary": None,
    }

    # --- Step 5: SiLU ---
    if fuse_silu:
        result["out_silu"] = F.silu(out)

    # --- Step 6: RMSNorm + Rotary (QKV projection mode) ---
    if norm_q is not None:
        # Split output into Q, K, V heads
        num_heads_qk = norm_q.shape[0] // head_dim if norm_q.ndim > 1 else N // head_dim
        # Assume output is [M, N] where N = num_q_heads * head_dim + num_k_heads * head_dim + num_v_heads * head_dim
        # For FLUX: Q and K each have some heads, V has the rest
        # Apply RMSNorm per head to Q and K portions
        out_normed = out.clone()

        # Apply RMSNorm to Q heads
        q_dim = N  # In the general case, Q/K/V dimensions are model-specific
        if norm_q is not None and norm_k is not None:
            # Reshape to heads for RMSNorm
            out_heads = out_normed.reshape(M, -1, head_dim)
            num_total_heads = out_heads.shape[1]
            # Determine Q/K split (assume Q and K have equal heads, rest is V)
            num_qk_heads = num_total_heads // 3  # Rough heuristic
            for h in range(num_qk_heads):
                out_heads[:, h, :] = rms_norm_reference(out_heads[:, h, :], norm_q)
            for h in range(num_qk_heads, 2 * num_qk_heads):
                out_heads[:, h, :] = rms_norm_reference(out_heads[:, h, :], norm_k)

            # Apply rotary embeddings to Q and K
            if rotary_emb_sin is not None and rotary_emb_cos is not None:
                for h in range(2 * num_qk_heads):
                    out_heads[:, h, :] = apply_rotary_emb_reference(
                        out_heads[:, h, :], rotary_emb_sin, rotary_emb_cos
                    )
            result["out_normed_rotary"] = out_heads.reshape(M, N)

    # --- Step 7: Output quantization (GELU + shift + smooth + quantize) ---
    if smooth_factor is not None:
        activated = gelu_reference(out)
        if not fp4:
            activated = activated + SHIFT_GELU
        smoothed = activated / smooth_factor.float().unsqueeze(0)

        if fp4:
            q_packed, q_scales = quantize_to_nvfp4(smoothed.to(torch.float32))
        else:
            q_packed, q_scales = quantize_to_int4(smoothed.to(torch.float32), unsigned=not fp4)
        result["out_quantized"] = q_packed
        result["out_oscales"] = q_scales

    # --- Step 8: LoRA down-projection for next layer ---
    if lora_down is not None:
        # Compute on the (possibly GELU-activated) output
        src = gelu_reference(out) if smooth_factor is not None else out
        result["lora_act_out"] = src.float() @ lora_down.float()  # (M, R)

    return result


def quantize_w4a4_act_fuse_lora_reference(
    input: torch.Tensor,
    lora_down: Optional[torch.Tensor] = None,
    smooth: Optional[torch.Tensor] = None,
    fuse_glu: bool = False,
    fp4: bool = False,
) -> dict[str, torch.Tensor]:
    """Reference implementation of activation quantization with fused LoRA.

    Quantizes 16-bit activations to 4-bit, optionally applying a smooth factor
    and computing the LoRA down-projection simultaneously.

    Computation:

    1. Apply smooth factor: ``x = input * smooth``
    2. (Optional) Apply GELU: ``x = gelu(x)``
    3. Quantize to INT4 or NVFP4
    4. Compute LoRA down-projection: ``lora_act = input @ lora_down``

    Parameters
    ----------
    input : torch.Tensor, shape ``(M, K)``, dtype ``float16 / bfloat16``
        Input activations.
    lora_down : torch.Tensor or None, shape ``(K, R)``
        LoRA down-projection weight.
    smooth : torch.Tensor or None, shape ``(K,)``
        Smooth factor applied element-wise to input channels.
    fuse_glu : bool
        If ``True``, apply GELU activation before quantization.
    fp4 : bool
        If ``True``, use NVFP4 quantization; else INT4.

    Returns
    -------
    dict with keys:
        ``"output"`` : torch.Tensor, shape ``(M, K // 2)``, dtype ``uint8``
            Packed quantized activations.
        ``"oscales"`` : torch.Tensor
            Per-group quantization scales.
        ``"lora_act_out"`` : torch.Tensor or None, shape ``(M, R)``, dtype ``float32``
            LoRA down-projection output.
    """
    x = input.float()

    # Step 1: Smooth factor
    if smooth is not None:
        x = x * smooth.float().unsqueeze(0)

    # Step 2: Optional GELU
    if fuse_glu:
        x = gelu_reference(x)

    # Step 3: Quantize
    if fp4:
        packed, oscales = quantize_to_nvfp4(x, group_size=NVFP4_GROUP_SIZE)
    else:
        packed, oscales = quantize_to_int4(x, group_size=INT4_GROUP_SIZE, unsigned=False)

    # Step 4: LoRA down-projection (on original smoothed input, not quantized)
    lora_act_out = None
    if lora_down is not None:
        lora_act_out = x @ lora_down.float()

    return {
        "output": packed,
        "oscales": oscales,
        "lora_act_out": lora_act_out,
    }


def attention_fp16_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference implementation of scaled dot-product attention.

    Parameters
    ----------
    q : torch.Tensor, shape ``(B, H, T_q, D)``
        Query tensor.
    k : torch.Tensor, shape ``(B, H, T_kv, D)``
        Key tensor.
    v : torch.Tensor, shape ``(B, H, T_kv, D)``
        Value tensor.
    scale : float or None
        Attention scale factor (default ``1 / sqrt(D)``).

    Returns
    -------
    torch.Tensor, shape ``(B, T_q, H * D)``
        Attention output, reshaped to ``(B, T_q, H * D)``.
    """
    B, H, T_q, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Scaled dot-product attention
    attn_weights = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_weights, v.float())  # (B, H, T_q, D)

    # Reshape: (B, H, T_q, D) → (B, T_q, H * D)
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T_q, H * D)
    return attn_out.to(q.dtype)


def gemv_awq_reference(
    in_feats: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    """Reference implementation of AWQ W4A16 GEMV / GEMM.

    Dequantizes the 4-bit weight matrix and performs a standard matrix
    multiplication with 16-bit activations.

    Parameters
    ----------
    in_feats : torch.Tensor, shape ``(M, K)``, dtype ``float16 / bfloat16``
        Input features.
    qweight : torch.Tensor, shape ``(N // 4, K // 2)``, dtype ``int32``
        Packed 4-bit weights.
    scales : torch.Tensor, shape ``(K // group_size, N)``
        Per-group scaling factors.
    zeros : torch.Tensor, shape ``(K // group_size, N)``
        Per-group zero points.
    group_size : int
        Quantization group size.

    Returns
    -------
    torch.Tensor, shape ``(M, N)``, dtype matching *in_feats*
        Output tensor.
    """
    weight = dequantize_awq(qweight, scales, zeros, group_size)
    out = in_feats.float() @ weight.float().T
    return out.to(in_feats.dtype)


def gemm_awq_reference(
    in_feats: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Reference implementation of AWQ GEMM.

    Identical to :func:`gemv_awq_reference` but kept as a separate entry point
    to mirror the CUDA API which distinguishes GEMV and GEMM paths.

    Parameters
    ----------
    in_feats : torch.Tensor, shape ``(M, K)``
        Input features.
    qweight : torch.Tensor, shape ``(N // 4, K // 2)``, dtype ``int32``
        Packed 4-bit weights.
    scales : torch.Tensor, shape ``(K // group_size, N)``
        Per-group scaling factors.
    zeros : torch.Tensor, shape ``(K // group_size, N)``
        Per-group zero points.
    group_size : int
        Quantization group size (default 128 for GEMM path).

    Returns
    -------
    torch.Tensor, shape ``(M, N)``
        Output tensor.
    """
    return gemv_awq_reference(in_feats, qweight, scales, zeros, group_size)


# ===================================================================
# 5. Fused Operators — Reference Implementations
# ===================================================================


def fused_gelu_mlp_reference(
    x: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor],
    fc2_weight: torch.Tensor,
    fc2_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reference implementation of fused GELU MLP.

    Computes ``fc2(gelu(fc1(x)))`` using standard PyTorch operations.

    Parameters
    ----------
    x : torch.Tensor, shape ``(B, S, C_in)``
        Input tensor.
    fc1_weight : torch.Tensor, shape ``(C_hidden, C_in)``
        First linear layer weight (dequantized).
    fc1_bias : torch.Tensor or None, shape ``(C_hidden,)``
        First linear layer bias.
    fc2_weight : torch.Tensor, shape ``(C_out, C_hidden)``
        Second linear layer weight (dequantized).
    fc2_bias : torch.Tensor or None, shape ``(C_out,)``
        Second linear layer bias.

    Returns
    -------
    torch.Tensor, shape ``(B, S, C_out)``
        Output tensor.
    """
    h = F.linear(x, fc1_weight, fc1_bias)
    h = gelu_reference(h)
    out = F.linear(h, fc2_weight, fc2_bias)
    return out


def fused_qkv_norm_rotary_reference(
    x: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: Optional[torch.Tensor],
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    rotary_emb_sin: torch.Tensor,
    rotary_emb_cos: torch.Tensor,
    num_heads: int,
    head_dim: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference implementation of fused QKV projection with RMSNorm and rotary.

    Computes:

    1. QKV projection: ``qkv = x @ proj_weight.T + proj_bias``
    2. Split into Q, K, V
    3. RMSNorm on Q and K (per head)
    4. Rotary embeddings on Q and K

    Parameters
    ----------
    x : torch.Tensor, shape ``(B, S, C_in)``
        Input tensor.
    proj_weight : torch.Tensor, shape ``(C_out, C_in)``
        QKV projection weight (dequantized).
    proj_bias : torch.Tensor or None, shape ``(C_out,)``
        QKV projection bias.
    norm_q_weight : torch.Tensor, shape ``(head_dim,)``
        RMSNorm weight for Q.
    norm_k_weight : torch.Tensor, shape ``(head_dim,)``
        RMSNorm weight for K.
    rotary_emb_sin : torch.Tensor, shape ``(S, head_dim // 2)``
        Sine of rotary embedding.
    rotary_emb_cos : torch.Tensor, shape ``(S, head_dim // 2)``
        Cosine of rotary embedding.
    num_heads : int
        Number of attention heads (for Q and K each).
    head_dim : int
        Dimension per attention head.
    eps : float
        RMSNorm epsilon.

    Returns
    -------
    torch.Tensor, shape ``(B, S, C_out)``
        Output with RMSNorm and rotary applied to Q and K.
    """
    B, S, C_in = x.shape
    # QKV projection
    qkv = F.linear(x, proj_weight, proj_bias)  # (B, S, C_out)
    C_out = qkv.shape[-1]

    # Split into Q, K, V heads
    q_dim = num_heads * head_dim
    k_dim = num_heads * head_dim
    v_dim = C_out - q_dim - k_dim

    q = qkv[..., :q_dim].reshape(B, S, num_heads, head_dim)
    k = qkv[..., q_dim : q_dim + k_dim].reshape(B, S, num_heads, head_dim)
    v_start = q_dim + k_dim
    v = qkv[..., v_start:]

    # RMSNorm per head
    q = rms_norm_reference(q, norm_q_weight, eps)
    k = rms_norm_reference(k, norm_k_weight, eps)

    # Rotary embeddings on Q and K
    sin = rotary_emb_sin.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D//2)
    cos = rotary_emb_cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D//2)
    q = apply_rotary_emb_reference(q, sin, cos)
    k = apply_rotary_emb_reference(k, sin, cos)

    # Reassemble
    q = q.reshape(B, S, q_dim)
    k = k.reshape(B, S, k_dim)
    out = torch.cat([q, k, v], dim=-1)
    return out
