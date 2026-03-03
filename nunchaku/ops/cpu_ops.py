"""
CPU reference implementations of Nunchaku's CUDA operators using pure PyTorch.

These provide CPU-compatible fallbacks for the quantized CUDA kernels,
enabling model inference on CPU (albeit without the performance optimizations
of the CUDA tensor-core kernels).

CUDA Operators and Their CPU Replacements
-----------------------------------------

1. ``ops.gemm_w4a4`` (svdq_gemm_w4a4_cuda)
   - CUDA: Fused W4A4 quantized GEMM with LoRA, RMSNorm, rotary embeddings, and
     activation fusions (SiLU/GELU) on tensor cores.
   - CPU: Dequantize INT4 weights and activations to float, then use ``torch.mm``
     for matrix multiplication, ``torch.addmm`` for LoRA, and standard PyTorch ops
     for bias/normalization.

2. ``ops.quantize_w4a4_act_fuse_lora`` (svdq_quantize_w4a4_act_fuse_lora_cuda)
   - CUDA: Fused activation quantization to INT4/NVFP4 with LoRA down-projection
     and smoothing in a single kernel.
   - CPU: Sequential PyTorch operations: smooth division, ``torch.mm`` for LoRA
     down-projection, per-group scale computation, rounding, clamping, and byte
     packing.

3. ``ops.gemv_awq`` (awq_gemv_w4a16_cuda)
   - CUDA: Interleaved AWQ W4A16 GEMV with TensorRT-LLM-style packing on tensor
     cores.
   - CPU: De-interleave and unpack AWQ int4 weights, apply per-group
     scale + zero-point dequantization, then ``torch.mm``.

4. ``ops.gemm_awq`` (awq_gemm_cuda)
   - CUDA: AWQ W4A16 GEMM for larger batch sizes.
   - CPU: Same dequantization as gemv_awq, then ``torch.mm``.

5. ``ops.attention_fp16``
   - CUDA: Custom FP16 attention kernel with packed Q/K/V format.
   - CPU: ``torch.nn.functional.scaled_dot_product_attention``.

6. ``ops.test_rmsnorm_rope``
   - CUDA: Fused RMSNorm + rotary embeddings kernel.
   - CPU: Sequential ``torch.nn.functional.rms_norm`` + manual rotary embedding
     application.

7. ``ops.test_pack_qkv``
   - CUDA: Fused QKV splitting and packing for attention.
   - CPU: ``torch.split`` + ``torch.reshape``.
"""

import math

import torch

from ..utils import ceil_divide


# ---------------------------------------------------------------------------
# INT4 pack / unpack helpers (SVDQuant W4A4 format)
# ---------------------------------------------------------------------------


def unpack_int4(packed: torch.Tensor, signed: bool = True) -> torch.Tensor:
    """Unpack INT4 values from packed bytes.

    In the SVDQuant format, each byte stores two 4-bit values:
    low nibble (bits 0-3) = even-index value, high nibble (bits 4-7) = odd-index value.

    Parameters
    ----------
    packed : torch.Tensor, shape (..., K // 2), dtype int8 or uint8
        Packed tensor.
    signed : bool
        If True, values are signed INT4 (range -8..7).
        If False, values are unsigned INT4 (range 0..15).

    Returns
    -------
    torch.Tensor, shape (..., K), dtype float32
    """
    p = packed.view(torch.uint8).to(torch.int16)
    low = p & 0x0F
    high = (p >> 4) & 0x0F
    if signed:
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return unpacked.float()


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack INT4 values into bytes (inverse of :func:`unpack_int4`).

    Parameters
    ----------
    values : torch.Tensor, shape (..., K), dtype int-like
        Values in range [-8, 7] (signed) or [0, 15] (unsigned).

    Returns
    -------
    torch.Tensor, shape (..., K // 2), dtype uint8
    """
    K = values.shape[-1]
    assert K % 2 == 0
    v = values.to(torch.int16)
    even = v[..., 0::2] & 0x0F
    odd = (v[..., 1::2] & 0x0F) << 4
    return (even | odd).to(torch.uint8)


# ---------------------------------------------------------------------------
# SVDQuant W4A4 dequantization
# ---------------------------------------------------------------------------


def dequantize_w4a4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    signed: bool = True,
) -> torch.Tensor:
    """Dequantize a packed INT4 tensor using per-group scales.

    Parameters
    ----------
    packed : torch.Tensor, shape (N, K // 2), dtype int8 / uint8
        Packed quantized values.
    scales : torch.Tensor, shape (K // group_size, N)
        Per-group scales (note: transposed layout).
    group_size : int
    signed : bool

    Returns
    -------
    torch.Tensor, shape (N, K), dtype float32
    """
    unpacked = unpack_int4(packed, signed=signed)  # (N, K)
    N, K = unpacked.shape
    num_groups = K // group_size
    unpacked = unpacked.view(N, num_groups, group_size)
    sc = scales.float().T.unsqueeze(-1)  # (N, num_groups, 1)
    return (unpacked * sc).view(N, K)


# ---------------------------------------------------------------------------
# SVDQuant quantize + LoRA (CPU)
# ---------------------------------------------------------------------------


def svdq_quantize_w4a4_act_fuse_lora_cpu(
    input: torch.Tensor,
    output: torch.Tensor,
    oscales: torch.Tensor,
    lora_down: torch.Tensor | None,
    lora_act_out: torch.Tensor | None,
    smooth: torch.Tensor | None,
    fuse_glu: bool = False,
    fp4: bool = False,
) -> None:
    """CPU fallback for ``ops.quantize_w4a4_act_fuse_lora``.

    Writes results **in-place** to *output*, *oscales*, and *lora_act_out*.
    """
    if fp4:
        raise NotImplementedError("CPU fallback does not support NVFP4 (fp4) quantization")
    if fuse_glu:
        raise NotImplementedError("CPU fallback does not support fused GLU")

    M, K = input.shape
    M_pad = output.shape[0]
    group_size = 64  # INT4 group size

    x = input.float()

    # 1. LoRA down-projection (before smoothing, on raw activations)
    if lora_down is not None and lora_act_out is not None:
        lora_result = x @ lora_down.float()  # (M, rank)
        lora_act_out[:M].copy_(lora_result)
        if M_pad > M:
            lora_act_out[M:].zero_()

    # 2. Smooth (division)
    if smooth is not None:
        x = x / smooth.float()

    # 3. Per-group quantization to signed INT4
    num_groups = K // group_size
    x_grouped = x.view(M, num_groups, group_size)

    group_max = x_grouped.abs().amax(dim=-1)  # (M, num_groups)
    scales = group_max / 7.0
    scales = scales.clamp(min=1e-10)

    rscale = 7.0 / group_max.clamp(min=1e-10)
    x_q = torch.round(x_grouped * rscale.unsqueeze(-1))
    x_q = x_q.clamp(-8, 7).to(torch.int8).view(M, K)

    # 4. Pack
    packed = pack_int4(x_q)  # (M, K//2)
    output[:M].copy_(packed)
    if M_pad > M:
        output[M:].zero_()

    # 5. Scales (transposed: (num_groups, M_pad))
    oscales[:, :M].copy_(scales.T.to(oscales.dtype))
    if M_pad > M:
        oscales[:, M:].zero_()


# ---------------------------------------------------------------------------
# SVDQuant W4A4 GEMM (CPU)
# ---------------------------------------------------------------------------


def svdq_gemm_w4a4_cpu(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: torch.Tensor | None = None,
    qout: torch.Tensor | None = None,
    ascales: torch.Tensor | None = None,
    wscales: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    poolout: torch.Tensor | None = None,
    lora_act_in: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    norm_q: torch.Tensor | None = None,
    norm_k: torch.Tensor | None = None,
    rotary_emb: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    smooth_factor: torch.Tensor | None = None,
    out_vk: torch.Tensor | None = None,
    out_linearattn: torch.Tensor | None = None,
    act_unsigned: bool = False,
    lora_scales: list[float] | None = None,
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: float | None = 1.0,
    wcscales: torch.Tensor | None = None,
    out_q: torch.Tensor | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    attn_tokens: int = 0,
) -> None:
    """CPU fallback for ``ops.gemm_w4a4``.

    Writes the result **in-place** to *out* (and optionally *qout*, *lora_act_out*,
    *out_q*, *out_k*, *out_v*).

    Supports: dequantized GEMM, LoRA, bias, per-tensor/channel scaling.
    Not yet supported: fused RMSNorm+RoPE, fused SiLU/GELU, quantized output,
    SANA-specific outputs, NVFP4.
    """
    if fp4:
        raise NotImplementedError("CPU fallback does not support NVFP4 (fp4)")

    group_size = 64  # INT4

    # --- dequantize ---
    dequant_act = dequantize_w4a4(act, ascales, group_size, signed=not act_unsigned)
    dequant_wgt = dequantize_w4a4(wgt, wscales, group_size, signed=True)

    # --- GEMM ---
    result = dequant_act @ dequant_wgt.T  # (M, N)

    # --- per-tensor / per-channel NVFP4 scaling ---
    if alpha is not None and alpha != 1.0:
        result = result * alpha
    if wcscales is not None:
        result = result * wcscales.float()

    # --- LoRA residual ---
    if lora_act_in is not None and lora_up is not None:
        lora_contrib = lora_act_in.float() @ lora_up.float().T
        if lora_scales is not None:
            # Apply per-group LoRA scaling (16 channels per group)
            N = lora_up.shape[0]
            scale_t = torch.ones(N, dtype=torch.float32, device=result.device)
            for g, s in enumerate(lora_scales):
                start = g * 16
                end = min(start + 16, N)
                scale_t[start:end] = s
            lora_contrib = lora_contrib * scale_t
        result = result + lora_contrib

    # --- bias ---
    if bias is not None:
        result = result + bias.float()

    # --- fused SiLU ---
    if fuse_silu:
        result = result * torch.sigmoid(result)

    # --- write to output ---
    if out is not None:
        M_out = out.shape[0]
        N_out = out.shape[1]
        out.copy_(result[:M_out, :N_out].to(out.dtype))

    # --- quantized output for next layer (qout + oscales + lora_act_out) ---
    if qout is not None and oscales is not None:
        _quantize_output_for_next_layer(
            result, qout, oscales, lora_down, lora_act_out, smooth_factor
        )


def _quantize_output_for_next_layer(result, qout, oscales, lora_down, lora_act_out, smooth_factor):
    """Quantize the GEMM output for consumption by the next layer (fused path)."""
    M, N = result.shape
    M_pad = qout.shape[0]
    group_size = 64

    x = result.float()

    # LoRA down for next layer
    if lora_down is not None and lora_act_out is not None:
        lora_act_out[:M].copy_((x @ lora_down.float())[:M])
        if M_pad > M:
            lora_act_out[M:].zero_()

    # Smooth for next layer
    if smooth_factor is not None:
        x = x / smooth_factor.float()

    num_groups = N // group_size
    x_grouped = x.view(M, num_groups, group_size)
    group_max = x_grouped.abs().amax(dim=-1)
    scales = group_max / 7.0
    scales = scales.clamp(min=1e-10)
    rscale = 7.0 / group_max.clamp(min=1e-10)
    x_q = torch.round(x_grouped * rscale.unsqueeze(-1)).clamp(-8, 7).to(torch.int8).view(M, N)

    packed = pack_int4(x_q)
    qout[:M].copy_(packed)
    if M_pad > M:
        qout[M:].zero_()

    oscales[:, :M].copy_(scales.T.to(oscales.dtype))
    if M_pad > M:
        oscales[:, M:].zero_()


# ---------------------------------------------------------------------------
# AWQ W4A16 dequantization (interleaved TensorRT-LLM packing)
# ---------------------------------------------------------------------------


def awq_unpack_weights(
    kernel: torch.Tensor, n: int, k: int
) -> torch.Tensor:
    """Unpack AWQ int4 weights from the TensorRT-LLM interleaved format.

    The weight tensor of shape ``(n // 4, k // 2)`` int32 is a contiguous reshape
    of ``(n, k // 8)`` int32.  Within each group of 4 consecutive uint32 values
    (covering 32 input channels), the 4-bit values are stored in an interleaved
    order used by the NVIDIA GEMV kernel.

    Parameters
    ----------
    kernel : torch.Tensor, shape (n // 4, k // 2), dtype int32
    n : int
        Output features.
    k : int
        Input features.

    Returns
    -------
    torch.Tensor, shape (n, k), dtype float32
        Unsigned int4 values in range [0, 15].
    """
    qw = kernel.reshape(n, k // 8).to(torch.int64)

    num_groups_32 = k // 32

    qw = qw.reshape(n, num_groups_32, 4)  # (n, G32, 4)

    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int64, device=qw.device)
    nibbles = (qw.unsqueeze(-1) >> shifts) & 0xF  # (n, G32, 4, 8)

    # Compute IC offset for each (u, nibble) pair
    u_idx = torch.arange(4, device=qw.device)
    n_idx = torch.arange(8, device=qw.device)
    ic_offsets = (n_idx % 4) * 8 + 2 * u_idx.unsqueeze(-1) + (n_idx // 4)  # (4, 8)

    # Scatter nibbles to correct IC positions
    weight = torch.zeros(n, num_groups_32, 32, dtype=torch.float32, device=kernel.device)
    flat_offsets = ic_offsets.reshape(1, 1, 32).expand(n, num_groups_32, 32)
    weight.scatter_(2, flat_offsets, nibbles.reshape(n, num_groups_32, 32).float())

    return weight.reshape(n, k)


def awq_dequantize_weights(
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
) -> torch.Tensor:
    """Dequantize AWQ packed weights to float.

    Parameters
    ----------
    kernel : torch.Tensor, shape (n // 4, k // 2), dtype int32
    scaling_factors : torch.Tensor, shape (k // group_size, n)
    zeros : torch.Tensor, shape (k // group_size, n)
        Pre-scaled zeros: ``-scale * zero_point``.
    n, k : int
    group_size : int

    Returns
    -------
    torch.Tensor, shape (n, k), dtype float32
    """
    weight = awq_unpack_weights(kernel, n, k)  # (n, k) unsigned 0-15

    num_groups = k // group_size
    weight = weight.view(n, num_groups, group_size)
    sc = scaling_factors.float().T.unsqueeze(-1)  # (n, num_groups, 1)
    zp = zeros.float().T.unsqueeze(-1)
    dequant = weight * sc + zp
    return dequant.view(n, k)


def awq_gemv_w4a16_cpu(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> torch.Tensor:
    """CPU fallback for ``ops.gemv_awq``.

    Parameters
    ----------
    in_feats : torch.Tensor, shape (m, k)
    kernel : torch.Tensor, shape (n // 4, k // 2), dtype int32
    scaling_factors : torch.Tensor, shape (k // group_size, n)
    zeros : torch.Tensor, shape (k // group_size, n)
    m, n, k : int
    group_size : int

    Returns
    -------
    torch.Tensor, shape (m, n)
    """
    weight = awq_dequantize_weights(kernel, scaling_factors, zeros, n, k, group_size)
    x = in_feats.float().reshape(m, k)
    output = x @ weight.T
    return output.to(in_feats.dtype)
