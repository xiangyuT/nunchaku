"""
Triton kernel implementations for Nunchaku quantized operations.

These kernels provide a portable, high-performance alternative to the native
CUDA C++ kernels.  They can run on any backend supported by Triton (NVIDIA
CUDA and Intel XPU).

The kernels are used automatically when Triton is installed and the native
CUDA extension is not available, or when explicitly selected via the
``NUNCHAKU_BACKEND=triton`` environment variable.

.. note::
    These kernels operate on the same packed INT4/UINT8 weight format used by
    the rest of nunchaku.  They are *not* numerically identical to the native
    CUDA kernels but should produce results of comparable quality.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def is_triton_available() -> bool:
    """Return ``True`` when the Triton compiler is importable."""
    return HAS_TRITON


# ── Triton kernel: W4A4 dequantize + GEMM ────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _dequant_gemm_w4a4_kernel(
        # Pointers
        act_ptr,        # packed activations  [M, K//2] int8
        wgt_ptr,        # packed weights      [N, K//2] int8
        ascales_ptr,    # activation scales   [K//G, M] fp16/bf16
        wscales_ptr,    # weight scales       [K//G, N] fp16/bf16
        out_ptr,        # output              [M, N]    fp16/bf16
        bias_ptr,       # bias                [N]       fp16/bf16 (or null)
        # Dimensions
        M, N, K: tl.constexpr,
        # Strides (in elements)
        stride_am, stride_ak,   # act strides
        stride_wn, stride_wk,   # wgt strides
        stride_asm, stride_ask, # ascales strides (K//G, M)
        stride_wsn, stride_wsk, # wscales strides (K//G, N)
        stride_om, stride_on,   # out strides
        # Meta
        GROUP_SIZE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Triton kernel for W4A4 dequantized GEMM.

        Each program instance computes a BLOCK_M × BLOCK_N tile of the output.
        Activations and weights are stored as packed INT4 pairs (two values per
        int8 byte) and are unpacked + dequantized on the fly.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K is the *unpacked* feature dimension; the packed dimension is K//2
        K_PACKED = K // 2
        num_groups = K // GROUP_SIZE

        for k_start in range(0, K_PACKED, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)

            # ── load packed activations [BLOCK_M, BLOCK_K] ──
            mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K_PACKED)
            a_packed = tl.load(
                act_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=mask_a, other=0
            ).to(tl.int8)

            # unpack: low nibble and high nibble
            a_lo = ((a_packed << 4) >> 4)  # sign-extend lower nibble
            a_hi = (a_packed >> 4)          # sign-extend upper nibble

            # interleave to [BLOCK_M, BLOCK_K * 2]
            a_unpacked_lo = a_lo.to(tl.float32)
            a_unpacked_hi = a_hi.to(tl.float32)

            # ── load packed weights [BLOCK_N, BLOCK_K] ──
            mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K_PACKED)
            w_packed = tl.load(
                wgt_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=mask_w, other=0
            ).to(tl.int8)

            w_lo = ((w_packed << 4) >> 4)
            w_hi = (w_packed >> 4)
            w_unpacked_lo = w_lo.to(tl.float32)
            w_unpacked_hi = w_hi.to(tl.float32)

            # ── dequantize with scales ──
            # For each packed element at position k in [0, K//2),
            # the unpacked positions are 2*k (lo) and 2*k+1 (hi).
            # The group index for unpacked position p is p // GROUP_SIZE.
            unpacked_offs_lo = 2 * offs_k  # unpacked positions for low nibble
            unpacked_offs_hi = 2 * offs_k + 1

            group_lo = unpacked_offs_lo // GROUP_SIZE
            group_hi = unpacked_offs_hi // GROUP_SIZE

            # load activation scales [BLOCK_K] for each group and M
            # ascales layout: [K//G, M]
            mask_as_lo = (group_lo[None, :] < num_groups) & (offs_m[:, None] < M)
            as_lo = tl.load(
                ascales_ptr + group_lo[None, :] * stride_ask + offs_m[:, None] * stride_asm,
                mask=mask_as_lo, other=1.0
            ).to(tl.float32)

            mask_as_hi = (group_hi[None, :] < num_groups) & (offs_m[:, None] < M)
            as_hi = tl.load(
                ascales_ptr + group_hi[None, :] * stride_ask + offs_m[:, None] * stride_asm,
                mask=mask_as_hi, other=1.0
            ).to(tl.float32)

            # load weight scales
            mask_ws_lo = (group_lo[None, :] < num_groups) & (offs_n[:, None] < N)
            ws_lo = tl.load(
                wscales_ptr + group_lo[None, :] * stride_wsk + offs_n[:, None] * stride_wsn,
                mask=mask_ws_lo, other=1.0
            ).to(tl.float32)

            mask_ws_hi = (group_hi[None, :] < num_groups) & (offs_n[:, None] < N)
            ws_hi = tl.load(
                wscales_ptr + group_hi[None, :] * stride_wsk + offs_n[:, None] * stride_wsn,
                mask=mask_ws_hi, other=1.0
            ).to(tl.float32)

            # dequantize: value * act_scale * wgt_scale
            a_deq_lo = a_unpacked_lo * as_lo   # [BLOCK_M, BLOCK_K]
            a_deq_hi = a_unpacked_hi * as_hi
            w_deq_lo = w_unpacked_lo * ws_lo   # [BLOCK_N, BLOCK_K]
            w_deq_hi = w_unpacked_hi * ws_hi

            # accumulate: out[m, n] += sum_k( a_deq[m,k] * w_deq[n,k] )
            acc += tl.dot(a_deq_lo, tl.trans(w_deq_lo))
            acc += tl.dot(a_deq_hi, tl.trans(w_deq_hi))

        # ── add bias ──
        if HAS_BIAS:
            bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += bias_vals[None, :]

        # ── store output ──
        mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(tl.float16),
            mask=mask_out,
        )

    @triton.jit
    def _quantize_act_kernel(
        # Pointers
        x_ptr,            # input         [M, K]  fp16/bf16
        out_ptr,          # packed output [M, K//2] uint8
        scales_ptr,       # scales        [K//G, M] fp16/bf16
        lora_down_ptr,    # LoRA down     [K, R]  fp16/bf16
        lora_out_ptr,     # LoRA output   [M, R]  fp32
        smooth_ptr,       # smooth factor [K]     fp16/bf16 (or null)
        # Dimensions
        M, K, R: tl.constexpr,
        # Strides
        stride_xm, stride_xk,
        stride_om, stride_ok,
        stride_sm, stride_sk,
        stride_ldm, stride_ldk,
        stride_lom, stride_lok,
        # Meta
        GROUP_SIZE: tl.constexpr,
        HAS_SMOOTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Triton kernel for INT4 activation quantization with optional smoothing.

        Each program computes quantization for a BLOCK_M × K tile of the input.
        """
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

        num_groups = K // GROUP_SIZE

        for g in range(num_groups):
            offs_k = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

            # load input tile [BLOCK_M, GROUP_SIZE]
            mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask, other=0.0,
            ).to(tl.float32)

            # optional smooth
            if HAS_SMOOTH:
                smooth = tl.load(smooth_ptr + offs_k, mask=offs_k < K, other=1.0).to(tl.float32)
                x = x * smooth[None, :]

            # per-row absmax within this group
            absmax = tl.max(tl.abs(x), axis=1)  # [BLOCK_M]
            absmax = tl.maximum(absmax, 1e-10)
            scale = absmax / 7.0  # [BLOCK_M]

            # quantize to int4 range [-8, 7]
            x_scaled = x / absmax[:, None] * 7.0

            # clamp and round
            x_q = tl.minimum(tl.maximum(x_scaled + 0.5, -8.0), 7.0)
            # floor instead of round for simplicity
            x_q = (x_q - 0.5).to(tl.int8)

            # store scales [K//G, M] layout
            scale_mask = offs_m < M
            tl.store(
                scales_ptr + g * stride_sk + offs_m * stride_sm,
                scale.to(tl.float16),
                mask=scale_mask,
            )

            # pack pairs of int4 into uint8
            # even indices in low nibble, odd indices in high nibble
            for p in range(0, GROUP_SIZE, 2):
                lo = tl.load(
                    x_ptr + offs_m * stride_xm + (g * GROUP_SIZE + p) * stride_xk,
                    mask=offs_m < M, other=0,
                ).to(tl.int8)
                hi = tl.load(
                    x_ptr + offs_m * stride_xm + (g * GROUP_SIZE + p + 1) * stride_xk,
                    mask=offs_m < M, other=0,
                ).to(tl.int8)
                packed = ((hi << 4) | (lo & 0xF)).to(tl.uint8)
                tl.store(
                    out_ptr + offs_m * stride_om + (g * GROUP_SIZE // 2 + p // 2) * stride_ok,
                    packed,
                    mask=offs_m < M,
                )


def triton_dequant_gemm_w4a4(
    act: Tensor,
    wgt: Tensor,
    ascales: Tensor,
    wscales: Tensor,
    out: Tensor | None = None,
    bias: Tensor | None = None,
    group_size: int = 64,
) -> Tensor:
    """Triton-based W4A4 dequantized GEMM.

    Parameters
    ----------
    act : Tensor, shape (M, K // 2), dtype int8
        Packed INT4 activations.
    wgt : Tensor, shape (N, K // 2), dtype int8
        Packed INT4 weights.
    ascales : Tensor, shape (K // group_size, M)
        Activation scales.
    wscales : Tensor, shape (K // group_size, N)
        Weight scales.
    out : Tensor or None, shape (M, N)
        Output tensor. Allocated if None.
    bias : Tensor or None, shape (N,)
        Optional bias.
    group_size : int
        Quantization group size (default: 64).

    Returns
    -------
    Tensor, shape (M, N)
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed. Install with: pip install triton")

    M, K_packed = act.shape
    N = wgt.shape[0]
    K = K_packed * 2

    if out is None:
        out = torch.empty(M, N, dtype=torch.float16, device=act.device)

    # Choose block sizes
    BLOCK_M = min(32, M) if M < 32 else 32
    BLOCK_N = min(32, N) if N < 32 else 32
    BLOCK_K = min(32, K_packed) if K_packed < 32 else 32

    # Ensure block sizes are powers of 2 for Triton
    for size in [16, 32, 64]:
        if BLOCK_M <= size:
            BLOCK_M = size
            break
    for size in [16, 32, 64]:
        if BLOCK_N <= size:
            BLOCK_N = size
            break
    for size in [16, 32, 64]:
        if BLOCK_K <= size:
            BLOCK_K = size
            break

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _dequant_gemm_w4a4_kernel[grid](
        act, wgt, ascales, wscales, out,
        bias if bias is not None else act,  # dummy pointer when no bias
        M, N, K,
        act.stride(0), act.stride(1),
        wgt.stride(0), wgt.stride(1),
        ascales.stride(1), ascales.stride(0),
        wscales.stride(1), wscales.stride(0),
        out.stride(0), out.stride(1),
        GROUP_SIZE=group_size,
        HAS_BIAS=(bias is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out


def triton_quantize_w4a4_act(
    input: Tensor,
    smooth: Tensor | None = None,
    lora_down: Tensor | None = None,
    fp4: bool = False,
    pad_size: int = 256,
) -> tuple[Tensor, Tensor, Tensor]:
    """Triton-based activation quantization to INT4.

    This is a simplified version that uses PyTorch for the quantization
    (since element-wise quantization does not benefit much from Triton)
    but is structured to integrate with the Triton dispatch path.

    Parameters
    ----------
    input : Tensor, shape (M, K), dtype float16/bfloat16
        Input activations.
    smooth : Tensor or None, shape (K,)
        Optional smoothing factor.
    lora_down : Tensor or None, shape (K, R)
        LoRA down-projection weights.
    fp4 : bool
        If True, use group_size=16; else group_size=64.
    pad_size : int
        Pad batch to multiples of this value.

    Returns
    -------
    tuple of (output, oscales, lora_act_out)
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed. Install with: pip install triton")

    batch_size, channels = input.shape
    group_size = 16 if fp4 else 64
    rank = lora_down.shape[1] if lora_down is not None else 1
    batch_size_pad = ((batch_size + pad_size - 1) // pad_size) * pad_size

    x = input.clone()
    if smooth is not None:
        x = x * smooth.unsqueeze(0)

    # Per-group absmax quantization using PyTorch
    # (element-wise ops don't benefit much from custom Triton)
    num_groups = channels // group_size
    x_grouped = x.reshape(batch_size, num_groups, group_size)
    absmax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax.squeeze(-1) / 7.0
    x_quant = torch.clamp(torch.round(x_grouped / absmax * 7.0), -8, 7).to(torch.int8)
    x_quant_flat = x_quant.reshape(batch_size, channels)

    # Pack pairs of int4 into uint8
    packed = (x_quant_flat[:, 1::2] << 4) | (x_quant_flat[:, ::2] & 0xF)
    packed = packed.to(torch.uint8)

    output = torch.zeros(batch_size_pad, channels // 2, dtype=torch.uint8, device=input.device)
    output[:batch_size].copy_(packed)

    if fp4:
        oscales = torch.zeros(channels // group_size, batch_size_pad, dtype=torch.float8_e4m3fn, device=input.device)
    else:
        oscales = torch.zeros(channels // group_size, batch_size_pad, dtype=input.dtype, device=input.device)
    oscales[:, :batch_size] = scales.permute(1, 0).to(oscales.dtype)

    lora_act_out = torch.zeros(batch_size_pad, rank, dtype=torch.float32, device=input.device)
    if lora_down is not None and rank > 0:
        lora_act_out[:batch_size] = (x[:batch_size].float() @ lora_down.float())

    return output, oscales, lora_act_out
