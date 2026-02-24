"""
Python wrappers for Nunchaku's high-performance quantized GEMM (General Matrix-Matrix Multiplication) kernels.

On CUDA devices the wrapper delegates to the native C++/CUDA extension.
On other devices (e.g. Intel XPU) a pure-PyTorch fallback is used
automatically so that the rest of the model code stays device-agnostic.
"""

import math

import torch


def _is_cuda_backend_available() -> bool:
    """Return True when the native CUDA extension can be loaded."""
    try:
        from .._C import ops as _ops  # noqa: F401

        return True
    except Exception:
        return False


def svdq_gemm_w4a4_cuda(
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
):
    """
    Quantized GEMM using SVDQuant W4A4 CUDA kernel, with support for LoRA, rotary embeddings, normalization, and fused activations.

    Parameters
    ----------
    act : torch.Tensor, shape (M, K // 2), dtype int8
        Packed input activations.
    wgt : torch.Tensor, shape (N, K // 2), dtype int8
        Packed quantized weights.
    out : torch.Tensor or None, shape (M, N), dtype float16 or bfloat16, optional
        Output tensor for the linear layer.
    qout : torch.Tensor or None, shape (M, N // 2), dtype int8, optional
        Packed quantized input for the next layer.
    ascales : torch.Tensor or None, shape (K // G, M), dtype float16/bfloat16 (INT4) or float8_e4m3fn (NVFP4), optional
        Activation scales.
    wscales : torch.Tensor or None, shape (K // G, N), dtype float16/bfloat16 (INT4) or float8_e4m3fn (NVFP4), optional
        Weight scales.
    oscales : torch.Tensor or None, shape (N // G, M), dtype float16/bfloat16 (INT4) or float8_e4m3fn (NVFP4), optional
        Output scales.
    poolout : torch.Tensor or None, optional
        Reserved for future use.
    lora_act_in : torch.Tensor or None, shape (M, R), dtype float32, optional
        LoRA down-projection activations.
    lora_up : torch.Tensor or None, shape (N, R), dtype float16 or bfloat16, optional
        Packed LoRA up-projection weights.
    lora_down : torch.Tensor or None, shape (N, R), dtype float16 or bfloat16, optional
        Packed LoRA down-projection weights for the next layer.
    lora_act_out : torch.Tensor or None, shape (M, R), dtype float32, optional
        Output for LoRA down-projection in the next layer.
    norm_q : torch.Tensor or None, shape (HEAD_DIM,), dtype float16 or bfloat16, optional
        Query RMS normalization.
    norm_k : torch.Tensor or None, shape (HEAD_DIM,), dtype float16 or bfloat16, optional
        Key RMS normalization.
    rotary_emb : torch.Tensor or None, shape (M, HEAD_DIM // 2, 2, 2), dtype float32, optional
        Packed rotary embeddings.
    bias : torch.Tensor or None, shape (N,), dtype float16 or bfloat16, optional
        Bias tensor.
    smooth_factor : torch.Tensor or None, shape (N,), dtype float16 or bfloat16, optional
        Smoothing factor for quantization in the next layer.
    out_vk : torch.Tensor or None, optional
        Used only in SANA. Leave as None.
    out_linearattn : torch.Tensor or None, optional
        Used only in SANA. Leave as None.
    act_unsigned : bool, default=False
        If True, activations are unsigned (e.g., after GeLU, shifted by 0.171875). This is only used for INT4 to enable unsigned INT4 activation quantization for better quantization quality.
    lora_scales : list of float or None, optional
        Per-group LoRA scaling factors (16 channels per group). Defaults to 1.0 per group.
    fuse_silu : bool, default=False
        If True, fuse SiLU activation.
    fp4 : bool, default=False
        If True, use 4-bit floating point quantization (NVFP4).
    alpha : float or None, default=1.0
        Per-tensor scaling factor for NVFP4.
    wcscales : torch.Tensor or None, shape (N,), dtype float8_e4m3fn, optional
        Per-channel scaling for NVFP4.
    out_q : torch.Tensor or None, shape (B, H, M, D), dtype int8, optional
        Packed quantized Q for attention (used in ``nunchaku-fp16`` attention).
    out_k : torch.Tensor or None, shape (B, H, M, D), dtype int8, optional
        Packed quantized K for attention (used in ``nunchaku-fp16`` attention).
    out_v : torch.Tensor or None, shape (B, H, M, D), dtype int8, optional
        Packed quantized V for attention (used in ``nunchaku-fp16`` attention).
    attn_tokens : int, default=0
        Number of attention tokens.

    Returns
    -------
    None
        Results are written in-place to the provided output tensors.

    Notes
    -----
    Notations:

    - M: batch size (input tokens)
    - K: input channels (feature dimension)
    - N: output channels
    - G: group size (64 for INT4, 16 for NVFP4)
    - R: LoRA rank
    - B: batch size for attention
    - H: number of heads
    - D: head dimension
    """
    # Determine the device of the primary tensor to decide backend.
    _device_type = act.device.type if act is not None else "cuda"

    if _device_type == "cuda" and _is_cuda_backend_available():
        from .._C import ops

        if lora_scales is None:
            rank = lora_up.shape[1]
            lora_scales = [1.0] * math.ceil(rank / 16)
        if alpha is None:
            alpha = 1.0
        ops.gemm_w4a4(
            act,
            wgt,
            out,
            qout,
            ascales,
            wscales,
            oscales,
            poolout,
            lora_act_in,
            lora_up,
            lora_down,
            lora_act_out,
            norm_q,
            norm_k,
            rotary_emb,
            bias,
            smooth_factor,
            out_vk,
            out_linearattn,
            act_unsigned,
            lora_scales,
            fuse_silu,
            fp4,
            alpha,
            wcscales,
            out_q,
            out_k,
            out_v,
            attn_tokens,
        )
    else:
        from .torch_fallback import svdq_gemm_w4a4_fallback

        if lora_scales is None and lora_up is not None:
            rank = lora_up.shape[1]
            lora_scales = [1.0] * math.ceil(rank / 16)
        if alpha is None:
            alpha = 1.0
        svdq_gemm_w4a4_fallback(
            act=act,
            wgt=wgt,
            out=out,
            qout=qout,
            ascales=ascales,
            wscales=wscales,
            oscales=oscales,
            poolout=poolout,
            lora_act_in=lora_act_in,
            lora_up=lora_up,
            lora_down=lora_down,
            lora_act_out=lora_act_out,
            norm_q=norm_q,
            norm_k=norm_k,
            rotary_emb=rotary_emb,
            bias=bias,
            smooth_factor=smooth_factor,
            out_vk=out_vk,
            out_linearattn=out_linearattn,
            act_unsigned=act_unsigned,
            lora_scales=lora_scales,
            fuse_silu=fuse_silu,
            fp4=fp4,
            alpha=alpha,
            wcscales=wcscales,
            out_q=out_q,
            out_k=out_k,
            out_v=out_v,
            attn_tokens=attn_tokens,
        )
