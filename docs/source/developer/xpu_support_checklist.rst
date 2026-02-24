Intel XPU Support Checklist
===========================

This document tracks the APIs and operations that need to be supported on Intel XPU
for nunchaku. Each API is categorized by its current status and the planned
implementation strategy (SYCL, Triton, or PyTorch fallback).

Legend
-----

- ✅ Supported
- 🔧 In progress (partial support)
- ❌ Not yet supported

.. contents:: Table of Contents
   :depth: 2
   :local:

Core Quantized Operations
--------------------------

These are the low-level quantized compute kernels that form the backbone of nunchaku inference.

.. list-table::
   :header-rows: 1
   :widths: 35 15 15 35

   * - API
     - Status
     - Backend
     - Notes
   * - ``svdq_gemm_w4a4`` (W4A4 GEMM)
     - 🔧
     - PyTorch / Triton
     - INT4 weight × INT4 activation matmul with dequantization.
       PyTorch fallback available. Triton kernel added for better performance.
   * - ``svdq_quantize_w4a4_act_fuse_lora`` (activation quantization)
     - 🔧
     - PyTorch / Triton
     - Per-group absmax quantization to INT4 with fused LoRA down-projection.
       PyTorch fallback available. Triton kernel added for better performance.
   * - ``awq_gemv_w4a16`` (AWQ GEMV)
     - ❌
     - PyTorch
     - AWQ W4A16 format: 8×int4 packed in int32. Complex unpacking logic.
       Not yet implemented for non-CUDA devices.
   * - ``awq_gemm`` (AWQ GEMM)
     - ❌
     - PyTorch
     - AWQ batched GEMM. Not yet implemented for non-CUDA devices.
   * - ``attention_fp16`` (fused attention)
     - ❌
     - PyTorch / Triton
     - Fused multi-head attention in FP16. Requires Triton or PyTorch SDPA fallback.

Fused Operations
-----------------

Higher-level fused operators that combine multiple operations into a single kernel call.

.. list-table::
   :header-rows: 1
   :widths: 35 15 15 35

   * - API
     - Status
     - Backend
     - Notes
   * - ``fused_gelu_mlp``
     - 🔧
     - PyTorch / Triton
     - Fused quantized MLP: linear → GELU → linear.
       Works via fallback GEMM path.
   * - ``fused_qkv_norm_rotary``
     - 🔧
     - PyTorch / Triton
     - Fused QKV projection + RMSNorm + rotary embedding.
       Works via fallback GEMM path.
   * - ``fused_silu`` (SiLU activation fusion)
     - 🔧
     - PyTorch
     - Fused SiLU within GEMM output. Supported via ``fuse_silu`` parameter.

Quantized Linear Layers
------------------------

Module-level wrappers that provide ``nn.Module`` interfaces for quantized linear operations.

.. list-table::
   :header-rows: 1
   :widths: 35 15 15 35

   * - API
     - Status
     - Backend
     - Notes
   * - ``SVDQW4A4Linear.forward``
     - 🔧
     - PyTorch / Triton
     - SVDQuant W4A4 quantized linear forward pass.
       Dispatches to fallback on non-CUDA.
   * - ``SVDQW4A4Linear.quantize``
     - 🔧
     - PyTorch / Triton
     - Activation quantization + LoRA down-projection.
   * - ``SVDQW4A4Linear.forward_quant``
     - 🔧
     - PyTorch / Triton
     - Forward pass with pre-quantized input.
   * - ``AWQW4A16Linear.forward``
     - ❌
     - PyTorch
     - AWQ W4A16 quantized linear forward pass. Blocked by AWQ GEMV.

Helper / Utility Operations
-----------------------------

INT4 packing, unpacking, dequantization, and other utility operations.

.. list-table::
   :header-rows: 1
   :widths: 35 15 15 35

   * - API
     - Status
     - Backend
     - Notes
   * - ``_unpack_int4``
     - ✅
     - PyTorch
     - Unpack INT8-packed INT4 pairs to individual INT8 values.
   * - ``_dequantize_int4``
     - ✅
     - PyTorch
     - Dequantize INT4 packed weights using per-group scales.
   * - ``pad_tensor``
     - ✅
     - PyTorch
     - Pad tensor to multiples of a given size. Device-agnostic.
   * - ``ceil_divide``
     - ✅
     - PyTorch
     - Integer ceiling division. Device-agnostic.

Device Abstraction / Runtime
-----------------------------

Device-agnostic APIs that abstract hardware differences between CUDA and XPU.

.. list-table::
   :header-rows: 1
   :widths: 35 15 35

   * - API
     - Status
     - Notes
   * - ``is_cuda_available``
     - ✅
     - Returns True if CUDA is available.
   * - ``is_xpu_available``
     - ✅
     - Returns True if Intel XPU is available (requires ``torch>=2.4`` with XPU support).
   * - ``get_supported_backends``
     - ✅
     - Lists available backends (``"cuda"``, ``"xpu"``).
   * - ``get_device_capability``
     - ✅
     - Returns ``DeviceCapability`` for CUDA (SM version); placeholder for XPU.
   * - ``get_device_memory``
     - ✅
     - Queries total device memory for CUDA and XPU.
   * - ``DeviceStream``
     - ✅
     - Unified stream wrapper for CUDA/XPU.
   * - ``DeviceEvent``
     - ✅
     - Unified event wrapper for CUDA/XPU.
   * - ``create_stream`` / ``create_event``
     - ✅
     - Factory functions for streams and events.
   * - ``current_stream``
     - ✅
     - Returns the current backend stream.
   * - ``stream_context``
     - ✅
     - Context manager for stream scope.
   * - ``empty_cache``
     - ✅
     - Releases unused cached memory on the device.
   * - ``synchronize_device``
     - ✅
     - Blocks until all pending device work is finished.

Model-Level Utilities
----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 35

   * - API
     - Status
     - Notes
   * - ``get_precision``
     - ✅
     - Determines quantization precision. Non-CUDA defaults to ``"int4"``.
   * - ``is_turing``
     - ✅
     - Returns ``False`` for non-CUDA devices.
   * - ``get_gpu_memory``
     - ✅
     - Supports CUDA and XPU via ``torch.xpu``.
   * - ``check_hardware_compatibility``
     - ✅
     - Validates quantization config for CUDA and XPU.
   * - ``CPUOffloadManager``
     - ✅
     - Supports CUDA and XPU via device abstraction layer.

Triton Kernel Support
----------------------

Triton kernels provide a portable, high-performance alternative to CUDA kernels
for operations that can run on Intel XPU (via Triton's XPU backend).

.. list-table::
   :header-rows: 1
   :widths: 35 15 35

   * - Kernel
     - Status
     - Notes
   * - ``triton_dequant_gemm_w4a4``
     - 🔧
     - Triton-based W4A4 GEMM with INT4 dequantization. Available when ``triton`` is installed.
   * - ``triton_quantize_w4a4_act``
     - 🔧
     - Triton-based activation quantization to INT4. Available when ``triton`` is installed.

Backend Selection Priority
---------------------------

The ops dispatch layer selects the best available backend in this order:

1. **CUDA native** (``nunchaku._C``): Best performance on NVIDIA GPUs.
2. **Triton**: Portable high-performance kernels on CUDA and Intel XPU.
3. **PyTorch fallback**: Runs on any PyTorch-supported device (CPU, XPU, CUDA without extension).

To force a specific backend, set the environment variable::

    NUNCHAKU_BACKEND=triton   # force Triton kernels
    NUNCHAKU_BACKEND=torch    # force PyTorch fallback

Next Steps
----------

1. Complete AWQ W4A16 GEMV fallback implementation
2. Add Triton attention kernel for Intel XPU
3. Performance benchmarking on Intel XPU hardware
4. End-to-end model inference validation on Intel XPU
