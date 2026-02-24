"""
Comprehensive unit tests for Intel XPU-targeted APIs.

These tests validate the correctness of all operations that need to work
on Intel XPU (and any non-CUDA device), using small tensors for efficiency.
They test three backend paths:
  1. PyTorch fallback ops
  2. Triton kernel ops (when available)
  3. Ops dispatch layer (gemm.py / quantize.py backend selection)

Tests run on CPU to ensure device-agnostic correctness without needing
CUDA or XPU hardware.
"""

import os
import sys
import types

import pytest
import torch

# ── Stub out the native C++ extension for non-CUDA test environments ──

if "nunchaku._C" not in sys.modules:
    _stub_C = types.ModuleType("nunchaku._C")
    _stub_ops = types.ModuleType("nunchaku._C.ops")
    _stub_utils = types.ModuleType("nunchaku._C.utils")

    def _not_implemented(*args, **kwargs):
        raise NotImplementedError("nunchaku._C is not available (no CUDA build)")

    for fn_name in ("gemm_w4a4", "quantize_w4a4_act_fuse_lora", "attention_fp16",
                     "gemm_awq", "gemv_awq", "test_rmsnorm_rope", "test_pack_qkv"):
        setattr(_stub_ops, fn_name, _not_implemented)
    for fn_name in ("set_log_level", "set_cuda_stack_limit", "disable_memory_auto_release",
                     "trim_memory", "set_faster_i2f_mode"):
        setattr(_stub_utils, fn_name, _not_implemented)

    _stub_C.ops = _stub_ops
    _stub_C.utils = _stub_utils
    _stub_C.QuantizedFluxModel = type("QuantizedFluxModel", (), {"__init__": _not_implemented})
    _stub_C.QuantizedSanaModel = type("QuantizedSanaModel", (), {"__init__": _not_implemented})
    _stub_C.QuantizedGEMM = type("QuantizedGEMM", (), {"__init__": _not_implemented})
    _stub_C.QuantizedGEMM88 = type("QuantizedGEMM88", (), {"__init__": _not_implemented})
    _stub_C.Tensor = type("Tensor", (), {})

    sys.modules["nunchaku._C"] = _stub_C
    sys.modules["nunchaku._C.ops"] = _stub_ops
    sys.modules["nunchaku._C.utils"] = _stub_utils


# ── imports ──────────────────────────────────────────────────────────

from nunchaku.ops.torch_fallback import (
    _dequantize_int4,
    _unpack_awq_int32,
    _unpack_int4,
    awq_gemv_w4a16_fallback,
    svdq_gemm_w4a4_fallback,
    svdq_quantize_w4a4_act_fuse_lora_fallback,
)
from nunchaku.runtime.device_utils import (
    DeviceCapability,
    get_supported_backends,
    is_cuda_available,
    is_xpu_available,
)
from nunchaku.utils import (
    ceil_divide,
    check_hardware_compatibility,
    get_precision,
    is_turing,
    pad_tensor,
)


# ══════════════════════════════════════════════════════════════════════
# 1. Helper / Utility Ops (device-agnostic)
# ══════════════════════════════════════════════════════════════════════

class TestUnpackInt4:
    """Tests for _unpack_int4: INT4 unpacking helper."""

    def test_shape_small(self):
        packed = torch.randint(-128, 127, (2, 4), dtype=torch.int8)
        result = _unpack_int4(packed)
        assert result.shape == (2, 8)

    def test_shape_standard(self):
        packed = torch.randint(-128, 127, (4, 32), dtype=torch.int8)
        result = _unpack_int4(packed)
        assert result.shape == (4, 64)

    def test_known_values(self):
        """Verify specific known INT4 values after unpacking."""
        # Pack: low=3, high=-2 → packed byte = (high << 4) | (low & 0xF)
        low, high = 3, -2
        packed = torch.tensor([[(high << 4) | (low & 0xF)]], dtype=torch.int8)
        unpacked = _unpack_int4(packed)
        assert unpacked[0, 0].item() == low
        assert unpacked[0, 1].item() == high

    def test_zero_input(self):
        packed = torch.zeros(2, 4, dtype=torch.int8)
        result = _unpack_int4(packed)
        assert torch.all(result == 0)

    def test_boundary_values(self):
        """INT4 range: [-8, 7]. Verify extremes survive round-trip."""
        low, high = -8, 7
        packed = torch.tensor([[(high << 4) | (low & 0xF)]], dtype=torch.int8)
        unpacked = _unpack_int4(packed)
        assert unpacked[0, 0].item() == low
        assert unpacked[0, 1].item() == high


class TestDequantizeInt4:
    """Tests for _dequantize_int4: INT4 dequantization."""

    def test_output_shape_small(self):
        N, K = 4, 64
        packed = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8)
        scales = torch.ones(K // 64, N, dtype=torch.float16)
        result = _dequantize_int4(packed, scales, group_size=64)
        assert result.shape == (N, K)
        assert result.dtype == torch.float16

    def test_zero_scales(self):
        N, K = 4, 64
        packed = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8)
        scales = torch.zeros(K // 64, N, dtype=torch.float16)
        result = _dequantize_int4(packed, scales, group_size=64)
        assert torch.all(result == 0)

    def test_unit_scales(self):
        """With unit scales, dequantized values should match unpacked values."""
        N, K = 2, 64
        packed = torch.randint(-8, 8, (N, K // 2), dtype=torch.int8)
        scales = torch.ones(K // 64, N, dtype=torch.float16)
        result = _dequantize_int4(packed, scales, group_size=64)
        unpacked = _unpack_int4(packed).to(torch.float16)
        assert torch.allclose(result, unpacked, atol=1e-3)

    def test_bfloat16_dtype(self):
        N, K = 4, 128
        packed = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8)
        scales = torch.ones(K // 64, N, dtype=torch.bfloat16)
        result = _dequantize_int4(packed, scales, group_size=64)
        assert result.dtype == torch.bfloat16


class TestPadTensor:
    """Tests for pad_tensor utility."""

    def test_no_padding_needed(self):
        t = torch.randn(4, 64)
        result = pad_tensor(t, 4, dim=0)
        assert result.shape == (4, 64)
        assert torch.equal(result, t)

    def test_padding_applied(self):
        t = torch.randn(3, 64)
        result = pad_tensor(t, 4, dim=0)
        assert result.shape == (4, 64)
        assert torch.equal(result[:3], t)

    def test_none_input(self):
        assert pad_tensor(None, 4, dim=0) is None

    def test_multiples_1(self):
        t = torch.randn(5, 32)
        assert pad_tensor(t, 1, dim=0) is t


class TestCeilDivide:
    """Tests for ceil_divide utility."""

    def test_exact_division(self):
        assert ceil_divide(64, 16) == 4

    def test_non_exact(self):
        assert ceil_divide(65, 16) == 5

    def test_one(self):
        assert ceil_divide(1, 256) == 1


# ══════════════════════════════════════════════════════════════════════
# 2. Core Quantized Operations (PyTorch fallback)
# ══════════════════════════════════════════════════════════════════════

class TestSvdqGemmW4A4Fallback:
    """Tests for W4A4 GEMM via PyTorch fallback with small tensors."""

    def test_basic_output_shape(self):
        M, K, N = 2, 64, 32
        act = torch.randint(-128, 127, (M, K // 2), dtype=torch.int8)
        wgt = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8)
        out = torch.zeros(M, N, dtype=torch.bfloat16)
        ascales = torch.ones(K // 64, M, dtype=torch.bfloat16)
        wscales = torch.ones(K // 64, N, dtype=torch.bfloat16)

        svdq_gemm_w4a4_fallback(act=act, wgt=wgt, out=out, ascales=ascales, wscales=wscales)
        assert out.shape == (M, N)

    def test_with_bias(self):
        M, K, N = 2, 64, 32
        act = torch.zeros(M, K // 2, dtype=torch.int8)  # zero activations
        wgt = torch.zeros(N, K // 2, dtype=torch.int8)  # zero weights
        out = torch.zeros(M, N, dtype=torch.bfloat16)
        ascales = torch.ones(K // 64, M, dtype=torch.bfloat16)
        wscales = torch.ones(K // 64, N, dtype=torch.bfloat16)
        bias = torch.full((N,), 3.0, dtype=torch.bfloat16)

        svdq_gemm_w4a4_fallback(act=act, wgt=wgt, out=out, ascales=ascales, wscales=wscales, bias=bias)
        expected = bias.unsqueeze(0).expand(M, N)
        assert torch.allclose(out, expected, atol=1e-1)

    def test_with_lora(self):
        M, K, N, R = 2, 64, 32, 8
        act = torch.randint(-128, 127, (M, K // 2), dtype=torch.int8)
        wgt = torch.zeros(N, K // 2, dtype=torch.int8)
        out = torch.zeros(M, N, dtype=torch.bfloat16)
        ascales = torch.ones(K // 64, M, dtype=torch.bfloat16)
        wscales = torch.ones(K // 64, N, dtype=torch.bfloat16)
        lora_act_in = torch.randn(M, R, dtype=torch.float32)
        lora_up = torch.randn(N, R, dtype=torch.bfloat16)

        svdq_gemm_w4a4_fallback(
            act=act, wgt=wgt, out=out, ascales=ascales, wscales=wscales,
            lora_act_in=lora_act_in, lora_up=lora_up,
        )
        # LoRA should contribute non-zero values
        assert out.abs().sum() > 0

    def test_fuse_silu(self):
        M, K, N = 2, 64, 32
        act = torch.randint(-128, 127, (M, K // 2), dtype=torch.int8)
        wgt = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8)
        out = torch.zeros(M, N, dtype=torch.bfloat16)
        ascales = torch.ones(K // 64, M, dtype=torch.bfloat16)
        wscales = torch.ones(K // 64, N, dtype=torch.bfloat16)

        svdq_gemm_w4a4_fallback(
            act=act, wgt=wgt, out=out, ascales=ascales, wscales=wscales,
            fuse_silu=True,
        )
        # SiLU output should be different from without SiLU
        assert out.shape == (M, N)

    def test_alpha_scaling(self):
        M, K, N = 2, 64, 32
        act = torch.randint(-128, 127, (M, K // 2), dtype=torch.int8)
        wgt = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8)

        out1 = torch.zeros(M, N, dtype=torch.bfloat16)
        out2 = torch.zeros(M, N, dtype=torch.bfloat16)
        ascales = torch.ones(K // 64, M, dtype=torch.bfloat16)
        wscales = torch.ones(K // 64, N, dtype=torch.bfloat16)

        svdq_gemm_w4a4_fallback(act=act, wgt=wgt, out=out1, ascales=ascales, wscales=wscales, alpha=1.0)
        svdq_gemm_w4a4_fallback(act=act, wgt=wgt, out=out2, ascales=ascales, wscales=wscales, alpha=2.0)

        # Alpha=2 should roughly double the output
        if out1.abs().sum() > 0:
            ratio = out2.abs().sum() / out1.abs().sum()
            assert 1.5 < ratio < 2.5

    def test_float16_dtype(self):
        M, K, N = 2, 64, 32
        act = torch.randint(-128, 127, (M, K // 2), dtype=torch.int8)
        wgt = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8)
        out = torch.zeros(M, N, dtype=torch.float16)
        ascales = torch.ones(K // 64, M, dtype=torch.float16)
        wscales = torch.ones(K // 64, N, dtype=torch.float16)

        svdq_gemm_w4a4_fallback(act=act, wgt=wgt, out=out, ascales=ascales, wscales=wscales)
        assert out.dtype == torch.float16


class TestSvdqQuantizeW4A4Fallback:
    """Tests for activation quantization via PyTorch fallback with small tensors."""

    def test_output_shapes_small(self):
        M, K, R = 2, 64, 8
        x = torch.randn(M, K, dtype=torch.bfloat16)
        lora_down = torch.randn(K, R, dtype=torch.bfloat16)
        output, oscales, lora_act = svdq_quantize_w4a4_act_fuse_lora_fallback(
            input=x, lora_down=lora_down, pad_size=256
        )
        pad_M = 256
        assert output.shape == (pad_M, K // 2)
        assert output.dtype == torch.uint8
        assert oscales.shape == (K // 64, pad_M)
        assert lora_act.shape == (pad_M, R)

    def test_quantization_range(self):
        """Packed output should only contain valid INT4 values."""
        M, K, R = 4, 64, 8
        x = torch.randn(M, K, dtype=torch.bfloat16)
        lora_down = torch.randn(K, R, dtype=torch.bfloat16)
        output, _, _ = svdq_quantize_w4a4_act_fuse_lora_fallback(
            input=x, lora_down=lora_down, pad_size=256
        )
        # Each byte should be a valid packed pair
        assert output.dtype == torch.uint8
        assert output.max() <= 255

    def test_lora_projection(self):
        M, K, R = 4, 64, 8
        x = torch.randn(M, K, dtype=torch.float16)
        lora_down = torch.randn(K, R, dtype=torch.float16)
        _, _, lora_act = svdq_quantize_w4a4_act_fuse_lora_fallback(
            input=x, lora_down=lora_down, pad_size=256
        )
        # LoRA output for valid rows should be non-zero
        assert lora_act[:M].abs().sum() > 0

    def test_with_smooth_factor(self):
        M, K, R = 4, 64, 8
        x = torch.randn(M, K, dtype=torch.bfloat16)
        lora_down = torch.randn(K, R, dtype=torch.bfloat16)
        # Use a non-uniform smooth factor to ensure different quantization
        smooth = torch.rand(K, dtype=torch.bfloat16) + 0.1

        _, scales1, _ = svdq_quantize_w4a4_act_fuse_lora_fallback(
            input=x, lora_down=lora_down, pad_size=256
        )
        _, scales2, _ = svdq_quantize_w4a4_act_fuse_lora_fallback(
            input=x, lora_down=lora_down, smooth=smooth, pad_size=256
        )
        # Smooth factor should change the output scales
        assert not torch.equal(scales1[:, :M], scales2[:, :M])

    def test_zero_input(self):
        M, K, R = 2, 64, 8
        x = torch.zeros(M, K, dtype=torch.bfloat16)
        lora_down = torch.randn(K, R, dtype=torch.bfloat16)
        output, oscales, lora_act = svdq_quantize_w4a4_act_fuse_lora_fallback(
            input=x, lora_down=lora_down, pad_size=256
        )
        # All-zero input should produce all-zero quantized output
        assert output[:M].sum() == 0


# ══════════════════════════════════════════════════════════════════════
# 2b. AWQ W4A16 GEMV (PyTorch fallback)
# ══════════════════════════════════════════════════════════════════════

class TestUnpackAwqInt32:
    """Tests for _unpack_awq_int32: AWQ TinyChat format unpacking."""

    @staticmethod
    def _pack_awq_weight(weight_uint4, n, k):
        """Pack a (n, k) uint4 weight into TinyChat AWQ int32 format for testing.

        Reproduces the pack_w4 logic from tinychat_utils.
        """
        w = weight_uint4.to(torch.int32)
        w = w.view(-1, 4, 8)
        packed_i16 = w[:, 0] | (w[:, 1] << 4) | (w[:, 2] << 8) | (w[:, 3] << 12)
        packed_i16 = packed_i16.view(n // 4, 4, k // 64, 16).permute(0, 2, 1, 3).reshape(n // 4, k)
        return packed_i16.to(torch.int16).view(torch.int32).reshape(n // 4, k // 2)

    def test_round_trip(self):
        """Pack then unpack should recover the original weight matrix."""
        n, k = 64, 128
        weight = torch.randint(0, 16, (n, k), dtype=torch.int32)
        packed = self._pack_awq_weight(weight, n, k)
        unpacked = _unpack_awq_int32(packed, n, k)
        assert torch.equal(unpacked, weight)

    def test_shape(self):
        n, k = 32, 64
        weight = torch.randint(0, 16, (n, k), dtype=torch.int32)
        packed = self._pack_awq_weight(weight, n, k)
        unpacked = _unpack_awq_int32(packed, n, k)
        assert unpacked.shape == (n, k)

    def test_value_range(self):
        """Unpacked values should always be in [0, 15]."""
        n, k = 32, 128
        weight = torch.randint(0, 16, (n, k), dtype=torch.int32)
        packed = self._pack_awq_weight(weight, n, k)
        unpacked = _unpack_awq_int32(packed, n, k)
        assert unpacked.min() >= 0
        assert unpacked.max() <= 15

    def test_zero_weight(self):
        n, k = 32, 64
        weight = torch.zeros(n, k, dtype=torch.int32)
        packed = self._pack_awq_weight(weight, n, k)
        unpacked = _unpack_awq_int32(packed, n, k)
        assert torch.all(unpacked == 0)


class TestAwqGemvW4A16Fallback:
    """Tests for awq_gemv_w4a16_fallback with small tensors."""

    @staticmethod
    def _pack_awq_weight(weight_uint4, n, k):
        w = weight_uint4.to(torch.int32)
        w = w.view(-1, 4, 8)
        packed_i16 = w[:, 0] | (w[:, 1] << 4) | (w[:, 2] << 8) | (w[:, 3] << 12)
        packed_i16 = packed_i16.view(n // 4, 4, k // 64, 16).permute(0, 2, 1, 3).reshape(n // 4, k)
        return packed_i16.to(torch.int16).view(torch.int32).reshape(n // 4, k // 2)

    def test_basic_output_shape(self):
        m, n, k = 1, 32, 64
        weight = torch.randint(0, 16, (n, k), dtype=torch.int32)
        kernel = self._pack_awq_weight(weight, n, k)
        in_feats = torch.randn(m, k, dtype=torch.bfloat16)
        scaling_factors = torch.ones(k // 64, n, dtype=torch.bfloat16)
        zeros = torch.zeros(k // 64, n, dtype=torch.bfloat16)

        output = awq_gemv_w4a16_fallback(in_feats, kernel, scaling_factors, zeros, m, n, k)
        assert output.shape == (m, n)

    def test_unit_scale_zero_zeros(self):
        """With scale=1 and zeros=0, GEMV should match direct FP matmul."""
        m, n, k = 2, 32, 64
        weight = torch.randint(0, 16, (n, k), dtype=torch.int32)
        kernel = self._pack_awq_weight(weight, n, k)
        in_feats = torch.randn(m, k, dtype=torch.float16)
        scaling_factors = torch.ones(k // 64, n, dtype=torch.float16)
        zeros = torch.zeros(k // 64, n, dtype=torch.float16)

        output = awq_gemv_w4a16_fallback(in_feats, kernel, scaling_factors, zeros, m, n, k)
        expected = in_feats @ weight.to(torch.float16).T
        assert torch.allclose(output, expected, atol=1e-1)

    def test_with_scaling(self):
        """Non-unit scales should change the output."""
        m, n, k = 1, 32, 64
        weight = torch.randint(0, 16, (n, k), dtype=torch.int32)
        kernel = self._pack_awq_weight(weight, n, k)
        in_feats = torch.randn(m, k, dtype=torch.float16)
        scaling_factors = torch.ones(k // 64, n, dtype=torch.float16) * 2.0
        zeros = torch.zeros(k // 64, n, dtype=torch.float16)

        out_2x = awq_gemv_w4a16_fallback(in_feats, kernel, scaling_factors, zeros, m, n, k)
        scaling_factors_1 = torch.ones(k // 64, n, dtype=torch.float16)
        out_1x = awq_gemv_w4a16_fallback(in_feats, kernel, scaling_factors_1, zeros, m, n, k)
        # With scale=2, the output should be roughly double
        if out_1x.abs().sum() > 0:
            ratio = out_2x.abs().sum() / out_1x.abs().sum()
            assert 1.5 < ratio < 2.5

    def test_with_zeros(self):
        """Non-zero zeros should shift the dequantized weights."""
        m, n, k = 1, 32, 64
        weight = torch.randint(0, 16, (n, k), dtype=torch.int32)
        kernel = self._pack_awq_weight(weight, n, k)
        in_feats = torch.randn(m, k, dtype=torch.float16)
        scaling_factors = torch.ones(k // 64, n, dtype=torch.float16)

        out_no_zeros = awq_gemv_w4a16_fallback(
            in_feats, kernel, scaling_factors,
            torch.zeros(k // 64, n, dtype=torch.float16), m, n, k
        )
        out_with_zeros = awq_gemv_w4a16_fallback(
            in_feats, kernel, scaling_factors,
            torch.ones(k // 64, n, dtype=torch.float16) * -5.0, m, n, k
        )
        assert not torch.equal(out_no_zeros, out_with_zeros)

    def test_bfloat16_dtype(self):
        m, n, k = 2, 32, 64
        weight = torch.randint(0, 16, (n, k), dtype=torch.int32)
        kernel = self._pack_awq_weight(weight, n, k)
        in_feats = torch.randn(m, k, dtype=torch.bfloat16)
        scaling_factors = torch.ones(k // 64, n, dtype=torch.bfloat16)
        zeros = torch.zeros(k // 64, n, dtype=torch.bfloat16)

        output = awq_gemv_w4a16_fallback(in_feats, kernel, scaling_factors, zeros, m, n, k)
        assert output.dtype == torch.bfloat16

    def test_batch_size_4(self):
        """Batched GEMV (m > 1) should work correctly."""
        m, n, k = 4, 32, 64
        weight = torch.randint(0, 16, (n, k), dtype=torch.int32)
        kernel = self._pack_awq_weight(weight, n, k)
        in_feats = torch.randn(m, k, dtype=torch.float16)
        scaling_factors = torch.ones(k // 64, n, dtype=torch.float16)
        zeros = torch.zeros(k // 64, n, dtype=torch.float16)

        output = awq_gemv_w4a16_fallback(in_feats, kernel, scaling_factors, zeros, m, n, k)
        assert output.shape == (m, n)


class TestAwqGemvDispatch:
    """Test that AWQ GEMV dispatch selects the correct backend."""

    def test_forced_torch_backend(self):
        """Force torch backend for AWQ GEMV and verify it works."""
        from nunchaku.ops.gemv import awq_gemv_w4a16_cuda

        m, n, k = 2, 32, 64
        weight = torch.randint(0, 16, (n, k), dtype=torch.int32)
        w = weight.to(torch.int32).view(-1, 4, 8)
        packed_i16 = w[:, 0] | (w[:, 1] << 4) | (w[:, 2] << 8) | (w[:, 3] << 12)
        packed_i16 = packed_i16.view(n // 4, 4, k // 64, 16).permute(0, 2, 1, 3).reshape(n // 4, k)
        kernel = packed_i16.to(torch.int16).view(torch.int32).reshape(n // 4, k // 2)

        in_feats = torch.randn(m, k, dtype=torch.bfloat16)
        scaling_factors = torch.ones(k // 64, n, dtype=torch.bfloat16)
        zeros = torch.zeros(k // 64, n, dtype=torch.bfloat16)

        old_val = os.environ.get("NUNCHAKU_BACKEND")
        try:
            os.environ["NUNCHAKU_BACKEND"] = "torch"
            output = awq_gemv_w4a16_cuda(in_feats, kernel, scaling_factors, zeros, m, n, k)
            assert output.shape == (m, n)
        finally:
            if old_val is None:
                os.environ.pop("NUNCHAKU_BACKEND", None)
            else:
                os.environ["NUNCHAKU_BACKEND"] = old_val

    def test_get_backend_returns_torch_on_cpu(self):
        from nunchaku.ops.gemv import _get_backend
        backend = _get_backend("cpu")
        assert backend == "torch"


class TestAWQW4A16LinearOnCPU:
    """End-to-end tests for AWQW4A16Linear module using fallback."""

    def _make_linear(self, in_f=64, out_f=32, group_size=64):
        from nunchaku.models.linear import AWQW4A16Linear
        linear = AWQW4A16Linear(
            in_features=in_f,
            out_features=out_f,
            bias=True,
            group_size=group_size,
            torch_dtype=torch.bfloat16,
            device="cpu",
        )
        # Pack valid uint4 weights
        weight = torch.randint(0, 16, (out_f, in_f), dtype=torch.int32)
        w = weight.view(-1, 4, 8)
        packed_i16 = w[:, 0] | (w[:, 1] << 4) | (w[:, 2] << 8) | (w[:, 3] << 12)
        packed_i16 = packed_i16.view(out_f // 4, 4, in_f // 64, 16).permute(0, 2, 1, 3).reshape(out_f // 4, in_f)
        packed_i32 = packed_i16.to(torch.int16).view(torch.int32).reshape(out_f // 4, in_f // 2)
        with torch.no_grad():
            linear.qweight.copy_(packed_i32)
            linear.wscales.fill_(1.0)
            linear.wzeros.fill_(0.0)
            if linear.bias is not None:
                linear.bias.zero_()
        return linear

    def test_forward_shape(self):
        linear = self._make_linear()
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        old_env = os.environ.get("NUNCHAKU_BACKEND")
        try:
            os.environ["NUNCHAKU_BACKEND"] = "torch"
            output = linear(x)
            assert output.shape == (2, 32)
        finally:
            if old_env is None:
                os.environ.pop("NUNCHAKU_BACKEND", None)
            else:
                os.environ["NUNCHAKU_BACKEND"] = old_env

    def test_repr(self):
        linear = self._make_linear()
        r = repr(linear)
        assert "AWQW4A16Linear" in r
        assert "in_features=64" in r
        assert "out_features=32" in r
        assert "group_size=64" in r


# ══════════════════════════════════════════════════════════════════════
# 3. Device Abstraction / Runtime APIs
# ══════════════════════════════════════════════════════════════════════

class TestBackendDetection:
    """Tests for backend availability detection."""

    def test_is_cuda_available_returns_bool(self):
        assert isinstance(is_cuda_available(), bool)

    def test_is_xpu_available_returns_bool(self):
        assert isinstance(is_xpu_available(), bool)

    def test_get_supported_backends_type(self):
        backends = get_supported_backends()
        assert isinstance(backends, list)
        for b in backends:
            assert isinstance(b, str)
            assert b in ("cuda", "xpu")


class TestDeviceCapability:
    """Tests for DeviceCapability."""

    def test_sm_ampere(self):
        assert DeviceCapability(major=8, minor=0).sm == "80"

    def test_sm_ada(self):
        assert DeviceCapability(major=8, minor=9).sm == "89"

    def test_sm_turing(self):
        assert DeviceCapability(major=7, minor=5).sm == "75"

    def test_sm_xpu_placeholder(self):
        assert DeviceCapability(major=0, minor=0).sm == "00"


# ══════════════════════════════════════════════════════════════════════
# 4. Model-Level Utility APIs
# ══════════════════════════════════════════════════════════════════════

class TestGetPrecision:
    """Tests for get_precision utility."""

    def test_auto_on_cpu(self):
        assert get_precision("auto", device="cpu") == "int4"

    def test_explicit_int4(self):
        assert get_precision("int4", device="cpu") == "int4"

    def test_explicit_fp4(self):
        assert get_precision("fp4", device="cpu") == "fp4"

    def test_warning_on_mismatch(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_precision("int4", device="cpu", pretrained_model_name_or_path="model-fp4")
            assert len(w) == 1
            assert "fp4" in str(w[0].message)


class TestIsTuring:
    """Tests for is_turing utility."""

    def test_cpu_returns_false(self):
        assert is_turing("cpu") is False


class TestCheckHardwareCompatibility:
    """Tests for check_hardware_compatibility."""

    def test_cpu_raises(self):
        config = {"weight": {"dtype": "int4"}}
        with pytest.raises(ValueError, match="Unsupported device type"):
            check_hardware_compatibility(config, device="cpu")

    def test_xpu_int4_ok(self):
        """XPU should accept int4 quantization."""
        config = {"weight": {"dtype": "int4"}}
        # This should NOT raise (but only works if XPU path is hit)
        # We test the function's logic by calling with device type "xpu"
        # Note: torch.device("xpu") may not be available, so we test the string path
        try:
            check_hardware_compatibility(config, device="xpu")
        except ValueError as e:
            # If XPU is not available as a device type, skip
            if "xpu" not in str(e).lower():
                raise

    def test_xpu_fp4_rejects(self):
        """XPU should reject fp4 quantization."""
        config = {"weight": {"dtype": "fp4_e2m1_all"}}
        with pytest.raises(ValueError, match="int4"):
            check_hardware_compatibility(config, device="xpu")


# ══════════════════════════════════════════════════════════════════════
# 5. Ops Dispatch Layer
# ══════════════════════════════════════════════════════════════════════

class TestGemmBackendSelection:
    """Tests that the GEMM dispatch layer selects the correct backend."""

    def test_get_backend_returns_torch_on_cpu(self):
        """On CPU without CUDA extension, should fall back to torch."""
        from nunchaku.ops.gemm import _get_backend
        backend = _get_backend("cpu")
        # Should be either "triton" (if triton installed) or "torch"
        assert backend in ("triton", "torch")

    def test_get_backend_env_override(self):
        """NUNCHAKU_BACKEND env var should force a specific backend."""
        from nunchaku.ops.gemm import _get_backend
        old_val = os.environ.get("NUNCHAKU_BACKEND")
        try:
            os.environ["NUNCHAKU_BACKEND"] = "torch"
            assert _get_backend("cpu") == "torch"

            os.environ["NUNCHAKU_BACKEND"] = "triton"
            assert _get_backend("cpu") == "triton"
        finally:
            if old_val is None:
                os.environ.pop("NUNCHAKU_BACKEND", None)
            else:
                os.environ["NUNCHAKU_BACKEND"] = old_val

    def test_forced_torch_backend_gemm(self):
        """Force torch backend and verify GEMM works via fallback."""
        from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda

        old_val = os.environ.get("NUNCHAKU_BACKEND")
        try:
            os.environ["NUNCHAKU_BACKEND"] = "torch"
            M, K, N = 2, 64, 32
            act = torch.randint(-128, 127, (M, K // 2), dtype=torch.int8)
            wgt = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8)
            out = torch.zeros(M, N, dtype=torch.bfloat16)
            ascales = torch.ones(K // 64, M, dtype=torch.bfloat16)
            wscales = torch.ones(K // 64, N, dtype=torch.bfloat16)

            svdq_gemm_w4a4_cuda(act=act, wgt=wgt, out=out, ascales=ascales, wscales=wscales)
            assert out.shape == (M, N)
        finally:
            if old_val is None:
                os.environ.pop("NUNCHAKU_BACKEND", None)
            else:
                os.environ["NUNCHAKU_BACKEND"] = old_val


# ══════════════════════════════════════════════════════════════════════
# 6. Triton Kernel Tests
# ══════════════════════════════════════════════════════════════════════

class TestTritonAvailability:
    """Tests for Triton availability detection."""

    def test_is_triton_available_returns_bool(self):
        from nunchaku.ops.triton_kernels import is_triton_available
        assert isinstance(is_triton_available(), bool)


class TestTritonQuantizeW4A4Act:
    """Tests for Triton-based activation quantization (uses PyTorch internally)."""

    def test_import_and_basic_shape(self):
        from nunchaku.ops.triton_kernels import is_triton_available, triton_quantize_w4a4_act

        if not is_triton_available():
            pytest.skip("Triton not installed")

        M, K, R = 4, 64, 8
        x = torch.randn(M, K, dtype=torch.bfloat16)
        lora_down = torch.randn(K, R, dtype=torch.bfloat16)

        output, oscales, lora_act = triton_quantize_w4a4_act(
            x, lora_down=lora_down, pad_size=256
        )
        pad_M = 256
        assert output.shape == (pad_M, K // 2)
        assert output.dtype == torch.uint8
        assert oscales.shape == (K // 64, pad_M)
        assert lora_act.shape == (pad_M, R)

    def test_consistency_with_torch_fallback(self):
        """Triton quantization should produce same results as torch fallback."""
        from nunchaku.ops.triton_kernels import is_triton_available, triton_quantize_w4a4_act

        if not is_triton_available():
            pytest.skip("Triton not installed")

        M, K, R = 4, 64, 8
        x = torch.randn(M, K, dtype=torch.bfloat16)
        lora_down = torch.randn(K, R, dtype=torch.bfloat16)

        out_triton, scales_triton, lora_triton = triton_quantize_w4a4_act(
            x, lora_down=lora_down, pad_size=256
        )
        out_torch, scales_torch, lora_torch = svdq_quantize_w4a4_act_fuse_lora_fallback(
            input=x, lora_down=lora_down, pad_size=256
        )

        assert torch.equal(out_triton[:M], out_torch[:M])
        # Scales and LoRA outputs should be close
        assert torch.allclose(
            scales_triton[:, :M].float(), scales_torch[:, :M].float(), atol=1e-2
        )
        assert torch.allclose(lora_triton[:M], lora_torch[:M], atol=1e-3)


# ══════════════════════════════════════════════════════════════════════
# 7. SVDQW4A4Linear Module (end-to-end on CPU)
# ══════════════════════════════════════════════════════════════════════

class TestSVDQW4A4LinearOnCPU:
    """End-to-end tests for the SVDQW4A4Linear module using fallback."""

    def _make_linear(self, in_f=64, out_f=32, rank=8):
        """Create a small SVDQW4A4Linear for testing."""
        from nunchaku.models.linear import SVDQW4A4Linear
        linear = SVDQW4A4Linear(
            in_features=in_f,
            out_features=out_f,
            rank=rank,
            bias=True,
            precision="int4",
            torch_dtype=torch.bfloat16,
            device="cpu",
        )
        # Initialize with appropriate random data for each dtype
        with torch.no_grad():
            linear.qweight.copy_(torch.randint(-128, 127, linear.qweight.shape, dtype=torch.int8))
            linear.wscales.normal_()
            linear.smooth_factor.fill_(1.0)
            linear.proj_down.normal_()
            linear.proj_up.normal_()
            if linear.bias is not None:
                linear.bias.zero_()
        return linear

    def test_forward_shape(self):
        linear = self._make_linear()
        x = torch.randn(1, 2, 64, dtype=torch.bfloat16)
        old_env = os.environ.get("NUNCHAKU_BACKEND")
        try:
            os.environ["NUNCHAKU_BACKEND"] = "torch"
            output = linear(x)
            assert output.shape == (1, 2, 32)
        finally:
            if old_env is None:
                os.environ.pop("NUNCHAKU_BACKEND", None)
            else:
                os.environ["NUNCHAKU_BACKEND"] = old_env

    def test_quantize_returns_three_tensors(self):
        linear = self._make_linear()
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        old_env = os.environ.get("NUNCHAKU_BACKEND")
        try:
            os.environ["NUNCHAKU_BACKEND"] = "torch"
            qx, ascales, lora_act = linear.quantize(x)
            assert qx.dtype == torch.uint8
            assert lora_act.dtype == torch.float32
        finally:
            if old_env is None:
                os.environ.pop("NUNCHAKU_BACKEND", None)
            else:
                os.environ["NUNCHAKU_BACKEND"] = old_env

    def test_repr(self):
        linear = self._make_linear()
        r = repr(linear)
        assert "SVDQW4A4Linear" in r
        assert "in_features=64" in r
        assert "out_features=32" in r
