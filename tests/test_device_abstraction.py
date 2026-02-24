"""
Unit tests for the Nunchaku runtime device abstraction layer and PyTorch fallback ops.

These tests are designed to run on any machine (CPU-only is sufficient) since
they exercise the device abstraction API surface and the PyTorch fallback
kernels without requiring a CUDA or XPU device.

NOTE: We mock the ``nunchaku._C`` extension before importing any nunchaku
modules so that the test suite can run without a CUDA build.
"""

import sys
import types

import pytest
import torch

# Provide a stub for the native extension so that imports succeed on
# machines where the CUDA extension has not been compiled.
if "nunchaku._C" not in sys.modules:
    _stub_C = types.ModuleType("nunchaku._C")
    _stub_ops = types.ModuleType("nunchaku._C.ops")
    _stub_utils = types.ModuleType("nunchaku._C.utils")

    # Stub out functions imported elsewhere in the package
    def _not_implemented(*args, **kwargs):
        raise NotImplementedError("nunchaku._C is not available (no CUDA build)")

    for fn_name in ("gemm_w4a4", "quantize_w4a4_act_fuse_lora", "attention_fp16", "gemm_awq", "gemv_awq",
                     "test_rmsnorm_rope", "test_pack_qkv"):
        setattr(_stub_ops, fn_name, _not_implemented)
    for fn_name in ("set_log_level", "set_cuda_stack_limit", "disable_memory_auto_release",
                     "trim_memory", "set_faster_i2f_mode"):
        setattr(_stub_utils, fn_name, _not_implemented)

    _stub_C.ops = _stub_ops
    _stub_C.utils = _stub_utils
    # Stub the C++ model classes
    _stub_C.QuantizedFluxModel = type("QuantizedFluxModel", (), {"__init__": _not_implemented})
    _stub_C.QuantizedSanaModel = type("QuantizedSanaModel", (), {"__init__": _not_implemented})
    _stub_C.QuantizedGEMM = type("QuantizedGEMM", (), {"__init__": _not_implemented})
    _stub_C.QuantizedGEMM88 = type("QuantizedGEMM88", (), {"__init__": _not_implemented})
    _stub_C.Tensor = type("Tensor", (), {})

    sys.modules["nunchaku._C"] = _stub_C
    sys.modules["nunchaku._C.ops"] = _stub_ops
    sys.modules["nunchaku._C.utils"] = _stub_utils

# ── runtime device_utils tests ────────────────────────────────────────

from nunchaku.runtime.device_utils import (
    DeviceCapability,
    DeviceEvent,
    DeviceStream,
    get_supported_backends,
    is_cuda_available,
    is_xpu_available,
)


class TestBackendDetection:
    """Tests for backend availability detection."""

    def test_is_cuda_available_returns_bool(self):
        assert isinstance(is_cuda_available(), bool)

    def test_is_xpu_available_returns_bool(self):
        assert isinstance(is_xpu_available(), bool)

    def test_get_supported_backends_returns_list(self):
        backends = get_supported_backends()
        assert isinstance(backends, list)
        # every entry must be a string
        for b in backends:
            assert isinstance(b, str)


class TestDeviceCapability:
    """Tests for DeviceCapability dataclass."""

    def test_sm_property(self):
        cap = DeviceCapability(major=8, minor=9)
        assert cap.sm == "89"

    def test_sm_turing(self):
        cap = DeviceCapability(major=7, minor=5)
        assert cap.sm == "75"

    def test_unknown_device(self):
        cap = DeviceCapability(major=0, minor=0)
        assert cap.sm == "00"


# ── torch_fallback op tests ──────────────────────────────────────────

from nunchaku.ops.torch_fallback import (
    _dequantize_int4,
    _unpack_int4,
    awq_gemv_w4a16_fallback,
    svdq_gemm_w4a4_fallback,
    svdq_quantize_w4a4_act_fuse_lora_fallback,
)


class TestUnpackInt4:
    """Tests for INT4 unpacking helper."""

    def test_output_shape(self):
        packed = torch.randint(-128, 127, (4, 32), dtype=torch.int8)
        unpacked = _unpack_int4(packed)
        assert unpacked.shape == (4, 64)

    def test_round_trip_values(self):
        """Pack two known int4 values and verify unpacking recovers them."""
        low_val = torch.tensor([3], dtype=torch.int8)
        high_val = torch.tensor([-2], dtype=torch.int8)
        packed = (high_val << 4) | (low_val & 0xF)
        unpacked = _unpack_int4(packed.unsqueeze(0))
        # The unpacking should produce [low, high] per element
        assert unpacked[0, 0].item() == 3
        assert unpacked[0, 1].item() == -2


class TestDequantizeInt4:
    """Tests for INT4 dequantization helper."""

    def test_output_shape(self):
        N, K = 16, 128
        packed_weight = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8)
        scales = torch.ones(K // 64, N, dtype=torch.float16)
        result = _dequantize_int4(packed_weight, scales, group_size=64)
        assert result.shape == (N, K)

    def test_zero_scales_produce_zeros(self):
        N, K = 8, 64
        packed_weight = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8)
        scales = torch.zeros(K // 64, N, dtype=torch.float16)
        result = _dequantize_int4(packed_weight, scales, group_size=64)
        assert torch.all(result == 0)


class TestSvdqGemmW4A4Fallback:
    """Tests for the fallback W4A4 GEMM."""

    def test_basic_output_shape(self):
        M, K, N = 4, 128, 64
        act = torch.randint(-128, 127, (M, K // 2), dtype=torch.int8)
        wgt = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8)
        out = torch.zeros(M, N, dtype=torch.bfloat16)
        ascales = torch.ones(K // 64, M, dtype=torch.bfloat16)
        wscales = torch.ones(K // 64, N, dtype=torch.bfloat16)

        svdq_gemm_w4a4_fallback(
            act=act,
            wgt=wgt,
            out=out,
            ascales=ascales,
            wscales=wscales,
        )
        # out should have been written (at least not all zeros for random input)
        assert out.shape == (M, N)

    def test_with_bias(self):
        M, K, N = 4, 128, 64
        act = torch.randint(-128, 127, (M, K // 2), dtype=torch.int8)
        wgt = torch.zeros(N, K // 2, dtype=torch.int8)
        out = torch.zeros(M, N, dtype=torch.bfloat16)
        ascales = torch.ones(K // 64, M, dtype=torch.bfloat16)
        wscales = torch.ones(K // 64, N, dtype=torch.bfloat16)
        bias = torch.ones(N, dtype=torch.bfloat16) * 5.0

        svdq_gemm_w4a4_fallback(
            act=act,
            wgt=wgt,
            out=out,
            ascales=ascales,
            wscales=wscales,
            bias=bias,
        )
        # With zero weights, output should be approximately bias
        assert torch.allclose(out, bias.unsqueeze(0).expand(M, N), atol=1e-1)


class TestSvdqQuantizeW4A4ActFuseLoraFallback:
    """Tests for the fallback activation quantization."""

    def test_output_shapes(self):
        M, K, R = 4, 128, 16
        x = torch.randn(M, K, dtype=torch.bfloat16)
        lora_down = torch.randn(K, R, dtype=torch.bfloat16)
        output, oscales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_fallback(
            input=x, lora_down=lora_down, pad_size=256
        )
        pad_M = 256  # ceil(4/256)*256
        assert output.shape == (pad_M, K // 2)
        assert output.dtype == torch.uint8
        assert oscales.shape == (K // 64, pad_M)
        assert lora_act_out.shape == (pad_M, R)

    def test_lora_projection_non_zero(self):
        M, K, R = 8, 128, 16
        x = torch.randn(M, K, dtype=torch.float16)
        lora_down = torch.randn(K, R, dtype=torch.float16)
        _, _, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_fallback(
            input=x, lora_down=lora_down, pad_size=256
        )
        # The LoRA projection should produce non-zero output for random input
        assert lora_act_out[:M].abs().sum() > 0


class TestAwqGemvW4A16Fallback:
    """Tests for the fallback AWQ GEMV."""

    def test_raises_not_implemented(self):
        m, n, k = 1, 64, 128
        in_feats = torch.randn(m, k, dtype=torch.float16)
        kernel = torch.randint(-2**31, 2**31 - 1, (n // 4, k // 2), dtype=torch.int32)
        scaling_factors = torch.ones(k // 64, n, dtype=torch.float16)
        zeros = torch.zeros(k // 64, n, dtype=torch.float16)

        with pytest.raises(NotImplementedError, match="AWQ W4A16 GEMV fallback is not yet implemented"):
            awq_gemv_w4a16_fallback(in_feats, kernel, scaling_factors, zeros, m, n, k)


# ── utils device-agnostic tests ──────────────────────────────────────

from nunchaku.utils import get_precision, is_turing


class TestGetPrecisionDeviceAgnostic:
    """Tests that get_precision works for non-CUDA devices."""

    def test_explicit_int4(self):
        assert get_precision("int4", device="cpu") == "int4"

    def test_explicit_fp4(self):
        assert get_precision("fp4", device="cpu") == "fp4"

    def test_auto_on_cpu_defaults_to_int4(self):
        assert get_precision("auto", device="cpu") == "int4"


class TestIsTuringDeviceAgnostic:
    """Tests that is_turing works for non-CUDA devices."""

    def test_cpu_returns_false(self):
        assert is_turing("cpu") is False


# ── check_hardware_compatibility tests ───────────────────────────────

from nunchaku.utils import check_hardware_compatibility


class TestCheckHardwareCompatibility:
    """Tests for the updated check_hardware_compatibility."""

    def test_unsupported_device_raises(self):
        config = {"weight": {"dtype": "int4"}}
        with pytest.raises(ValueError, match="Unsupported device type"):
            check_hardware_compatibility(config, device="cpu")
