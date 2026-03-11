"""
Unit tests for CPU reference implementations of Nunchaku's CUDA operators.

These tests verify mathematical correctness of the CPU fallback paths by:
1. Creating known weight/activation tensors
2. Manually quantizing with known parameters
3. Running through the CPU implementation
4. Comparing against expected float reference results

All tests use minimal shapes (e.g., 64×64) to keep execution fast.
"""

import math

import pytest
import torch

from nunchaku.ops.cpu_ops import (
    awq_dequantize_weights,
    awq_gemv_w4a16_cpu,
    awq_unpack_weights,
    dequantize_w4a4,
    pack_int4,
    svdq_gemm_w4a4_cpu,
    svdq_quantize_w4a4_act_fuse_lora_cpu,
    unpack_int4,
)
from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
from nunchaku.ops.gemv import awq_gemv_w4a16_cuda
from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
from nunchaku.utils import ceil_divide


# ---------------------------------------------------------------------------
# INT4 pack / unpack roundtrip
# ---------------------------------------------------------------------------


class TestPackUnpackInt4:
    """Test INT4 packing and unpacking roundtrip."""

    def test_pack_unpack_signed_roundtrip(self):
        """Pack signed INT4 values, then unpack, and verify equality."""
        values = torch.tensor([[-8, -1, 0, 1, 7, -3, 4, 6]], dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, signed=True)
        assert torch.equal(unpacked, values.float())

    def test_pack_unpack_unsigned_roundtrip(self):
        """Pack unsigned INT4 values, then unpack, verify equality."""
        values = torch.tensor([[0, 15, 1, 14, 7, 8, 3, 12]], dtype=torch.int16)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, signed=False)
        assert torch.equal(unpacked, values.float())

    def test_pack_unpack_batch(self):
        """Test with a small batch of INT4 values."""
        torch.manual_seed(42)
        values = torch.randint(-8, 8, (4, 64), dtype=torch.int8)
        packed = pack_int4(values)
        assert packed.shape == (4, 32)
        unpacked = unpack_int4(packed, signed=True)
        assert torch.equal(unpacked, values.float())

    def test_low_high_nibble_order(self):
        """Verify that even indices go to low nibble and odd to high nibble."""
        values = torch.tensor([[3, -2]], dtype=torch.int8)
        packed = pack_int4(values)
        # low nibble = 3 & 0xF = 3, high nibble = (-2 & 0xF) << 4 = 14 << 4 = 224
        # byte = 3 | 224 = 227
        byte_val = packed.view(torch.uint8).item()
        assert (byte_val & 0x0F) == 3  # low nibble
        assert ((byte_val >> 4) & 0x0F) == 14  # high nibble = -2 as unsigned = 14


# ---------------------------------------------------------------------------
# SVDQuant W4A4 dequantization
# ---------------------------------------------------------------------------


class TestDequantizeW4A4:
    """Test INT4 dequantization with per-group scales."""

    def test_basic_dequant(self):
        """Dequantize a small tensor and verify against manual computation."""
        # 1 row, 64 columns (1 group of 64)
        N, K, group_size = 1, 64, 64
        # Create known INT4 values: all 1
        values = torch.ones(N, K, dtype=torch.int8)
        packed = pack_int4(values)
        # Scale = 2.0 for the single group
        scales = torch.tensor([[2.0]])  # (K//group_size=1, N=1)
        result = dequantize_w4a4(packed, scales, group_size, signed=True)
        assert result.shape == (1, 64)
        assert torch.allclose(result, torch.full((1, 64), 2.0))

    def test_multi_group_dequant(self):
        """Test dequantization with multiple groups."""
        N, K, group_size = 2, 128, 64
        # Group 0: all 3, scale 1.0 → dequant = 3.0
        # Group 1: all -2, scale 0.5 → dequant = -1.0
        values = torch.zeros(N, K, dtype=torch.int8)
        values[:, :64] = 3
        values[:, 64:] = -2
        packed = pack_int4(values)

        scales = torch.zeros(2, N)  # (K//group_size=2, N=2)
        scales[0, :] = 1.0
        scales[1, :] = 0.5
        result = dequantize_w4a4(packed, scales, group_size, signed=True)
        assert torch.allclose(result[:, :64], torch.full((N, 64), 3.0))
        assert torch.allclose(result[:, 64:], torch.full((N, 64), -1.0))


# ---------------------------------------------------------------------------
# SVDQuant quantize + LoRA (CPU)
# ---------------------------------------------------------------------------


class TestSVDQQuantize:
    """Test SVDQuant W4A4 activation quantization + LoRA down-projection."""

    def _run_quantize(self, M=4, K=64, rank=4, pad_size=256, use_smooth=True):
        """Helper that runs quantization and returns all outputs."""
        torch.manual_seed(0)
        x = torch.randn(M, K, dtype=torch.bfloat16)
        lora_down = torch.randn(K, rank, dtype=torch.bfloat16)
        smooth = torch.randn(K, dtype=torch.bfloat16).abs() + 0.1 if use_smooth else None

        M_pad = ceil_divide(M, pad_size) * pad_size
        output = torch.empty(M_pad, K // 2, dtype=torch.uint8)
        oscales = torch.empty(K // 64, M_pad, dtype=torch.bfloat16)
        lora_act_out = torch.empty(M_pad, rank, dtype=torch.float32)

        svdq_quantize_w4a4_act_fuse_lora_cpu(
            x, output, oscales, lora_down, lora_act_out, smooth
        )
        return x, output, oscales, lora_act_out, lora_down, smooth

    def test_output_shapes(self):
        """Verify output tensor shapes."""
        M, K, rank = 4, 64, 4
        x, output, oscales, lora_act_out, _, _ = self._run_quantize(M, K, rank)
        M_pad = 256
        assert output.shape == (M_pad, K // 2)
        assert oscales.shape == (K // 64, M_pad)
        assert lora_act_out.shape == (M_pad, rank)

    def test_quantize_dequantize_roundtrip(self):
        """Quantize then dequantize and check it's close to the smoothed input."""
        M, K, rank = 4, 64, 4
        x, output, oscales, lora_act_out, lora_down, smooth = self._run_quantize(M, K, rank)

        # Dequantize
        dequant = dequantize_w4a4(output[:M], oscales[:, :M], group_size=64, signed=True)

        # The dequantized values should approximate x / smooth
        x_smoothed = x.float() / smooth.float()
        # Relative error should be small (quantization introduces ~1/7 = 14% error at most)
        abs_err = (dequant - x_smoothed).abs()
        # Each value is quantized to one of 15 levels, so max error ≈ scale / 2
        max_scale = oscales[:, :M].float().abs().max().item()
        assert abs_err.max().item() < max_scale, f"Max error {abs_err.max()} exceeds scale {max_scale}"

    def test_lora_down_projection(self):
        """Verify LoRA down-projection is computed on raw input (before smoothing)."""
        M, K, rank = 4, 64, 4
        x, output, oscales, lora_act_out, lora_down, smooth = self._run_quantize(M, K, rank)

        expected_lora = x.float() @ lora_down.float()
        actual_lora = lora_act_out[:M]
        assert torch.allclose(actual_lora, expected_lora, atol=1e-3, rtol=1e-3)

    def test_scales_positive(self):
        """Verify all scales are positive."""
        _, _, oscales, _, _, _ = self._run_quantize()
        # Only check the active rows (first M columns)
        assert (oscales[:, :4].float() >= 0).all()

    def test_dispatches_via_wrapper(self):
        """Verify the wrapper function dispatches to CPU implementation."""
        torch.manual_seed(0)
        M, K, rank = 4, 64, 4
        x = torch.randn(M, K, dtype=torch.bfloat16)
        lora_down = torch.randn(K, rank, dtype=torch.bfloat16)
        smooth = torch.randn(K, dtype=torch.bfloat16).abs() + 0.1

        output, oscales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x, lora_down=lora_down, smooth=smooth
        )
        assert output.device.type == "cpu"
        assert oscales.shape[0] == K // 64
        assert lora_act_out.shape[1] == rank


# ---------------------------------------------------------------------------
# SVDQuant W4A4 GEMM (CPU)
# ---------------------------------------------------------------------------


class TestSVDQGemm:
    """Test SVDQuant W4A4 GEMM on CPU."""

    def _make_quantized_data(self, M=4, K=64, N=64, rank=4):
        """Create quantized weights and activations for testing."""
        torch.manual_seed(42)
        group_size = 64

        # Create float weight matrix and quantize it
        W_float = torch.randn(N, K, dtype=torch.float32) * 0.1
        num_groups = K // group_size
        W_grouped = W_float.view(N, num_groups, group_size)
        w_max = W_grouped.abs().amax(dim=-1, keepdim=True)
        wscales_val = w_max / 7.0
        W_q = torch.round(W_grouped / wscales_val.clamp(min=1e-10)).clamp(-8, 7).to(torch.int8)
        wscales = wscales_val.squeeze(-1).T.to(torch.bfloat16)  # (num_groups, N)
        qweight = pack_int4(W_q.view(N, K))

        # Create float activation matrix and quantize it
        X_float = torch.randn(M, K, dtype=torch.float32)
        X_grouped = X_float.view(M, num_groups, group_size)
        x_max = X_grouped.abs().amax(dim=-1, keepdim=True)
        ascales_val = x_max / 7.0
        X_q = torch.round(X_grouped / ascales_val.clamp(min=1e-10)).clamp(-8, 7).to(torch.int8)
        ascales = ascales_val.squeeze(-1).T.to(torch.bfloat16)  # (num_groups, M)
        qact = pack_int4(X_q.view(M, K))

        # LoRA
        lora_act_in = torch.randn(M, rank, dtype=torch.float32)
        lora_up = torch.randn(N, rank, dtype=torch.bfloat16)
        bias = torch.randn(N, dtype=torch.bfloat16)

        return {
            "W_float": W_float,
            "X_float": X_float,
            "qweight": qweight,
            "qact": qact,
            "wscales": wscales,
            "ascales": ascales,
            "lora_act_in": lora_act_in,
            "lora_up": lora_up,
            "bias": bias,
            "W_q": W_q.view(N, K),
            "X_q": X_q.view(M, K),
        }

    def test_gemm_basic(self):
        """Test basic quantized GEMM without LoRA or bias."""
        M, K, N, rank = 4, 64, 64, 4
        data = self._make_quantized_data(M, K, N, rank)
        out = torch.zeros(M, N, dtype=torch.bfloat16)

        svdq_gemm_w4a4_cpu(
            act=data["qact"],
            wgt=data["qweight"],
            out=out,
            ascales=data["ascales"],
            wscales=data["wscales"],
            lora_act_in=data["lora_act_in"],
            lora_up=data["lora_up"],
        )

        # Reference: dequant_act @ dequant_wgt.T + lora
        dequant_act = dequantize_w4a4(data["qact"], data["ascales"], 64, signed=True)
        dequant_wgt = dequantize_w4a4(data["qweight"], data["wscales"], 64, signed=True)
        expected = dequant_act @ dequant_wgt.T + data["lora_act_in"] @ data["lora_up"].float().T
        assert torch.allclose(out.float(), expected.to(torch.bfloat16).float(), atol=0.1, rtol=0.05)

    def test_gemm_with_bias(self):
        """Test quantized GEMM with bias."""
        M, K, N, rank = 4, 64, 64, 4
        data = self._make_quantized_data(M, K, N, rank)
        out = torch.zeros(M, N, dtype=torch.bfloat16)

        svdq_gemm_w4a4_cpu(
            act=data["qact"],
            wgt=data["qweight"],
            out=out,
            ascales=data["ascales"],
            wscales=data["wscales"],
            lora_act_in=data["lora_act_in"],
            lora_up=data["lora_up"],
            bias=data["bias"],
        )

        dequant_act = dequantize_w4a4(data["qact"], data["ascales"], 64, signed=True)
        dequant_wgt = dequantize_w4a4(data["qweight"], data["wscales"], 64, signed=True)
        expected = dequant_act @ dequant_wgt.T + data["lora_act_in"] @ data["lora_up"].float().T + data["bias"].float()
        assert torch.allclose(out.float(), expected.to(torch.bfloat16).float(), atol=0.1, rtol=0.05)

    def test_gemm_dispatches_via_wrapper(self):
        """Verify wrapper dispatches to CPU when input is on CPU."""
        M, K, N, rank = 4, 64, 64, 4
        data = self._make_quantized_data(M, K, N, rank)
        out = torch.zeros(M, N, dtype=torch.bfloat16)

        svdq_gemm_w4a4_cuda(
            act=data["qact"],
            wgt=data["qweight"],
            out=out,
            ascales=data["ascales"],
            wscales=data["wscales"],
            lora_act_in=data["lora_act_in"],
            lora_up=data["lora_up"],
        )
        assert out.abs().sum().item() > 0  # Not all zeros → something was computed


# ---------------------------------------------------------------------------
# AWQ W4A16 dequantization and GEMV
# ---------------------------------------------------------------------------


class TestAWQ:
    """Test AWQ weight unpacking, dequantization, and GEMV."""

    def _make_awq_data(self, n=64, k=64, group_size=64):
        """Create AWQ-format packed weights from known float weights."""
        import ctypes

        torch.manual_seed(7)
        num_groups = k // group_size

        # Create float weights
        W_float = torch.randn(n, k, dtype=torch.float32) * 0.5

        # Compute per-group scales and zeros
        W_grouped = W_float.view(n, num_groups, group_size)
        w_min = W_grouped.amin(dim=-1)  # (n, num_groups)
        w_max = W_grouped.amax(dim=-1)
        scales = (w_max - w_min) / 15.0
        scales = scales.clamp(min=1e-10)
        zero_point = torch.round(-w_min / scales).clamp(0, 15)

        # Quantize to uint4
        W_q = torch.round(W_grouped / scales.unsqueeze(-1) + zero_point.unsqueeze(-1))
        W_q = W_q.clamp(0, 15).to(torch.int64).view(n, k)

        # Pack into AWQ interleaved format
        num_groups_32 = k // 32
        kernel = torch.zeros(n, k // 8, dtype=torch.int32)
        for oc in range(n):
            for g32 in range(num_groups_32):
                for u in range(4):
                    uint32_val = 0
                    for ni in range(8):
                        ic_offset = (ni % 4) * 8 + 2 * u + (ni // 4)
                        ic = g32 * 32 + ic_offset
                        val = W_q[oc, ic].item()
                        uint32_val |= int(val) << (ni * 4)
                    # Convert unsigned 32-bit to signed 32-bit for torch.int32
                    signed_val = ctypes.c_int32(uint32_val).value
                    kernel[oc, g32 * 4 + u] = signed_val

        kernel = kernel.reshape(n // 4, k // 2)

        # Store scales and zeros in the format expected by the kernel
        # scales: (k // group_size, n), zeros: (k // group_size, n)
        # zeros = -scale * zero_point (pre-scaled)
        scaling_factors = scales.T.to(torch.bfloat16)  # (num_groups, n)
        zeros_pre = (-scales * zero_point).T.to(torch.bfloat16)

        return {
            "W_float": W_float,
            "W_q": W_q,
            "kernel": kernel,
            "scaling_factors": scaling_factors,
            "zeros": zeros_pre,
            "scales_raw": scales,
            "zero_point": zero_point,
        }

    def test_unpack_roundtrip(self):
        """Verify that packing then unpacking recovers the original uint4 values."""
        n, k = 64, 64
        data = self._make_awq_data(n, k)
        unpacked = awq_unpack_weights(data["kernel"], n, k)
        assert torch.equal(unpacked, data["W_q"].float())

    def test_dequantize_close_to_float(self):
        """Verify dequantized weights are close to original float weights."""
        n, k, group_size = 64, 64, 64
        data = self._make_awq_data(n, k, group_size)
        dequant = awq_dequantize_weights(
            data["kernel"], data["scaling_factors"], data["zeros"], n, k, group_size
        )
        # Quantization error: each value is one of 16 levels, max error ≈ scale/2
        max_scale = data["scales_raw"].max().item()
        assert (dequant - data["W_float"]).abs().max().item() < max_scale

    def test_gemv_basic(self):
        """Test AWQ GEMV against float reference."""
        m, n, k, group_size = 2, 64, 64, 64
        data = self._make_awq_data(n, k, group_size)
        torch.manual_seed(99)
        x = torch.randn(m, k, dtype=torch.bfloat16)

        output = awq_gemv_w4a16_cpu(
            x, data["kernel"], data["scaling_factors"], data["zeros"], m, n, k, group_size
        )

        # Reference: x @ dequant_W.T
        dequant_W = awq_dequantize_weights(
            data["kernel"], data["scaling_factors"], data["zeros"], n, k, group_size
        )
        expected = x.float() @ dequant_W.T
        assert torch.allclose(output.float(), expected.to(torch.bfloat16).float(), atol=0.05, rtol=0.01)

    def test_gemv_dispatches_via_wrapper(self):
        """Verify wrapper function dispatches to CPU."""
        m, n, k, group_size = 2, 64, 64, 64
        data = self._make_awq_data(n, k, group_size)
        torch.manual_seed(99)
        x = torch.randn(m, k, dtype=torch.bfloat16)

        output = awq_gemv_w4a16_cuda(
            x, data["kernel"], data["scaling_factors"], data["zeros"], m, n, k, group_size
        )
        assert output.shape == (m, n)
        assert output.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# SVDQW4A4Linear end-to-end (CPU)
# ---------------------------------------------------------------------------


class TestSVDQLinearCPU:
    """Test the full SVDQW4A4Linear forward pass on CPU."""

    def _make_linear(self, in_features=64, out_features=64, rank=4):
        """Create a SVDQW4A4Linear with random quantized weights on CPU."""
        from nunchaku.models.linear import SVDQW4A4Linear

        torch.manual_seed(123)
        layer = SVDQW4A4Linear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            bias=True,
            precision="int4",
            torch_dtype=torch.bfloat16,
            device="cpu",
        )
        # Fill with random quantized weights
        with torch.no_grad():
            layer.qweight.copy_(torch.randint(-128, 127, layer.qweight.shape, dtype=torch.int8))
            layer.wscales.copy_(torch.randn_like(layer.wscales).abs() * 0.01)
            layer.smooth_factor.copy_(torch.randn_like(layer.smooth_factor).abs() + 0.1)
            layer.proj_down.copy_(torch.randn_like(layer.proj_down) * 0.01)
            layer.proj_up.copy_(torch.randn_like(layer.proj_up) * 0.01)
            layer.bias.copy_(torch.randn_like(layer.bias) * 0.01)
        return layer

    def test_forward_runs(self):
        """Test that forward pass runs without errors on CPU."""
        layer = self._make_linear()
        x = torch.randn(1, 2, 64, dtype=torch.bfloat16)
        out = layer(x)
        assert out.shape == (1, 2, 64)
        assert out.device.type == "cpu"

    def test_forward_deterministic(self):
        """Test that two identical runs produce the same output."""
        layer = self._make_linear()
        x = torch.randn(1, 2, 64, dtype=torch.bfloat16)
        out1 = layer(x)
        out2 = layer(x)
        assert torch.equal(out1, out2)

    def test_forward_close_to_dequant_reference(self):
        """Compare forward output to manual dequant → matmul → LoRA reference."""
        layer = self._make_linear()
        x = torch.randn(1, 2, 64, dtype=torch.bfloat16)
        out = layer(x)

        # Manual reference
        x_2d = x.view(2, 64)

        # LoRA down-projection (before smoothing)
        lora_act = x_2d.float() @ layer.proj_down.float()

        # Smooth then quantize
        x_smooth = x_2d.float() / layer.smooth_factor.float()
        x_grouped = x_smooth.view(2, 1, 64)
        gmax = x_grouped.abs().amax(dim=-1, keepdim=True)
        scale = gmax / 7.0
        scale = scale.clamp(min=1e-10)
        x_q = torch.round(x_grouped / scale).clamp(-8, 7)
        # Dequant: x_q * scale ≈ x_smooth
        x_dequant = (x_q * scale).view(2, 64)

        # Dequant weights
        dequant_wgt = dequantize_w4a4(layer.qweight, layer.wscales, 64, signed=True)

        # GEMM + LoRA + bias
        expected = x_dequant @ dequant_wgt.T + lora_act @ layer.proj_up.float().T + layer.bias.float()

        # Due to quantization, allow generous tolerance
        assert torch.allclose(out.view(2, 64).float(), expected.to(torch.bfloat16).float(), atol=1.0, rtol=0.5)


# ---------------------------------------------------------------------------
# AWQW4A16Linear end-to-end (CPU)
# ---------------------------------------------------------------------------


class TestAWQLinearCPU:
    """Test the full AWQW4A16Linear forward pass on CPU."""

    def _make_linear(self, in_features=64, out_features=64, group_size=64):
        """Create an AWQW4A16Linear with random quantized weights on CPU."""
        import ctypes

        from nunchaku.models.linear import AWQW4A16Linear

        torch.manual_seed(456)
        layer = AWQW4A16Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            group_size=group_size,
            torch_dtype=torch.bfloat16,
            device="cpu",
        )
        # Pack known weights into AWQ format
        n, k = out_features, in_features
        num_groups_32 = k // 32

        # Random uint4 weights
        W_q = torch.randint(0, 16, (n, k), dtype=torch.int64)
        kernel = torch.zeros(n, k // 8, dtype=torch.int32)
        for oc in range(n):
            for g32 in range(num_groups_32):
                for u in range(4):
                    uint32_val = 0
                    for ni in range(8):
                        ic_offset = (ni % 4) * 8 + 2 * u + (ni // 4)
                        ic = g32 * 32 + ic_offset
                        val = W_q[oc, ic].item()
                        uint32_val |= int(val) << (ni * 4)
                    signed_val = ctypes.c_int32(uint32_val).value
                    kernel[oc, g32 * 4 + u] = signed_val

        with torch.no_grad():
            layer.qweight.copy_(kernel.reshape(n // 4, k // 2))
            layer.wscales.copy_(torch.randn(k // group_size, n, dtype=torch.bfloat16).abs() * 0.01)
            layer.wzeros.copy_(-layer.wscales * 8)  # zero point = 8
            layer.bias.copy_(torch.randn(n, dtype=torch.bfloat16) * 0.01)
        return layer

    def test_forward_runs(self):
        """Test that forward pass runs without errors on CPU."""
        layer = self._make_linear()
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        out = layer(x)
        assert out.shape == (2, 64)
        assert out.device.type == "cpu"

    def test_forward_deterministic(self):
        """Two identical runs produce the same output."""
        layer = self._make_linear()
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        out1 = layer(x)
        out2 = layer(x)
        assert torch.equal(out1, out2)
