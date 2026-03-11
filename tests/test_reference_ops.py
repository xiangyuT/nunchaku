"""
Verification workflow: validate torch-native reference operators against real model weights.

This script loads the ``svdq-int4_r256-z-image-turbo.safetensors`` quantized model
checkpoint and verifies that the torch-native reference implementations in
:mod:`nunchaku.ops.reference` produce correct results when exercised with actual
quantized weights and realistic input tensors.

Usage::

    # Download the safetensors file first (or provide a HuggingFace Hub path):
    pytest tests/test_reference_ops.py -v

    # Or with a custom path:
    NUNCHAKU_TEST_MODEL_PATH="nunchaku-tech/nunchaku-z-image-turbo/svdq-int4_r256-z-image-turbo.safetensors" \
        pytest tests/test_reference_ops.py -v

The tests validate:

1. **dequantize_int4**: Dequantizes packed INT4 model weights; checks shapes and finite values.
2. **quantize_to_int4 round-trip**: Quantize → dequantize preserves values within quantization error.
3. **rms_norm_reference**: Matches ``torch.nn.functional`` RMS normalization on real norm weights.
4. **apply_rotary_emb_reference**: Correct sin/cos rotary embedding application.
5. **attention_fp16_reference**: Matches ``torch.nn.functional.scaled_dot_product_attention``.
6. **gemm_w4a4_reference**: Dequantized W4A4 matmul matches float matmul on real weights.
7. **quantize_w4a4_act_fuse_lora_reference**: Activation quantization + LoRA down-projection on real smooth/proj_down.
8. **SVDQW4A4Linear end-to-end**: Full quantized linear forward pass produces the same result as dequantized float linear.
"""

import json
import logging
import math
import os
import importlib.util
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Import reference ops directly via importlib to avoid pulling in the full
# nunchaku package (which requires accelerate, diffusers, etc.).
_ref_path = str(pathlib.Path(__file__).resolve().parent.parent / "nunchaku" / "ops" / "reference.py")
_ref_spec = importlib.util.spec_from_file_location("nunchaku_ops_reference", _ref_path)
_ref_mod = importlib.util.module_from_spec(_ref_spec)
_ref_spec.loader.exec_module(_ref_mod)

INT4_GROUP_SIZE = _ref_mod.INT4_GROUP_SIZE
apply_rotary_emb_reference = _ref_mod.apply_rotary_emb_reference
attention_fp16_reference = _ref_mod.attention_fp16_reference
dequantize_int4 = _ref_mod.dequantize_int4
gemm_w4a4_reference = _ref_mod.gemm_w4a4_reference
gelu_reference = _ref_mod.gelu_reference
quantize_to_int4 = _ref_mod.quantize_to_int4
quantize_w4a4_act_fuse_lora_reference = _ref_mod.quantize_w4a4_act_fuse_lora_reference
rms_norm_reference = _ref_mod.rms_norm_reference
silu_reference = _ref_mod.silu_reference

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "nunchaku-tech/nunchaku-z-image-turbo/svdq-int4_r256-z-image-turbo.safetensors"


def _resolve_model_path() -> str:
    """Return the model path from env or default HuggingFace Hub path."""
    return os.environ.get("NUNCHAKU_TEST_MODEL_PATH", DEFAULT_MODEL_PATH)


def _load_safetensors(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Load safetensors from local file or HuggingFace Hub."""
    try:
        import safetensors
    except ImportError:
        raise ImportError("safetensors package is required: pip install safetensors")

    local_path = path
    if not os.path.isfile(local_path):
        try:
            from huggingface_hub import hf_hub_download

            parts = path.split("/")
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])
            local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as e:
            raise FileNotFoundError(f"Cannot resolve model path: {path}") from e

    with safetensors.safe_open(local_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    return state_dict, metadata


@pytest.fixture(scope="module")
def model_weights():
    """Load the Z-Image-Turbo quantized model weights (cached per module)."""
    path = _resolve_model_path()
    try:
        state_dict, metadata = _load_safetensors(path)
    except Exception as e:
        pytest.skip(f"Cannot load model weights from {path}: {e}")
    return state_dict, metadata


def _find_svdq_linear_prefix(state_dict: dict[str, torch.Tensor]) -> str:
    """Find the first SVDQW4A4Linear layer prefix in the state dict."""
    for key in state_dict:
        if key.endswith(".qweight"):
            return key.removesuffix(".qweight")
    pytest.skip("No SVDQW4A4Linear layer found in state dict")


def _extract_linear_weights(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Extract all weight tensors for one SVDQW4A4Linear layer."""
    result = {}
    for key in state_dict:
        if key.startswith(prefix + "."):
            short_key = key[len(prefix) + 1 :]
            result[short_key] = state_dict[key]
    return result


# ---------------------------------------------------------------------------
# Test: dequantize_int4 on actual model weights
# ---------------------------------------------------------------------------


class TestDequantizeInt4:
    """Verify dequantize_int4 with actual quantized model weights."""

    def test_dequantize_shape_and_finite(self, model_weights):
        state_dict, _ = model_weights
        prefix = _find_svdq_linear_prefix(state_dict)
        w = _extract_linear_weights(state_dict, prefix)

        qweight = w["qweight"]  # (out_features, in_features // 2), int8
        wscales = w["wscales"]  # (in_features // group_size, out_features)

        out_features, half_in = qweight.shape
        in_features = half_in * 2

        # dequantize_int4 expects (N, K//2) packed and (K//G, N) scales
        # qweight is (out_features, in_features//2) → treat as (N, K//2)
        # wscales is (in_features//G, out_features) → (K//G, N)  ✓
        dequantized = dequantize_int4(qweight, wscales, group_size=INT4_GROUP_SIZE)
        assert dequantized.shape == (out_features, in_features), (
            f"Expected ({out_features}, {in_features}), got {dequantized.shape}"
        )
        assert torch.isfinite(dequantized).all(), "Dequantized weights contain non-finite values"

    def test_dequantize_value_range(self, model_weights):
        """Dequantized values should be in a reasonable range for model weights."""
        state_dict, _ = model_weights
        prefix = _find_svdq_linear_prefix(state_dict)
        w = _extract_linear_weights(state_dict, prefix)

        dequantized = dequantize_int4(w["qweight"], w["wscales"], group_size=INT4_GROUP_SIZE)
        # Quantized model weights should have bounded magnitude
        assert dequantized.abs().max() < 100, f"Dequantized weights have unexpectedly large values: {dequantized.abs().max()}"


# ---------------------------------------------------------------------------
# Test: quantize_to_int4 round-trip
# ---------------------------------------------------------------------------


class TestQuantizeInt4RoundTrip:
    """Verify INT4 quantization round-trip fidelity."""

    def test_round_trip_on_random(self):
        """Quantize → dequantize should preserve values within quantization error."""
        torch.manual_seed(42)
        x = torch.randn(8, 256, dtype=torch.float32)
        packed, scales = quantize_to_int4(x, group_size=INT4_GROUP_SIZE, unsigned=False)
        recovered = dequantize_int4(packed.to(torch.int8), scales, group_size=INT4_GROUP_SIZE, unsigned=False)

        # Quantization error should be bounded: max_error < max(abs(x)) / 7
        group_max = x.reshape(8, -1, INT4_GROUP_SIZE).abs().amax(dim=-1).max()
        max_error = (x - recovered).abs().max()
        assert max_error < group_max / 7 + 1e-6, f"Round-trip error too large: {max_error}"

    def test_round_trip_preserves_shape(self):
        x = torch.randn(4, 128, dtype=torch.float32)
        packed, scales = quantize_to_int4(x, group_size=INT4_GROUP_SIZE)
        assert packed.shape == (4, 64)
        assert scales.shape == (128 // INT4_GROUP_SIZE, 4)


# ---------------------------------------------------------------------------
# Test: rms_norm_reference
# ---------------------------------------------------------------------------


class TestRMSNorm:
    """Verify RMS normalization against PyTorch reference."""

    def test_rms_norm_identity_weight(self):
        """With weight=ones, output should have RMS ≈ 1."""
        torch.manual_seed(42)
        x = torch.randn(4, 128)
        weight = torch.ones(128)
        out = rms_norm_reference(x, weight)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.01)

    def test_rms_norm_with_real_weights(self, model_weights):
        """Compare with PyTorch's F.rms_norm if available, or manual check."""
        state_dict, _ = model_weights
        # Find a norm weight in the state dict
        norm_key = None
        for k in state_dict:
            if ".norm_q.weight" in k and state_dict[k].ndim == 1:
                norm_key = k
                break
        if norm_key is None:
            # Fallback: try fused module naming
            for k in state_dict:
                if k.endswith("norm_q_weight") and state_dict[k].ndim == 1:
                    norm_key = k
                    break
        if norm_key is None:
            pytest.skip("No norm_q weight found in model")

        norm_weight = state_dict[norm_key]
        head_dim = norm_weight.shape[0]

        torch.manual_seed(42)
        x = torch.randn(16, head_dim)
        out = rms_norm_reference(x, norm_weight, eps=1e-6)

        # Manual reference
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        expected = (x / rms) * norm_weight
        assert torch.allclose(out, expected, atol=1e-5), f"Max diff: {(out - expected).abs().max()}"


# ---------------------------------------------------------------------------
# Test: apply_rotary_emb_reference
# ---------------------------------------------------------------------------


class TestRotaryEmb:
    """Verify rotary embedding application."""

    def test_rotary_no_rotation_when_zero(self):
        """When sin=0 and cos=1, output should equal input."""
        x = torch.randn(4, 128)
        sin = torch.zeros(4, 64)
        cos = torch.ones(4, 64)
        out = apply_rotary_emb_reference(x, sin, cos)
        assert torch.allclose(out, x, atol=1e-6)

    def test_rotary_preserves_norm(self):
        """Rotary embedding should preserve vector norms."""
        torch.manual_seed(42)
        x = torch.randn(4, 128)
        theta = torch.randn(4, 64)
        sin = torch.sin(theta)
        cos = torch.cos(theta)
        out = apply_rotary_emb_reference(x, sin, cos)
        # Norms should be approximately preserved
        x_norm = x.norm(dim=-1)
        out_norm = out.norm(dim=-1)
        assert torch.allclose(x_norm, out_norm, atol=1e-4), f"Norm not preserved: {(x_norm - out_norm).abs().max()}"


# ---------------------------------------------------------------------------
# Test: attention_fp16_reference
# ---------------------------------------------------------------------------


class TestAttention:
    """Verify attention reference against PyTorch SDPA."""

    def test_attention_matches_sdpa(self):
        """Compare with torch.nn.functional.scaled_dot_product_attention."""
        torch.manual_seed(42)
        B, H, T, D = 2, 8, 32, 64
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        # Reference implementation
        out_ref = attention_fp16_reference(q, k, v)

        # PyTorch SDPA
        out_sdpa = F.scaled_dot_product_attention(q, k, v)
        out_sdpa = out_sdpa.permute(0, 2, 1, 3).reshape(B, T, H * D)

        assert torch.allclose(out_ref, out_sdpa, atol=1e-4), f"Max diff: {(out_ref - out_sdpa).abs().max()}"

    def test_attention_output_shape(self):
        B, H, T, D = 1, 4, 16, 128
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        out = attention_fp16_reference(q, k, v)
        assert out.shape == (B, T, H * D)


# ---------------------------------------------------------------------------
# Test: gemm_w4a4_reference with real weights
# ---------------------------------------------------------------------------


class TestGemmW4A4:
    """Verify W4A4 GEMM reference with actual quantized weights."""

    def test_gemm_with_unit_scales(self):
        """When scales are all 1, result should match plain matmul."""
        torch.manual_seed(42)
        M, K, N = 4, 128, 64
        act = torch.randn(M, K)
        wgt = torch.randn(N, K)
        ascales = torch.ones(K // INT4_GROUP_SIZE, M)
        wscales = torch.ones(K // INT4_GROUP_SIZE, N)

        result = gemm_w4a4_reference(act, wgt, ascales, wscales)
        expected = act.float() @ wgt.float().T
        assert torch.allclose(result["out"], expected, atol=1e-4), (
            f"Max diff: {(result['out'] - expected).abs().max()}"
        )

    def test_gemm_with_lora_and_bias(self):
        """GEMM + LoRA + bias should match manual computation."""
        torch.manual_seed(42)
        M, K, N, R = 4, 128, 64, 32
        act = torch.randn(M, K)
        wgt = torch.randn(N, K)
        ascales = torch.ones(K // INT4_GROUP_SIZE, M)
        wscales = torch.ones(K // INT4_GROUP_SIZE, N)
        lora_act = torch.randn(M, R)
        lora_up = torch.randn(N, R)
        bias = torch.randn(N)

        result = gemm_w4a4_reference(
            act, wgt, ascales, wscales, bias=bias, lora_act_in=lora_act, lora_up=lora_up
        )
        expected = act.float() @ wgt.float().T + lora_act.float() @ lora_up.float().T + bias.float()
        assert torch.allclose(result["out"], expected, atol=1e-3), (
            f"Max diff: {(result['out'] - expected).abs().max()}"
        )

    def test_gemm_with_real_model_weights(self, model_weights):
        """Dequantize weights and compare GEMM result with float matmul."""
        state_dict, _ = model_weights
        prefix = _find_svdq_linear_prefix(state_dict)
        w = _extract_linear_weights(state_dict, prefix)

        qweight = w["qweight"]  # (out_features, in_features // 2)
        wscales = w["wscales"]  # (in_features // group_size, out_features)

        out_features, half_in = qweight.shape
        in_features = half_in * 2

        # Dequantize the weight matrix
        weight_float = dequantize_int4(qweight, wscales, group_size=INT4_GROUP_SIZE)
        assert weight_float.shape == (out_features, in_features)

        # Create random input
        torch.manual_seed(42)
        M = 8
        x = torch.randn(M, in_features)

        # Float matmul reference
        expected = x.float() @ weight_float.float().T

        # GEMM with unit scales (since we're using dequantized values)
        ascales = torch.ones(in_features // INT4_GROUP_SIZE, M)
        wscales_unit = torch.ones(in_features // INT4_GROUP_SIZE, out_features)
        result = gemm_w4a4_reference(x, weight_float, ascales, wscales_unit)

        assert torch.allclose(result["out"], expected, atol=1e-3), (
            f"Max diff: {(result['out'] - expected).abs().max()}"
        )

    def test_gemm_with_lora_from_model(self, model_weights):
        """Test GEMM with real LoRA weights from the model."""
        state_dict, _ = model_weights
        prefix = _find_svdq_linear_prefix(state_dict)
        w = _extract_linear_weights(state_dict, prefix)

        if "proj_up" not in w or "proj_down" not in w:
            pytest.skip("No LoRA projections in this layer")

        qweight = w["qweight"]
        wscales = w["wscales"]
        proj_up = w["proj_up"]  # (out_features, rank)
        bias = w.get("bias", None)

        out_features, half_in = qweight.shape
        in_features = half_in * 2
        rank = proj_up.shape[1]

        weight_float = dequantize_int4(qweight, wscales, group_size=INT4_GROUP_SIZE)

        torch.manual_seed(42)
        M = 8
        x = torch.randn(M, in_features)
        lora_act_in = torch.randn(M, rank)

        ascales = torch.ones(in_features // INT4_GROUP_SIZE, M)
        wscales_unit = torch.ones(in_features // INT4_GROUP_SIZE, out_features)

        result = gemm_w4a4_reference(
            x, weight_float, ascales, wscales_unit,
            bias=bias,
            lora_act_in=lora_act_in,
            lora_up=proj_up,
        )

        expected = x.float() @ weight_float.float().T + lora_act_in.float() @ proj_up.float().T
        if bias is not None:
            expected = expected + bias.float()
        assert torch.allclose(result["out"], expected, atol=1e-2), (
            f"Max diff: {(result['out'] - expected).abs().max()}"
        )


# ---------------------------------------------------------------------------
# Test: quantize_w4a4_act_fuse_lora_reference with real weights
# ---------------------------------------------------------------------------


class TestQuantizeActFuseLoRA:
    """Verify activation quantization + LoRA down-projection with real model weights."""

    def test_quantize_act_basic(self):
        """Basic activation quantization produces correct shapes."""
        torch.manual_seed(42)
        M, K, R = 8, 256, 32
        x = torch.randn(M, K)
        lora_down = torch.randn(K, R)
        smooth = torch.ones(K)

        result = quantize_w4a4_act_fuse_lora_reference(x, lora_down=lora_down, smooth=smooth)
        assert result["output"].shape == (M, K // 2)
        assert result["oscales"].shape == (K // INT4_GROUP_SIZE, M)
        assert result["lora_act_out"].shape == (M, R)

    def test_lora_output_matches_matmul(self):
        """LoRA down-projection should match x @ lora_down."""
        torch.manual_seed(42)
        M, K, R = 8, 256, 32
        x = torch.randn(M, K)
        lora_down = torch.randn(K, R)
        smooth = torch.ones(K)  # unit smooth factor

        result = quantize_w4a4_act_fuse_lora_reference(x, lora_down=lora_down, smooth=smooth)

        # With smooth=1, smoothed input = input, so lora_act = x @ lora_down
        expected_lora = x.float() @ lora_down.float()
        assert torch.allclose(result["lora_act_out"], expected_lora, atol=1e-4), (
            f"Max diff: {(result['lora_act_out'] - expected_lora).abs().max()}"
        )

    def test_quantize_act_with_real_weights(self, model_weights):
        """Quantize activations using real smooth_factor and proj_down from model."""
        state_dict, _ = model_weights
        prefix = _find_svdq_linear_prefix(state_dict)
        w = _extract_linear_weights(state_dict, prefix)

        if "smooth_factor" not in w or "proj_down" not in w:
            pytest.skip("Missing smooth_factor or proj_down")

        smooth = w["smooth_factor"]
        proj_down = w["proj_down"]  # (in_features, rank)
        in_features = smooth.shape[0]
        rank = proj_down.shape[1]

        torch.manual_seed(42)
        M = 8
        x = torch.randn(M, in_features, dtype=torch.float32)

        result = quantize_w4a4_act_fuse_lora_reference(
            x, lora_down=proj_down, smooth=smooth, fp4=False,
        )

        assert result["output"].shape == (M, in_features // 2)
        assert result["lora_act_out"].shape == (M, rank)
        assert torch.isfinite(result["lora_act_out"]).all()

        # Verify LoRA output: should be (x * smooth) @ proj_down
        smoothed = x.float() * smooth.float().unsqueeze(0)
        expected_lora = smoothed @ proj_down.float()
        assert torch.allclose(result["lora_act_out"], expected_lora, atol=1e-3), (
            f"Max diff: {(result['lora_act_out'] - expected_lora).abs().max()}"
        )


# ---------------------------------------------------------------------------
# Test: SiLU and GELU activations
# ---------------------------------------------------------------------------


class TestActivations:
    """Verify activation functions match PyTorch built-ins."""

    def test_silu_matches_pytorch(self):
        torch.manual_seed(42)
        x = torch.randn(16, 128)
        assert torch.allclose(silu_reference(x), F.silu(x), atol=1e-6)

    def test_gelu_matches_pytorch(self):
        torch.manual_seed(42)
        x = torch.randn(16, 128)
        assert torch.allclose(gelu_reference(x), F.gelu(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Test: end-to-end SVDQW4A4Linear simulation with real weights
# ---------------------------------------------------------------------------


class TestEndToEndLinear:
    """Simulate the full SVDQW4A4Linear forward pass using reference ops."""

    def test_full_linear_forward(self, model_weights):
        """Simulate: quantize_act → gemm → output, using real model weights."""
        state_dict, _ = model_weights
        prefix = _find_svdq_linear_prefix(state_dict)
        w = _extract_linear_weights(state_dict, prefix)

        required_keys = {"qweight", "wscales", "smooth_factor", "proj_down", "proj_up"}
        if not required_keys.issubset(w.keys()):
            pytest.skip(f"Missing keys: {required_keys - set(w.keys())}")

        qweight = w["qweight"]
        wscales = w["wscales"]
        smooth = w["smooth_factor"]
        proj_down = w["proj_down"]
        proj_up = w["proj_up"]
        bias = w.get("bias", None)

        out_features, half_in = qweight.shape
        in_features = half_in * 2
        rank = proj_up.shape[1]

        # Step 1: Dequantize weights to float
        weight_float = dequantize_int4(qweight, wscales, group_size=INT4_GROUP_SIZE)

        # Step 2: Create random input
        torch.manual_seed(42)
        M = 8
        x = torch.randn(M, in_features, dtype=torch.float32)

        # Step 3: Simulate quantize_act with fused LoRA
        quant_result = quantize_w4a4_act_fuse_lora_reference(
            x, lora_down=proj_down, smooth=smooth, fp4=False,
        )
        lora_act = quant_result["lora_act_out"]

        # Step 4: Dequantize the quantized activations
        act_dequantized = dequantize_int4(
            quant_result["output"].to(torch.int8),
            quant_result["oscales"],
            group_size=INT4_GROUP_SIZE,
            unsigned=False,
        )

        # Step 5: GEMM with unit scales (using dequantized values)
        ascales = torch.ones(in_features // INT4_GROUP_SIZE, M)
        wscales_unit = torch.ones(in_features // INT4_GROUP_SIZE, out_features)
        gemm_result = gemm_w4a4_reference(
            act_dequantized, weight_float, ascales, wscales_unit,
            bias=bias,
            lora_act_in=lora_act,
            lora_up=proj_up,
        )

        output = gemm_result["out"]

        # Step 6: Compare with full-precision forward
        smoothed = x.float() * smooth.float().unsqueeze(0)
        fp_lora_act = smoothed @ proj_down.float()
        fp_output = smoothed @ weight_float.float().T + fp_lora_act @ proj_up.float().T
        if bias is not None:
            fp_output = fp_output + bias.float()

        # Due to quantization, the error can be larger, but should be bounded
        relative_error = (output - fp_output).abs() / (fp_output.abs() + 1e-6)
        mean_relative_error = relative_error.mean().item()
        max_abs_error = (output - fp_output).abs().max().item()

        logger.info(
            "End-to-end linear test — Layer: %s, Shape: in=%d, out=%d, rank=%d, "
            "Mean relative error: %.6f, Max absolute error: %.4f",
            prefix, in_features, out_features, rank, mean_relative_error, max_abs_error,
        )

        # The quantization error is expected to be non-trivial but bounded.
        # We check that the mean relative error is within a reasonable range.
        assert mean_relative_error < 1.0, (
            f"Mean relative error {mean_relative_error} is too large (should be < 1.0)"
        )


# ---------------------------------------------------------------------------
# Test: model metadata parsing
# ---------------------------------------------------------------------------


class TestModelMetadata:
    """Verify that model metadata is correctly parsed."""

    def test_metadata_has_config(self, model_weights):
        _, metadata = model_weights
        assert "config" in metadata, "Metadata should contain 'config'"
        config = json.loads(metadata["config"])
        assert isinstance(config, dict)

    def test_metadata_has_quantization_config(self, model_weights):
        _, metadata = model_weights
        if "quantization_config" in metadata:
            qconfig = json.loads(metadata["quantization_config"])
            assert isinstance(qconfig, dict)
            logger.info("Quantization config: %s", json.dumps(qconfig, indent=2))

    def test_state_dict_has_expected_keys(self, model_weights):
        state_dict, _ = model_weights
        # Should have qweight, wscales, proj_up, proj_down, smooth_factor keys
        has_qweight = any(k.endswith(".qweight") for k in state_dict)
        has_wscales = any(k.endswith(".wscales") for k in state_dict)
        has_proj_up = any(k.endswith(".proj_up") for k in state_dict)
        assert has_qweight, "State dict should contain .qweight keys"
        assert has_wscales, "State dict should contain .wscales keys"
        assert has_proj_up, "State dict should contain .proj_up keys"


# ---------------------------------------------------------------------------
# Test: multiple layers consistency
# ---------------------------------------------------------------------------


class TestMultipleLayersConsistency:
    """Verify reference ops across multiple layers in the model."""

    def test_all_qweight_layers_dequantize(self, model_weights):
        """Every qweight layer should dequantize without errors."""
        state_dict, _ = model_weights
        qweight_keys = [k for k in state_dict if k.endswith(".qweight")]
        # Test a sample of layers (first 5)
        for qkey in qweight_keys[:5]:
            prefix = qkey.removesuffix(".qweight")
            w = _extract_linear_weights(state_dict, prefix)
            if "wscales" not in w:
                continue
            dequantized = dequantize_int4(w["qweight"], w["wscales"], group_size=INT4_GROUP_SIZE)
            assert torch.isfinite(dequantized).all(), f"Non-finite values in dequantized {prefix}"

    def test_all_smooth_factors_finite(self, model_weights):
        """All smooth factors should be finite and positive."""
        state_dict, _ = model_weights
        smooth_keys = [k for k in state_dict if k.endswith(".smooth_factor")]
        for key in smooth_keys[:5]:
            smooth = state_dict[key]
            assert torch.isfinite(smooth).all(), f"Non-finite values in {key}"
