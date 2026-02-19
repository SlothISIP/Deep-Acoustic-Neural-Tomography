"""Unit tests for Phase 3 inverse model components.

Tests
-----
    test_sdf_decoder_output_shape:  verify (B, 1) output
    test_inverse_model_predict_sdf: verify predict_sdf method
    test_eikonal_on_known_sdf:     circle SDF -> |grad s| = 1 analytically
    test_iou_perfect:              identical SDFs -> IoU = 1.0
    test_iou_empty:                both empty -> IoU = 1.0
    test_iou_partial:              partial overlap -> 0 < IoU < 1
    test_cycle_gradient_flow:      verify gradients reach auto-decoder codes
    test_p_inc_torch_vs_scipy:     asymptotic p_inc matches scipy for large kr

Usage
-----
    python -m pytest tests/test_inverse_model.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inverse_model import (
    InverseModel,
    SDFDecoder,
    build_inverse_model,
    compute_p_inc_torch,
    compute_sdf_iou,
    eikonal_loss,
    sdf_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def sdf_decoder(device):
    """Small SDF decoder for testing."""
    return SDFDecoder(
        d_cond=32, d_hidden=64, n_blocks=2,
        n_fourier=16, fourier_sigma=10.0, dropout=0.0,
    ).to(device)


@pytest.fixture
def inverse_model(device):
    """Small inverse model for testing."""
    return build_inverse_model(
        n_scenes=3, d_cond=32, d_hidden=64,
        n_blocks=2, n_fourier=16, fourier_sigma=10.0, dropout=0.0,
    ).to(device)


# ---------------------------------------------------------------------------
# SDFDecoder tests
# ---------------------------------------------------------------------------
class TestSDFDecoder:
    def test_output_shape(self, sdf_decoder, device):
        """SDFDecoder produces (B, 1) output."""
        B = 64
        xy = torch.randn(B, 2, device=device)
        z = torch.randn(B, 32, device=device)
        out = sdf_decoder(xy, z)
        assert out.shape == (B, 1), f"Expected (64, 1), got {out.shape}"

    def test_output_finite(self, sdf_decoder, device):
        """SDFDecoder output is finite."""
        xy = torch.randn(128, 2, device=device)
        z = torch.randn(128, 32, device=device)
        out = sdf_decoder(xy, z)
        assert torch.all(torch.isfinite(out)), "Non-finite SDF output"

    def test_different_z_different_output(self, sdf_decoder, device):
        """Different latent codes produce different SDF predictions."""
        xy = torch.randn(32, 2, device=device)
        z1 = torch.randn(32, 32, device=device)
        z2 = torch.randn(32, 32, device=device)
        out1 = sdf_decoder(xy, z1)
        out2 = sdf_decoder(xy, z2)
        # Very unlikely to be exactly equal for random codes
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Different latent codes should produce different outputs"
        )


# ---------------------------------------------------------------------------
# InverseModel tests
# ---------------------------------------------------------------------------
class TestInverseModel:
    def test_predict_sdf_shape(self, inverse_model, device):
        """predict_sdf returns (B, 1)."""
        xy = torch.randn(100, 2, device=device)
        sdf = inverse_model.predict_sdf(0, xy)
        assert sdf.shape == (100, 1)

    def test_predict_sdf_different_scenes(self, inverse_model, device):
        """Different scenes produce different SDF predictions."""
        xy = torch.randn(50, 2, device=device)
        sdf0 = inverse_model.predict_sdf(0, xy)
        sdf1 = inverse_model.predict_sdf(1, xy)
        # Codes are initialized randomly, so outputs should differ
        assert not torch.allclose(sdf0, sdf1, atol=1e-6)

    def test_get_code(self, inverse_model):
        """get_code returns correct shape."""
        z = inverse_model.get_code(0)
        assert z.shape == (32,)  # d_cond=32

    def test_count_parameters(self, inverse_model):
        """count_parameters returns positive integer."""
        n = inverse_model.count_parameters()
        assert n > 0
        assert isinstance(n, int)


# ---------------------------------------------------------------------------
# Eikonal loss tests
# ---------------------------------------------------------------------------
class TestEikonalLoss:
    def test_eikonal_on_circle_sdf(self, device):
        """Circle SDF satisfies |grad s| = 1 analytically.

        For a circle of radius R centered at origin:
            s(x, y) = sqrt(x^2 + y^2) - R
            grad s = (x, y) / sqrt(x^2 + y^2)
            |grad s| = 1  (everywhere except origin)
        """
        R = 0.5
        # Random points away from origin
        xy = torch.randn(256, 2, device=device) * 0.5 + 0.3
        xy.requires_grad_(True)

        r = torch.sqrt(xy[:, 0:1] ** 2 + xy[:, 1:2] ** 2 + 1e-10)
        sdf_circle = r - R  # (256, 1)

        loss = eikonal_loss(sdf_circle, xy)
        # Circle SDF should have near-zero Eikonal loss
        assert loss.item() < 1e-4, (
            f"Eikonal loss on circle SDF should be ~0, got {loss.item():.4e}"
        )

    def test_eikonal_on_linear_sdf(self, device):
        """Linear SDF s(x,y) = ax + by with ||(a,b)|| = 1 has |grad s| = 1."""
        xy = torch.randn(256, 2, device=device)
        xy.requires_grad_(True)

        # Direction vector with unit norm
        direction = torch.tensor([0.6, 0.8], device=device)  # |dir| = 1.0
        sdf_linear = (xy @ direction).unsqueeze(-1)  # (256, 1)

        loss = eikonal_loss(sdf_linear, xy)
        assert loss.item() < 1e-6, (
            f"Eikonal loss on linear SDF should be ~0, got {loss.item():.4e}"
        )


# ---------------------------------------------------------------------------
# SDF loss tests
# ---------------------------------------------------------------------------
class TestSDFLoss:
    def test_sdf_loss_zero_for_perfect(self, device):
        """Identical predictions give zero loss."""
        sdf = torch.randn(100, 1, device=device)
        loss = sdf_loss(sdf, sdf)
        assert loss.item() < 1e-7

    def test_sdf_loss_positive_for_mismatch(self, device):
        """Mismatched predictions give positive loss."""
        pred = torch.randn(100, 1, device=device)
        gt = torch.randn(100, 1, device=device)
        loss = sdf_loss(pred, gt)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# IoU tests
# ---------------------------------------------------------------------------
class TestIoU:
    def test_iou_perfect(self, device):
        """Identical SDFs give IoU = 1.0."""
        sdf = torch.tensor([-1.0, -0.5, 0.5, 1.0], device=device)
        iou = compute_sdf_iou(sdf, sdf)
        assert iou == 1.0

    def test_iou_empty(self, device):
        """Both SDFs all-positive (no interior) gives IoU = 1.0."""
        sdf = torch.tensor([0.1, 0.5, 1.0, 2.0], device=device)
        iou = compute_sdf_iou(sdf, sdf)
        assert iou == 1.0

    def test_iou_no_overlap(self, device):
        """Non-overlapping interiors give IoU = 0.0."""
        pred = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=device)
        gt = torch.tensor([1.0, 1.0, -1.0, -1.0], device=device)
        iou = compute_sdf_iou(pred, gt)
        assert iou == 0.0

    def test_iou_partial_overlap(self, device):
        """Partial overlap gives 0 < IoU < 1."""
        pred = torch.tensor([-1.0, -1.0, -1.0, 1.0], device=device)
        gt = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=device)
        iou = compute_sdf_iou(pred, gt)
        # Intersection: 2, Union: 3 -> IoU = 2/3
        assert abs(iou - 2.0 / 3.0) < 1e-6

    def test_iou_with_2d_input(self, device):
        """IoU handles (N, 1) shaped inputs."""
        sdf = torch.tensor([[-1.0], [0.5], [-0.3], [1.0]], device=device)
        iou = compute_sdf_iou(sdf, sdf)
        assert iou == 1.0


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------
class TestGradientFlow:
    def test_cycle_gradient_reaches_codes(self, inverse_model, device):
        """Verify gradients flow from SDF loss back to auto-decoder codes."""
        xy = torch.randn(32, 2, device=device)
        scene_idx = 0

        sdf_pred = inverse_model.predict_sdf(scene_idx, xy)
        loss = sdf_pred.sum()
        loss.backward()

        # Check that auto-decoder codes have gradient
        code_grad = inverse_model.auto_decoder_codes.weight.grad
        assert code_grad is not None, (
            "No gradient on auto_decoder_codes after backward"
        )
        assert code_grad[scene_idx].abs().sum() > 0, (
            "Gradient on code for scene_idx should be non-zero"
        )

    def test_eikonal_creates_graph(self, inverse_model, device):
        """Eikonal loss creates computation graph for second-order optimization."""
        xy = torch.randn(32, 2, device=device, requires_grad=True)
        sdf_pred = inverse_model.predict_sdf(0, xy)
        loss = eikonal_loss(sdf_pred, xy)

        # Should be able to backward without error
        loss.backward()
        assert inverse_model.auto_decoder_codes.weight.grad is not None


# ---------------------------------------------------------------------------
# p_inc tests
# ---------------------------------------------------------------------------
class TestPIncTorch:
    def test_p_inc_shape(self, device):
        """compute_p_inc_torch returns correct shapes."""
        B = 64
        x_src = torch.randn(B, 2, device=device)
        x_rcv = torch.randn(B, 2, device=device) + 2.0
        k = torch.full((B,), 50.0, device=device)

        p_re, p_im = compute_p_inc_torch(x_src, x_rcv, k)
        assert p_re.shape == (B,)
        assert p_im.shape == (B,)

    def test_p_inc_finite(self, device):
        """p_inc values are finite."""
        x_src = torch.zeros(10, 2, device=device)
        x_rcv = torch.randn(10, 2, device=device) + 1.0
        k = torch.full((10,), 40.0, device=device)

        p_re, p_im = compute_p_inc_torch(x_src, x_rcv, k)
        assert torch.all(torch.isfinite(p_re))
        assert torch.all(torch.isfinite(p_im))

    def test_p_inc_vs_scipy(self, device):
        """Asymptotic p_inc matches scipy for large kr (kr > 20).

        p_inc = -(i/4) H_0^{(1)}(kr)
        """
        from scipy.special import hankel1

        x_src = torch.tensor([[0.0, 0.0]], device=device)
        x_rcv = torch.tensor([[0.5, 0.0]], device=device)  # r = 0.5
        k_val = 80.0  # kr = 40
        k = torch.tensor([k_val], device=device)

        p_re, p_im = compute_p_inc_torch(x_src, x_rcv, k)

        # Scipy reference
        r = 0.5
        kr = k_val * r
        H0 = hankel1(0, kr)
        p_ref = -0.25j * H0
        p_ref_re = float(p_ref.real)
        p_ref_im = float(p_ref.imag)

        # Tolerance: asymptotic has O(1/kr) correction, ~6% at kr=40.
        # For Helmholtz loss (weight 1e-4) this accuracy is sufficient.
        np.testing.assert_allclose(
            p_re.item(), p_ref_re, rtol=0.06,
            err_msg=f"Re(p_inc) mismatch: torch={p_re.item():.6f}, scipy={p_ref_re:.6f}",
        )
        np.testing.assert_allclose(
            p_im.item(), p_ref_im, rtol=0.06,
            err_msg=f"Im(p_inc) mismatch: torch={p_im.item():.6f}, scipy={p_ref_im:.6f}",
        )
