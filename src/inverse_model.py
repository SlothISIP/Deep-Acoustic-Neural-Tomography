"""Phase 3 Inverse Model: Sound -> Geometry.

Architecture
------------
    InverseModel
    +-- auto_decoder_codes: nn.Embedding(n_scenes, d_cond)
    |   Per-scene learnable latent code z_i (DeepSDF approach)
    +-- SDFDecoder
    |   +-- FourierFeatureEncoder(dim=2, n=128, sigma=10)
    |   +-- cond_proj: Linear(fourier_dim + d_cond, d_hidden) + GELU
    |   +-- blocks: 6 x ResidualBlock(d_hidden)
    |   +-- head: LayerNorm(d_hidden) -> Linear(d_hidden, 1)  [SDF in meters]
    +-- frozen_forward: TransferFunctionModel (Phase 2, eval mode, no grad)

Loss Functions
--------------
    L_sdf:       L1 near boundary (|s| < delta), L2 elsewhere
    L_eikonal:   mean((|grad(s)| - 1)^2) via autograd
    L_cycle:     ||p_pred - p_gt||^2 / ||p_gt||^2  through frozen forward
    L_helmholtz: mean(|laplacian(p) + k^2 p|^2)  at exterior points (FP32)
    L_z_reg:     ||z||^2  latent code regularization

Why auto-decoder?
    With only 15 training scenes, an acoustic encoder lacks diversity to
    learn a meaningful pressure->geometry mapping.  Auto-decoder (Park et
    al., 2019) optimizes per-scene latent codes directly via backprop.
    An encoder can be added in Phase 4 for generalization.

Reference
---------
    Park et al. (2019) "DeepSDF: Learning Continuous Signed Distance
    Functions for Shape Representation"
"""

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.forward_model import (
    FourierFeatureEncoder,
    ResidualBlock,
    TransferFunctionModel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0


# ---------------------------------------------------------------------------
# SDF Decoder
# ---------------------------------------------------------------------------
class SDFDecoder(nn.Module):
    """Conditional SDF decoder: (x, y) + z -> signed distance [m].

    Maps 2D spatial coordinates conditioned on a latent code z to the
    signed distance field value.  Uses Fourier features for positional
    encoding and residual blocks for the main network.

    Parameters
    ----------
    d_cond : int
        Latent code dimension.
    d_hidden : int
        Hidden layer width.
    n_blocks : int
        Number of residual blocks.
    n_fourier : int
        Number of Fourier features for spatial encoding.
    fourier_sigma : float
        Fourier feature bandwidth [m^-1].
    dropout : float
        Dropout rate in residual blocks.
    """

    def __init__(
        self,
        d_cond: int = 256,
        d_hidden: int = 256,
        n_blocks: int = 6,
        n_fourier: int = 128,
        fourier_sigma: float = 10.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_cond = d_cond
        self.d_hidden = d_hidden

        # Fourier features for 2D spatial input
        self.encoder = FourierFeatureEncoder(
            input_dim=2,
            n_features=n_fourier,
            sigma=fourier_sigma,
        )
        fourier_dim = self.encoder.output_dim  # 2*n_fourier + 2

        # Conditioning projection: concat(Fourier(xy), z) -> d_hidden
        self.cond_proj = nn.Sequential(
            nn.Linear(fourier_dim + d_cond, d_hidden),
            nn.GELU(),
        )

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(d_hidden, dropout=dropout) for _ in range(n_blocks)]
        )

        # Output head: SDF value [m]
        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        xy_m: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Predict SDF value at spatial coordinates.

        Parameters
        ----------
        xy_m : torch.Tensor, shape (B, 2)
            Spatial coordinates [m].
        z : torch.Tensor, shape (B, d_cond)
            Per-sample latent code.

        Returns
        -------
        sdf : torch.Tensor, shape (B, 1)
            Signed distance [m]. Negative inside body, positive outside.
        """
        feat = self.encoder(xy_m)  # (B, 2*n_fourier + 2)
        h = torch.cat([feat, z], dim=-1)  # (B, fourier_dim + d_cond)
        h = self.cond_proj(h)  # (B, d_hidden)

        for block in self.blocks:
            h = block(h)  # (B, d_hidden)

        return self.head(h)  # (B, 1)


# ---------------------------------------------------------------------------
# Inverse Model
# ---------------------------------------------------------------------------
class InverseModel(nn.Module):
    """Phase 3 inverse model: acoustic observations -> SDF geometry.

    Uses auto-decoder approach (DeepSDF): per-scene learnable latent codes
    are optimized jointly with the SDF decoder via backprop.

    Multi-body support: scenes with K>1 codes use smooth-min composition
    to represent disjoint geometry (e.g., S12 dual parallel bars).

    Parameters
    ----------
    n_scenes : int
        Number of scenes (for auto-decoder codes).
    d_cond : int
        Latent code dimension.
    d_hidden : int
        SDF decoder hidden width.
    n_blocks : int
        SDF decoder depth.
    n_fourier : int
        Fourier features for spatial encoding.
    fourier_sigma : float
        Fourier feature bandwidth [m^-1].
    dropout : float
        Dropout rate in SDF decoder.
    codes_per_scene : dict, optional
        Mapping {scene_idx: K} for multi-code scenes.
        Default: all scenes get K=1.
    smooth_min_alpha : float
        Sharpness of smooth-min (log-sum-exp) approximation.
        Higher = closer to hard min. Default 50.0.
    """

    def __init__(
        self,
        n_scenes: int = 15,
        d_cond: int = 256,
        d_hidden: int = 256,
        n_blocks: int = 6,
        n_fourier: int = 128,
        fourier_sigma: float = 10.0,
        dropout: float = 0.05,
        codes_per_scene: Optional[Dict[int, int]] = None,
        smooth_min_alpha: float = 50.0,
    ) -> None:
        super().__init__()
        self.n_scenes = n_scenes
        self.d_cond = d_cond
        self.smooth_min_alpha = smooth_min_alpha

        # Build code allocation: how many codes per scene
        if codes_per_scene is None:
            codes_per_scene = {}
        self.codes_per_scene = codes_per_scene  # {scene_idx: K}

        # Compute total number of codes and per-scene ranges
        total_codes = 0
        self._scene_code_ranges: Dict[int, Tuple[int, int]] = {}
        for si in range(n_scenes):
            k = codes_per_scene.get(si, 1)
            self._scene_code_ranges[si] = (total_codes, total_codes + k)
            total_codes += k
        self._total_codes = total_codes

        # Auto-decoder: variable-size code table
        self.auto_decoder_codes = nn.Embedding(total_codes, d_cond)
        nn.init.normal_(self.auto_decoder_codes.weight, std=0.01)

        # SDF decoder (shared across all codes)
        self.sdf_decoder = SDFDecoder(
            d_cond=d_cond,
            d_hidden=d_hidden,
            n_blocks=n_blocks,
            n_fourier=n_fourier,
            fourier_sigma=fourier_sigma,
            dropout=dropout,
        )

    def n_codes_for_scene(self, scene_idx: int) -> int:
        """Number of latent codes for a scene."""
        start, end = self._scene_code_ranges[scene_idx]
        return end - start

    def predict_sdf(
        self,
        scene_idx: int,
        xy_m: torch.Tensor,
    ) -> torch.Tensor:
        """Predict SDF at spatial coordinates for a given scene.

        For single-code scenes (K=1): standard forward pass.
        For multi-code scenes (K>1): predict SDF per code, compose
        via smooth-min (log-sum-exp approximation to min).

        Parameters
        ----------
        scene_idx : int
            0-indexed scene index into auto-decoder codes.
        xy_m : torch.Tensor, shape (B, 2)
            Spatial coordinates [m].

        Returns
        -------
        sdf : torch.Tensor, shape (B, 1)
            Signed distance prediction [m].
        """
        start, end = self._scene_code_ranges[scene_idx]
        K = end - start

        if K == 1:
            # Single code: original fast path
            z = self.auto_decoder_codes.weight[start]  # (d_cond,)
            z_exp = z.unsqueeze(0).expand(xy_m.shape[0], -1)  # (B, d_cond)
            return self.sdf_decoder(xy_m, z_exp)  # (B, 1)

        # Multi-code: predict SDF per code, compose via smooth-min
        B = xy_m.shape[0]
        sdf_stack = []  # K tensors of (B, 1)
        for ci in range(start, end):
            z_k = self.auto_decoder_codes.weight[ci]  # (d_cond,)
            z_exp = z_k.unsqueeze(0).expand(B, -1)  # (B, d_cond)
            sdf_k = self.sdf_decoder(xy_m, z_exp)  # (B, 1)
            sdf_stack.append(sdf_k)

        sdf_cat = torch.cat(sdf_stack, dim=-1)  # (B, K)

        # Smooth-min: sdf = -logsumexp(-alpha * sdf_cat) / alpha
        # Approximates min(sdf_1, ..., sdf_K) for large alpha
        alpha = self.smooth_min_alpha
        sdf_composed = -torch.logsumexp(
            -alpha * sdf_cat, dim=-1, keepdim=True
        ) / alpha  # (B, 1)

        return sdf_composed

    def get_code(self, scene_idx: int) -> torch.Tensor:
        """Get the latent code(s) for a scene (no copy).

        Parameters
        ----------
        scene_idx : int
            0-indexed scene index.

        Returns
        -------
        z : torch.Tensor
            shape (d_cond,) for K=1, shape (K, d_cond) for K>1.
        """
        start, end = self._scene_code_ranges[scene_idx]
        K = end - start
        if K == 1:
            return self.auto_decoder_codes.weight[start]
        return self.auto_decoder_codes.weight[start:end]  # (K, d_cond)

    @torch.no_grad()
    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_state_dict_compat(
        self,
        state_dict: dict,
        strict: bool = False,
    ) -> None:
        """Load checkpoint with backward compatibility for old format.

        Old format: auto_decoder_codes.weight has shape (n_scenes, d_cond)
        New format: auto_decoder_codes.weight has shape (total_codes, d_cond)

        When loading old checkpoints into multi-code model, single-code
        weights are copied to the first code of each scene. Extra codes
        are re-initialized.

        Parameters
        ----------
        state_dict : dict
            Checkpoint state dict.
        strict : bool
            If True, raise on missing/unexpected keys.
        """
        old_codes = state_dict.get("auto_decoder_codes.weight")
        new_codes = self.auto_decoder_codes.weight

        if old_codes is not None and old_codes.shape[0] != new_codes.shape[0]:
            # Old format: (n_scenes, d_cond) -> need to remap
            logger.info(
                "Remapping codes: old %s -> new %s",
                old_codes.shape, new_codes.shape,
            )
            remapped = torch.randn_like(new_codes.data) * 0.01
            n_old = old_codes.shape[0]
            for si in range(min(n_old, self.n_scenes)):
                start, end = self._scene_code_ranges[si]
                # Copy old code to first slot
                remapped[start] = old_codes[si]
                # Extra codes get small random perturbation of the original
                for ci in range(start + 1, end):
                    remapped[ci] = old_codes[si] + torch.randn_like(old_codes[si]) * 0.001
            state_dict = dict(state_dict)
            state_dict["auto_decoder_codes.weight"] = remapped

        self.load_state_dict(state_dict, strict=strict)


# ---------------------------------------------------------------------------
# Loss: SDF supervision
# ---------------------------------------------------------------------------
def sdf_loss(
    sdf_pred: torch.Tensor,
    sdf_gt: torch.Tensor,
    boundary_threshold_m: float = 0.1,
) -> torch.Tensor:
    """Mixed SDF loss: L1 near boundary, L2 elsewhere.

    Near-boundary samples get L1 loss for sharp zero-crossing recovery.
    Far-field samples get L2 loss for smooth regression.

    Parameters
    ----------
    sdf_pred : torch.Tensor, shape (B, 1)
    sdf_gt : torch.Tensor, shape (B, 1)
    boundary_threshold_m : float
        Distance threshold for boundary region [m].

    Returns
    -------
    loss : torch.Tensor, scalar
    """
    near = sdf_gt.abs() < boundary_threshold_m  # (B, 1)
    loss = torch.zeros(1, device=sdf_pred.device, dtype=sdf_pred.dtype)

    if near.any():
        loss = loss + nn.functional.l1_loss(
            sdf_pred[near], sdf_gt[near]
        )
    if (~near).any():
        loss = loss + nn.functional.mse_loss(
            sdf_pred[~near], sdf_gt[~near]
        )

    return loss.squeeze()


# ---------------------------------------------------------------------------
# Loss: Eikonal constraint  |grad(s)| = 1
# ---------------------------------------------------------------------------
def eikonal_loss(
    sdf_pred: torch.Tensor,
    xy_query: torch.Tensor,
) -> torch.Tensor:
    """Eikonal constraint: |grad(s)| = 1 everywhere.

    The signed distance function satisfies the Eikonal equation
    |nabla s(x)| = 1 at every point in space.

    Parameters
    ----------
    sdf_pred : torch.Tensor, shape (B, 1)
        SDF predictions (must be in computational graph of xy_query).
    xy_query : torch.Tensor, shape (B, 2), requires_grad=True
        Query points used to compute sdf_pred.

    Returns
    -------
    loss : torch.Tensor, scalar
        mean((|nabla s| - 1)^2)
    """
    grad_sdf = torch.autograd.grad(
        outputs=sdf_pred.sum(),
        inputs=xy_query,
        create_graph=True,
        retain_graph=True,
    )[0]  # (B, 2)

    grad_norm = grad_sdf.norm(dim=-1)  # (B,)
    return ((grad_norm - 1.0) ** 2).mean()


# ---------------------------------------------------------------------------
# Incident field (differentiable, asymptotic Hankel)
# ---------------------------------------------------------------------------
def compute_p_inc_torch(
    x_src: torch.Tensor,
    x_rcv: torch.Tensor,
    k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D incident field using asymptotic Hankel function.

    p_inc = -(i/4) H_0^{(1)}(kr)

    For kr >> 1 (valid for kr > 5):
        H_0^{(1)}(kr) ~ sqrt(2/(pi*kr)) exp(i(kr - pi/4))

    Yielding:
        Re(p_inc) =  (1/4) sqrt(2/(pi*kr)) sin(kr - pi/4)
        Im(p_inc) = -(1/4) sqrt(2/(pi*kr)) cos(kr - pi/4)

    Parameters
    ----------
    x_src : torch.Tensor, shape (B, 2)
        Source positions [m].
    x_rcv : torch.Tensor, shape (B, 2)
        Receiver positions [m].  May require grad for Helmholtz loss.
    k : torch.Tensor, shape (B,) or (B, 1)
        Wavenumber [rad/m].

    Returns
    -------
    p_inc_re : torch.Tensor, shape (B,)
    p_inc_im : torch.Tensor, shape (B,)
    """
    k_flat = k.reshape(-1)  # (B,)

    dx = x_rcv[:, 0] - x_src[:, 0]  # (B,)
    dy = x_rcv[:, 1] - x_src[:, 1]  # (B,)
    r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-20)  # (B,)
    kr = k_flat * r  # (B,)

    # Amplitude: (1/4) sqrt(2/(pi*kr)), clamp kr for safety
    amp = 0.25 * torch.sqrt(2.0 / (math.pi * kr.clamp(min=1.0)))  # (B,)

    # Phase: kr - pi/4
    phase = kr - math.pi / 4.0  # (B,)

    p_inc_re = amp * torch.sin(phase)   # (B,)
    p_inc_im = -amp * torch.cos(phase)  # (B,)

    return p_inc_re, p_inc_im


# ---------------------------------------------------------------------------
# Loss: Cycle-consistency through frozen forward model
# ---------------------------------------------------------------------------
def cycle_consistency_loss(
    inverse_model: InverseModel,
    forward_model: TransferFunctionModel,
    scene_idx: int,
    x_src: torch.Tensor,
    x_rcv: torch.Tensor,
    k: torch.Tensor,
    p_gt_re: torch.Tensor,
    p_gt_im: torch.Tensor,
    scale: float,
    fwd_scene_ids: torch.Tensor,
) -> torch.Tensor:
    """Cycle-consistency loss: SDF prediction -> forward model -> pressure.

    Path: z_i -> SDFDecoder(x_rcv, z_i) -> sdf_at_rcv
          -> frozen_forward(x_src, x_rcv, k, sdf_at_rcv, scene_id) -> T_pred
          -> p_total = p_inc * (1 + T_complex * scale) -> compare with p_gt

    Parameters
    ----------
    inverse_model : InverseModel
        Trainable inverse model.
    forward_model : TransferFunctionModel
        Frozen Phase 2 forward model (no grad).
    scene_idx : int
        0-indexed scene index for the inverse model.
    x_src : torch.Tensor, shape (B, 2)
        Source positions [m].
    x_rcv : torch.Tensor, shape (B, 2)
        Receiver positions [m].
    k : torch.Tensor, shape (B, 1)
        Wavenumber [rad/m].
    p_gt_re : torch.Tensor, shape (B,)
        Ground truth Re(p_total).
    p_gt_im : torch.Tensor, shape (B,)
        Ground truth Im(p_total).
    scale : float
        Per-scene transfer function RMS scale.
    fwd_scene_ids : torch.Tensor, shape (B,), long
        0-indexed scene IDs for the forward model's scene embedding.

    Returns
    -------
    loss : torch.Tensor, scalar
        ||p_pred - p_gt||^2 / ||p_gt||^2
    """
    B = x_rcv.shape[0]

    # Predict SDF at receiver positions (differentiable path to z)
    sdf_at_rcv = inverse_model.predict_sdf(scene_idx, x_rcv)  # (B, 1)

    # Forward model prediction (frozen weights, but differentiable w.r.t. sdf)
    with torch.no_grad():
        # Detach x_src and x_rcv for the forward model -- we only want
        # gradients through sdf_at_rcv -> auto_decoder_codes + sdf_decoder
        x_src_d = x_src.detach()
        x_rcv_d = x_rcv.detach()
        k_d = k.detach()

    # sdf_at_rcv still has grad; forward_from_coords builds features
    # using detached positions but differentiable sdf
    t_pred = forward_model.forward_from_coords(
        x_src_d, x_rcv_d, k_d, sdf_at_rcv, scene_ids=fwd_scene_ids,
    )  # (B, 2)

    t_re = t_pred[:, 0] * scale  # (B,) denormalized
    t_im = t_pred[:, 1] * scale  # (B,)

    # Incident field (no grad needed, just for reconstruction)
    with torch.no_grad():
        p_inc_re, p_inc_im = compute_p_inc_torch(x_src, x_rcv, k.squeeze(-1))

    # p_total = p_inc * (1 + T_complex)
    # Re: p_inc_re*(1+t_re) - p_inc_im*t_im
    # Im: p_inc_im*(1+t_re) + p_inc_re*t_im
    p_pred_re = p_inc_re * (1.0 + t_re) - p_inc_im * t_im  # (B,)
    p_pred_im = p_inc_im * (1.0 + t_re) + p_inc_re * t_im  # (B,)

    # Relative L2 loss
    diff_sq = (p_pred_re - p_gt_re) ** 2 + (p_pred_im - p_gt_im) ** 2  # (B,)
    ref_sq = p_gt_re ** 2 + p_gt_im ** 2  # (B,)

    return diff_sq.sum() / ref_sq.sum().clamp(min=1e-30)


# ---------------------------------------------------------------------------
# Loss: Helmholtz PDE residual
# ---------------------------------------------------------------------------
def helmholtz_residual(
    forward_model: TransferFunctionModel,
    sdf_decoder: SDFDecoder,
    z: torch.Tensor,
    x_src: torch.Tensor,
    x_rcv: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    fwd_scene_ids: torch.Tensor,
) -> torch.Tensor:
    """Helmholtz PDE residual: nabla^2 p + k^2 p = 0 in free space.

    Computes the PDE residual at exterior points using 2nd-order autograd.
    Requires FP32 for numerical stability of second derivatives.

    Parameters
    ----------
    forward_model : TransferFunctionModel
        Frozen Phase 2 forward model.
    sdf_decoder : SDFDecoder
        SDF decoder (part of inverse model).
    z : torch.Tensor, shape (d_cond,)
        Latent code for the scene.
    x_src : torch.Tensor, shape (B, 2)
        Source positions [m].  Detached.
    x_rcv : torch.Tensor, shape (B, 2)
        Evaluation points [m].  Will be made differentiable.
    k : torch.Tensor, shape (B, 1)
        Wavenumber [rad/m].
    scale : float
        Per-scene transfer function RMS scale.
    fwd_scene_ids : torch.Tensor, shape (B,), long
        0-indexed scene IDs for forward model embedding.

    Returns
    -------
    residual : torch.Tensor, scalar
        Normalized mean |nabla^2 p + k^2 p|^2 / (k^4 |p|^2 + eps).
    """
    B = x_rcv.shape[0]

    # Make evaluation points differentiable
    x_eval = x_rcv.detach().clone().requires_grad_(True)  # (B, 2)

    # Predict SDF at evaluation points (differentiable in x_eval)
    # Handle multi-code z: (K, d_cond) -> smooth-min composition
    if z.dim() == 1:
        z_exp = z.unsqueeze(0).expand(B, -1)  # (B, d_cond)
        sdf_eval = sdf_decoder(x_eval, z_exp)  # (B, 1)
    else:
        # z: (K, d_cond) -- multi-body
        K = z.shape[0]
        sdf_parts = []
        for ki in range(K):
            z_k = z[ki].unsqueeze(0).expand(B, -1)  # (B, d_cond)
            sdf_parts.append(sdf_decoder(x_eval, z_k))  # (B, 1)
        sdf_cat = torch.cat(sdf_parts, dim=-1)  # (B, K)
        alpha = 50.0
        sdf_eval = -torch.logsumexp(
            -alpha * sdf_cat, dim=-1, keepdim=True
        ) / alpha  # (B, 1)

    # Forward model prediction (differentiable in x_eval through sdf and coords)
    t_pred = forward_model.forward_from_coords(
        x_src.detach(), x_eval, k.detach(), sdf_eval,
        scene_ids=fwd_scene_ids,
    )  # (B, 2)

    t_re = t_pred[:, 0] * scale  # (B,)
    t_im = t_pred[:, 1] * scale  # (B,)

    # Incident field (differentiable in x_eval)
    k_flat = k.reshape(-1)  # (B,)
    p_inc_re, p_inc_im = compute_p_inc_torch(
        x_src.detach(), x_eval, k_flat,
    )

    # Total field: p = p_inc * (1 + T)
    p_re = p_inc_re * (1.0 + t_re) - p_inc_im * t_im  # (B,)
    p_im = p_inc_im * (1.0 + t_re) + p_inc_re * t_im  # (B,)

    # Laplacian of p_re: d^2 p_re / dx^2 + d^2 p_re / dy^2
    grad_p_re = torch.autograd.grad(
        p_re.sum(), x_eval, create_graph=True, retain_graph=True,
    )[0]  # (B, 2)

    d2_p_re_dx2 = torch.autograd.grad(
        grad_p_re[:, 0].sum(), x_eval, create_graph=True, retain_graph=True,
    )[0][:, 0]  # (B,)
    d2_p_re_dy2 = torch.autograd.grad(
        grad_p_re[:, 1].sum(), x_eval, create_graph=True, retain_graph=True,
    )[0][:, 1]  # (B,)
    laplacian_re = d2_p_re_dx2 + d2_p_re_dy2  # (B,)

    # Laplacian of p_im
    grad_p_im = torch.autograd.grad(
        p_im.sum(), x_eval, create_graph=True, retain_graph=True,
    )[0]  # (B, 2)

    d2_p_im_dx2 = torch.autograd.grad(
        grad_p_im[:, 0].sum(), x_eval, create_graph=True, retain_graph=True,
    )[0][:, 0]  # (B,)
    d2_p_im_dy2 = torch.autograd.grad(
        grad_p_im[:, 1].sum(), x_eval, create_graph=True, retain_graph=True,
    )[0][:, 1]  # (B,)
    laplacian_im = d2_p_im_dx2 + d2_p_im_dy2  # (B,)

    # Helmholtz residual: nabla^2 p + k^2 p
    k_sq = k_flat ** 2  # (B,)
    res_re = laplacian_re + k_sq * p_re  # (B,)
    res_im = laplacian_im + k_sq * p_im  # (B,)

    residual_sq = res_re ** 2 + res_im ** 2  # (B,)

    # Normalize by k^4 * |p|^2 for dimensionless residual
    p_sq = p_re ** 2 + p_im ** 2  # (B,)
    k4 = k_sq ** 2  # (B,)
    normalizer = (k4 * p_sq).clamp(min=1e-30)  # (B,)

    return (residual_sq / normalizer).mean()


# ---------------------------------------------------------------------------
# Metric: SDF IoU
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_sdf_iou(
    sdf_pred: torch.Tensor,
    sdf_gt: torch.Tensor,
) -> float:
    """Compute intersection-over-union of SDF interior regions.

    Interior is defined as SDF <= 0 (inside body).

    Parameters
    ----------
    sdf_pred : torch.Tensor, shape (N,) or (N, 1)
        Predicted SDF values.
    sdf_gt : torch.Tensor, shape (N,) or (N, 1)
        Ground truth SDF values.

    Returns
    -------
    iou : float
        IoU in [0, 1].  Returns 1.0 if both predictions are empty.
    """
    pred_inside = (sdf_pred.reshape(-1) <= 0)
    gt_inside = (sdf_gt.reshape(-1) <= 0)

    intersection = (pred_inside & gt_inside).sum().item()
    union = (pred_inside | gt_inside).sum().item()

    if union == 0:
        return 1.0  # both empty -> perfect agreement
    return intersection / union


# ---------------------------------------------------------------------------
# Metric: Chamfer & Hausdorff Distance (boundary-level SDF metrics)
# ---------------------------------------------------------------------------
def extract_zero_contour(
    sdf_grid: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """Extract zero-level contour points from SDF grid via sign changes.

    Uses linear interpolation on grid edges where SDF crosses zero.
    Implements marching-squares zero-crossing detection without external
    dependencies (no skimage required).

    Parameters
    ----------
    sdf_grid : np.ndarray, shape (Gx, Gy)
        Signed distance field on regular grid.
    grid_x : np.ndarray, shape (Gx,)
        X coordinates of grid.
    grid_y : np.ndarray, shape (Gy,)
        Y coordinates of grid.

    Returns
    -------
    contour_points : np.ndarray, shape (N, 2)
        Physical coordinates of zero-crossing points [m].
        Empty (0, 2) array if no crossings found.
    """
    points: list = []
    gx, gy = sdf_grid.shape

    # Horizontal edges: sign change along y-axis
    for i in range(gx):
        s_row = sdf_grid[i, :]  # (Gy,)
        sign_change = s_row[:-1] * s_row[1:] < 0  # (Gy-1,)
        j_indices = np.where(sign_change)[0]
        for j in j_indices:
            t = s_row[j] / (s_row[j] - s_row[j + 1])
            x = grid_x[i]
            y = grid_y[j] + t * (grid_y[j + 1] - grid_y[j])
            points.append([x, y])

    # Vertical edges: sign change along x-axis
    for j in range(gy):
        s_col = sdf_grid[:, j]  # (Gx,)
        sign_change = s_col[:-1] * s_col[1:] < 0  # (Gx-1,)
        i_indices = np.where(sign_change)[0]
        for i in i_indices:
            t = s_col[i] / (s_col[i] - s_col[i + 1])
            x = grid_x[i] + t * (grid_x[i + 1] - grid_x[i])
            y = grid_y[j]
            points.append([x, y])

    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    return np.array(points, dtype=np.float64)  # (N, 2)


def compute_chamfer_hausdorff(
    contour_pred: np.ndarray,
    contour_gt: np.ndarray,
) -> tuple:
    """Compute Chamfer distance and Hausdorff distance between contours.

    Chamfer distance (CD):
        CD = (1/|P|) Σ_p min_q ||p-q|| + (1/|Q|) Σ_q min_p ||q-p||
        Averaged bidirectional nearest-neighbor distance [m].

    Hausdorff distance (HD):
        HD = max( max_p min_q ||p-q||, max_q min_p ||q-p|| )
        Worst-case boundary deviation [m].

    Parameters
    ----------
    contour_pred : np.ndarray, shape (N, 2)
        Predicted boundary points [m].
    contour_gt : np.ndarray, shape (M, 2)
        Ground truth boundary points [m].

    Returns
    -------
    chamfer : float
        Chamfer distance [m].
    hausdorff : float
        Hausdorff distance [m].
    """
    from scipy.spatial import KDTree

    if len(contour_pred) == 0 or len(contour_gt) == 0:
        return float("inf"), float("inf")

    tree_gt = KDTree(contour_gt)
    tree_pred = KDTree(contour_pred)

    d_pred_to_gt, _ = tree_gt.query(contour_pred)  # (N,)
    d_gt_to_pred, _ = tree_pred.query(contour_gt)   # (M,)

    chamfer = (d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2.0
    hausdorff = max(d_pred_to_gt.max(), d_gt_to_pred.max())

    return float(chamfer), float(hausdorff)


def compute_sdf_boundary_errors(
    sdf_pred: np.ndarray,
    sdf_gt: np.ndarray,
    near_threshold_m: float = 0.1,
    far_threshold_m: float = 0.5,
) -> dict:
    """Compute SDF L1 error stratified by distance to boundary.

    Parameters
    ----------
    sdf_pred : np.ndarray, shape (N,)
        Predicted SDF values.
    sdf_gt : np.ndarray, shape (N,)
        Ground truth SDF values.
    near_threshold_m : float
        Distance threshold for "near boundary" region [m].
    far_threshold_m : float
        Distance threshold for "far from boundary" region [m].

    Returns
    -------
    dict with 'l1_near', 'l1_far', 'l1_overall', 'n_near', 'n_far'.
    """
    abs_gt = np.abs(sdf_gt)
    abs_diff = np.abs(sdf_pred - sdf_gt)

    near_mask = abs_gt < near_threshold_m
    far_mask = abs_gt > far_threshold_m

    l1_overall = float(abs_diff.mean())
    l1_near = float(abs_diff[near_mask].mean()) if near_mask.sum() > 0 else float("nan")
    l1_far = float(abs_diff[far_mask].mean()) if far_mask.sum() > 0 else float("nan")

    return {
        "l1_near": l1_near,
        "l1_far": l1_far,
        "l1_overall": l1_overall,
        "n_near": int(near_mask.sum()),
        "n_far": int(far_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_inverse_model(
    n_scenes: int = 15,
    d_cond: int = 256,
    d_hidden: int = 256,
    n_blocks: int = 6,
    n_fourier: int = 128,
    fourier_sigma: float = 10.0,
    dropout: float = 0.05,
    multi_body_scene_ids: Optional[Dict[int, int]] = None,
    inv_scene_id_map: Optional[Dict[int, int]] = None,
    smooth_min_alpha: float = 50.0,
) -> InverseModel:
    """Build the Phase 3 inverse model.

    Parameters
    ----------
    n_scenes : int
        Number of scenes for auto-decoder codes.
    d_cond : int
        Latent code dimension.
    d_hidden : int
        SDF decoder hidden width.
    n_blocks : int
        SDF decoder depth.
    n_fourier : int
        Fourier features for spatial encoding.
    fourier_sigma : float
        Fourier feature bandwidth [m^-1].
    dropout : float
        Dropout rate in SDF decoder.
    multi_body_scene_ids : dict, optional
        Mapping {scene_id: K} for multi-body scenes.
        E.g., {12: 2} means scene 12 gets 2 latent codes.
    inv_scene_id_map : dict, optional
        Mapping {scene_id: scene_idx} to convert scene_ids to 0-indexed.
        Required when multi_body_scene_ids uses scene_ids (not indices).
    smooth_min_alpha : float
        Sharpness of smooth-min (log-sum-exp) for multi-code composition.
        Higher = closer to hard min. Default 50.0.

    Returns
    -------
    model : InverseModel
    """
    # Convert multi_body_scene_ids (global IDs) to codes_per_scene (0-indexed)
    codes_per_scene: Optional[Dict[int, int]] = None
    if multi_body_scene_ids:
        if inv_scene_id_map is None:
            raise ValueError(
                "inv_scene_id_map required when multi_body_scene_ids is set"
            )
        codes_per_scene = {}
        for sid, k in multi_body_scene_ids.items():
            if sid in inv_scene_id_map:
                codes_per_scene[inv_scene_id_map[sid]] = k
            else:
                logger.warning(
                    "Multi-body scene_id %d not in scene map, skipping", sid,
                )

    model = InverseModel(
        n_scenes=n_scenes,
        d_cond=d_cond,
        d_hidden=d_hidden,
        n_blocks=n_blocks,
        n_fourier=n_fourier,
        fourier_sigma=fourier_sigma,
        dropout=dropout,
        codes_per_scene=codes_per_scene,
        smooth_min_alpha=smooth_min_alpha,
    )

    n_params = model.count_parameters()
    multi_str = ""
    if codes_per_scene:
        multi_str = f", multi-body: {codes_per_scene}"
    logger.info(
        "InverseModel built: %d parameters (%.2f MB), "
        "%d scenes x %d-dim codes, %d total codes%s",
        n_params,
        n_params * 4 / 1e6,
        n_scenes,
        d_cond,
        model._total_codes,
        multi_str,
    )

    return model
