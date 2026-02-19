"""Phase 2 Forward Model: Transfer Function Surrogate.

Architecture
------------
    p_total(x, x_s, k) = p_inc(x, x_s, k) * (1 + T_pred(x, x_s, k, geometry) * scale)

    T = p_scat / p_inc  (transfer function)

    p_inc = -(i/4) H_0^{(1)}(k |x - x_s|)   (2D free-space Green)

    T_pred = FourierMLP(x_s, y_s, x_r, y_r, k, sdf, dist, dx, dy)
    Output: (Re(T), Im(T)) normalized by per-scene RMS

Components
----------
    1. FourierFeatureEncoder -- random Gaussian positional encoding
       gamma(v) = [cos(2*pi*B*v), sin(2*pi*B*v)],  B ~ N(0, sigma^2)

    2. ResidualBlock -- pre-LayerNorm residual: x + MLP(LN(x))

    3. TransferFunctionModel -- Fourier features + Residual MLP

Why Transfer Function?
    Complex p_scat oscillates rapidly with k (~12 phase cycles across 2-8 kHz).
    Dividing by p_inc removes the dominant phase variation, yielding a smooth
    target that neural networks can learn (89.6% var explained vs 13% for raw).

Reference
---------
    Tancik et al. (2020) "Fourier Features Let Networks Learn HF Functions"
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0
N_RAW_FEATURES: int = 9  # (x_s, y_s, x_r, y_r, k, sdf, dist, dx, dy)


# ---------------------------------------------------------------------------
# Fourier Feature Encoder
# ---------------------------------------------------------------------------
class FourierFeatureEncoder(nn.Module):
    """Random Fourier feature encoding for positional inputs.

    Maps d-dimensional input v to 2m-dimensional features:

        gamma(v) = [cos(2*pi*B*v), sin(2*pi*B*v)]

    where B in R^{m x d}, B_ij ~ N(0, sigma^2).

    Supports multi-scale encoding: when sigma is a list, features are split
    across scales.  E.g. sigma=[10, 30, 90] dedicates n_features//3 to each
    scale, capturing both large-scale wave patterns and fine-scale near-edge
    diffraction.

    Parameters
    ----------
    input_dim : int
        Dimension of raw input vector.
    n_features : int
        Number of Fourier frequencies m.  Output dimension = 2*m + input_dim.
    sigma : float or list of float
        Standard deviation(s) of the random frequency matrix B  [m^-1].
        If a list, features are split across scales (multi-scale encoding).
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int = 256,
        sigma: float = 30.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.output_dim = 2 * n_features + input_dim  # cos + sin + raw passthrough

        # Fixed random matrix -- NOT a learned parameter
        if isinstance(sigma, (list, tuple)):
            # Multi-scale Fourier features: split across sigma values
            n_scales = len(sigma)
            n_per = n_features // n_scales
            B_parts = []
            for i, s in enumerate(sigma):
                # Last scale gets remaining features
                n_f = n_per if i < n_scales - 1 else n_features - i * n_per
                B_parts.append(torch.randn(n_f, input_dim) * s)  # (n_f, d)
            B = torch.cat(B_parts, dim=0)  # (m, d)
        else:
            B = torch.randn(n_features, input_dim) * sigma  # (m, d)
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input coordinates with Fourier features + raw passthrough.

        Parameters
        ----------
        x : torch.Tensor, shape (..., d)

        Returns
        -------
        features : torch.Tensor, shape (..., 2*m + d)
        """
        proj = 2.0 * math.pi * (x @ self.B.T)  # (..., m)
        return torch.cat(
            [torch.cos(proj), torch.sin(proj), x], dim=-1
        )  # (..., 2m + d)


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-LayerNorm residual block: x + Dropout(FC2(GELU(FC1(LN(x))))).

    Uses pre-norm (LayerNorm before linear layers) following modern
    Transformer convention for stable training of deep networks.

    Parameters
    ----------
    dim : int
        Hidden dimension.
    dropout : float
        Dropout rate applied after second linear layer.
    """

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize fc2 near zero for stable residual at init
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: x + Dropout(FC2(GELU(FC1(LN(x)))))."""
        h = self.norm(x)  # (B, dim)
        h = self.act(self.fc1(h))  # (B, dim)
        h = self.drop(self.fc2(h))  # (B, dim)
        return x + h  # (B, dim)


# ---------------------------------------------------------------------------
# Transfer Function Model
# ---------------------------------------------------------------------------
class TransferFunctionModel(nn.Module):
    """Predicts transfer function T = p_scat / p_inc.

    Uses random Fourier feature encoding followed by a residual MLP
    to predict the real and imaginary parts of the normalized transfer
    function.

    Total field reconstruction:
        T_complex = (Re + j*Im) * scene_scale
        p_total = p_inc * (1 + T_complex)

    Parameters
    ----------
    d_in : int
        Raw input dimension (9 features).
    d_hidden : int
        Hidden layer width.
    d_out : int
        Output dimension (2 for Re/Im of T).
    n_blocks : int
        Number of residual blocks.
    n_fourier : int
        Number of Fourier frequencies (output = 2*n_fourier + d_in).
    fourier_sigma : float
        Fourier feature bandwidth [m^-1].
    dropout : float
        Dropout rate in residual blocks (0.0 = no dropout).
    n_scenes : int
        Number of scenes for scene embedding (0 = no embedding).
    scene_emb_dim : int
        Dimension of learned scene embedding vector.
    """

    def __init__(
        self,
        d_in: int = N_RAW_FEATURES,
        d_hidden: int = 768,
        d_out: int = 2,
        n_blocks: int = 6,
        n_fourier: int = 256,
        fourier_sigma: float = 30.0,  # float or list of float for multi-scale
        dropout: float = 0.0,
        n_scenes: int = 15,
        scene_emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.n_blocks = n_blocks
        self.n_scenes = n_scenes

        # Fourier feature encoder
        self.encoder = FourierFeatureEncoder(
            input_dim=d_in,
            n_features=n_fourier,
            sigma=fourier_sigma,
        )
        feat_dim = self.encoder.output_dim  # 2*n_fourier + d_in

        # Scene embedding (gives model explicit scene identity)
        self.scene_emb_dim = scene_emb_dim if n_scenes > 0 else 0
        if n_scenes > 0:
            self.scene_embedding = nn.Embedding(n_scenes, scene_emb_dim)
            feat_dim += scene_emb_dim

        # Input projection: feat_dim -> d_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_hidden),
            nn.GELU(),
        )

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(d_hidden, dropout=dropout) for _ in range(n_blocks)]
        )

        # Output head: LayerNorm -> Linear -> 2
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out),
        )

        # Input normalization buffers (set via set_normalization)
        self.register_buffer("input_mean", torch.zeros(d_in))
        self.register_buffer("input_std", torch.ones(d_in))

    def set_normalization(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        """Set z-score normalization statistics from training data.

        Parameters
        ----------
        mean : torch.Tensor, shape (d_in,)
        std : torch.Tensor, shape (d_in,)
        """
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)

    def forward(
        self,
        inputs: torch.Tensor,
        scene_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict normalized transfer function (Re(T), Im(T)).

        Parameters
        ----------
        inputs : torch.Tensor, shape (B, d_in)
            Raw features: (x_s, y_s, x_r, y_r, k, sdf, dist, dx, dy).
        scene_ids : torch.Tensor, optional, shape (B,), dtype=long
            0-indexed scene IDs for scene embedding.

        Returns
        -------
        t_pred : torch.Tensor, shape (B, 2)
            (Re, Im) of predicted transfer function (normalized by scene scale).
        """
        # Z-score normalization
        x = (inputs - self.input_mean) / self.input_std  # (B, d_in)

        # Fourier encode + raw passthrough
        x_enc = self.encoder(x)  # (B, 2*n_fourier + d_in)

        # Concatenate scene embedding if available
        if self.n_scenes > 0 and scene_ids is not None:
            s_emb = self.scene_embedding(scene_ids)  # (B, scene_emb_dim)
            x_enc = torch.cat([x_enc, s_emb], dim=-1)  # (B, feat_dim + emb_dim)

        # Residual MLP
        h = self.input_proj(x_enc)  # (B, d_hidden)
        for block in self.blocks:
            h = block(h)  # (B, d_hidden)

        return self.output_head(h)  # (B, 2)

    def forward_from_coords(
        self,
        x_src: torch.Tensor,
        x_rcv: torch.Tensor,
        k: torch.Tensor,
        sdf_rcv: torch.Tensor,
        scene_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward from raw physical coordinates (for PDE loss).

        Builds derived features (dist, dx, dy) from differentiable x_rcv
        so that torch.autograd can compute spatial derivatives.

        Parameters
        ----------
        x_src : torch.Tensor, shape (B, 2)
            Source positions [m].  Detached (no grad).
        x_rcv : torch.Tensor, shape (B, 2)
            Receiver positions [m].  requires_grad=True for PDE loss.
        k : torch.Tensor, shape (B, 1)
            Wavenumber [rad/m].
        sdf_rcv : torch.Tensor, shape (B, 1)
            SDF at receiver position [m].
        scene_ids : torch.Tensor, optional, shape (B,), dtype=long
            0-indexed scene IDs for scene embedding.

        Returns
        -------
        t_pred : torch.Tensor, shape (B, 2)
        """
        dx = x_rcv[:, 0:1] - x_src[:, 0:1]  # (B, 1)
        dy = x_rcv[:, 1:2] - x_src[:, 1:2]  # (B, 1)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-30)  # (B, 1)

        inputs = torch.cat(
            [x_src, x_rcv, k, sdf_rcv, dist, dx, dy], dim=-1
        )  # (B, 9)

        return self.forward(inputs, scene_ids=scene_ids)

    @torch.no_grad()
    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_transfer_model(
    d_hidden: int = 768,
    n_blocks: int = 6,
    n_fourier: int = 256,
    fourier_sigma: float = 30.0,
    dropout: float = 0.0,
    n_scenes: int = 15,
    scene_emb_dim: int = 32,
    d_out: int = 2,
) -> TransferFunctionModel:
    """Build the Phase 2 transfer function model with specified hyperparameters.

    Parameters
    ----------
    d_hidden : int
        Hidden layer width.
    n_blocks : int
        Number of residual blocks.
    n_fourier : int
        Number of Fourier features.
    fourier_sigma : float or list of float
        Fourier feature bandwidth [m^-1].  Pass a list for multi-scale
        encoding, e.g. [10, 30, 90].
    dropout : float
        Dropout rate in residual blocks.
    n_scenes : int
        Number of scenes for embedding (0 = no embedding).
    scene_emb_dim : int
        Scene embedding dimension.
    d_out : int
        Output dimension.  2 for cartesian (Re, Im), 3 for log-polar
        (log|R|, cos(angle), sin(angle)).

    Returns
    -------
    model : TransferFunctionModel
    """
    model = TransferFunctionModel(
        d_in=N_RAW_FEATURES,
        d_hidden=d_hidden,
        d_out=d_out,
        n_blocks=n_blocks,
        n_fourier=n_fourier,
        fourier_sigma=fourier_sigma,
        dropout=dropout,
        n_scenes=n_scenes,
        scene_emb_dim=scene_emb_dim,
    )

    n_params = model.count_parameters()
    logger.info(
        "TransferFunctionModel built: %d parameters (%.2f MB)",
        n_params,
        n_params * 4 / 1e6,
    )

    return model
