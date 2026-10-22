"""Motion-driven spatial modulation (MDSM) from a complete cardiac cycle.

Provides per-pixel FiLM modulation to the U-Net regularizer, driven by an
image-domain motion signal computed from inter-frame magnitude differences.
The implementation follows Eq. (8)--(9) of the CineMamba manuscript.

Design summary:
- Motion map: from x_prev (B, T, H, W, 2) complex cascade input,
  compute a per-pixel magnitude-variation surrogate m ∈ (B, 1, H, W)
  from all T differences, including the last-to-first wrap-around. Normalized
  per-sample and clipped to [-3, 3].
- MDSM FiLM: at each U-Net level ℓ with resolution (h_ℓ, w_ℓ), resize
  m via bilinear interp and pass through a 1×1 conv (single-in → 2·D_ℓ out)
  to produce spatial FiLM parameters (γ, β) ∈ (B, D_ℓ, h_ℓ, w_ℓ).
- Zero-init on the final 1×1 conv layer so γ=1, β=0 at start — MDSM
  starts as an identity mapping.

Application (in MambaPromptUnet):
    U_out(t, d, i, j) = γ(d, i, j) · U_ct(t, d, i, j) + β(d, i, j)
Broadcast across T — spatial condition is time-invariant within a cascade
(temporal state evolution is handled separately by CP-Mamba).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Motion-density map — one call per cascade, then reused across levels.
# -----------------------------------------------------------------------------


def compute_motion_map(
    x_prev_complex: torch.Tensor,
    clip_range: float = 3.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute per-pixel motion-density map from cascade input image.

    Args:
        x_prev_complex: (B, T, H, W, 2) complex-as-last-dim image tensor,
            i.e. the coil-combined reconstruction estimate at the current
            cascade's entry (typically from sens_reduce).
        clip_range: half-range for symmetric clipping after z-normalization.
        eps: numerical stability in std computation.

    Returns:
        m: (B, 1, H, W) float. Zero-mean, unit-std (approximately),
        clipped to [-clip_range, clip_range].
    """
    B, T, H, W, two = x_prev_complex.shape
    assert two == 2, f"expected complex-as-last-dim, got last-dim {two}"

    # Magnitude: (B, T, H, W)
    mag = torch.sqrt(x_prev_complex[..., 0] ** 2 + x_prev_complex[..., 1] ** 2 + eps)

    if T < 2:
        # Not enough frames — return zero motion map.
        return torch.zeros(B, 1, H, W, device=x_prev_complex.device, dtype=mag.dtype)

    # Eq. (8): T cyclic differences, including frame T -> frame 1.
    diffs = (mag.roll(shifts=-1, dims=1) - mag).abs()
    m = diffs.mean(dim=1, keepdim=True)

    # Per-sample z-normalize then clip.
    mean = m.mean(dim=(-2, -1), keepdim=True)
    std = m.std(dim=(-2, -1), keepdim=True).clamp(min=eps)
    m = (m - mean) / std
    m = m.clamp(-clip_range, clip_range)

    return m


# -----------------------------------------------------------------------------
# Per-level FiLM head — cheap 1×1 conv from single-channel motion → 2·D FiLM.
# -----------------------------------------------------------------------------


class MDSMFiLMHead(nn.Module):
    """Bilinear-downsample the motion map to a level's spatial resolution,
    then produce per-pixel FiLM (γ, β) via a 1×1 conv.

    Zero-init on the final conv so γ=1, β=0 at start (identity at init).
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        # Single-channel motion → 2·D_ℓ FiLM parameters
        self.conv = nn.Conv2d(1, 2 * out_channels, kernel_size=1, bias=True)
        # Zero-init both weight and bias so gamma=1, beta=0 at start.
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        motion_map: torch.Tensor,  # (B, 1, H, W) at full resolution
        target_hw: tuple,           # (h_ℓ, w_ℓ)
    ) -> tuple:
        """Returns (gamma, beta), each (B, D, h_ℓ, w_ℓ)."""
        # Downsample to target level resolution
        m_ds = F.interpolate(
            motion_map,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        # 1×1 conv → 2·D channels; split
        raw = self.conv(m_ds)  # (B, 2D, h, w)
        # Split into γ (multiplicative, centered at 1) and β (additive, at 0)
        gamma_delta, beta = raw.chunk(2, dim=1)
        # Add 1 to γ_delta so the whole (γ) is (1 + δγ); starts at 1.
        gamma = 1.0 + gamma_delta
        return gamma, beta


# -----------------------------------------------------------------------------
# Application: broadcast FiLM across T and apply to a (B*T, D, h, w) tensor.
# -----------------------------------------------------------------------------


def apply_mdsm(
    x_bt: torch.Tensor,        # (B*T, D, h, w)
    gamma: torch.Tensor,       # (B, D, h, w)
    beta: torch.Tensor,        # (B, D, h, w)
    num_frames: int,
) -> torch.Tensor:
    """Apply per-pixel FiLM with time-uniform γ, β to a T-in-batch tensor.

    Broadcasts γ, β across all T frames of each sample.
    """
    BT, D, H, W = x_bt.shape
    T = num_frames
    B = BT // T
    assert gamma.shape == (B, D, H, W), (
        f"gamma shape {gamma.shape} != expected ({B}, {D}, {H}, {W})"
    )

    # Expand γ, β from (B, D, h, w) to (B, T, D, h, w) then flatten to (B*T, D, h, w)
    gamma_bt = gamma.unsqueeze(1).expand(B, T, D, H, W).reshape(BT, D, H, W)
    beta_bt = beta.unsqueeze(1).expand(B, T, D, H, W).reshape(BT, D, H, W)

    return gamma_bt * x_bt + beta_bt
