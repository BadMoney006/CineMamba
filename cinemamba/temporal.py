"""Cardiac-phase-conditioned temporal Mamba (CP-Mamba) block.

Temporal state evolution described in Section 3.3 of the CineMamba paper.

Given a 5D feature tensor (B, T, D, h, w), CP-Mamba runs Mamba's selective
scan along the temporal axis T, with the SSM discretization step Δ additively
conditioned on the cardiac phase φ_t. Supports bidirectional scan with
diastolic-anchor initialization.

Key design:
- Reshape (B, T, D, h, w) → (B·h·w, T, D). Each spatial location becomes
  an independent length-T time series, runnable through Mamba's batched
  selective scan without a custom CUDA kernel.
- Δ_t = softplus(W_Δ u_t + W_φ γ(φ_t) + b_Δ), where γ is a Fourier
  feature embedding of the cardiac phase.
- Bidirectional: two independent SSMs (forward + backward), outputs
  concatenated and projected.
- Diastolic anchor: h_0 = h_{T+1} = MLP(u_{t*}), t* = argmin_t φ_t.

Interface: forward(x, phase) → out
  x:     (B, T, D, h, w) float
  phase: (B, T)          float in [0, 1)
  out:   (B, T, D, h, w) float, residual-added

The anchored scan has a PyTorch implementation; the zero-state scan uses
``mamba_ssm`` when available and otherwise uses the same PyTorch recurrence.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Fourier feature embedding for cardiac phase φ ∈ [0, 1)
# -----------------------------------------------------------------------------


class FourierPhaseEmbedding(nn.Module):
    """γ(φ) = [sin(2π k φ), cos(2π k φ)] for k = 1..L. Output dim = 2L.

    Not learned — a pure positional embedding.
    """

    def __init__(self, num_frequencies: int = 4):
        super().__init__()
        self.num_frequencies = num_frequencies
        # k = 1, 2, ..., L (integer harmonics of the fundamental)
        self.register_buffer(
            "freqs",
            torch.arange(1, num_frequencies + 1, dtype=torch.float32) * 2.0 * math.pi,
            persistent=False,
        )

    @property
    def out_dim(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phase: (..., T) tensor, values in [0, 1).
        Returns:
            gamma: (..., T, 2L) tensor.
        """
        # phase.unsqueeze(-1): (..., T, 1); freqs: (L,) → broadcast to (..., T, L)
        angles = phase.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# -----------------------------------------------------------------------------
# Selective SSM utility — one-directional scan with explicit (Δ, B, C)
# -----------------------------------------------------------------------------


def _selective_scan_direction(
    u: torch.Tensor,          # (BS, D, T)
    delta: torch.Tensor,      # (BS, D, T)
    A: torch.Tensor,          # (D, N)
    B: torch.Tensor,          # (BS, N, T)
    C: torch.Tensor,          # (BS, N, T)
    D_skip: torch.Tensor,     # (D,)
    z: Optional[torch.Tensor] = None,   # (BS, D, T) for gating
    delta_bias: Optional[torch.Tensor] = None,  # (D,)
    initial_hidden: Optional[torch.Tensor] = None,  # (BS, D, N) — anchor init
) -> torch.Tensor:
    """One-directional selective scan wrapper.

    Delegates to mamba_ssm.ops.selective_scan_interface.selective_scan_fn when
    initial_hidden is None (the fast CUDA path). If initial_hidden is provided
    we fall back to a Python-level scan with the same discretization scheme,
    because the reference CUDA kernel does not expose an h0 argument.

    The Python fallback is O(T) memory + O(T) sequential steps in autograd; T
    is small (≤ 20 frames) so this is acceptable during training.

    All tensors are contiguous and float32.
    """
    if initial_hidden is None:
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except ImportError:
            initial_hidden = torch.zeros(
                u.shape[0], u.shape[1], A.shape[-1], device=u.device, dtype=u.dtype
            )
        else:
            return selective_scan_fn(
                u, delta, A, B, C, D_skip, z, delta_bias, delta_softplus=True
            )

    # ---- Python fallback with anchor initialization ----
    # Discretization: Δ̄ = softplus(Δ + Δ_bias); Ā = exp(Δ̄ A); B̄ = Δ̄ B
    # Matches the selective_scan reference recurrence (delta * B * u).
    BS, D, T = u.shape
    N = A.shape[-1]

    if delta_bias is not None:
        delta = delta + delta_bias.view(1, D, 1)
    delta = F.softplus(delta)  # (BS, D, T) positive

    # Ā_t (BS, D, T, N) = exp(delta_t[:, :, :, None] * A[None, :, None, :])
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-2))  # (BS, D, T, N)
    # B̄_t u_t (BS, D, T, N) = delta_t[:, :, :, None] * B_t[:, None, :, :] * u_t[:, :, :, None]
    # B is (BS, N, T) → transpose to (BS, T, N) for broadcasting
    B_bt = B.transpose(1, 2)  # (BS, T, N)
    dB_u = delta.unsqueeze(-1) * B_bt.unsqueeze(1) * u.unsqueeze(-1)  # (BS, D, T, N)

    # Sequential scan
    h = initial_hidden  # (BS, D, N)
    ys = []
    C_bt = C.transpose(1, 2)  # (BS, T, N)
    for t in range(T):
        h = dA[:, :, t] * h + dB_u[:, :, t]  # (BS, D, N)
        # y_t = C_t · h_t + D · u_t
        y_t = (h * C_bt[:, t].unsqueeze(1)).sum(dim=-1)  # (BS, D)
        ys.append(y_t)
    y = torch.stack(ys, dim=-1)  # (BS, D, T)
    y = y + D_skip.view(1, D, 1) * u  # skip connection

    if z is not None:
        y = y * F.silu(z)

    return y


# -----------------------------------------------------------------------------
# CP-Mamba block — cardiac-phase-conditioned temporal state evolution
# -----------------------------------------------------------------------------


class CPMambaBlock(nn.Module):
    """Cine-temporal selective-scan Mamba block with phase-conditioned Δ.

    Args:
        d_model: input/output feature dimension D of the block.
        d_state: SSM hidden state dimension N. Default 16.
        d_conv: local temporal conv kernel size (over T). Default 4.
        expand: inner-dim expansion factor (Mamba convention). Default 2.
        num_phase_freqs: L in the Fourier phase embedding (dim = 2L).
        bidirectional: run forward+backward scans and concatenate.
        diastolic_anchor: initialize hidden states from the estimated
            end-diastolic frame instead of zeros.
        dt_min, dt_max, dt_init_floor: initialization range for Δ projection,
            matching Mamba's defaults.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_phase_freqs: int = 4,
        bidirectional: bool = True,
        diastolic_anchor: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.bidirectional = bidirectional
        self.diastolic_anchor = diastolic_anchor
        self.num_directions = 2 if bidirectional else 1

        # Standard Mamba block components
        self.norm = nn.LayerNorm(d_model)
        # Linear projection D → 2·d_inner → split to (a, g). a is fed to SSM,
        # g is the multiplicative gate.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depth-wise Conv1D over T for local temporal receptive field before SSM.
        # We run this per-direction so each direction has its own conv weights.
        self.conv1d = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    kernel_size=d_conv,
                    padding=d_conv - 1,  # causal-ish; we trim to T after conv
                    groups=self.d_inner,
                )
                for _ in range(self.num_directions)
            ]
        )
        self.act = nn.SiLU()

        # Fourier phase embedding: shared across directions.
        self.phase_embed = FourierPhaseEmbedding(num_frequencies=num_phase_freqs)
        phase_dim = self.phase_embed.out_dim  # 2L

        # SSM parameter projections — one set per direction.
        # x_proj produces (dt_rank + 2 * d_state) from d_inner.
        self.dt_rank = math.ceil(d_model / 16)  # Mamba default
        self.x_proj = nn.ModuleList(
            [
                nn.Linear(
                    self.d_inner, self.dt_rank + 2 * d_state, bias=False
                )
                for _ in range(self.num_directions)
            ]
        )
        # Δ projection: dt_rank → d_inner (Mamba original)
        self.dt_proj = nn.ModuleList(
            [nn.Linear(self.dt_rank, self.d_inner, bias=True) for _ in range(self.num_directions)]
        )
        # NEW: phase → d_inner. Adds physiological conditioning to Δ.
        self.dt_phase_proj = nn.ModuleList(
            [nn.Linear(phase_dim, self.d_inner, bias=False) for _ in range(self.num_directions)]
        )

        # Initialize dt_proj to place log(dt) in [log(dt_min), log(dt_max)],
        # matching Mamba's defaults.
        for dt_proj in self.dt_proj:
            dt_init_std = self.dt_rank ** -0.5
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True

        # Initialize dt_phase_proj near zero so phase contribution starts small
        # and grows during training (analogous to zero-init trick in DPG).
        for dt_ph in self.dt_phase_proj:
            nn.init.normal_(dt_ph.weight, std=1e-3)

        # A parameter — S4D real init, one per direction.
        # Store log(A) to keep A negative-real.
        self.A_log = nn.ParameterList()
        self.D = nn.ParameterList()
        for _ in range(self.num_directions):
            A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_log.append(nn.Parameter(torch.log(A)))
            self.D.append(nn.Parameter(torch.ones(self.d_inner)))

        # Diastolic anchor MLP: u_{t*} → initial hidden state h_0 ∈ R^{d_inner × d_state}
        if diastolic_anchor:
            self.anchor_mlp = nn.Sequential(
                nn.Linear(self.d_inner, self.d_inner),
                nn.GELU(),
                nn.Linear(self.d_inner, self.d_inner * d_state),
            )
            # Zero-init the final layer so anchor starts as zero hidden state
            # (matches the non-anchor behavior at init).
            nn.init.zeros_(self.anchor_mlp[-1].weight)
            nn.init.zeros_(self.anchor_mlp[-1].bias)

        # Output projection: concat over directions → d_model.
        self.out_proj = nn.Linear(self.num_directions * self.d_inner, d_model, bias=False)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def _direction_forward(
        self,
        a: torch.Tensor,        # (BS, T, d_inner) — pre-SSM feature
        phase: torch.Tensor,    # (BS, T)
        direction: int,         # 0 = forward, 1 = backward
    ) -> torch.Tensor:          # (BS, T, d_inner)
        BS, T, d_inner = a.shape

        # ---- Direction-specific input ordering ----
        if direction == 1:
            a = a.flip(dims=[1])
            phase = phase.flip(dims=[1])

        # ---- Conv1d over T (needs (BS, d_inner, T)) ----
        x_conv = a.transpose(1, 2)                     # (BS, d_inner, T)
        x_conv = self.conv1d[direction](x_conv)[:, :, :T]  # trim padding
        x_conv = self.act(x_conv)
        a = x_conv.transpose(1, 2)                     # (BS, T, d_inner)

        # ---- Parameter projections ----
        # x_proj: d_inner → (dt_rank + 2·N)
        x_dbl = self.x_proj[direction](a)                     # (BS, T, dt_rank+2N)
        dt_raw, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # Δ path: content-driven + phase-driven
        dt = self.dt_proj[direction](dt_raw)                  # (BS, T, d_inner)
        gamma = self.phase_embed(phase)                       # (BS, T, 2L)
        dt = dt + self.dt_phase_proj[direction](gamma)        # (BS, T, d_inner)

        # ---- Prepare selective_scan_fn inputs ----
        u = a.transpose(1, 2).contiguous()             # (BS, d_inner, T)
        delta = dt.transpose(1, 2).contiguous()        # (BS, d_inner, T)
        B_t = B.transpose(1, 2).contiguous()           # (BS, N, T)
        C_t = C.transpose(1, 2).contiguous()           # (BS, N, T)
        A = -torch.exp(self.A_log[direction].float())  # (d_inner, N)  A < 0
        D_skip = self.D[direction].float()             # (d_inner,)

        # ---- Anchor initialization ----
        initial_hidden = None
        if self.diastolic_anchor:
            # t* = argmin_t phase (nearest end-diastole per-sample).
            # Indices in this scan's ordering: since we flipped for backward,
            # argmin on the flipped phase yields the correct in-order t*.
            t_star = phase.argmin(dim=1)  # (BS,)
            # Gather u at t*: a is (BS, T, d_inner)
            batch_idx = torch.arange(BS, device=a.device)
            u_star = a[batch_idx, t_star]  # (BS, d_inner)
            h0 = self.anchor_mlp(u_star)   # (BS, d_inner * d_state)
            initial_hidden = h0.view(BS, self.d_inner, self.d_state)

        # ---- Selective scan ----
        y = _selective_scan_direction(
            u=u,
            delta=delta,
            A=A,
            B=B_t,
            C=C_t,
            D_skip=D_skip,
            z=None,          # gate is applied outside per Mamba block convention
            delta_bias=None,  # already added via dt_proj bias inside dt
            initial_hidden=initial_hidden,
        )
        # y shape: (BS, d_inner, T)
        y = y.transpose(1, 2)  # (BS, T, d_inner)

        # ---- Undo direction flip ----
        if direction == 1:
            y = y.flip(dims=[1])
        return y

    def forward(
        self,
        x: torch.Tensor,       # (B, T, D, h, w)
        phase: torch.Tensor,   # (B, T) in [0, 1)
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D, h, w) float. Feature tensor at some U-Net level.
            phase: (B, T) float in [0, 1). Cardiac phase per frame.

        Returns:
            (B, T, D, h, w). Residual is added: out = x + f(x, phase).
        """
        B, T, D, H, W = x.shape
        assert D == self.d_model, f"d_model mismatch: expected {self.d_model}, got {D}"
        assert phase.shape == (B, T), (
            f"phase shape {phase.shape} != expected ({B}, {T})"
        )

        # ---- (B, T, D, h, w) → (B·h·w, T, D) ----
        # Move T to axis 1, then flatten spatial into batch.
        # Original layout: (B, T, D, H, W)
        # Target: (B*H*W, T, D)
        x_perm = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, T, D)
        x_flat = x_perm.view(B * H * W, T, D)

        # phase must be broadcast to every spatial position.
        phase_flat = phase.unsqueeze(1).unsqueeze(2).expand(B, H, W, T).reshape(B * H * W, T)

        # ---- Residual path ----
        residual = x_flat  # (BS, T, D)

        # ---- Pre-norm + input projection ----
        h = self.norm(x_flat)                     # (BS, T, D)
        ag = self.in_proj(h)                      # (BS, T, 2·d_inner)
        a, g = ag.chunk(2, dim=-1)                # each (BS, T, d_inner)

        # ---- Directional scans ----
        outs = []
        for direction in range(self.num_directions):
            y_dir = self._direction_forward(a, phase_flat, direction)
            outs.append(y_dir)
        y = torch.cat(outs, dim=-1) if self.num_directions > 1 else outs[0]

        # ---- Gate ----
        # For bidirectional the gate is applied to the concatenated output.
        # Replicate g across directions so shapes align.
        g_full = g.repeat(1, 1, self.num_directions)   # (BS, T, dirs·d_inner)
        y = y * F.silu(g_full)

        # ---- Output projection ----
        y = self.out_proj(y)                      # (BS, T, D)

        # ---- Residual add ----
        y = residual + y

        # ---- Reshape back to 5D ----
        y = y.view(B, H, W, T, D).permute(0, 3, 4, 1, 2).contiguous()  # (B, T, D, H, W)
        return y
