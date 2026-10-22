import torch
import torch.nn as nn
import fastmri
from typing import List, Tuple, Optional
from .promptmr import PromptMRBlock, SensitivityModel
from .regularizer import MambaPromptUnet
from .motion import compute_motion_map

# Wraps the image-domain regularizer with normalization and complex bookkeeping.
#
# Supports two batch-flatten strategies:
# - use_cp_mamba=False (baseline): T is flattened into channel, giving a
#   (B, 2T, H, W) tensor to the U-Net. Matches PromptMR original.
# - use_cp_mamba=True: T is flattened into batch, giving a (B*T, 2, H, W)
#   tensor. Every frame is processed independently in space, then CP-Mamba
#   blocks inside the U-Net re-establish temporal coupling.
class NormMambaPromptUnet(nn.Module):
    def __init__(
        self,
        num_adj_slices=10,
        n_feat0=48,
        img_size=256,
        use_checkpoint: bool = False,
        use_t_to_batch: bool = False,
        use_cp_mamba: bool = True,
        cp_mamba_kwargs: dict = None,
        use_mdsm: bool = True,
    ):
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.use_t_to_batch = use_t_to_batch
        self.use_cp_mamba = use_cp_mamba
        self.use_mdsm = use_mdsm

        # CP-Mamba and MDSM require an explicit temporal dimension.
        if use_cp_mamba and not use_t_to_batch:
            raise ValueError("use_cp_mamba=True requires use_t_to_batch=True (need explicit T axis).")
        if use_mdsm and not use_cp_mamba:
            raise ValueError("use_mdsm=True requires use_cp_mamba=True.")

        # in_chans depends on flatten strategy.
        # T-to-batch: every frame goes through U-Net separately, so 2 channels
        # (real + imag) per frame.
        # T-in-channel (PromptMR original): all frames stacked as channel, so 2*T.
        in_chans = 2 if use_t_to_batch else num_adj_slices * 2

        self.net = MambaPromptUnet(
            in_chans=in_chans,
            out_chans=in_chans,
            n_feat0=n_feat0,
            img_size=img_size,
            use_checkpoint=use_checkpoint,
            use_cp_mamba=use_cp_mamba,
            cp_mamba_kwargs=cp_mamba_kwargs,
            use_mdsm=use_mdsm,
        )

    def complex_to_chan_dim_flat_channel(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, 2) → (B, 2T, H, W) — original PromptMR behavior."""
        b, c, h, w, two = x.shape
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim_flat_channel(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2T, H, W) → (B, T, H, W, 2)."""
        b, c2, h, w = x.shape
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def complex_to_chan_dim_flat_batch(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, 2) → (B*T, 2, H, W) — CP-Mamba mode."""
        b, t, h, w, _ = x.shape
        return x.permute(0, 1, 4, 2, 3).contiguous().view(b * t, 2, h, w)

    def chan_complex_to_last_dim_flat_batch(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """(B*T, 2, H, W) → (B, T, H, W, 2)."""
        bt, two, h, w = x.shape
        assert bt == B * T and two == 2, f"unexpected shape {x.shape} for B={B} T={T}"
        return x.view(B, T, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)
        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1) + 1e-8
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def forward(self, x: torch.Tensor, phase: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, 2) complex-as-last-dim image after sens_reduce.
            phase: (B, T) required if use_cp_mamba=True. Ignored otherwise.
        Returns:
            (B, T, H, W, 2)
        """
        B, T, H, W, _ = x.shape

        # Recomputed from the cascade-entry image at every cascade.
        motion_map = None
        if self.use_mdsm:
            motion_map = compute_motion_map(x)  # (B, 1, H, W)

        if self.use_t_to_batch:
            # T→batch layout: (B, T, H, W, 2) → (B*T, 2, H, W). Every frame
            # traverses the spatial U-Net independently; CP-Mamba restores T.
            x_flat = self.complex_to_chan_dim_flat_batch(x)   # (B*T, 2, H, W)
            x_flat, mean, std = self.norm(x_flat)              # per-frame norm

            if self.use_cp_mamba:
                assert phase is not None, "use_cp_mamba=True but phase not provided"
                x_out = self.net(
                    x_flat, phase=phase, num_frames=T, motion_map=motion_map,
                )
            else:
                # T-to-batch without temporal scan: C1b ablation.
                x_out = self.net(x_flat)

            x_out = self.unnorm(x_out, mean, std)
            x_out = self.chan_complex_to_last_dim_flat_batch(x_out, B, T)
        else:
            # T-in-channel layout (PromptMR original): (B, T, H, W, 2) → (B, 2T, H, W).
            x_flat = self.complex_to_chan_dim_flat_channel(x)
            x_flat, mean, std = self.norm(x_flat)
            x_out = self.net(x_flat)
            x_out = self.unnorm(x_out, mean, std)
            x_out = self.chan_complex_to_last_dim_flat_channel(x_out)

        return x_out

class CineMamba(nn.Module):
    """Eight-cascade CineMamba reconstruction network.

    Three orthogonal flags control the ablation matrix:
    - `use_t_to_batch`: flatten strategy. False = PromptMR original
      (T packed into channel). True = every frame goes through U-Net
      independently (needed by CP-Mamba and MDSM).
    - `use_cp_mamba`: insert CP-Mamba block after each U-Net level to
      re-establish temporal coupling along T. Requires use_t_to_batch=True.
    - `use_mdsm`: insert MDSM FiLM after each CP-Mamba block.

    Ablation configurations:
    - C1  baseline:      t_to_batch=F, cp_mamba=F, mdsm=F  (PromptMR original)
    - C1b flatten-only:  t_to_batch=T, cp_mamba=F, mdsm=F  (isolates flatten confound)
    - C2  +CP-Mamba(t/T): t_to_batch=T, cp_mamba=T, mdsm=F
    - C3  +CP-Mamba(φ):   t_to_batch=T, cp_mamba=T, mdsm=F
    - C4  full:           t_to_batch=T, cp_mamba=T, mdsm=T

    Sensitivity map estimation always uses the T-in-channel form — coil
    sensitivity is a static per-slice property that does not benefit from
    temporal Mamba conditioning.
    """
    def __init__(self,
                 num_cascades: int = 8,
                 num_adj_slices: int = 10,
                 img_size: int = 256,
                 n_feat0: int = 48,
                 use_checkpoint: bool = True,
                 use_t_to_batch: bool = True,
                 use_cp_mamba: bool = True,
                 cp_mamba_kwargs: dict = None,
                 use_mdsm: bool = True):
        super().__init__()

        # Consistency checks
        if use_cp_mamba and not use_t_to_batch:
            raise ValueError("use_cp_mamba=True requires use_t_to_batch=True")
        if use_mdsm and not use_cp_mamba:
            raise ValueError("use_mdsm=True requires use_cp_mamba=True")

        self.sens_net = SensitivityModel(
            num_adj_slices=num_adj_slices,
            n_feat0=24,
            mask_center=True
        )

        self.cascades = nn.ModuleList([
            PromptMRBlock(
                model=NormMambaPromptUnet(
                    num_adj_slices=num_adj_slices,
                    n_feat0=n_feat0,
                    img_size=img_size,
                    use_checkpoint=use_checkpoint,
                    use_t_to_batch=use_t_to_batch,
                    use_cp_mamba=use_cp_mamba,
                    cp_mamba_kwargs=cp_mamba_kwargs,
                    use_mdsm=use_mdsm,
                ),
                num_adj_slices=num_adj_slices
            )
            for _ in range(num_cascades)
        ])

        self.use_checkpoint = use_checkpoint
        self.num_adj_slices = num_adj_slices
        self.use_t_to_batch = use_t_to_batch
        self.use_cp_mamba = use_cp_mamba
        self.use_mdsm = use_mdsm

    def forward(self, masked_kspace, mask, acc_factor_idx=None, phase=None):
        """
        Args:
            masked_kspace: (B, T, Coil, H, W, 2)
            mask: (B, T, Coil, H, W)
            acc_factor_idx: (B,) long — accepted for interface compatibility
                but not used (ConditionEncoder removed).
            phase: (B, T) float in [0, 1). Required for CP-Mamba.
        """
        # --- Reshape (Flatten Time & Coil for sens_net) ---
        if masked_kspace.dim() == 6:
            B, T, C, H, W, two = masked_kspace.shape
            masked_kspace = masked_kspace.view(B, T * C, H, W, 2)

            if mask.dim() == 5:
                mask = mask.view(B, T * C, H, W)

        if mask.dim() == 4:
            mask = mask.unsqueeze(-1)

        if not mask.dtype == torch.bool:
            mask = mask > 0

        if self.use_cp_mamba and phase is None:
            raise ValueError("use_cp_mamba=True but phase not provided to forward()")

        # 1. Sensitivity map estimation (always T-in-channel form)
        if self.use_checkpoint and self.training:
            sens_maps = torch.utils.checkpoint.checkpoint(
                self.sens_net, masked_kspace, mask, None, use_reentrant=False)
        else:
            sens_maps = self.sens_net(masked_kspace, mask, None)

        # 2. Iterative reconstruction
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            if self.use_checkpoint and self.training:
                if self.use_cp_mamba:
                    _phase = phase
                    def make_fn(cas, ph):
                        def fn(kp, mk, m, sm):
                            return cas(kp, mk, m, sm, phase=ph)
                        return fn
                    kspace_pred = torch.utils.checkpoint.checkpoint(
                        make_fn(cascade, _phase),
                        kspace_pred, masked_kspace, mask, sens_maps,
                        use_reentrant=False,
                    )
                else:
                    # C1 or C1b: no phase needed, PromptMRBlock's regular signature.
                    kspace_pred = torch.utils.checkpoint.checkpoint(
                        cascade,
                        kspace_pred, masked_kspace, mask, sens_maps,
                        use_reentrant=False
                    )
            else:
                kspace_pred = cascade(
                    kspace_pred, masked_kspace, mask, sens_maps, phase=phase,
                )

        # 3. Final output image
        final_image = self.cascades[0].sens_reduce(kspace_pred, sens_maps)
        return fastmri.complex_abs(final_image)
