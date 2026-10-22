import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .mambair import BasicLayer, VSSBlock
from .promptmr import PromptBlock, conv
from .temporal import CPMambaBlock
from .motion import MDSMFiLMHead, apply_mdsm

class MambaLayerWrapper(nn.Module):
    """
    一个包装器，用于将 MambaIR 的 BasicLayer (处理 B, L, C) 
    适配到 U-Net 的 (B, C, H, W) 流程中。
    """
    def __init__(self, dim, depth, input_resolution, mlp_ratio=2., drop_path=0., use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution # (H, W)
        
        # 使用 mambaIR.py 中的 BasicLayer
        self.layer = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=use_checkpoint
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H, W, C) -> (B, L, C)
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        
        # 传入 BasicLayer，需要提供 x_size=(H, W)
        # 注意：如果分辨率变化（因为 U-Net 的多尺度），
        # 这里的 input_resolution 可能需要动态调整或在 init 时固定
        # MambaIR 的实现通常依赖固定的 input_resolution 来做位置编码或初始化
        # 这里我们假设在不同尺度实例化不同的 Wrapper
        
        out = self.layer(x_flat, (H, W))
        
        # (B, L, C) -> (B, H, W, C) -> (B, C, H, W)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out

class MambaDownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depth, input_resolution, kernel_size=3, bias=False, use_checkpoint=False):
        super().__init__()
        
        # 1. Mamba 特征提取 (替换原有的 Encoder CAB 堆叠)
        # 注意：输入分辨率在这一层是固定的
        self.encoder = MambaLayerWrapper(
            dim=in_dim, 
            depth=depth, 
            input_resolution=input_resolution,
            use_checkpoint=use_checkpoint,
        )
        
        # 2. 下采样 (保持 PromptMR 的 Conv 下采样)
        self.down = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=bias)

    def forward(self, x):
        enc = self.encoder(x)
        down = self.down(enc)
        return down, enc

class MambaUpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, prompt_dim, depth, input_resolution, bias=False, use_checkpoint=False):
        super().__init__()
        
        # 计算融合后的通道数
        fuse_dim = in_dim + prompt_dim
        
        # 1. 融合 Prompt 的 Mamba 层 (替换原有的 fuse CAB)
        self.fuse_mamba = MambaLayerWrapper(
            dim=fuse_dim,
            depth=depth, 
            input_resolution=input_resolution, # 此时还是低分辨率
            use_checkpoint=use_checkpoint,
        )
        
        # 2. 降维 (Reduce)
        self.reduce = nn.Conv2d(fuse_dim, in_dim, kernel_size=1, bias=bias)
        
        # 3. 上采样 (Upsample)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False)
        )
        
        # 4. Refine Mamba 层 (替换原有的 ca CAB)
        # 上采样后的分辨率
        up_resolution = (input_resolution[0]*2, input_resolution[1]*2)
        self.refine_mamba = MambaLayerWrapper(
            dim=out_dim,
            depth=depth,
            input_resolution=up_resolution,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x, prompt_dec, skip):
        # x: 来自深层 (低分辨率)
        # prompt_dec: 当前层的 Prompt
        # skip: 来自 Encoder 的跳跃连接 (高分辨率)
        
        # 1. 拼接 Prompt
        x = torch.cat([x, prompt_dec], dim=1)
        
        # 2. 融合特征 (Mamba)
        x = self.fuse_mamba(x)
        
        # 3. 降维
        x = self.reduce(x)
        
        # 4. 上采样并叠加 Skip
        x = self.up(x) + skip
        
        # 5. 精修特征 (Mamba)
        x = self.refine_mamba(x)
        
        return x

class MambaSkipBlock(nn.Module):
    def __init__(self, dim, depth, input_resolution, use_checkpoint=False):
        super().__init__()
        if depth == 0:
            self.skip_op = nn.Identity()
        else:
            self.skip_op = MambaLayerWrapper(dim, depth, input_resolution, use_checkpoint=use_checkpoint)

    def forward(self, x):
        return self.skip_op(x)

class MambaPromptUnet(nn.Module):
    def __init__(self,
                 in_chans=10,
                 out_chans=10,
                 n_feat0=48,
                 img_size=256, # 必须指定输入图像大小，以便计算每层的分辨率
                 feature_dim = [72, 96, 120],
                 prompt_dim = [24, 48, 72],
                 len_prompt = [5, 5, 5],
                 prompt_size = [64, 32, 16],
                 # Mamba 深度配置 (替代原有的 n_enc_cab 等)
                 depths_enc = [2, 2, 2],
                 depths_dec = [2, 2, 2],
                 depths_bottleneck = 2,
                 depths_skip = 1,
                 learnable_input_prompt=False,
                 bias=False,
                 use_checkpoint: bool = False,
                 # Paper ablations may disable the temporal and spatial modules.
                 use_cp_mamba: bool = False,
                 cp_mamba_kwargs: dict = None,
                 use_mdsm: bool = False,
                 ):
        super(MambaPromptUnet, self).__init__()

        # 0. 特征提取 (浅层特征)
        self.use_checkpoint = use_checkpoint
        self.feat_extract = conv(in_chans, n_feat0, kernel_size=3, bias=bias)

        # 计算各级分辨率
        res_0 = (img_size, img_size)
        res_1 = (img_size // 2, img_size // 2)
        res_2 = (img_size // 4, img_size // 4)
        res_3 = (img_size // 8, img_size // 8)

        # 1. Encoder (基于 Mamba)
        # Level 1: n_feat0 -> feature_dim[0]
        self.enc_level1 = MambaDownBlock(n_feat0, feature_dim[0], depths_enc[0], res_0, bias=bias, use_checkpoint=use_checkpoint)
        # Level 2: feature_dim[0] -> feature_dim[1]
        self.enc_level2 = MambaDownBlock(feature_dim[0], feature_dim[1], depths_enc[1], res_1, bias=bias, use_checkpoint=use_checkpoint)
        # Level 3: feature_dim[1] -> feature_dim[2]
        self.enc_level3 = MambaDownBlock(feature_dim[1], feature_dim[2], depths_enc[2], res_2, bias=bias, use_checkpoint=use_checkpoint)

        # 2. Skip Connections
        self.skip_attn1 = MambaSkipBlock(n_feat0, depths_skip, res_0, use_checkpoint=use_checkpoint)
        self.skip_attn2 = MambaSkipBlock(feature_dim[0], depths_skip, res_1, use_checkpoint=use_checkpoint)
        self.skip_attn3 = MambaSkipBlock(feature_dim[1], depths_skip, res_2, use_checkpoint=use_checkpoint)

        # 3. Bottleneck (基于 Mamba)
        self.bottleneck = MambaLayerWrapper(feature_dim[2], depths_bottleneck, res_3, use_checkpoint=use_checkpoint)

        # 4. Decoder (基于 Mamba + Prompt)
        # Level 3: feature_dim[2] -> feature_dim[1]
        # PromptBlock signature: (prompt_dim, prompt_len, prompt_size, lin_dim)
        # PromptMR decoder prompts.
        self.prompt_level3 = PromptBlock(prompt_dim=prompt_dim[2], prompt_len=len_prompt[2], prompt_size=prompt_size[2], lin_dim=feature_dim[2], learnable_input_prompt=learnable_input_prompt)
        self.dec_level3 = MambaUpBlock(feature_dim[2], feature_dim[1], prompt_dim[2], depths_dec[2], input_resolution=res_3, bias=bias, use_checkpoint=use_checkpoint)

        # Level 2: feature_dim[1] -> feature_dim[0]
        self.prompt_level2 = PromptBlock(prompt_dim=prompt_dim[1], prompt_len=len_prompt[1], prompt_size=prompt_size[1], lin_dim=feature_dim[1], learnable_input_prompt=learnable_input_prompt)
        self.dec_level2 = MambaUpBlock(feature_dim[1], feature_dim[0], prompt_dim[1], depths_dec[1], input_resolution=res_2, bias=bias, use_checkpoint=use_checkpoint)

        # Level 1: feature_dim[0] -> n_feat0
        self.prompt_level1 = PromptBlock(prompt_dim=prompt_dim[0], prompt_len=len_prompt[0], prompt_size=prompt_size[0], lin_dim=feature_dim[0], learnable_input_prompt=learnable_input_prompt)
        self.dec_level1 = MambaUpBlock(feature_dim[0], n_feat0, prompt_dim[0], depths_dec[0], input_resolution=res_1, bias=bias, use_checkpoint=use_checkpoint)

        # 5. 输出层
        self.conv_last = conv(n_feat0, out_chans, 5, bias=bias)

        # 6. CP-Mamba blocks: three encoder sites, bottleneck, three decoder sites.
        # plus bottleneck plus each decoder level output. Dims match the tensor
        # that flows out of the corresponding U-Net block.
        #
        # IMPORTANT dim map: MambaDownBlock.forward returns (down, enc) where
        # `enc` has *in_dim* channels (before downsampling). So the skip that
        # flows out of enc_level_i has channels = the *input* dim to that level:
        #   enc1: n_feat0        (input to enc_level1)
        #   enc2: feature_dim[0] (input to enc_level2)
        #   enc3: feature_dim[1] (input to enc_level3)
        #   bottleneck: feature_dim[2]
        #   dec3 out: feature_dim[1] (out_dim of MambaUpBlock at level 3)
        #   dec2 out: feature_dim[0]
        #   dec1 out: n_feat0
        self.use_cp_mamba = use_cp_mamba
        if use_cp_mamba:
            ck = dict(
                d_state=16, d_conv=4, expand=2,
                num_phase_freqs=4, bidirectional=True, diastolic_anchor=True,
            )
            ck.update(cp_mamba_kwargs or {})
            self.ct_enc1 = CPMambaBlock(d_model=n_feat0,        **ck)
            self.ct_enc2 = CPMambaBlock(d_model=feature_dim[0], **ck)
            self.ct_enc3 = CPMambaBlock(d_model=feature_dim[1], **ck)
            self.ct_bot  = CPMambaBlock(d_model=feature_dim[2], **ck)
            self.ct_dec3 = CPMambaBlock(d_model=feature_dim[1], **ck)
            self.ct_dec2 = CPMambaBlock(d_model=feature_dim[0], **ck)
            self.ct_dec1 = CPMambaBlock(d_model=n_feat0,        **ck)

        # 7. MDSM FiLM heads follow CP-Mamba at the same seven sites.
        self.use_mdsm = use_mdsm
        if use_mdsm:
            assert use_cp_mamba, "use_mdsm=True requires use_cp_mamba=True"
            self.mdsc_enc1 = MDSMFiLMHead(out_channels=n_feat0)
            self.mdsc_enc2 = MDSMFiLMHead(out_channels=feature_dim[0])
            self.mdsc_enc3 = MDSMFiLMHead(out_channels=feature_dim[1])
            self.mdsc_bot  = MDSMFiLMHead(out_channels=feature_dim[2])
            self.mdsc_dec3 = MDSMFiLMHead(out_channels=feature_dim[1])
            self.mdsc_dec2 = MDSMFiLMHead(out_channels=feature_dim[0])
            self.mdsc_dec1 = MDSMFiLMHead(out_channels=n_feat0)

    # -----------------------------------------------------------------
    def _checkpoint_call(self, fn, *args):
        if not (self.use_checkpoint and self.training):
            return fn(*args)

        def wrapped(*inner_args):
            device_type = inner_args[0].device.type if inner_args and isinstance(inner_args[0], torch.Tensor) else "cuda"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                return fn(*inner_args)

        return torch.utils.checkpoint.checkpoint(
            wrapped,
            *args,
            use_reentrant=False,
        )

    # -----------------------------------------------------------------
    # CP-Mamba injection helper: apply the block over the temporal
    # axis of a (B*T, D, h, w) tensor. Reshapes back to (B*T, D, h, w)
    # afterwards so downstream blocks remain unchanged.
    # -----------------------------------------------------------------
    def _apply_cp_mamba(
        self,
        x: torch.Tensor,           # (B*T, D, h, w)
        phase: torch.Tensor,        # (B, T) — same for every spatial position
        num_frames: int,
        ct_block: nn.Module,
    ) -> torch.Tensor:
        BT, D, H, W = x.shape
        T = num_frames
        B = BT // T
        x5 = x.view(B, T, D, H, W)
        x5 = self._checkpoint_call(ct_block, x5, phase)  # (B, T, D, h, w)
        return x5.view(BT, D, H, W)

    def _apply_mdsm(
        self,
        x: torch.Tensor,           # (B*T, D, h, w)
        motion_map: torch.Tensor,   # (B, 1, H_full, W_full)
        num_frames: int,
        head: nn.Module,
    ) -> torch.Tensor:
        """Compute FiLM (γ, β) from motion_map at the current level's
        resolution and apply it to a (B*T, D, h, w) tensor.
        """
        BT, D, H, W = x.shape
        gamma, beta = head(motion_map, target_hw=(H, W))
        return apply_mdsm(x, gamma, beta, num_frames)

    def forward(self, x, phase=None, num_frames=None, motion_map=None):
        # x shape: (B*T, C, H, W) — T frames flattened into batch by
        # NormMambaPromptUnet. When use_cp_mamba=True the caller must also
        # provide `phase` of shape (B, T) and `num_frames = T`.
        # When use_mdsm=True the caller must additionally provide
        # `motion_map` of shape (B, 1, H_full, W_full) computed once from
        # the cascade input image.

        if self.use_cp_mamba:
            assert phase is not None, "use_cp_mamba=True but phase not provided"
            assert num_frames is not None, "use_cp_mamba=True but num_frames not provided"
        if self.use_mdsm:
            assert motion_map is not None, "use_mdsm=True but motion_map not provided"

        # 0. 浅层特征
        x = self.feat_extract(x)

        # 1. Encoder
        enc1 = self.enc_level1.encoder(x)
        if self.use_cp_mamba:
            enc1 = self._apply_cp_mamba(enc1, phase, num_frames, self.ct_enc1)
            if self.use_mdsm:
                enc1 = self._apply_mdsm(enc1, motion_map, num_frames, self.mdsc_enc1)
        x_down1 = self.enc_level1.down(enc1)
        enc2 = self.enc_level2.encoder(x_down1)
        if self.use_cp_mamba:
            enc2 = self._apply_cp_mamba(enc2, phase, num_frames, self.ct_enc2)
            if self.use_mdsm:
                enc2 = self._apply_mdsm(enc2, motion_map, num_frames, self.mdsc_enc2)
        x_down2 = self.enc_level2.down(enc2)
        enc3 = self.enc_level3.encoder(x_down2)
        if self.use_cp_mamba:
            enc3 = self._apply_cp_mamba(enc3, phase, num_frames, self.ct_enc3)
            if self.use_mdsm:
                enc3 = self._apply_mdsm(enc3, motion_map, num_frames, self.mdsc_enc3)
        x_down3 = self.enc_level3.down(enc3)

        # 2. Bottleneck
        x_bot = self.bottleneck(x_down3)
        if self.use_cp_mamba:
            x_bot = self._apply_cp_mamba(x_bot, phase, num_frames, self.ct_bot)
            if self.use_mdsm:
                x_bot = self._apply_mdsm(x_bot, motion_map, num_frames, self.mdsc_bot)

        # 3. Decoder with vanilla PromptMR PromptBlock (no c, no prev_prompt)
        dec_prompt3 = self.prompt_level3(x_bot)
        x_up3 = self.dec_level3(x_bot, dec_prompt3, self.skip_attn3(enc3))
        if self.use_cp_mamba:
            x_up3 = self._apply_cp_mamba(x_up3, phase, num_frames, self.ct_dec3)
            if self.use_mdsm:
                x_up3 = self._apply_mdsm(x_up3, motion_map, num_frames, self.mdsc_dec3)

        dec_prompt2 = self.prompt_level2(x_up3)
        x_up2 = self.dec_level2(x_up3, dec_prompt2, self.skip_attn2(enc2))
        if self.use_cp_mamba:
            x_up2 = self._apply_cp_mamba(x_up2, phase, num_frames, self.ct_dec2)
            if self.use_mdsm:
                x_up2 = self._apply_mdsm(x_up2, motion_map, num_frames, self.mdsc_dec2)

        dec_prompt1 = self.prompt_level1(x_up2)
        x_up1 = self.dec_level1(x_up2, dec_prompt1, self.skip_attn1(enc1))
        if self.use_cp_mamba:
            x_up1 = self._apply_cp_mamba(x_up1, phase, num_frames, self.ct_dec1)
            if self.use_mdsm:
                x_up1 = self._apply_mdsm(x_up1, motion_map, num_frames, self.mdsc_dec1)

        # 4. 输出
        out = self.conv_last(x_up1)
        return out

