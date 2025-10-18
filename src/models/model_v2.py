import torch
import torch.nn as nn

from src.models.vision_transformer import VisionTransformer
from functools import partial


class VisionTransformerDecoder(nn.Module):
    """
    Non auto-regressive pixel decoder.

    (TODO) [...]
    """

    def __init__(
        self,
        T,
        dim_out,
        num_layers,
        num_heads,
        num_target_channels,
        H_patches,
        W_patches,
        vjepa_size_in,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.T = T
        self.H_patches = H_patches
        self.W_patches = W_patches
        self.num_target_channels = num_target_channels
        self.patch_size = 16
        self.vjepa_size_in = vjepa_size_in

        self.dim_out = dim_out

        self.act = nn.GELU()

        self.time_expansion = nn.Conv2d(
            4,
            num_target_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv_regression = nn.ConvTranspose2d(
            in_channels=dim_out,
            out_channels=1,
            kernel_size=6,
            stride=2,
        )

        self.vit_decoder = VisionTransformer(
            img_size=(32, 32),
            patch_size=2,
            in_chans=num_target_channels,  # 16
            embed_dim=dim_out,  # 1024
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_rope=True,
            tubelet_size=1,
            ignore_patches=True,
            use_activation_checkpointing=False,
        )

    def forward(self, x):
        B, _, _ = x.shape
        x = x.view(
            B, self.T, self.vjepa_size_in * self.vjepa_size_in, self.dim_out
        )  # From  (B, 4*196, 2048) to (B, 4, 196, 2048)
        x = self.time_expansion(
            x
        )  # From (B, 4, 196, 1024) into (B, 16, 196, 1024) i.e., time axis expansion
        x = self.act(x)
        x = x.view(
            -1,
            self.num_target_channels * self.vjepa_size_in * self.vjepa_size_in,
            self.dim_out,
        )  # From (B, 16, 196, 1024) to (B, 16*196, 1024)
        x = self.vit_decoder(
            x,
            T=self.num_target_channels,
            tokenize=False,
            H_patches=self.H_patches,
            W_patches=self.W_patches,
        )
        x = x.view(
            B * self.num_target_channels,
            self.dim_out,
            self.vjepa_size_in,
            self.vjepa_size_in,
        )  # From (B, 16, 196, 1024) to (B*16, 1024, 14, 14)
        x = self.conv_regression(x)
        x = x.view(B, self.num_target_channels, 1, x.size(-2), x.size(-1))
        return x


class ModelWrapperV2(nn.Module):
    def __init__(
        self,
        vjepa,
        patch_size,
        dim_out=1024,
        num_heads=16,
        num_decoder_layers=4,
        num_target_channels=16,
        vjepa_size_in=14,
        vjepa_size_out=18,
        num_frames=4,
        image_size=32,
    ):
        super(ModelWrapperV2, self).__init__()
        self.vjepa = vjepa
        self.patch_size = patch_size
        self.num_target_channels = num_target_channels
        self.vjepa_size_in = vjepa_size_in
        self.vjepa_size_out = vjepa_size_out
        self.dim_out = dim_out
        self.image_size = image_size

        self.vit_decoder = VisionTransformerDecoder(
            T=num_frames,
            vjepa_size_in=vjepa_size_in,
            dim_out=dim_out,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            H_patches=self.image_size // self.patch_size,
            W_patches=self.image_size // self.patch_size,
            num_target_channels=num_target_channels,
        )

    def forward(self, x):
        B, C, T, H, W = x.shape  # (B, T=4, 11, 252, 252)
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        vjepa_out = self.vjepa(
            x=x,
            tokenize=True,
            T=T,
            H_patches=H_patches,
            W_patches=W_patches,
        )
        regressed = self.vit_decoder(vjepa_out)  # B, 16, 1, 252, 252
        return regressed
