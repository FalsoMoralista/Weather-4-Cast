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
        image_size=32,
        n_bins=513,
        patch_size=2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.T = T
        self.H_patches = H_patches
        self.W_patches = W_patches
        self.num_target_channels = num_target_channels
        self.patch_size = patch_size
        self.n_bins = n_bins
        self.image_size = image_size
        self.vjepa_size_in = vjepa_size_in
        self.num_patches = self.H_patches * self.W_patches

        self.dim_out = dim_out

        self.act = nn.ReLU()

        self.time_expansion = nn.Conv2d(
            self.T,
            self.num_target_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.expansion_norm = nn.LayerNorm(self.dim_out)

        self.vit_decoder = VisionTransformer(
            img_size=(32, 32),
            patch_size=self.patch_size,
            in_chans=num_target_channels,  # 16
            embed_dim=dim_out,
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

        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
            nn.Conv3d(
                in_channels=self.num_target_channels,
                out_channels=self.num_target_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.upscale_norm = nn.LayerNorm(self.dim_out)

        self.conv_regression = nn.Conv3d(
            in_channels=self.dim_out,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.regression_norm = nn.BatchNorm2d(self.num_target_channels)

        self.conv_bins = nn.Conv2d(
            in_channels=self.num_target_channels,
            out_channels=self.n_bins,
            kernel_size=(self.image_size, self.image_size),
            stride=1,
        )

    def forward(self, z):
        B = z.shape[0]
        z = z.view(B, self.T, self.num_patches, self.dim_out)
        z = self.time_expansion(z)
        z = self.act(z)
        z = self.expansion_norm(z)
        z = z.view(
            B,
            self.num_target_channels * self.num_patches,
            self.dim_out,
        )
        z = self.vit_decoder(
            z,
            self.num_target_channels,
            H_patches=self.H_patches,
            W_patches=self.W_patches,
            tokenize=False,
        )
        z = z.view(
            B,
            self.num_target_channels,
            self.H_patches,
            self.W_patches,
            self.dim_out,
        )
        z = z.permute(0, 1, 4, 2, 3)
        z = self.upscale(z)
        z = z.permute(0, 1, 3, 4, 2)
        z = self.upscale_norm(z)
        z = z.permute(0, 4, 1, 2, 3)
        z = self.conv_regression(z)
        z = z.permute(0, 2, 3, 4, 1)
        z = z.squeeze(-1)
        z = self.act(z)
        z = self.regression_norm(z)
        z = self.conv_bins(z)
        z = z.squeeze((-1, -2))
        z = self.act(z)
        return z


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
        num_frames=4,
        image_size=32,
        n_bins=129,
    ):
        super(ModelWrapperV2, self).__init__()
        self.vjepa = vjepa
        self.patch_size = patch_size
        self.num_target_channels = num_target_channels
        self.vjepa_size_in = vjepa_size_in
        self.dim_out = dim_out
        self.image_size = image_size
        self.n_bins = n_bins

        self.vit_decoder = VisionTransformerDecoder(
            T=num_frames,
            vjepa_size_in=vjepa_size_in,
            dim_out=dim_out,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            H_patches=self.image_size // self.patch_size,
            W_patches=self.image_size // self.patch_size,
            num_target_channels=num_target_channels,
            image_size=image_size,
            patch_size=patch_size,
            n_bins=n_bins,
        )

    def forward(self, x):
        B, C, T, H, W = x.shape  # (B, T=4, 11, 32, 32)
        x = x.permute(0, 2, 1, 3, 4)  # B, C, T, H, W
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        vjepa_out = self.vjepa(
            x=x,
            tokenize=True,
            T=T,
            H_patches=H_patches,
            W_patches=W_patches,
        )
        regressed = self.vit_decoder(vjepa_out)  # B, 16, 513
        return regressed
