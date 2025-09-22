from collections import OrderedDict

import torch
import torch.nn as nn

import torchvision.transforms as T
from src.models.vision_transformer import VisionTransformer


class ReductionView(nn.Module):
    def __init__(self, B, T, vjepa_size_in, dim_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.B = B
        self.T = T
        self.vjepa_size_in = vjepa_size_in
        self.dim_out = dim_out

    def forward(self, x):
        B, _, _ = x.shape
        return x.view(
            B,
            self.T,
            self.vjepa_size_in * self.vjepa_size_in,
            self.dim_out,
        )


class StretcherView(nn.Module):
    def __init__(
        self,
        dim_out,
        num_target_channels,
        vjepa_size_in,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dim_out = dim_out
        self.num_target_channels = num_target_channels
        self.vjepa_size_in = vjepa_size_in

    def forward(self, x):
        return x.view(
            -1,
            self.num_target_channels * self.vjepa_size_in * self.vjepa_size_in,
            self.dim_out // 2,
        )


class DecoderVisionTransformer(nn.Module):
    def __init__(
        self,
        B,
        T,
        dim_out,
        num_layers,
        num_heads,
        num_target_channels,
        H_patches,
        W_patches,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.B = B
        self.T = T
        self.H_patches = H_patches
        self.W_patches = W_patches
        self.num_target_channels = num_target_channels
        self.vision_decoder = VisionTransformer(
            img_size=(224, 224),
            patch_size=16,
            in_chans=num_target_channels,  # 16
            embed_dim=dim_out // 2,  # 1024
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            batch_first=True,
            use_rope=True,
            tubelet_size=1,
            ignore_patches=True,
            use_activation_checkpointing=True
        )

    def forward(self, x):
        return self.vision_decoder(
            x,
            T=self.num_target_channels,
            tokenize=False,
            H_patches=self.H_patches,
            W_patches=self.W_patches,
        )


class ModelWrapper(nn.Module):
    def __init__(
        self,
        backbone,
        vjepa,
        patch_size,
        dim_in=4096,
        dim_out=2048,
        num_heads=16,
        num_layers=4,
        num_target_channels=16,
        vjepa_size_in=14,
        vjepa_size_out=18,
        last_linear_dimension=324,
        batch_size=2,
        num_frames=4,
    ):
        super(ModelWrapper, self).__init__()
        self.backbone = backbone
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.vjepa = vjepa
        self.downsample = nn.Conv2d(in_channels=11, out_channels=3, kernel_size=1)
        self.patch_size = patch_size
        self.num_target_channels = num_target_channels
        self.vjepa_size_in = vjepa_size_in
        self.vjepa_size_out = vjepa_size_out
        self.dim_out = dim_out

        # DinoV3 SAT normalization config
        # https://huggingface.co/facebook/dinov3-vit7b16-pretrain-sat493m/resolve/main/preprocessor_config.json
        self.normalize = T.Normalize(
            mean=[0.430, 0.411, 0.296],
            std=[0.213, 0.156, 0.143],
        )
        view_reduction = ReductionView(batch_size, num_frames, vjepa_size_in, dim_out)
        stretcher_view = StretcherView(dim_out, num_target_channels, vjepa_size_in)
        self.vision_decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dimension_reduction",
                        nn.Linear(dim_in, dim_out),
                    ),  # B, T*196, 2048
                    ("reduction_activation", nn.GELU()),
                    ("reduction_view", view_reduction),
                    (
                        "time_stretcher",
                        nn.Conv2d(
                            4,
                            num_target_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),  # B, 16, T*196, 2048
                    ("strecher_activation", nn.GELU()),
                    (
                        "second_dimension_reduction",
                        nn.Linear(dim_out, dim_out // 2),
                    ),  # 1024
                    ("second_reduction_activation", nn.GELU()),
                    ("stretcher_view", stretcher_view),
                    (
                        "decoder",
                        DecoderVisionTransformer(
                            batch_size,
                            num_frames,
                            dim_out,
                            num_layers,
                            num_heads,
                            num_target_channels,
                            224 // self.patch_size,
                            224 // self.patch_size,
                        ),
                    ),
                    ("regressor", nn.Linear(dim_out // 2, last_linear_dimension)),
                ]
            )
        )

    def forward(self, x):
        B, T, C, H, W = x.shape  # (2, 4, 11, 252, 252)
        x = x.view(B * T, C, H, W)  # [8, 11, 252, 252]
        x = self.downsample(x)
        x = self.normalize(x)

        with torch.inference_mode():
            features = self.backbone.forward_features(x)
        tokens = features["x_norm_patchtokens"]  # (B*T, num_patches, embed_dim)
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        tokens = tokens.reshape(B, T * tokens.size(1), tokens.size(2)).clone()
        vjepa_out = self.vjepa(
            x=tokens,
            tokenize=False,
            T=T,
            H_patches=H_patches,
            W_patches=W_patches,
        )
        regressed = self.vision_decoder(vjepa_out)  # B, 3136, 324
        
        out = regressed.view(
            B,
            self.num_target_channels,
            1,
            self.vjepa_size_out * self.vjepa_size_in,
            self.vjepa_size_out * self.vjepa_size_in,
        )

        return out
