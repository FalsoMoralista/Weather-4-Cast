import torch
import torch.nn as nn

import torchvision.transforms as T
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
        n_bins=129,
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
        self.n_bins = n_bins
        self.dim_out = dim_out

        self.act = nn.GELU()

        self.time_expansion = nn.Conv2d(
            4,
            num_target_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # self.conv_regression = nn.ConvTranspose2d(in_channels=1024, out_channels=1, kernel_size=18, stride=18)
        self.conv_regression = nn.ConvTranspose2d(
            in_channels=self.dim_out,
            out_channels=1,
            kernel_size=6,
            stride=2,
        )

        self.conv_bins = nn.Conv3d(
            in_channels=1,
            out_channels=self.n_bins,
            kernel_size=(1,32,32),
            stride=32
        )

        self.vit_decoder = VisionTransformer(
            img_size=(224, 224),
            patch_size=16,
            in_chans=num_target_channels,  # 16
            embed_dim=dim_out,  # 1024
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            batch_first=True,
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
        x = self.act(x).view(B, 1, self.num_target_channels, x.size(-2), x.size(-1))
        x = self.conv_bins(x).squeeze(2,3,4)
        #x = x.view(B, self.num_target_channels, x.size(-1)).squeeze(2,3,4) # (B, 16, 32,32)
        #x = x.view(B, self.num_target_channels, 1, x.size(-2), x.size(-1))
        return x


class ModelWrapper(nn.Module):
    def __init__(
        self,
        backbone,
        vjepa,
        patch_size,
        dim_out=384,
        num_heads=24,
        num_decoder_layers=4,
        num_target_channels=16,
        vjepa_size_in=14,
        vjepa_size_out=18,
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
        self.act = nn.GELU()

        self.squeeze = nn.Linear(in_features=1024, out_features=self.dim_out)

        self.vit_decoder = VisionTransformerDecoder(
            T=num_frames,
            vjepa_size_in=vjepa_size_in,
            dim_out=self.dim_out,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            H_patches=224 // patch_size,
            W_patches=224 // self.patch_size,
            num_target_channels=num_target_channels,
        )

        # DinoV3 SAT normalization config
        # https://huggingface.co/facebook/dinov3-vit7b16-pretrain-sat493m/resolve/main/preprocessor_config.json
        self.normalize = T.Normalize(
            mean=[0.430, 0.411, 0.296],
            std=[0.213, 0.156, 0.143],
        )

    def forward(self, x):
        B, T, C, H, W = x.shape  # (B, T=4, 11, 252, 252)
        x = x.view(B * T, C, H, W)  # [B * T, 11, 252, 252]
        x = self.downsample(x)
        x = self.normalize(x)

        with torch.inference_mode():
            features = self.backbone.forward_features(x)
        tokens = features["x_norm_patchtokens"]  # (B*T, num_patches, embed_dim)
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        tokens = tokens.reshape(
            B, T * tokens.size(1), tokens.size(2)
        ).clone()  # Inference mode tensors requires cloning for grad mode reutilisation
        vjepa_out = self.vjepa(
            x=tokens,
            tokenize=False,
            T=T,
            H_patches=H_patches,
            W_patches=W_patches,
        )

        squeezed_out = self.squeeze(vjepa_out) 
        squeezed_out = self.act(squeezed_out)
        regressed = self.vit_decoder(squeezed_out)  # B, 16, 1, 252, 252
        # print(f'regressed output size: {regressed.shape}',flush=True)

        # out = regressed.view(
        #    B,
        #    self.num_target_channels,
        #    self.vjepa_size_out * self.vjepa_size_in,
        #    self.vjepa_size_out * self.vjepa_size_in,
        # )

        return regressed
