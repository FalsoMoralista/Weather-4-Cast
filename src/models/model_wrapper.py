import torch
import torch.nn as nn

import torchvision.transforms as T


class ModelWrapper(nn.Module):
    def __init__(self, backbone, vjepa, patch_size):
        super(ModelWrapper, self).__init__()
        self.backbone = backbone
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.vjepa = vjepa
        self.downsample = nn.Conv2d(in_channels=11, out_channels=3, kernel_size=1)
        self.patch_size = patch_size

        # DinoV3 SAT normalization config
        # https://huggingface.co/facebook/dinov3-vit7b16-pretrain-sat493m/resolve/main/preprocessor_config.json
        self.normalize = T.Normalize(
            mean=[0.430, 0.411, 0.296],
            std=[0.213, 0.156, 0.143],
        )
        self.dim_reduction = nn.Linear(4096, 2048)  # B, T*196, 2048
        self.reduction_act = nn.GELU()
        self.time_strecher = nn.Conv3d(
            4,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # B, 16, T*196, 2048
        self.strecher_act = nn.GELU()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=2048, nhead=16, batch_first=True),
            num_layers=4,
        )

    def forward(self, x):
        B, T, C, H, W = x.shape  # (2, 4, 11, 252, 252)
        x = x.view(B * T, C, H, W)  # [8, 11, 252, 252]
        x = self.downsample(x)
        x = self.normalize(x)

        with torch.inference_mode():
            features = self.backbone.forward_features(x)
        tokens = features["x_norm_patchtokens"]  # (B*T, num_patches, embed_dim)
        print("tokens:", tokens.size())
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
        vjepa_reducted = self.dim_reduction(vjepa_out)
        vjepa_reducted = self.reduction_act(vjepa_reducted)

        vjepa_stretched = self.time_strecher(vjepa_reducted)
        vjepa_stretched = self.strecher_act(vjepa_stretched)

        out = self.decoder(
            tgt=vjepa_reducted,
            memory=vjepa_stretched.flatten(2).permute(0, 2, 1),
        )

        return out
