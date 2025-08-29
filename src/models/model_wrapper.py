import torch
import torch.nn as nn

import torchvision.transforms as T
from src.models.vision_transformer import VisionTransformer


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
        self.dim_reduction = nn.Linear(dim_in, dim_out)  # B, T*196, 2048
        self.reduction_act = nn.GELU()
        self.time_strecher = nn.Conv2d(
            4,
            num_target_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # B, 16, T*196, 2048
        self.strecher_act = nn.GELU()
        self.second_dim_reduction = nn.Linear(dim_out, dim_out // 2)  # 1024
        self.second_act = nn.GELU()
        # self.decoder_query = nn.Parameter(
        #     torch.randn(num_target_channels * vjepa_size_in * vjepa_size_in, dim_out)
        # )
        self.second_patch_size = 2
        self.decoder = VisionTransformer(
            img_size=(vjepa_size_in, vjepa_size_in),
            patch_size=self.second_patch_size,
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
        )
        self.regressor = nn.Linear(dim_out, last_linear_dimension)

    def forward(self, x):
        B, T, C, H, W = x.shape  # (2, 4, 11, 252, 252)
        x = x.view(B * T, C, H, W)  # [8, 11, 252, 252]
        x = self.downsample(x)
        x = self.normalize(x)

        with torch.inference_mode():
            features = self.backbone.forward_features(x)
        tokens = features["x_norm_patchtokens"]  # (B*T, num_patches, embed_dim)
        print("Tokens:", tokens.size())
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
        print("Vjepa output shape:", vjepa_out.shape, " it should be (B, 784, 4096)")
        vjepa_reducted = self.dim_reduction(vjepa_out)
        print(
            "Vjepa output reducted shape:",
            vjepa_reducted.shape,
            "it should be (B, 784, 2048)",
        )
        vjepa_reducted = self.reduction_act(vjepa_reducted)
        vjepa_reducted = vjepa_reducted.view(
            B,
            T,
            self.vjepa_size_in * self.vjepa_size_in,
            self.dim_out,
        )
        print("vjepa reshaped:", vjepa_reducted.shape, "it should be (B, 4, 196, 2084)")

        vjepa_stretched = self.time_strecher(vjepa_reducted)
        print(
            "Vjepa output stretched shape:",
            vjepa_stretched.shape,
            " it should be (B, 16, 196, 2048)",
        )
        vjepa_stretched = self.strecher_act(vjepa_stretched)

        vjepa_stretched = self.second_dim_reduction(vjepa_stretched)
        vjepa_stretched = self.second_act(vjepa_stretched)
        print(
            "Vjepa output second reducted shape:",
            vjepa_stretched.shape,
            "it should be (B, 16, 196, 1024)",
        )

        # query = self.decoder_query.unsqueeze(0).expand(B, -1, -1)
        # print("Query shape:", query.shape)
        stretched = vjepa_stretched.view(
            -1,
            self.num_target_channels * self.vjepa_size_in * self.vjepa_size_in,
            self.dim_out // 2,
        )
        print("Stretched shape:", stretched.shape, " it should be (B, 3136, 1024)")

        decoded = self.decoder(
            x=stretched,
            T=self.num_target_channels,
            tokenize=False,
            H_patches=self.vjepa_size_in // self.second_patch_size,
            W_patches=self.vjepa_size_in // self.second_patch_size,
        )
        print("Decoded shape:", decoded.shape, "it should be (B, 3136, 1024)")

        regressed = self.regressor(decoded)
        print("Regressed shape:", regressed.shape, "it should be (B, 3136, 324)")
        out = regressed.view(
            B,
            self.num_target_channels,
            1,
            self.vjepa_size_out * self.vjepa_size_in,
            self.vjepa_size_out * self.vjepa_size_in,
        )
        print("Final output shape:", out.shape, "it should be (B, 16, 1, 252, 252)")

        return out
