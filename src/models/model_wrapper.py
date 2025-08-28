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
        self.normalize = T.Normalize(mean=[0.430, 0.411, 0.296], std=[0.213, 0.156, 0.143]) 


    def forward(self, x):        
        B, T, C, H, W = x.shape  # (2, 4, 11, 252, 252)
        x = x.view(B * T, C, H, W) # [8, 11, 252, 252]
        x = self.downsample(x)
        x = self.normalize(x)

        with torch.inference_mode():
            features = self.backbone.forward_features(x)
        tokens = features["x_norm_patchtokens"]  # (B*T, num_patches, embed_dim)
        print('tokens:', tokens.size())
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        tokens = tokens.reshape(B, T * tokens.size(1), tokens.size(2)).clone()
        out = self.vjepa(x=tokens, tokenize=False,T=T, H_patches=H_patches, W_patches=W_patches)
        return out

