import torch
import torch.nn as nn


class ModelWrapper(nn.Module):

    def __init__(self, backbone, vjepa):
        super(ModelWrapper, self).__init__()
        self.backbone = backbone
        self.backbone.eval()
        self.vjepa = vjepa
        self.downsample = nn.Conv2d(in_channels=11, out_channels=3, kernel_size=1)

    def forward(self, x):        
        B, T, C, H, W = x.shape  # (2, 4, 11, 252, 252)
        x = x.view(B * T, C, H, W) # [8, 11, 252, 252]
        x = self.downsample(x)
        print('x shape, after conv:', x.size())
        with torch.inference_mode():
            features = self.backbone.forward_features(x)
            tokens = features["x_norm_patchtokens"]  # (B*T, num_patches, embed_dim)
            # combine temporal + patch dimensions in one reshape
            tokens = tokens.reshape(B, T * tokens.size(1), tokens.size(2))
            out = self.vjepa(tokens)
        return out

