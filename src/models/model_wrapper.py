import torch
import torch.nn as nn


class ModelWrapper(nn.Module):

    def __init__(self, backbone):
        super(ModelWrapper, self).__init__()
        self.backbone = backbone
        self.downsample = nn.Conv2d(in_channels=11, out_channels=3, kernel_size=1)

    def forward(self, x):        
        x = self.downsample(x)
        print('x shape, after conv:', x.size())
        with torch.inference_mode():
            features = self.backbone.forward_features(x)
            tokens = features["x_norm_patchtokens"]
        return tokens

