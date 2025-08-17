import torch
import torch.nn as nn


class JointModel(nn.Module):
    def __init__(self, backbone, projector, embed_dim=1024):
        super(JointModel, self).__init__()
        self.embed_dim = embed_dim
        self.projector = projector
        self.backbone = backbone
        self.change_backbone_state(freeze=True)

    def change_backbone_state(self, freeze):
        if freeze:
            for p_num, p in enumerate(self.backbone.parameters()):
                p.requires_grad = False
            self.frozen = True
            self.backbone.train(False)
        else:
            for p_num, p in enumerate(self.backbone.parameters()):
                p.requires_grad = True
            self.frozen = False
            self.backbone.train(True)

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                x_1 = self.backbone(x)
        else:
            x_1 = self.backbone(x)

        x_2 = self.projector(x_1)
        return x_1, x_2



class JointFTModel(nn.Module):
    def __init__(self, backbone, projector, embed_dim=1024):
        super(JointFTModel, self).__init__()
        self.embed_dim = embed_dim
        self.projector = projector
        self.backbone = backbone
        
        for p_num, p in enumerate(self.backbone.parameters()):
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.projector(x)
        return x
