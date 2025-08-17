import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.tensors import trunc_normal_

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, nb_classes):
        super(ClassificationHead, self).__init__()
        self.embed_dim = embed_dim
        self.nb_classes = nb_classes
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(embed_dim, nb_classes)        
        torch.nn.init.constant_(self.classifier.bias, 0)
        trunc_normal_(self.classifier.weight, std=2e-5)

    def forward(self, x):       
        return self.classifier(self.dropout(x))

class ClassificationModel(nn.Module):
    def __init__(self, vit_backbone, embed_dim, nb_classes, dinov2=False):
        super(ClassificationModel, self).__init__()        
        self.vit_encoder = vit_backbone
        self.dinov2 = dinov2
        self.nb_classes = nb_classes
        self.classifier = ClassificationHead(embed_dim=embed_dim, nb_classes=self.nb_classes)
    
    def forward(self, imgs):
        h = self.vit_encoder(imgs)
        if self.dinov2:
            classifier_logits = self.classifier(self.vit_encoder.norm(h))
            h = F.normalize(h, dim=1, p=2)
        else:    
            #h = F.layer_norm(h, (h.size(-1),)) # Normalize over feature-dim 
            h = torch.mean(h, dim=1).squeeze(dim=1)
            classifier_logits = self.classifier(F.layer_norm(h, (h.size(-1),)))
            h = F.normalize(h, dim=1, p=2) # FIXME refactor
        return 0, classifier_logits, h
    
