
# Author: Luciano Filho

import torch
import torch.nn  as nn
import torch.nn.functional as F
from src.models.autoencoder import MaskedAutoEncoder
from src.models.transformer_autoencoder import VisionTransformerAutoEncoder

from src.models.hierarchical_classifiers import JEHierarchicalClassifier
from src.models.joint_embedding_classifier import JointEmbeddingClassifier
from src.models.multi_head_attention_hierarchical_cls import MultiHeadAttentionHierarchicalCls
from src.models.multi_head_attention_classifier import MultiHeadAttentionClassifier
from src.models.paired_multi_head_attention_classifier import PairedCrossAttentionClassifier


class FGDCC(nn.Module):

    def __init__(self, vit_backbone, classifier, backbone_patch_mean=False, raw_features=False):
        super(FGDCC, self).__init__()        
        self.backbone_patch_mean = backbone_patch_mean
        self.vit_encoder = vit_backbone
        self.classifier = classifier
        self.raw_features = raw_features

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def forward(self, imgs, device, autoencoder=False, cold_start=False):

        if autoencoder:
            with torch.no_grad():
                h = self.vit_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),)) # Normalize over feature-dim 
            if cold_start:
                return 0,0, h
            else:
                parent_logits, subclass_logits, subclass_proj_embed = self.classifier(h)
                return parent_logits, subclass_logits, subclass_proj_embed

        else:
            h = self.vit_encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),)) # Normalize over feature-dim 

        parent_logits, subclass_logits, subclass_proj_embed = self.classifier(h)
                
        return parent_logits, subclass_logits, subclass_proj_embed


    def setup_autoencoder_features(self, imgs):
        h = self.vit_encoder(imgs)
        h = F.layer_norm(h, (h.size(-1),)) # Normalize over feature-dim 
        # Step 2. Forward into the hierarchical classifier
        _, _, subclass_proj_embed = self.classifier(h) 
        # Detach from the graph
        subclass_proj_embed = subclass_proj_embed.detach()
        return subclass_proj_embed

def get_model(embed_dim, drop_path, nb_classes, K_range, proj_embed_dim, pretrained_model, device, raw_features=False):
    cls = PairedCrossAttentionClassifier(input_dim=embed_dim,
                                      nb_classes=nb_classes,
                                      proj_embed_dim=proj_embed_dim,
                                      drop_path=drop_path,
                                      num_heads=8,
                                      nb_subclasses_per_parent=K_range)

    model = FGDCC(vit_backbone=pretrained_model, classifier=cls, raw_features=raw_features)
    model.to(device)
    return model                 