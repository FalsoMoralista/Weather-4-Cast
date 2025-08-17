import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.tensors import trunc_normal_
from src.models.vision_transformer import Block


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        B, N, _ = query.shape
        
        Q = self.query(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = (attn_weights @ V).transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)
        
        return attn_output
    

class PairedCrossAttentionClassifier(nn.Module):

    def __init__(self, input_dim, proj_embed_dim, nb_classes, drop_path, nb_subclasses_per_parent, num_heads=4):
        super(PairedCrossAttentionClassifier, self).__init__()
        self.proj_embed_dim = proj_embed_dim
        self.nb_classes = nb_classes
        self.nb_subclasses_per_parent = nb_subclasses_per_parent
        self.num_heads = num_heads
        
        self.proj_times = 1

        self.act = nn.GELU()
        
        # -- Classifier Embeddings
        self.subclass_proj = Block(dim=1280, num_heads=8, mlp_ratio=4.0, qkv_bias=False, drop=0.2)

        self.parent_cross_attention = MultiHeadCrossAttention(self.proj_times * proj_embed_dim, num_heads)
        
        # TODO: add back if necessary
        #self.subclass_cross_attention = MultiHeadCrossAttention(self.proj_times * proj_embed_dim, num_heads)

        #self.parent_feature_selection = nn.Linear((self.proj_times + 1) * proj_embed_dim, input_dim)
        #self.subclass_feature_selection = nn.Linear((self.proj_times + 1) * proj_embed_dim, input_dim)
        
        self.parent_classifier = nn.Linear(proj_embed_dim, nb_classes)
        
        self.subclass_classifier = nn.Linear(proj_embed_dim, nb_classes)        

        self.head_drop = nn.Dropout(drop_path)
        
        self.init_weight()

    def init_weight(self):

        #trunc_normal_(self.subclass_proj[0].weight, std=2e-5)
        self.init_std = 2e-5
        self._init_weights(self.subclass_proj)

        #trunc_normal_(self.parent_feature_selection.weight, std=2e-5)
        #trunc_normal_(self.subclass_feature_selection.weight, std=2e-5)

        trunc_normal_(self.parent_cross_attention.key.weight, std=2e-5)
        trunc_normal_(self.parent_cross_attention.query.weight, std=2e-5)
        trunc_normal_(self.parent_cross_attention.value.weight, std=2e-5)

        trunc_normal_(self.parent_classifier.weight, std=2e-5)
        trunc_normal_(self.subclass_classifier.weight, std=2e-5)        

        torch.nn.init.constant_(self.parent_cross_attention.key.bias, 0)
        torch.nn.init.constant_(self.parent_cross_attention.query.bias, 0)
        torch.nn.init.constant_(self.parent_cross_attention.value.bias, 0)
        
        #torch.nn.init.constant_(self.parent_feature_selection.bias, 0)
        #torch.nn.init.constant_(self.subclass_feature_selection.bias, 0)

        torch.nn.init.constant_(self.parent_classifier.bias, 0)
        torch.nn.init.constant_(self.subclass_classifier.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, h):
        
        B, N, C = h.size() # [batch_size, num_patches, embed_dim]

        subclass_proj_embed = self.subclass_proj(h) # output shape [B, 256, 1280]
        subclass_proj_embed = F.layer_norm(subclass_proj_embed, (subclass_proj_embed.size(-1),))

        # Cross-attention to integrate subclass features into parent features
        integrated_parent_features = self.parent_cross_attention(h, subclass_proj_embed, subclass_proj_embed)
        integrated_parent_features = F.layer_norm(integrated_parent_features, (integrated_parent_features.size(-1),)) # Normalize over feature-dim 
        # TODO: remove #integrated_parent_features = torch.mean(integrated_parent_features, dim=1).squeeze(dim=1)

        # TODO: change naming, guess that the opposite makes more sense
        # TODO: compute cross att between subclass_projection and parent classifier embedding 
        
        #integrated_subclass_features = self.subclass_cross_attention(subclass_proj_embed, h, h) # compute cross-attention between sub-class and parent features
        #integrated_subclass_features = F.layer_norm(integrated_subclass_features, (integrated_subclass_features.size(-1),)) # Normalize over feature-dim 
        

        #parent_proj_embed = torch.cat((h, integrated_parent_features), dim=-1)
        #parent_proj_embed = self.parent_feature_selection(parent_proj_embed)
        
        parent_proj_embed = h  + integrated_parent_features

        parent_proj_embed = self.head_drop(torch.mean(parent_proj_embed, dim=1).squeeze(dim=1))
        
        #subclass_proj_embed = torch.cat((subclass_proj_embed, integrated_subclass_features), dim=-1)
        #subclass_proj_embed = self.subclass_feature_selection(subclass_proj_embed)

        #subclass_proj_embed = subclass_proj_embed + integrated_subclass_features # TODO: reconsider dual attention

        parent_logits = self.parent_classifier(parent_proj_embed)  # Shape (batch_size, num_parents)
        
        subclass_logits = self.subclass_classifier(self.head_drop(torch.mean(subclass_proj_embed, dim=1).squeeze(dim=1))) # Input with dropout instead of replacing the variable with a dropped-out vector
        
        # TODO fix dropout and norm mechanisms following the vit implementation
        return parent_logits, subclass_logits, subclass_proj_embed