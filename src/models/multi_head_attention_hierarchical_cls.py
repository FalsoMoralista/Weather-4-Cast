
from src.utils.tensors import trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParentClassifier(nn.Module):
    def __init__(self, input_dim ,nb_parent_classes):
        super(ParentClassifier, self).__init__()

        self.fc = nn.Linear(input_dim, nb_parent_classes)    
        trunc_normal_(self.fc.weight, std=2e-5)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
            
    def forward(self, x):
        x = self.fc(x)
        return x

class SubClassClassifier(nn.Module):
    def __init__(self, input_dim, nb_subclasses):
        super(SubClassClassifier, self).__init__()

        self.fc = nn.Linear(input_dim, nb_subclasses)
        
        trunc_normal_(self.fc.weight, std=2e-5)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        x = self.fc(x)        
        return x

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

class MultiHeadAttentionHierarchicalCls(nn.Module):

    def __init__(self, input_dim, proj_embed_dim, nb_classes, drop_path, nb_subclasses_per_parent, num_heads=4):
        super(MultiHeadAttentionHierarchicalCls, self).__init__()
        self.proj_embed_dim = proj_embed_dim
        self.nb_classes = nb_classes
        self.nb_subclasses_per_parent = nb_subclasses_per_parent
        self.num_heads = num_heads
        
        self.proj_times = 4

        self.act = nn.GELU()
        
        # -- Classifier Embeddings
        self.parent_proj = nn.Sequential(
            nn.Linear(input_dim, self.proj_times * proj_embed_dim),
            nn.GELU()  # Apply GELU after projection
        )

        self.subclass_proj = nn.Sequential(
            nn.Linear(input_dim, self.proj_times * proj_embed_dim),
            nn.GELU()
        )

        self.cross_attention = MultiHeadCrossAttention(self.proj_times * proj_embed_dim, num_heads)

        trunc_normal_(self.parent_proj[0].weight, std=2e-5)
        trunc_normal_(self.subclass_proj[0].weight, std=2e-5)
        if self.subclass_proj[0].bias is not None and self.parent_proj[0].bias is not None:
            torch.nn.init.constant_(self.parent_proj[0].bias, 0)
            torch.nn.init.constant_(self.subclass_proj[0].bias, 0)

        self.head_drop = nn.Dropout(drop_path)

        self.parent_classifier = ParentClassifier(proj_embed_dim, nb_classes)
        self.child_classifiers = nn.ModuleList(
            [nn.ModuleList(
                [SubClassClassifier(proj_embed_dim, nb_subclasses=nb_subclasses) for _ in range(nb_classes)]
            ) for nb_subclasses in nb_subclasses_per_parent]    
        )

        self.parent_feature_selection = nn.Sequential(
            nn.LayerNorm((self.proj_times + 1) * proj_embed_dim),
            nn.Linear((self.proj_times + 1) * proj_embed_dim, input_dim)
        )
        
        self.subclass_feature_selection = nn.Sequential(
            nn.LayerNorm(self.proj_times * proj_embed_dim),
            nn.Linear(self.proj_times * proj_embed_dim, input_dim)
        )        

    def forward(self, h, device):
        
        B, N, C = h.size() # [batch_size, num_patches, embed_dim]

        parent_proj_embed = self.parent_proj(h) # output shape [B, 256, 5120]
        subclass_proj_embed = self.subclass_proj(h) # output shape [B, 256, 5120]

        # Cross-attention to integrate subclass features into parent features
        integrated_features = self.cross_attention(parent_proj_embed, subclass_proj_embed, subclass_proj_embed)
        integrated_features = F.layer_norm(integrated_features, (integrated_features.size(-1),)) # Normalize over feature-dim 
        
        parent_proj_embed = torch.cat((h, integrated_features), dim=-1)
        parent_proj_embed = self.parent_feature_selection(parent_proj_embed)
        parent_proj_embed = torch.mean(parent_proj_embed, dim=1).squeeze(dim=1) # Take the mean over patch-level representation and squeeze
        parent_proj_embed = F.layer_norm(parent_proj_embed, (parent_proj_embed.size(-1),))
        parent_proj_embed = self.head_drop(parent_proj_embed)

        subclass_proj_embed = self.subclass_feature_selection(subclass_proj_embed)
        subclass_proj_embed = torch.mean(subclass_proj_embed, dim=1).squeeze(dim=1)
        subclass_proj_embed = F.layer_norm(subclass_proj_embed, (subclass_proj_embed.size(-1),))
        subclass_proj_embed = self.head_drop(subclass_proj_embed)

        parent_logits = self.parent_classifier(parent_proj_embed)  # Shape (batch_size, num_parents)
        parent_probs = F.softmax(parent_logits, dim=1)  # Softmax over class dimension

        # The parent class prediction allows to select the index for the correspondent subclass classifier
        y_hat = torch.argmax(parent_probs, dim=1)  # Argmax over class dimension: Shape (batch_size)

        # TODO: vectorize
        # Use the predicted parent class to select the corresponding child classifier
        child_logits = [torch.zeros(B, num, device=device) for num in self.nb_subclasses_per_parent] # Each element within child_logits is associated to a classifier with K outputs.
        for i in range(len(self.nb_subclasses_per_parent)):
            for j in range(B): # Iterate over each sample in the batch                   
                # We will make predictions for each value of K belonging to num_children_per_parent (e.g., [2,3,4,5]) 
                logits = self.child_classifiers[i][y_hat[j]](subclass_proj_embed[j]) 
                child_logits[i][j] = logits        

        return parent_logits, child_logits, parent_proj_embed, subclass_proj_embed


'''
1. Layer Normalization Placement

    Before Attention Layers: It’s common in transformer architectures to apply LayerNorm before attention layers. This is known as Pre-LayerNorm, which helps stabilize training by normalizing the input to each layer.
    After Attention and Projection: Applying LayerNorm after the cross-attention and projection layers is also beneficial. This normalizes the output, making the model more robust to the scale of the input.

2. GELU Activation Placement

    After Linear Projections: The GELU activation is typically placed after linear projection layers (e.g., parent_proj, subclass_proj) to introduce non-linearity. This is important for capturing complex relationships.
    After Cross-Attention: Applying GELU after the cross-attention layer’s output can help in enhancing non-linearity further.

'''