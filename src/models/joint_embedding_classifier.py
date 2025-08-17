# Author Luciano Filho

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.tensors import trunc_normal_


class JointEmbeddingClassifier(nn.Module):

    def __init__(self, input_dim, nb_classes, drop_path, proj_embed_dim, num_children_per_parent):
        super(JointEmbeddingClassifier, self).__init__()
        
        self.proj_embed_dim = proj_embed_dim
        self.nb_classes = nb_classes
        self.num_children_per_parent = num_children_per_parent
        
        self.parent_proj = nn.Linear(input_dim, self.proj_embed_dim)
        self.subclass_proj = nn.Linear(input_dim, self.proj_embed_dim)
        init_linear(self.parent_proj)
        init_linear(self.subclass_proj)
        
        self.head_drop = nn.Dropout(drop_path)

        self.parent_classifier = ParentClassifier(proj_embed_dim, nb_classes)

        self.child_classifiers = nn.ModuleList(
            [nn.ModuleList(
                [SubClassClassifier(input_dim=proj_embed_dim, num_subclasses=num_children) for _ in range(nb_classes)]
            ) for num_children in num_children_per_parent]    
        )        

    # This classifier uses the dataset label to index the subclass classifiers.
    def forward(self, x, y, device):

        parent_proj_embed = self.parent_proj(self.head_drop(x))
        parent_logits = self.parent_classifier(self.head_drop(parent_proj_embed))
    
        child_proj_embed = self.subclass_proj(self.head_drop(x))

        # Use the predicted parent class to select the corresponding child classifier
        child_logits = [torch.zeros(x.size(0), num, device=device) for num in self.num_children_per_parent] # Each element within child_logits is associated to a classifier with K outputs.
        
        # TODO: 
        # 1 - vectorize
        # 2 - make child logits a batch
        for i in range(len(self.num_children_per_parent)):
            # Iterate over each sample in the batch
            for j in range(x.size(0)): 
                # We will make predictions for each value of K belonging to num_children_per_parent (e.g., [2,3,4,5]) 
                child_logits[i][j] = self.child_classifiers[i][y[j]](child_proj_embed[j])

        return parent_logits, child_logits, parent_proj_embed, child_proj_embed
    
def init_linear(layer):
    trunc_normal_(layer.weight, std=2e-5)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, 0)    
    
class ParentClassifier(nn.Module):
    def __init__(self, input_dim, nb_classes):
        super(ParentClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, nb_classes) 
        init_linear(self.fc)       

    def forward(self, x):
        x = self.fc(x)
        return x

class SubClassClassifier(nn.Module):
    def __init__(self, input_dim, num_subclasses=0):
        super(SubClassClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_subclasses)
        init_linear(self.fc)

    def forward(self, x):
        x = self.fc(x)        
        return x
