
# Author: Luciano Filho

import logging
import sys
import copy

import torch

import src.models.vision_transformer as vit
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_
import torch.nn  as nn
import torch.nn.functional as F
from torch import inf 

import src.models.autoencoder as AE


class ParentClassifier(nn.Module):
    def __init__(self, input_dim, proj_embed_dim ,num_parents):
        super(ParentClassifier, self).__init__()
        self.proj_embed_dim = proj_embed_dim
        self.fc = nn.Linear(input_dim, num_parents)
        self.proj = nn.Linear(num_parents, self.proj_embed_dim)
        self.drop = nn.Dropout(0.2)
        trunc_normal_(self.proj.weight, std=2e-5)
        trunc_normal_(self.fc.weight, std=2e-5)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias, 0)
            torch.nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        x = F.layer_norm(x, (x.size(-1),))  # normalize over feature-dim 
        return x, self.proj(self.drop(x))

class ChildClassifier(nn.Module):
    def __init__(self, input_dim, proj_embed_dim=None, num_children=0):
        super(ChildClassifier, self).__init__()
        self.proj_embed_dim = proj_embed_dim
        self.fc = nn.Linear(input_dim, num_children)
        self.proj = nn.Linear(num_children, proj_embed_dim)
        self.drop = nn.Dropout(0.2)
        
        trunc_normal_(self.proj.weight, std=2e-5)
        trunc_normal_(self.fc.weight, std=2e-5)

        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias, 0)
            torch.nn.init.constant_(self.proj.bias, 0)
    
    def forward(self, x):
        x = self.fc(x)        
        x = F.layer_norm(x, (x.size(-1),))  # normalize over feature-dim 
        return x, self.proj(self.drop(x))

'''
    Hierarchical Classifier
    ----------
    num_parents: int
        Number of parent classes.

    num_children_per_parent: list 

        This allows for model selection, i.e., finding the optimal value of K.
        
        We associate 1 classifier of shape (embed_dim, K) for each K in the range within num_children_per_parent. Each one will then be used 
        to predict the K-means assignment correspondent to each value of K and the one to make the most accurate prediction will be selected 
        for backpropagation.   

        One concern with this approach is that the probability that randomness plays a bigger role in the subclass prediction
        increases inversely with the number of K. In other words, if a random subclass classifier learns how to produce one hot 
        vectors, regardless of its semantic meaning, the chance that it correctly guess a subclass is 25% for K=2. If it does 
        learns that every subclass classifier with output size = K = 2 could either be [0, 1] or [1, 0], the probability
        that it generates a correct output at random is 50%. One way to prevent this from happening perhaps is by forcing it
        to make predictions in the space of the cartesian product between K-means assingments and ground truth labels.

'''
class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim, num_parents, drop_path, proj_embed_dim, num_children_per_parent):
        super(HierarchicalClassifier, self).__init__()
        self.proj_embed_dim = proj_embed_dim
        self.head_drop = nn.Dropout(drop_path)
        self.num_parents = num_parents
        self.num_children_per_parent = num_children_per_parent
        self.parent_classifier = ParentClassifier(input_dim, proj_embed_dim, num_parents)
        self.child_classifiers = nn.ModuleList(
            [nn.ModuleList(
                [ChildClassifier(input_dim, proj_embed_dim ,num_children) for _ in range(num_parents)]
            ) for num_children in num_children_per_parent]    
        )

    def forward(self, x, device):
        x = self.head_drop(x)
        parent_logits, parent_proj_embeddings = self.parent_classifier(x)  # Shape (batch_size, num_parents)
        parent_probs = F.softmax(parent_logits, dim=1)  # Softmax over class dimension

        # The parent class prediction allows to select the index for the correspondent subclass classifier
        parent_class = torch.argmax(parent_probs, dim=1)  # Argmax over class dimension: Shape (batch_size)

        # Use the predicted parent class to select the corresponding child classifier
        child_logits = [torch.zeros(x.size(0), num, device=device) for num in self.num_children_per_parent] # Each element within child_logits is associated to a classifier with K outputs.
        child_proj_embeddings = [torch.zeros(x.size(0), self.proj_embed_dim, device=device) for num in self.num_children_per_parent] # Each element within child_logits is associated to a classifier with K outputs.
        for i in range(len(self.num_children_per_parent)):
            for j in range(x.size(0)): # Iterate over each sample in the batch                   
                # We will make predictions for each value of K belonging to num_children_per_parent (e.g., [2,3,4,5]) 
                logits, proj_embeddings = self.child_classifiers[i][parent_class[j]](x[j])
                child_logits[i][j] = logits
                child_proj_embeddings[i][j] = proj_embeddings
        return parent_logits, child_logits, parent_proj_embeddings, child_proj_embeddings


class JEHierarchicalClassifier(nn.Module):
    def __init__(self, input_dim, num_parents, drop_path, proj_embed_dim, num_children_per_parent):
        super(JEHierarchicalClassifier, self).__init__()
        self.proj_embed_dim = proj_embed_dim
        self.num_parents = num_parents
        self.num_children_per_parent = num_children_per_parent

        # -- Joint Embedding
        self.parent_proj = nn.Linear(input_dim, proj_embed_dim)
        self.child_proj = nn.Linear(input_dim, proj_embed_dim)
        trunc_normal_(self.parent_proj.weight, std=2e-5)
        trunc_normal_(self.child_proj.weight, std=2e-5)
        if self.child_proj.bias is not None and self.parent_proj.bias is not None:
            torch.nn.init.constant_(self.parent_proj.bias, 0)
            torch.nn.init.constant_(self.child_proj.bias, 0)

        self.parent_classifier = JEParentClassifier(proj_embed_dim, num_parents)

        self.head_drop = nn.Dropout(drop_path)

        self.child_classifiers = nn.ModuleList(
            [nn.ModuleList(
                [JEChildClassifier(proj_embed_dim, num_children=num_children) for _ in range(num_parents)]
            ) for num_children in num_children_per_parent]    
        )

    def forward(self, x, device):
        x = self.head_drop(x)

        parent_proj_emb = self.parent_proj(x)
        parent_proj_emb = F.layer_norm(parent_proj_emb, (parent_proj_emb.size(-1),))  # normalize over feature-dim 
        parent_proj_emb = self.head_drop(parent_proj_emb)

        child_proj_emb = self.child_proj(x)
        child_proj_emb = F.layer_norm(child_proj_emb, (child_proj_emb.size(-1),))  # normalize over feature-dim 
        child_proj_emb = self.head_drop(child_proj_emb)

        parent_logits = self.parent_classifier(parent_proj_emb)  # Shape (batch_size, num_parents)
        parent_probs = F.softmax(parent_logits, dim=1)  # Softmax over class dimension

        # The parent class prediction allows to select the index for the correspondent subclass classifier
        parent_class = torch.argmax(parent_probs, dim=1)  # Argmax over class dimension: Shape (batch_size)

        # TODO: vectorize
        # Use the predicted parent class to select the corresponding child classifier
        child_logits = [torch.zeros(x.size(0), num, device=device) for num in self.num_children_per_parent] # Each element within child_logits is associated to a classifier with K outputs.
        for i in range(len(self.num_children_per_parent)):
            for j in range(x.size(0)): # Iterate over each sample in the batch                   
                # We will make predictions for each value of K belonging to num_children_per_parent (e.g., [2,3,4,5]) 
                logits = self.child_classifiers[i][parent_class[j]](child_proj_emb[j]) 
                child_logits[i][j] = logits
        return parent_logits, child_logits, parent_proj_emb, child_proj_emb

# TODO: FIXME colocar as projeções no target encoder?
class JEParentClassifier(nn.Module):
    def __init__(self, input_dim ,num_parents):
        super(JEParentClassifier, self).__init__()

        self.fc = nn.Linear(input_dim, num_parents)    
        trunc_normal_(self.fc.weight, std=2e-5)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias, 0)
            
    def forward(self, x):
        x = self.fc(x)
        x = F.layer_norm(x, (x.size(-1),))  # normalize over feature-dim 
        return x

class JEChildClassifier(nn.Module):
    def __init__(self, input_dim, num_children):
        super(JEChildClassifier, self).__init__()

        self.fc = nn.Linear(input_dim, num_children)
        trunc_normal_(self.fc.weight, std=2e-5)

        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        x = self.fc(x)        
        x = F.layer_norm(x, (x.size(-1),))  # normalize over feature-dim 
        return x
