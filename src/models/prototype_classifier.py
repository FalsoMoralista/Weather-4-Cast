import torch 
import faiss
import torch.nn as nn
import torch.nn.functional as F
import faiss.contrib.torch_utils

from collections import defaultdict
from src.models.prototype_layer import PrototypeLayer
from src.models.vision_transformer import Attention

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()



class CustomVICReg(nn.Module):
    '''
        Sim:
            Effect: This encourages the model to project x closer to its class-specific centroids in embedding space — akin to pulling features toward learned anchors.
            Comment: Since you're doing this for multiple K values (i.e., multiple hypotheses about the number of clusters), you're implicitly promoting multi-scale structure learning.
        Var:
            ensuring that each feature dimension has non-zero variance.
        Cov: 
            it prevents all centroids from collapsing to the same subspace,reduces redundancy.
    '''

    def __init__(self, num_features, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, compute_std_x=False, compute_cov_x=False):
        super().__init__()
        self.num_features = num_features
        self.compute_std_x = compute_std_x
        self.compute_cov_x = compute_cov_x
        
        self.sim_coeff = sim_coeff # lambda
        self.std_coeff = std_coeff # mu
        self.cov_coeff = cov_coeff # nu

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x, nearest_prototypes, prototype_space):
        '''
            Pulls together x to its corresponding nearest prototypes, while mantaining
            the variance of each generated prototype feature (variance) above some
            threshold and decorrelate each pair of feature variables in the generated
            prototype space.  
        
            Args:
                x: batch of samples (B, embed_dim)
                nearest_prototypes: cluster assignments for x (B, len(K_range), embed_dim)
                prototype_space:  spanned prototype space (B, sum(K_range), embed_dim) 
        '''
        B = x.size(0)
        
        repr_loss = 0
        if self.sim_coeff > 0:
            K = nearest_prototypes.size(1)
            x_expanded = x.unsqueeze(1).repeat(1, K, 1) # Create K copies of x, transform into (B, K, embed_dim) for compatibility 
            repr_loss = F.mse_loss(x_expanded, nearest_prototypes) # Pulls together each sample to its nearest centroid

        # Center embeddings
        x = x - x.mean(dim=0)
        prototype_space = prototype_space - prototype_space.mean(dim=0)
        
        epsilon = 1e-4

        # Per feature variance across the batch
        std_prototypes = torch.sqrt(prototype_space.var(dim=0) + epsilon)
        std_loss = torch.mean(F.relu(1 - std_prototypes)) # Hinge loss
        
        if self.compute_std_x:
            std_x = torch.sqrt(x.var(dim=0) + epsilon)
            std_loss = std_loss / 2 + torch.mean(F.relu(1 - std_x)) / 2
        
        K = prototype_space.size(1)
        prototype_space = prototype_space.view(B * K, self.num_features) # Flatten into 2D.
        cov_prototypes = (prototype_space.T @ prototype_space) / (prototype_space.size(0) - 1) # 1/(N - 1) for unbiased estimate 
        
        cov_loss = self.off_diagonal(cov_prototypes).pow_(2).sum().div(self.num_features)
        
        if self.compute_cov_x:
            cov_x = (x.T @ x) / (B - 1)
            cov_loss += self.off_diagonal(cov_x).pow_(2).sum().div(
                self.num_features
            ) 

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss    

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads=8, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.norm1 = norm_layer(embed_dim)
        self.attention = Attention(dim=embed_dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=proj_drop)
    
    def forward(self, x, return_attention=False):
        y, attn = self.attention(self.norm1(x))
        if return_attention:
            return attn
        x = x + y
        return x
        
class PrototypeBasedClassifier(nn.Module):

    def __init__(self, nb_classes, backbone, attn_drop=0., proj_drop=0.,  embed_dim=1024, K_range=[2,3,4,5], device=None, feature_bank=None):
        super(PrototypeBasedClassifier,self).__init__()
        self.nb_classes = nb_classes
        self.embed_dim = embed_dim
        self.K_range = K_range
        self.backbone = backbone
        self.norm_layer=nn.LayerNorm(embed_dim)
        self.prototype_layer = PrototypeNetwork(nb_classes=nb_classes,
                                             K_range=K_range,
                                             feature_bank=feature_bank,
                                             device=device,
                                             embed_dim=embed_dim)
        self.VCR = CustomVICReg(num_features=embed_dim)
        self.attention_blocks = nn.ModuleList([
            Block(embed_dim=embed_dim,attn_drop=attn_drop, proj_drop=proj_drop) for _ in range(nb_classes)
        ])
        self.cls_head = nn.Linear(embed_dim, nb_classes)
        

    def forward(self, x):
        B = x.size(0)
        
        # Step 1: Feature extraction
        x = self.backbone(x) # TODO (1): verify if batch normalization is applied by default or has to be done by hand. 
                              # TODO (2): Gather the last N outputs to recover the prototype representation.  
        
        # Step 3:  Obtain prototype representations
        nearest_centroids , prototype_spaces, y  = self.prototype_layer(x) 

        # Step 4: Compute VICReg loss
        
        vcr_reg = self.VCR(x, nearest_centroids, prototype_spaces.clone())

        # Step 5: Batch samples by class label
        unique_classes = torch.unique(y, sorted=True)
        
        # Step 6: Prepare output tensor for all conditioned prototypes
        conditioned_prototypes = torch.zeros(
            B, sum(self.K_range), self.embed_dim, device=x.device
        )
    
        # Step 7: Apply per-class attention in batch
        for class_id in unique_classes:
            class_id = int(class_id)  # convert scalar tensor to Python int
            indices_tensor = (y == class_id).nonzero(as_tuple=True)[0]  # shape: (N_class,)
            
            attention_block = self.attention_blocks[class_id]
            class_prototypes = prototype_spaces[indices_tensor]  # (N_class, sum(K_range), embed_dim)
            
            attended = attention_block(class_prototypes)  # (N_class, sum(K_range), embed_dim)
            conditioned_prototypes[indices_tensor] = attended

        # Step 8: Normalize and classify
        conditioned_prototypes = F.normalize(conditioned_prototypes, p=2, dim=2)
        conditioned_prototypes = conditioned_prototypes.mean(1)
        #conditioned_prototypes = conditioned_prototypes.view(-1, self.embed_dim)  # flatten across batch dimension
        logits = self.cls_head(conditioned_prototypes)  # (B, nb_classes)

        return logits, vcr_reg


class PrototypeNetwork(nn.Module):
    def __init__(self, nb_classes, K_range, embed_dim, device=None, feature_bank=None):
        super(PrototypeNetwork, self).__init__()
        self.nb_classes=nb_classes
        self.K_range=K_range
        self.embed_dim=embed_dim

        self.prototypes = nn.ModuleList([
            nn.ParameterList([nn.Parameter(torch.randn(K, embed_dim)) for K in K_range]) for _ in range(nb_classes)
        ])

        if feature_bank is not None:
            logger.info('Running K-means initialization...')
            self.k_means_initialization(feature_bank=feature_bank, device=device)
            logger.info('Done...')
        
    def k_means_initialization(self, feature_bank, device):
        resources = faiss.StandardGpuResources()
        for class_id in range(self.nb_classes):
            xb = torch.stack(feature_bank[class_id]).to(device)    
            for K_idx, K in enumerate(self.K_range):
                centroids = faiss_k_means(xb=xb, K=K, resources=resources)
                for i in range(len(self.prototypes[class_id][K_idx])):
                    self.prototypes[class_id][K_idx][i].data.copy_(centroids[i])
                
    def get_prototype_group(self):   
        return [
            torch.stack([
                self.prototypes[class_id][k_idx]
                for class_id in range(self.nb_classes)
            ])  # shape: (nb_classes, K, embed_dim)
            for k_idx in range(len(self.K_range))
        ]

    def forward(self, x):
        B = x.size(0)
        prototype_groups = self.get_prototype_group()  # list of length len(K_range), each (nb_classes, K, embed_dim)
        
        nearest_prototypes = torch.zeros(
            B, len(self.K_range), self.embed_dim, device=x.device
        )
        
        # To build the prototype set (B, sum(K_range), embed_dim), accumulate slices in a list
        prototype_set_list = []
        
        for i, prototype_group in enumerate(prototype_groups):
            nb_classes, K, embed_dim = prototype_group.shape
            
            # Flatten prototypes to (nb_classes * K, embed_dim)
            prototypes_flat = prototype_group.view(-1, embed_dim)
            
            # Compute distances between x (B, embed_dim) and prototypes_flat (nb_classes*K, embed_dim)
            dists = torch.cdist(x, prototypes_flat, p=2)  # (B, nb_classes * K)
            
            # Find nearest prototype index per sample (B,)
            idx_nearest = torch.argmin(dists, dim=1)  # (B,)
            
            # Find class index for nearest prototype by integer division by K
            class_indices = idx_nearest // K  # (B,) each element in [0, nb_classes-1]
            
            # Get the nearest prototypes vectors (B, embed_dim)
            nearest = prototypes_flat[idx_nearest]  # (B, embed_dim)
            nearest_prototypes[:, i, :] = nearest
            
            # For each sample in batch, gather all prototypes of that sample's class for this K
            # Initialize a tensor (B, K, embed_dim) to hold these
            prototypes_for_samples = torch.zeros(B, K, embed_dim, device=x.device)
            
            # Loop over batch to gather class prototypes
            for b in range(B):
                prototypes_for_samples[b] = prototype_group[class_indices[b]]
            
            # prototypes_for_samples: (B, K, embed_dim)
            # Reshape to (B, K * embed_dim) if you want to concatenate all K's later
            # But better keep (B, K, embed_dim) and concatenate along K dimension at the end
            prototype_set_list.append(prototypes_for_samples)
        
        # Concatenate along K dimension across all K_range entries
        # prototype_set_list elements: each (B, K, embed_dim)
        prototype_set = torch.cat(prototype_set_list, dim=1)  # (B, sum(K_range), embed_dim)
        
        return nearest_prototypes, prototype_set, class_indices



def faiss_k_means(xb, K, resources, embed_dim=1024, n_redo=10, max_iter=300, normalize_prototypes=False):
    device = xb.device

    kmeans = faiss.Kmeans(
            d=embed_dim,         
            k=K,           # number of clusters
            nredo=n_redo,           # number of cluster redos
            niter=max_iter,  
            verbose=False,        
            min_points_per_centroid = 0 # avoiding warnings concerning minimum amount of points
    )

    # Set CPU index as the base for the GPU index
    cpu_index = faiss.IndexFlatL2(embed_dim)
    # Transfer to GPU
    config = faiss.GpuIndexFlatConfig()
    gpu_index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
    
    # Use the GPU index in kmeans
    kmeans.index = gpu_index
    # Enable GPU clustering
    kmeans.gpu = True            
    kmeans.train(xb.cpu().numpy()) 
    
    centroids = torch.from_numpy(kmeans.centroids)
    if normalize_prototypes:
        centroids = F.normalize(centroids, p=2, dim=1) # Normalize centroids

    return centroids.to(device)