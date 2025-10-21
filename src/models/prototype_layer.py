import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tensors import trunc_normal_

#import faiss 
#import faiss.contrib.torch_utils



def init_weights(m, init_std=2e-5):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=init_std)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        trunc_normal_(m.weight, std=init_std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def k_means_init(xb, K, embed_dim=1024, n_redo=30, max_iter=300, normalize_prototypes=False):
    device = xb.device

    kmeans = faiss.Kmeans(
            d=embed_dim,         
            k=K,           # number of clusters
            nredo=n_redo,           # number of cluster redos
            niter=max_iter,  
            verbose=False,        
            min_points_per_centroid = K # avoiding warnings concerning minimum amount of points
    )

    # Set CPU index as the base for the GPU index
    cpu_index = faiss.IndexFlatL2(embed_dim)

    # Transfer to GPU
    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = device 
    gpu_index = faiss.index_cpu_to_gpu(resources, device, cpu_index)
    
    # Use the GPU index in kmeans
    kmeans.index = gpu_index
    # Enable GPU clustering
    kmeans.gpu = True
            
    kmeans.train(xb) 
    
    centroids = torch.from_numpy(kmeans.centroids)
    if normalize_prototypes:
        centroids = F.normalize(centroids, p=2, dim=1) # Normalize centroids

    return centroids.to(device)


class PrototypeMatrix(nn.Module):
    def __init__(self, embed_dim, K):
        super(PrototypeMatrix, self).__init__()
        self.embed_dim = embed_dim
        self.K = K 
        self.prototype_matrix = nn.Linear(embed_dim, embed_dim * K)
        init_weights(self.prototype_matrix)

    def forward(self, x):
        return self.prototype_matrix(x).view(x.size(0), self.K,  self.embed_dim)
        

class PrototypeLayer(nn.Module):
    def __init__(self, nb_classes, feature_bank=None, normalize_prototypes=False, embed_dim=1024, K_range=[2,3,4,5]):
        super(PrototypeLayer, self).__init__()
        self.nb_classes = nb_classes
        self.normalize_prototypes = normalize_prototypes
        self.embed_dim = embed_dim
        self.K_range = K_range
        self.prototype_space = nn.ModuleList([PrototypeMatrix(embed_dim=self.embed_dim, K=sum(K_range)) for _ in range(self.nb_classes)])

    def forward(self, x, y):
        '''
            Return the nearest prototypes for each sample in the batch (for each K hypothesis) and the corresponding
            prototype space for each class (y) in the batch.
            
            Args:
                x: a batch of images (B, embed_dim)
                y: a batch of labels (B)

            Returns:
                nearest_prototypes: Tensor of shape (B, len(K_range), embed_dim) 
                prototype_set: Tensor of shape (B, sum(K_range), embed_dim), the spanned prototype space
                selected_indices: Tensor of shape (B, len(K_range)) with prototype indices                                
        '''

        unique_classes = y.unique()
        prototype_dictionary = {}
        for cls in unique_classes:
            x_cls = x[y == cls]
            prototype_space = self.prototype_space[cls](x_cls.mean(dim=0, keepdim=True)).squeeze(0) # Spans the prototype space for class i, from sample x_i 
            prototype_dictionary[int(cls)] = prototype_space

        x = F.normalize(x, p=2, dim=1) # Unit sphere projection for cosine similarity
        nearest_prototypes, prototype_set, selected_indices = [], [], []
        for x_i, y_i in zip(x,y):
            y_i = int(y_i) # Class label
            x_i = x_i.unsqueeze(0) # shape: (1, D)
                    
            prototype_space = prototype_dictionary[int(y_i)] # Retrieve the prototype space for class i
            prototype_set.append(prototype_space)
            
            normalized_prototypes = F.normalize(prototype_space, p=2, dim=1)
            
            prev_K = 0
            selected_idx, selected_prototypes = [], []
            for K in self.K_range:
                prototype_subset = normalized_prototypes[prev_K: prev_K + K] # Select each prototype subset e.g., [0:2], then [2:5], then [5:9], and so on ...
                similarities = F.cosine_similarity(x_i, prototype_subset, dim=1) # TODO add A soft version option (e.g., softmax over cosine sim) could smooth gradients and reduce variance.
                nearest_idx = torch.argmax(similarities)
                selected_prototypes.append(prototype_subset[nearest_idx])
                selected_idx.append(nearest_idx + prev_K)
                prev_K += K
            nearest_prototypes.append(torch.stack(selected_prototypes))
            selected_indices.append(torch.tensor(selected_idx, device=x.device))

        return torch.stack(nearest_prototypes), torch.stack(prototype_set), torch.stack(selected_indices)



