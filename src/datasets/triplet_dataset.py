import os
import subprocess
import time
import numpy as np
from logging import getLogger
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Optional

GLOBAL_SEED = 0
logger = getLogger()
import os
import PIL
from torchvision import datasets, transforms
from timm.data import create_transform

def build_transform(is_train):
    # Imagenet stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    input_size = 224
    
    # train transform
    if is_train:
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.0,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
        return transform
    
    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC), # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_dataset(is_train, test, image_folder):
    transform = build_transform(is_train)
    if not test:
        root = os.path.join(image_folder, 'train' if is_train else 'val')
    else:
        root = os.path.join(image_folder, 'test')
        logger.info('Test dataset created @%s' % root)
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset

# Adaptive Triplet Dataset Classes
class AdaptiveTripletImageFolder(Dataset):
    """
    Adaptation of ImageFolder for triplet learning with adaptive negative sampling
    """
    
    def __init__(self, 
                 root: str,
                 transform=None,
                 initial_centroids: Optional[np.ndarray] = None,
                 quantities=None,
                 keys=None,
                 feature_dim: int = 512):
        """
        Args:
            root: Root directory path (like ImageFolder)
            transform: Transform to apply to images
            initial_centroids: Initial centroids array of shape (n_classes, feature_dim)
            feature_dim: Dimension of feature vectors for centroids
        """
        # Use ImageFolder to get the data structure
        self.imagefolder = datasets.ImageFolder(root, transform=None)  # We'll apply transform manually
        self.transform = transform
        self.feature_dim = feature_dim

        # Cache stuff
        self.quantities=quantities
        self.keys=list(keys)
        
        # Extract images and labels
        self.samples = self.imagefolder.samples
        self.targets = self.imagefolder.targets
        self.classes = self.imagefolder.classes
        self.class_to_idx = self.imagefolder.class_to_idx
        
        # Group samples by class for efficient sampling
        self.class_to_indices = defaultdict(list)
        for idx, target in enumerate(self.targets):
            self.class_to_indices[target].append(idx)
        
        self.n_classes = len(self.classes)
        
        # Initialize centroids
        if initial_centroids is not None:
            assert initial_centroids.shape == (self.n_classes, feature_dim), \
                f"Centroids shape {initial_centroids.shape} doesn't match expected ({self.n_classes}, {feature_dim})"
            self.centroids = initial_centroids
        else:
            # Random initialization
            self.centroids = np.random.randn(self.n_classes, feature_dim)
            logger.info(f"Initialized random centroids with shape {self.centroids.shape}")
        
        # KNN model for finding nearest classes
        self.knn = NearestNeighbors(n_neighbors=min(self.n_classes, 5), metric='cosine')
        self._update_knn()
        
        logger.info(f"AdaptiveTripletImageFolder created with {len(self.samples)} samples, "
                   f"{self.n_classes} classes, feature_dim={feature_dim}")
    
    @torch.no_grad
    def update_centroids(self, new_centroids, centroid_labels):
        new_centroids = torch.mean(new_centroids, dim=1)
        new_centroids = F.normalize(new_centroids, p=2, dim=-1) # unit sphere projection for cosine similarity
        for data_point, class_idx in zip(new_centroids, centroid_labels):
            key_idx = self.keys.index(class_idx)
            self.quantities[class_idx] += 1
            old_centroid = self.centroids[key_idx]
            self.centroids[key_idx] = old_centroid + ((data_point - old_centroid) / self.quantities[class_idx])
        
        self._update_knn()
    
    def _update_knn(self):
        """Update the KNN model with current centroids"""
        if self.centroids.shape[0] > 1:
            self.knn.fit(self.centroids)
    
    def _get_nearest_different_class(self, class_idx: int) -> int:
        """Get the nearest different class for a given class"""
        if self.n_classes < 2:
            return class_idx  # Fallback if only one class
        
        # Find nearest neighbors (first will be itself if k>1, otherwise nearest different)
        k = min(2, self.n_classes)
        distances, indices = self.knn.kneighbors([self.centroids[class_idx]], n_neighbors=k)
        
        # Return the first different class
        for idx in indices[0]:
            if idx != class_idx:
                return idx
        
        # Fallback: return a random different class
        available_classes = [i for i in range(self.n_classes) if i != class_idx]
        return random.choice(available_classes) if available_classes else class_idx

    def get_triplet_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
        """
        Sample a batch of triplets (anchors, positives, negatives)
        Returns tensors ready for training and the class indices for anchors and positives
        """
        anchors = []
        positives = []  
        negatives = []
        anchor_labels = []  # Class indices for anchors
        positive_labels = []  # Class indices for positives (should be same as anchors)
        
        for _ in range(batch_size):
            # Sample random anchor
            anchor_idx = random.randint(0, len(self.samples) - 1)
            anchor_path, anchor_label = self.samples[anchor_idx]
            anchor_img = self.imagefolder.loader(anchor_path)
            
            # Sample positive (different image, same class)
            positive_indices = [i for i in self.class_to_indices[anchor_label] if i != anchor_idx]
            if positive_indices:
                positive_idx = random.choice(positive_indices)
            else:
                positive_idx = anchor_idx  # Fallback if only one sample in class
            positive_path, positive_label = self.samples[positive_idx]
            positive_img = self.imagefolder.loader(positive_path)
            
            # Sample negative from nearest different class
            nearest_diff_class = self._get_nearest_different_class(anchor_label)
            negative_idx = random.choice(self.class_to_indices[nearest_diff_class])
            negative_path, _ = self.samples[negative_idx]
            negative_img = self.imagefolder.loader(negative_path)
            
            # Apply transforms
            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
            else:
                t = [transforms.Resize((224,224), interpolation=PIL.Image.BICUBIC), transforms.ToTensor()]
                to_tensor = transforms.Compose(t) 
                anchor_img = to_tensor(anchor_img)
                positive_img = to_tensor(positive_img)
                negative_img = to_tensor(negative_img)

            anchors.append(anchor_img)
            positives.append(positive_img)  
            negatives.append(negative_img)
            anchor_labels.append(anchor_label)  # Class index (int)
            positive_labels.append(positive_label)  # Class index (int)
        
        # Stack into tensors
        anchor_batch = torch.stack(anchors)
        positive_batch = torch.stack(positives)
        negative_batch = torch.stack(negatives)
        
        return anchor_batch, positive_batch, negative_batch, anchor_labels, positive_labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Standard getitem for compatibility (returns anchor, label)"""
        path, target = self.samples[idx]
        sample = self.imagefolder.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

class AdaptiveTripletDataLoader:
    """
    DataLoader wrapper for adaptive triplet sampling
    """
    
    def __init__(self, 
                 dataset: AdaptiveTripletImageFolder,
                 batch_size: int,
                 num_batches_per_epoch: int,
                 world_size: int = 1,
                 rank: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.world_size = world_size
        self.rank = rank
        
        # Adjust batches per process for distributed training
        if world_size > 1:
            self.num_batches_per_epoch = num_batches_per_epoch // world_size
            logger.info(f"Adjusted batches per epoch to {self.num_batches_per_epoch} for rank {rank}")
    
    def __iter__(self):
        # Set different random seed for each rank to avoid identical sampling
        if self.world_size > 1:
            random.seed(GLOBAL_SEED + self.rank + int(time.time()))
            np.random.seed(GLOBAL_SEED + self.rank + int(time.time()))
        
        for batch_idx in range(self.num_batches_per_epoch):
            yield self.dataset.get_triplet_batch(self.batch_size)
    
    def __len__(self):
        return self.num_batches_per_epoch
    
    def update_centroids(self, new_centroids: np.ndarray):
        """Update centroids in the dataset"""
        self.dataset.update_centroids(new_centroids)

# Modified function to create triplet dataset
def make_TripletDataset(
    batch_size,
    num_batches_per_epoch=None,
    pin_mem=True,
    quantities=None,
    keys=None, 
    num_workers=8,
    world_size=1,
    rank=0,
    image_folder=None,
    training=True,
    test=False,
    feature_dim=512,
    initial_centroids=None,
    **kwargs  # Catch other arguments for compatibility
):
    """
    Create adaptive triplet dataset and dataloader
    
    Args:
        batch_size: Batch size for triplet batches
        num_batches_per_epoch: Number of batches per epoch
        feature_dim: Dimension of centroids feature space
        initial_centroids: Optional initial centroids
        Other args: Same as make_GenericDataset for compatibility
    
    Returns:
        dataset: AdaptiveTripletImageFolder instance
        dataloader: AdaptiveTripletDataLoader instance  
        None: (no sampler needed since we handle sampling internally)
    """
    
    # Build transform
    transform = build_transform(is_train=training)
    
    # Set up data path
    if not test:
        root = os.path.join(image_folder, 'train' if training else 'val')
    else:
        root = os.path.join(image_folder, 'test')
        logger.info('Test dataset created @%s' % root)
    
    # Create adaptive triplet dataset
    dataset = AdaptiveTripletImageFolder(
        root=root,
        transform=transform,
        initial_centroids=initial_centroids,
        quantities=quantities,
        keys=keys,
        feature_dim=feature_dim
    )
    
    num_batches_per_epoch = len(dataset) // batch_size

    # Create adaptive dataloader
    dataloader = AdaptiveTripletDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        world_size=world_size,
        rank=rank
    )
    
    logger.info(f'Triplet dataset created with {len(dataset)} samples, '
               f'{dataset.n_classes} classes, {num_batches_per_epoch} batches/epoch')
    
    return dataset, dataloader, None

# Example usage function
def example_triplet_training():
    """Example of how to use the integrated triplet system"""
    
    # Create triplet dataset and dataloader
    dataset, dataloader, _ = make_TripletDataset(
        batch_size=32,
        num_batches_per_epoch=100,
        image_folder="/path/to/your/data",
        training=True,
        feature_dim=512,
        world_size=1,
        rank=0
    )
    
    print(f"Dataset has {dataset.n_classes} classes")
    print(f"Sample class names: {dataset.classes[:5]}")
    
    # Training loop
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}")
        
        for batch_idx, (anchors, positives, negatives) in enumerate(dataloader):
            print(f"  Batch {batch_idx + 1}: "
                  f"Anchors {anchors.shape}, Positives {positives.shape}, Negatives {negatives.shape}")
            
            # Your model training code here:
            # anchor_features = model(anchors)
            # positive_features = model(positives)
            # negative_features = model(negatives)
            # loss = triplet_loss(anchor_features, positive_features, negative_features)
            
            # Update centroids periodically (e.g., every 10 batches)
            if batch_idx % 10 == 0 and batch_idx > 0:
                # Compute new centroids from current model features
                # new_centroids = compute_centroids_from_model(model, dataset)
                new_centroids = np.random.randn(dataset.n_classes, 512)  # Dummy for example
                dataloader.update_centroids(new_centroids)
                print(f"    Updated centroids at batch {batch_idx + 1}")
            
            if batch_idx >= 2:  # Just show first few batches for demo
                break
