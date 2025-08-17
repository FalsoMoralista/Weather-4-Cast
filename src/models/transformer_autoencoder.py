
import torch
import torch.nn as nn
import torch.nn.functional as F

#import matplotlib.pyplot as plt
import numpy as np

import math

from src.models.vision_transformer import Block
from src.utils.tensors import trunc_normal_

def visualize_masked_input(masked_x, original_x, save_name, item_index=0):
    """
    Visualizes the masked input as a 2D heatmap, highlighting the areas that were masked.

    Args:
        masked_x (torch.Tensor): The input tensor after the mask has been applied, shape (batch_size, num_patches, embed_dim).
        original_x (torch.Tensor): The original input tensor before masking, shape (batch_size, num_patches, embed_dim).
        item_index (int): The index of the item in the batch to visualize.
    """
    # Select the item to visualize
    masked_input = masked_x[item_index].detach().cpu().numpy()
    original_input = original_x[item_index].detach().cpu().numpy()

    # Compute the mean of the embeddings for each patch to create a heatmap-like visualization
    heatmap_masked = np.mean(masked_input, axis=-1)  # Shape: (num_patches,)
    heatmap_original = np.mean(original_input, axis=-1)  # Shape: (num_patches,)

    # Reshape the heatmaps into a 2D grid (e.g., 16x16 for 256 patches)
    num_patches = int(np.sqrt(len(heatmap_masked)))
    heatmap_masked = heatmap_masked.reshape(num_patches, num_patches)
    heatmap_original = heatmap_original.reshape(num_patches, num_patches)

    # Plot the original and masked heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original input heatmap
    axes[0].imshow(heatmap_original, cmap='viridis', interpolation='nearest')
    axes[0].set_title("Original Input Heatmap")
    axes[0].axis('off')

    # Masked input heatmap
    axes[1].imshow(heatmap_masked, cmap='viridis', interpolation='nearest')
    axes[1].set_title("Masked Input Heatmap")
    axes[1].axis('off')

    # Show the color bar
    plt.colorbar(axes[1].imshow(heatmap_masked, cmap='viridis'), ax=axes, fraction=0.046, pad=0.04)
    plt.savefig(save_name) # show()

class VisionTransformerAutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        ratio = 4.0
        drop_rate = 0.25
        
        self.encoder = nn.Sequential(
            Block(dim=1280, num_heads=8, mlp_ratio=ratio, qkv_bias=False, drop=drop_rate),
            nn.Linear(1280, 1024),
            nn.GELU(),            
            Block(dim=1024, num_heads=8, mlp_ratio=ratio, qkv_bias=False, drop=drop_rate),
            nn.Linear(1024, 768),              
        )

        self.decoder = nn.Sequential(
            Block(dim=768, num_heads=8, mlp_ratio=ratio, qkv_bias=False, drop=drop_rate),
            nn.Linear(768, 1024),
            nn.GELU(),            
            Block(dim=1024, num_heads=8, mlp_ratio=ratio, qkv_bias=False, drop=drop_rate),
            nn.Linear(1024, 1280)
        )

        self.init_std=0.02
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        
        layer_id = 0
        layer = self.encoder[0]
        rescale(layer.attn.proj.weight.data, layer_id + 1)
        rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        layer_id = 1
        layer = self.encoder[3]
        rescale(layer.attn.proj.weight.data, layer_id + 1)
        rescale(layer.mlp.fc2.weight.data, layer_id + 1)


        layer_id = 0
        layer = self.decoder[0]
        rescale(layer.attn.proj.weight.data, layer_id + 1)
        rescale(layer.mlp.fc2.weight.data, layer_id + 1)
        
        layer_id = 1
        layer = self.decoder[3]
        rescale(layer.attn.proj.weight.data, layer_id + 1)
        rescale(layer.mlp.fc2.weight.data, layer_id + 1)

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

    def generate_patch_mask(self, batch_size, num_patches, embed_dim, device, mask_fraction=0.8):
        # Calculate the total number of patches to mask
        num_masked_patches = int(num_patches * mask_fraction)

        # Generate a batch of random indices for masking
        all_indices = torch.arange(num_patches, device=device)
        masked_indices = torch.stack([
            all_indices[torch.randperm(num_patches, device=device)[:num_masked_patches]]
            for _ in range(batch_size)
        ])

        # Create the mask with all `False` values
        mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)

        # Use advanced indexing to set the masked patches to `True`
        mask[torch.arange(batch_size, device=device).unsqueeze(1), masked_indices] = True

        # Expand the mask to match the embedding dimension
        mask = mask.unsqueeze(-1).expand(-1, -1, embed_dim)  # Shape: (batch_size, num_patches, embed_dim)

        return mask

    def forward(self, x, device):
        batch_size, num_patches, embed_dim = x.shape
        # Generate the patch mask
        mask = self.generate_patch_mask(batch_size, num_patches, embed_dim, device)
        
        # Mask the input: Set the masked patches to zero
        masked_input = x * (~mask)  # Use `~mask` to keep unmasked patches

        # Normalize the masked input
        masked_input = F.layer_norm(masked_input, (masked_input.size(-1),))

        # Encode and decode
        bottleneck_output = self.encoder(masked_input)
        bottleneck_output = F.layer_norm(bottleneck_output, (bottleneck_output.size(-1),))
        
        reconstructed_input = self.decoder(bottleneck_output)

        reconstructed_input = F.layer_norm(reconstructed_input, (embed_dim,))  # Normalize over feature dimension

        return reconstructed_input, bottleneck_output        