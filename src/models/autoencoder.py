
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1280, 1152),
            nn.GELU(),
            nn.Linear(1152, 1024),
            nn.GELU(),
            nn.Linear(1024 , 768)                        
        )

        self.decoder = torch.nn.Sequential(
            nn.Linear(768, 1024), 
            nn.GELU(),
            nn.Linear(1024, 1152), 
            nn.GELU(),
            nn.Linear(1152, 1280), 
        )

    def generate_mask(self, batch_size, feature_dim, device, mask_fraction=0.2):
        # Calculate the number of elements to mask
        total_elements = batch_size * feature_dim
        num_masked_elements = int(total_elements * mask_fraction)
        
        # Create a mask with all zeros
        mask = torch.zeros(total_elements, dtype=torch.bool, device=device)
        
        # Randomly select indices to be masked
        masked_indices = torch.randperm(total_elements, device=device)[:num_masked_elements]
        
        # Set the selected indices to 1 (masked)
        mask[masked_indices] = 1
        
        # Reshape the mask to the original input shape
        mask = mask.view(batch_size, feature_dim)
        
        return mask

    def forward(self, x, device):
        mask = self.generate_mask(x.size(0), feature_dim=1280, device=device)
        masked_input = x * (~mask) # Mask input    
        masked_input = F.layer_norm(masked_input, (masked_input.size(-1),))

        bottleneck_output = self.encoder(masked_input)
        
        reconstructed_input = self.decoder(bottleneck_output)
        reconstructed_input = F.layer_norm(reconstructed_input, (reconstructed_input.size(-1),))  # normalize over feature-dim 

        return reconstructed_input, bottleneck_output

def vanilla_autoencoder():
    return MaskedAutoEncoder()
