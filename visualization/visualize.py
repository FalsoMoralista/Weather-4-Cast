import h5py
import matplotlib.pyplot as plt
import numpy as np

# Example: Load a sample Weather4cast patch from an HDF5 file
# Replace 'path_to_file.h5' and dataset name with your actual file and dataset

file_path = '../../boxi_0015.train.reflbt0.ns.h5'

f = h5py.File(file_path, 'r')
data = f['rates.crop']

print("Data shape:", data.shape)  # Should be (11, 256, 256) or similar


# Plot each spectral band
num_bands = data.shape[0]

fig, axs = plt.subplots(3, 4, figsize=(16, 12))
axs = axs.flatten()

for i in range(num_bands):
    ax = axs[i]
    band_img = data[i]
    im = ax.imshow(band_img, cmap='gray')
    ax.set_title(f'Spectral Band {i+1}')
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Hide any empty subplots (if 11 bands in a 3x4 grid, last subplot empty)
for j in range(num_bands, len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.savefig('vis.png')
