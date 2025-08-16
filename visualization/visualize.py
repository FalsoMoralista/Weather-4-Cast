import h5py
import matplotlib.pyplot as plt
import numpy as np

file_path = '../dataset/w4c24/2019/HRIT/roxi_0008.cum1test19.reflbt0.ns.h5'

f = h5py.File(file_path, 'r')

print(f"Keys in the file:", list(f.keys()))

data = f['REFL-BT']

print("Data shape:", data.shape)

num_bands = data.shape[1]

fig, axs = plt.subplots(3, 4, figsize=(16, 12))
axs = axs.flatten()

first_image = data[0] # Getting the first image from batch

for i in range(num_bands):
    ax = axs[i]
    band_img = first_image[i]
    im = ax.imshow(band_img, cmap='gray')
    ax.set_title(f'Spectral Band {i+1}')
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for j in range(num_bands, len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.savefig('vis.png')
