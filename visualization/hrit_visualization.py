import os
import sys

import h5py
import matplotlib.pyplot as plt


class HritVisualizer:
    KEY = "REFL-BT"

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")
        self.file_path = path
        self._initialize()

    def _initialize(self):
        f = h5py.File(self.file_path, "r")
        self.data = f[self.KEY]
        self.shape = self.data.shape
        print(f"Keys in the file: {list(f.keys())}")
        print("Data shape:", self.shape)
        self.num_images = self.shape[0]
        self.num_bands = self.shape[1]

    def visualize_as_grid(
        self,
        image_idx: int,
        output_path: str = "bands_visualization.png",
    ):
        if image_idx < 0 or image_idx >= self.number_of_images:
            raise ValueError(
                f"Image index {image_idx} is out of bounds for the dataset, min: 0, max: {self.number_of_images} images."
            )

        fig, axs = plt.subplots(3, 4, figsize=(16, 12))
        axs = axs.flatten()

        image = self.data[image_idx]

        for i in range(self.num_bands):
            ax = axs[i]
            band_img = image[i]
            im = ax.imshow(band_img, cmap="gray")
            ax.set_title(f"Spectral Band {i + 1}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.savefig(output_path)
        fig.tight_layout()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python hrit_visualization.py <path_to_h5_file> <image_index> <visualization_type>"
        )
        sys.exit(1)
    path = sys.argv[1]
    visualizer = HritVisualizer(path)
    visualizer.visualize_as_grid(image_idx=0)
    print("-> Visualization saved to bands_visualization.png")
