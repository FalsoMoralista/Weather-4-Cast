import os
import sys

import h5py
import matplotlib.pyplot as plt


class HritVisualizer:
    KEY = "REFL-BT"

    IR_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7]
    VIS_CHANNELS = [8, 9]
    WV_CHANNELS = [10, 11]

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

    def visualize(
        self,
        image_idx: int,
        output_path: str = "bands_visualization.png",
    ):
        if image_idx < 0 or image_idx >= self.num_images:
            raise ValueError(
                f"Image index {image_idx} is out of bounds for the dataset, min: 0, max: {self.num_images} images."
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
    if len(sys.argv) != 3:
        print("Usage: python hrit_visualization.py <path_to_h5_file> <image_index>")
        sys.exit(1)
    path = sys.argv[1]
    image_index = int(sys.argv[2])
    visualizer = HritVisualizer(path)
    visualizer.visualize(image_idx=image_index)
    print("-> Visualization saved to bands_visualization.png")
