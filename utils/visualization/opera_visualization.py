import os
import sys

import h5py
import matplotlib.pyplot as plt
import seaborn as sns


class OperaVisualization:
    KEY = "rates.crop"

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
        output_path: str = "opera_visualization.png",
    ):
        if image_idx < 0 or image_idx >= self.num_images:
            raise ValueError(
                f"Image index {image_idx} is out of bounds for the dataset, min: 0, max: {self.num_images} images."
            )

        fig = plt.figure(figsize=(16, 12))

        image = self.data[image_idx][0]
        sns.heatmap(image, cmap="viridis", cbar=False)

        fig.tight_layout()
        fig.savefig(output_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python opera_visualization.py <path_to_h5_file> <image_index>")
        sys.exit(1)
    path = sys.argv[1]
    image_index = int(sys.argv[2])
    visualizer = OperaVisualization(path)
    visualizer.visualize(image_idx=image_index)
    print("-> Visualization saved to opera_visualization.png")
