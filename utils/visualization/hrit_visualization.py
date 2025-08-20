import os
import sys

import h5py
import matplotlib.pyplot as plt
import seaborn as sns


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

    def _plot_channels(self, image, channels: list[int], ax):
        for i, channel in enumerate(channels):
            band_img = image[channel]
            sns.heatmap(band_img, ax=ax[i], cmap="gray", cbar=False)
            ax[i].set_title(f"Channel {channel + 1}")
            ax[i].axis("off")

    def visualize(
        self,
        image_idx: int,
    ):
        if image_idx < 0 or image_idx >= self.num_images:
            raise ValueError(
                f"Image index {image_idx} is out of bounds for the dataset, min: 0, max: {self.num_images} images."
            )

        image = self.data[image_idx]

        ir_fig, ir_axs = plt.subplots(4, 2, figsize=(16, 12))
        ir_axs = ir_axs.flatten()

        self._plot_channels(image, self.IR_CHANNELS, ir_axs)

        vis_fig, vis_axs = plt.subplots(2, 1, figsize=(16, 12))
        vis_axs = vis_axs.flatten()

        self._plot_channels(image, self.VIS_CHANNELS, vis_axs)

        wv_fig, wv_axs = plt.subplots(2, 1, figsize=(16, 12))
        wv_axs = wv_axs.flatten()

        self._plot_channels(image, self.WV_CHANNELS, wv_axs)

        ir_fig.tight_layout()
        vis_fig.tight_layout()
        wv_fig.tight_layout()

        ir_fig.savefig("hrit_ir_bands_visualization.png")
        vis_fig.savefig("hrit_vis_bands_visualization.png")
        wv_fig.savefig("hrit_wv_bands_visualization.png")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python hrit_visualization.py <path_to_h5_file> <image_index>")
        sys.exit(1)
    path = sys.argv[1]
    image_index = int(sys.argv[2])
    visualizer = HritVisualizer(path)
    visualizer.visualize(image_idx=image_index)
    print("-> Visualizations saved")
