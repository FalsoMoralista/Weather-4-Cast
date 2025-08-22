import os
import sys

import h5py
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


class HritGifGenerator:
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

    def generate(self, channel: int):
        if channel < 0 or channel >= self.num_bands:
            raise ValueError(
                f"Channel {channel} is out of bounds. Must be between 0 and {self.num_bands - 1}."
            )

        if not os.path.exists("gifs"):
            os.makedirs("gifs")

        images = []
        for i in range(self.num_images):
            if i < 1000:
                continue
            if i == 1200:
                break
            band_img = self.data[i, channel, :]
            figure = plt.Figure(figsize=(2.5, 2.5), dpi=100)
            ax = figure.add_subplot()
            sns.heatmap(band_img, cmap="viridis", ax=ax, cbar=False)
            name = f"gifs/hrit_channel_{channel}_frame_{i}.jpg"
            ax.axis("off")
            figure.tight_layout()
            figure.savefig(name)

            img = Image.open(name)
            images.append(img)

        output_path = f"hrit_gif_{channel}.gif"

        if images:
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0,
            )
            print(f"-> GIF saved to {output_path}")
        else:
            print("-> No images to save.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python hrit_gif_generator.py <path_to_h5_file> <channel>")
        sys.exit(1)
    path = sys.argv[1]
    channel = int(sys.argv[2])
    generator = HritGifGenerator(path)
    print(f"-> Initialized GIF generator for file: {path}")
    print(
        f"-> Number of images: {generator.num_images}, Number of bands: {generator.num_bands}"
    )
    print(f"-> Channel selected: {channel}")
    generator.generate(channel)
    print(f"-> GIF generation completed. Image saved to hrit_animation_{channel}.gif")
