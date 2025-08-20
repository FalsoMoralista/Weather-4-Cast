import os
import sys

import h5py


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

    def generate(self, channel: str, output_path: str = "hrit_animation.gif"):
        pass


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python hrit_gif_generator.py <path_to_h5_file> <channel>")
        sys.exit(1)
    path = sys.argv[1]
    channel = sys.argv[2]
    generator = HritGifGenerator(path)
    print(f"-> Initialized GIF generator for file: {path}")
    print(
        f"-> Number of images: {generator.num_images}, Number of bands: {generator.num_bands}"
    )
    print(f"-> Channel selected: {channel}")
    generator.generate(channel)
