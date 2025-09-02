import os
import sys

import torch
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class OperaCleaning:
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

    def print(self):
        threshold = 0
        image_with_all_999 = 0
        num_of_999 = 0
        for i in range(self.num_images):
            if np.any(self.data[i] < threshold):
                num_of_999 = np.sum(self.data[i] == -9999000)
                wrong_pixels = np.sum(self.data[i] < threshold)
                if wrong_pixels == num_of_999:
                    image_with_all_999 += 1
                    print("ATENTION:")
                    print("All wrong pixels are -9999000")
                    print("This timestep should be removed from the dataset")
                    continue
                image_shape = self.data[i].shape
                print(
                    f"Image {i} has shape {image_shape} and contains values < {threshold}"
                )
                print(f"Pixels with values < {threshold}: {wrong_pixels}")
        print(
            f"Total images with ALL values -9999000: {image_with_all_999} out of {self.num_images}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean-rates.py <path_to_h5_file>")
        sys.exit(1)
    path = sys.argv[1]
    cleaner = OperaCleaning(path)
    cleaner.print()
