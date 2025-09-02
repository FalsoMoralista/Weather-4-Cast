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
        num_of_errors = 0
        for i in range(self.num_images):
            if np.any(self.data[i] < threshold):
                print("Type", type(self.data[i]))
                image_shape = self.data[i].shape
                print(
                    f"Image {i} has shape {image_shape} and contains values < {threshold}"
                )
                wrong_pixels = np.sum(self.data[i] < threshold)
                number_of_pixels = np.prod(image_shape)
                print(f"Pixels with values < {threshold}: {wrong_pixels}")
                print(
                    f"Min value: {np.min(self.data[i])}, Max value: {np.max(self.data[i])}"
                )
                if wrong_pixels == number_of_pixels:
                    num_of_errors += 1
        print(
            f"Total images with ALL values < {threshold}: {num_of_errors} out of {self.num_images}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean-rates.py <path_to_h5_file>")
        sys.exit(1)
    path = sys.argv[1]
    cleaner = OperaCleaning(path)
    cleaner.print()
