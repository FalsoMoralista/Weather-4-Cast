import sys
from pathlib import Path

import h5py
import numpy as np


class OperaCleaning:
    KEY = "rates.crop"

    def __init__(self, path: str):
        self.base_path = path
        self._initialize()

    def _initialize(self):
        self.files = Path(self.base_path).rglob(f"*{self.KEY}*")

    def clean_file(self, path: str):
        with h5py.File(path, "r+") as hf:
            data = hf[self.KEY]
            num_images = data.shape[0]
            num_bands = data.shape[1]
            print(
                f"Cleaning file: {path}, number of images: {num_images}, number of bands: {num_bands}"
            )
            for i in range(num_images):
                data[i][data[i] < 0] = 0
            hf[self.KEY] = data

    def clean(self):
        for file in self.files:
            print(f"Processing file: {file}")
            path = file.absolute()
            self.clean_file(path)
            self.print(path)

    def print(self, path: str):
        with h5py.File(path, "r") as hf:
            data = hf[self.KEY]
            num_images = data.shape[0]
            for i in range(num_images):
                if np.any(data[i] < 0):
                    print(f"File: {path} - Still have negative values")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean-rates.py <path_to_h5_file>")
        sys.exit(1)
    path = sys.argv[1]
    cleaner = OperaCleaning(path)
    cleaner.clean()
