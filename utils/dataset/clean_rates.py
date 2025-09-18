import sys
import itertools
from pathlib import Path

import h5py
import numpy as np
import torch


class OperaCleaning:
    KEY = "rates.crop"
    HRIT_KEY = "reflbt0.ns"

    def __init__(self, path: str):
        self.base_path = path
        self._initialize()

    def _initialize(self):
        self.files = itertools.chain(
            Path(self.base_path).rglob(f"*{self.KEY}*"),
            Path(self.base_path).rglob(f"*{self.HRIT_KEY}*"),
        )

    def clean_file(self, path: str):
        with h5py.File(path, "r+") as hf:
            data = hf[self.KEY]
            num_images = data.shape[0]
            num_bands = data.shape[1]
            print(
                f"Cleaning file: {path}, number of images: {num_images}, number of bands: {num_bands}",
                flush=True,
            )
            for i in range(num_images):
                arr = data[i][:]
                arr[arr < 0 | np.isnan(arr) | np.isinf(arr)] = 0
                data[i] = arr

    def clean(self):
        for file in self.files:
            print(f"Processing file: {file}", flush=True)
            path = file.absolute()
            self.clean_from_torch(path)
            # self.clean_file(path)
            self.print(path)

    def print(self, path: str):
        with h5py.File(path, "r") as hf:
            data = hf[self.KEY]
            num_images = data.shape[0]
            for i in range(num_images):
                if np.any(data[i] < 0):
                    print(f"File: {path} - Still have negative values", flush=True)
                    break
                if np.any(np.isnan(data[i])):
                    print(f"File: {path} - Still have NaN values", flush=True)
                    break
                if np.any(np.isinf(data[i])):
                    print(f"File: {path} - Still have Inf values", flush=True)
                    break

    def clean_from_torch(self, path: str):
        device = "cuda:0"
        batch_size = 128
        with h5py.File(path, "r+") as hf:
            data = hf[self.KEY]
            num_images = data.shape[0]
            num_bands = data.shape[1]
            print(
                f"Cleaning file: {path}, number of images: {num_images}, number of bands: {num_bands}",
                flush=True,
            )
            for start in range(0, num_images, batch_size):
                end = min(start + batch_size, num_images)
                arr = torch.tensor(data[start:end][:]).to(device)
                arr = torch.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                # data[start:end] = arr.to("cpu").numpy()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean-rates.py <path_to_h5_file>")
        sys.exit(1)
    path = sys.argv[1]
    cleaner = OperaCleaning(path)
    cleaner.clean()
