from pathlib import Path

import torch
from torch.utils.data import Dataset

import h5py


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._init_files()


class SslDataset(Dataset):
    ROOT: str = "/home/lucianodourado/weather-4-cast/dataset/w4c24"
    HRIT_KEY: str = "REFL-BT"
    OPERA_KEY: str = "rates.crop"
    type: str

    type_map = {
        "train": "train.reflbt0",
        "val": "val.reflbt0",
    }

    def __init__(self, type: str = "train", transform=None):
        self.type = type
        self.transform = transform

        root = Path(self.ROOT)
        prefix = self.type_map[type]
        self.file_paths = sorted(root.glob(f"**/*{prefix}*"))
        print(f"Found {len(self.file_paths)} files for {type} dataset")
        self.total_images = 0
        self._init_files()

    def _init_files(self):
        self.index = {}
        current_index = 0
        for file in self.file_paths:
            f = h5py.File(file, "r", swmr=True)
            image_count = f[self.HRIT_KEY].shape[0]
            self.index[file] = {
                "start": current_index,
                "end": current_index + image_count,
                "file": f,
            }
            opera_path = (
                str(file.absolute())
                .replace("HRIT", "OPERA")
                .replace("reflbt0", "rates.crop")
                .replace(".ns", "")
            )
            opera_data = h5py.File(opera_path, "r", swmr=True)
            self.index[file]["opera_file"] = opera_data
            self.total_images += image_count
            current_index += image_count

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        print(f"Fetching item at global index: {idx}")
        for file, data in self.index.items():
            start, end = data["start"], data["end"]
            print(f"Checking file: {file}, index range: ({start}, {end})")
            if start <= idx < end:
                dataset = data["file"]
                local_idx = idx - start
                image = dataset[self.HRIT_KEY][local_idx]
                image = torch.tensor(image, dtype=torch.float32)
                if self.transform:
                    image = self.transform(image)
                opera_dataset = data["opera_file"]
                opera_image = opera_dataset[self.OPERA_KEY][local_idx]
                opera_image = torch.tensor(opera_image, dtype=torch.float32)
                print(f"Loaded image {idx} from {file} at local index {local_idx}")
                return image, opera_image


if __name__ == "__main__":
    dataset = SslDataset(type="train")
    print(f"Dataset length: {len(dataset)}")
    hrit, opera = dataset[20308]
    print(f"HRIT shape: {hrit.shape}, OPERA shape: {opera.shape}")
