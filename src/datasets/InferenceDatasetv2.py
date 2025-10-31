from pathlib import Path
from logging import getLogger

import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import Dataset, DataLoader

import h5py

logger = getLogger()


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # dataset._init_files()


class InferenceDatasetV2(Dataset):
    ROOT: str = "/home/lucianodourado/weather-4-cast/dataset/w4c24"

    def __init__(
        self,
        dataset_path: str,
        type: str = "cum1test",
        input_size=(224, 224),
        transform=None,
    ):
        self.dataset_path = dataset_path
        self.transform = transform
        self.input_size = input_size
        root = Path(self.dataset_path)

        self.type = type

        self.file = list(root.glob(f"**/*{self.type}*"))
        print(f"Found {len(self.file)} files for type {self.type}")
        assert len(self.file) == 1, "Expected exactly one file"
        self.file = self.file[0]

        self.data = h5py.File(self.file, "r", swmr=True)
        print(f"Opened file {self.file}")
        print("Shape:", self.data["REFL-BT"].shape)

        self.length = self.data["REFL-BT"].shape[0]
        self.num_slices = self.length / 4

    def __len__(self):
        return int(self.num_slices)

    def __getitem__(self, idx):
        start_idx = idx * 4
        end_idx = start_idx + 4

        refl_bt = self.data["REFL-BT"][start_idx:end_idx, :, :]
        refl_bt = tensor(refl_bt, dtype=torch.float32)

        if self.input_size is not None:
            input = F.interpolate(
                refl_bt,
                size=self.input_size,
                mode="bicubic",
            )

        if self.transform:
            input = self.transform(input)

        return input


if __name__ == "__main__":
    dataset = InferenceDatasetV2(InferenceDatasetV2.ROOT, type="roxi_0008.cum1test19")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn,
    )

    for i, batch in enumerate(dataloader):
        print(i, batch.shape)
