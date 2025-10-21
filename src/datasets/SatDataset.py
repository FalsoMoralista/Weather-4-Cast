from pathlib import Path

from torch import tensor
from torch.utils.data import Dataset


import torch.nn.functional as F

import h5py

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._init_files()


def make_sat_dataset(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=False,
):
    dataset = SatDataset(
        SatDataset.ROOT,
        type="train" if training else "val",
        transform=transform,
    )

    logger.info("Sat dataset created")

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
        prefetch_factor=8,
        worker_init_fn=worker_init_fn,
    )
    logger.info("Sat Dataset dataloader created")

    return dataset, data_loader, dist_sampler


class SatDataset(Dataset):
    ROOT: str = "/home/lucianodourado/weather-4-cast/dataset/w4c24"
    HRIT_WINDOW_SIZE: int = 4
    HRIT_KEY: str = "REFL-BT"
    OPERA_KEY: str = "rates.crop"
    OPERA_WINDOW_SIZE: int = 16

    type: str
    dataset_path: str
    paths: list[Path]
    years: list[int]
    hrit_path: list[Path]
    opera_path: list[Path]
    hrit_index: dict[str, tuple[int, int]]

    def __init__(
        self,
        dataset_path: str,
        type: str = "train",
        input_size=(224, 224),
        transform=None,
    ):
        self.dataset_path = dataset_path
        self.transform = transform
        self.input_size = input_size
        root = Path(self.dataset_path)
        years = root.glob("20*")
        self.years = sorted([int(year.name) for year in years if year.is_dir()])

        self.paths = [root / str(year) for year in self.years]
        hrit_path = [p / "HRIT" for p in self.paths]
        self.hrit_path = self._sort_files_by_name(hrit_path)
        opera_path = [p / "OPERA" for p in self.paths]
        self.opera_path = self._sort_files_by_name(opera_path)

        self.type = type
        self.hrit_index = self._build_index(type)

    def _sort_files_by_name(self, files: list[Path]):
        return sorted(files, key=lambda x: str(x.absolute()))

    def _init_files(self):
        self.hrit_files = {
            f: h5py.File(f, "r", swmr=True) for f in self.hrit_index.keys()
        }
        opera_names = [
            (f, f.replace("HRIT", "OPERA").replace("reflbt0.ns", "rates.crop"))
            for f in self.hrit_index.keys()
        ]
        self.opera_files = {f: h5py.File(o, "r", swmr=True) for f, o in opera_names}

    def _get_hrit_files_by_type(self, type: str):
        files = []
        if type == "train":
            for p in self.hrit_path:
                files.extend(p.glob("*train.reflbt0*"))
        elif type == "val":
            for p in self.hrit_path:
                files.extend(p.glob("*val.reflbt0*"))
        else:
            raise ValueError(f"Unknown dataset type: {type}")
        return sorted(files, key=lambda x: x.name)

    def _get_hrit_size(self, type: str):
        size = 0
        files = self._get_hrit_files_by_type(type)
        for f in files:
            with h5py.File(f, "r", swmr=True) as file:
                size += (
                    file[self.HRIT_KEY].shape[0]
                    - self.OPERA_WINDOW_SIZE
                    - self.HRIT_WINDOW_SIZE
                    + 1
                )
        return size

    def _get_hrit_train_size(self):
        return self._get_hrit_size("train")

    def _get_hrit_val_size(self):
        return self._get_hrit_size("val")

    def __len__(self):
        if self.type == "train":
            return self._get_hrit_train_size()
        elif self.type == "val":
            return self._get_hrit_val_size()
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def _get_image_count(self, path: Path):
        with h5py.File(path, "r", swmr=True) as file:
            return (
                file[self.HRIT_KEY].shape[0]
                - self.OPERA_WINDOW_SIZE
                - self.HRIT_WINDOW_SIZE
                + 1
            )

    def _build_index(self, type: str):
        files = self._get_hrit_files_by_type(type)
        index = {}
        current_count = 0
        for _, f in enumerate(files):
            count = self._get_image_count(f)
            index[str(f.absolute())] = (current_count, count)
            current_count += count
        return index

    def __getitem__(self, idx):
        for file_name, (start, count) in self.hrit_index.items():
            if start <= idx < start + count:
                hrit = self.hrit_files[file_name]
                opera = self.opera_files[file_name]
                input_start = idx - start
                input_end = input_start + self.HRIT_WINDOW_SIZE

                input = hrit[self.HRIT_KEY][input_start:input_end]

                target_start = idx + self.HRIT_WINDOW_SIZE - start
                target_end = (
                    idx - start + self.HRIT_WINDOW_SIZE + self.OPERA_WINDOW_SIZE
                )
                target = opera[self.OPERA_KEY][target_start:target_end]

                input = F.interpolate(
                    tensor(input),
                    size=self.input_size,
                    mode="bicubic",
                )
                target = tensor(target).squeeze(1)

                if self.transform:
                    input, target = self.transform((input, target))
                return input, target


if __name__ == "__main__":
    training = True
    world_size = 1
    rank = 0
    batch_size = 4
    pin_mem = True
    num_workers = 16
    drop_last = False

    dataset = SatDataset(SatDataset.ROOT, type="train" if training else "val")

    print(f"Dataset length: {len(dataset)}", flush=True)

    logger.info("Sat dataset created")

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
    )

    print(f"Loader length: {len(loader)} - Batch size: {loader.batch_size}")

    for i, data in enumerate(loader):
        input, target = data
        logger.info(input.size(), target.size())
        logger.info(
            f"Batch {i} - Input shape: {input.shape}, Target shape: {target.shape}"
        )
