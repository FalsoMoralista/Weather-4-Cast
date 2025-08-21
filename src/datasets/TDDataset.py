from pathlib import Path

from torch.utils.data import Dataset

import h5py


class TDDataset(Dataset):
    ROOT = "/home/lucianodourado/weather-4-cast/dataset/w4c24"

    HRIT_KEY = "REFL-BT"

    def __init__(self, dataset_path: str, type: str = "train"):
        self.dataset_path = dataset_path
        root = Path(self.dataset_path)
        years = root.glob("20*")
        self.years = sorted([int(year.name) for year in years if year.is_dir()])

        self.paths = [root / str(year) for year in self.years]
        hrit_path = [p / "HRIT" for p in self.paths]
        self.hrit_path = self._sort_files_by_name(hrit_path)
        opera_path = [p / "OPERA" for p in self.paths]
        self.opera_path = self._sort_files_by_name(opera_path)

        self.type = type

    def _sort_files_by_name(self, files: list[Path]):
        return sorted(files, key=lambda x: x.name)

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
            with h5py.File(f, "r") as file:
                size += file[self.HRIT_KEY].shape[0]
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

    def __getitem__(self, idx):
        return None


if __name__ == "__main__":
    dataset = TDDataset(TDDataset.ROOT)
    print("HRIT Train Size:", dataset._get_hrit_train_size())
    print("HRIT Val Size:", dataset._get_hrit_val_size())
