from pathlib import Path

from torch.utils.data import Dataset

import h5py


class TDDataset(Dataset):
    ROOT = "/home/lucianodourado/weather-4-cast/dataset/w4c24"

    HRIT_KEY = "REFL-BT"

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        root = Path(self.dataset_path)
        years = root.glob("20*")
        self.years = sorted([int(year.name) for year in years if year.is_dir()])

        self.paths = [root / str(year) for year in self.years]
        hrit_path = [p / "HRIT" for p in self.paths]
        self.hrit_path = sorted(hrit_path, key=lambda x: x.name)
        opera_path = [p / "OPERA" for p in self.paths]
        self.opera_path = sorted(opera_path, key=lambda x: x.name)

    def _get_hrit_size(self, suffix: str):
        size = 0
        for p in self.hrit_path:
            files = p.glob(f"*{suffix}*")
            for f in files:
                with h5py.File(f, "r") as file:
                    size += file[self.HRIT_KEY].shape[0]
        return size

    def _get_hrit_train_size(self):
        return self._get_hrit_size("train.reflbt0")

    def _get_hrit_val_size(self):
        return self._get_hrit_size("val.reflbt0")

    def _get_hrit_test_size(self):
        return self._get_hrit_size("test.reflbt0")

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None


if __name__ == "__main__":
    dataset = TDDataset(TDDataset.ROOT)
    print("HRIT Train Size:", dataset._get_hrit_train_size())
    print("HRIT Val Size:", dataset._get_hrit_val_size())
    print("HRIT Test Size:", dataset._get_hrit_test_size())
