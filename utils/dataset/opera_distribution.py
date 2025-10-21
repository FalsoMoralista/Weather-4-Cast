from pathlib import Path

import h5py

PATH = "dataset/w4c24"
names = "rates.crop"
KEY = "rates.crop"


def get_files():
    files = Path(PATH).rglob(f"*{names}*")
    return list(files)


opera_files = get_files()

for opera in opera_files:
    dataset = h5py.File(opera, "r")
    print(f"Processing file: {opera}")
    data = dataset[KEY]
    print(f"Data shape: {data.shape}")
    window_size = 16
