import os
import sys
from pathlib import Path
import zipfile


def zip_dataset(dataset_name: str, last_file_index: int = -1):
    dataset_path = Path(f"./dataset/{dataset_name}")
    if not dataset_path.exists():
        print(f"Dataset path {dataset_path} does not exist.")
        return

    pack_path = Path("./dataset/packs")
    if not pack_path.exists():
        pack_path.mkdir(parents=True)

    files = dataset_path.glob("**/*")
    files_name = [file for file in files if file.is_file()]
    files_name = sorted(files_name, key=lambda x: x.stat().st_size, reverse=True)

    MAX_SIZE = 5 * 1024 * 1024 * 1024

    pack_idx = 0

    current_pack_size = 0
    current_pack_zip = zipfile.ZipFile(
        pack_path / f"{dataset_name}_{pack_idx}.zip", "w", zipfile.ZIP_DEFLATED
    )

    for idx, file in enumerate(files):
        if idx < last_file_index:
            continue
        file_size = os.path.getsize(file)
        if file_size > MAX_SIZE:
            pack_idx += 1
            large_file_zip = zipfile.ZipFile(
                pack_path / f"{dataset_name}_{pack_idx}.zip", "w", zipfile.ZIP_DEFLATED
            )
            large_file_zip.write(file, arcname=file.relative_to(dataset_path))
            large_file_zip.close()
            print(
                f"File {file} is larger than 5GB, packed separately. Pack index: {pack_idx}"
            )
            continue
        if current_pack_size + file_size > MAX_SIZE:
            current_pack_zip.close()
            pack_idx += 1
            current_pack_zip = zipfile.ZipFile(
                pack_path / f"{dataset_name}_{pack_idx}.zip", "w", zipfile.ZIP_DEFLATED
            )
            current_pack_size = 0
            print(f"Pack {pack_idx} reached 5GB, starting a new pack.")
        current_pack_zip.write(file, arcname=file.relative_to(dataset_path))
        current_pack_size += file_size

    if current_pack_size > 0:
        current_pack_zip.close()
        print(f"Final pack {pack_idx}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python zip-dataset.py <dataset_name> <last_file_index>")
        sys.exit(1)
    dataset_name = sys.argv[1]
    last_file_index = int(sys.argv[2])
    zip_dataset(dataset_name, last_file_index)
