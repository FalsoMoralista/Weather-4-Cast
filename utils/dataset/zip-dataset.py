import os
import sys
from pathlib import Path
import zipfile

def log_process(current_file_index, current_pack_index):
    log_file = Path("./process.log")
    with log_file.open("w") as f:
        f.write(f"{current_file_index} {current_pack_index}")
    print(f"Process logged to {log_file}")


def read_process():
    log_file = Path("./process.log")
    if not log_file.exists():
        return 0, -1
    with log_file.open("r") as f:
        content = f.read().strip()
        if content:
            current_file_index, current_pack_index = map(int, content.split())
            return current_file_index, current_pack_index
    return 0, -1

def zip_dataset(dataset_name: str, zips_to_generate: int = -1):
    dataset_path = Path(f"./dataset/{dataset_name}")
    if not dataset_path.exists():
        print(f"Dataset path {dataset_path} does not exist.")
        return

    pack_path = Path("./dataset/packs")
    if not pack_path.exists():
        pack_path.mkdir(parents=True)

    files = dataset_path.glob("**/*")
    files = sorted(files, key=lambda x: x.stat().st_size)

    MAX_SIZE = 5 * 1024 * 1024 * 1024

    current_pack_size = 0
    current_pack_zip = None

    generated_zips = 0
    current_file_index, pack_idx = read_process()

    for idx, file in enumerate(files):
        if file.is_dir():
            print(f"Skipping dir {file}")
            continue
        if idx < current_file_index:
            print(
                f"Skipping file {file} at index {idx} (current_file_index={current_file_index})"
            )
            continue
        current_file_index += 1
        file_size = os.path.getsize(file)
        if file_size > MAX_SIZE:
            pack_idx += 1
            large_file_zip = zipfile.ZipFile(
                pack_path / f"{dataset_name}-{pack_idx}.zip", "w", zipfile.ZIP_DEFLATED
            )
            large_file_zip.write(file, arcname=file.relative_to(dataset_path))
            large_file_zip.close()
            print(
                f"File {file} is larger than 5GB, packed separately. Pack index: {pack_idx}"
            )
            generated_zips += 1
            if generated_zips >= zips_to_generate:
                if not current_pack_zip:
                    print(f"Generated {generated_zips} zips, stopping as requested.")
                    log_process(current_file_index, pack_idx)
                    return
                else:
                    current_pack_zip.close()
                    print(f"Generated {generated_zips} zips, stopping as requested.")
                    log_process(current_file_index, pack_idx)
                    return
            continue
        if current_pack_size + file_size > MAX_SIZE:
            current_pack_zip.close()
            generated_zips += 1
            if zips_to_generate != -1 and generated_zips >= zips_to_generate:
                print(f"Generated {generated_zips} zips, stopping as requested.")
                log_process(current_file_index, pack_idx)
                return
            pack_idx += 1
            current_pack_zip = zipfile.ZipFile(
                pack_path / f"{dataset_name}-{pack_idx}.zip", "w", zipfile.ZIP_DEFLATED
            )
            current_pack_size = 0
            print(f"Pack {pack_idx} reached 5GB, starting a new pack.")
        if current_pack_zip is None:
            pack_idx += 1
            current_pack_zip = zipfile.ZipFile(
                pack_path / f"{dataset_name}-{pack_idx}.zip", "w", zipfile.ZIP_DEFLATED
            )
        current_pack_zip.write(file, arcname=file.relative_to(dataset_path))
        current_pack_size += file_size

    if current_pack_zip and current_pack_size > 0:
        current_pack_zip.close()
        log_process(current_file_index, pack_idx)
        print(f"Final pack {pack_idx}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python zip-dataset.py <dataset_name> <zips_to_generate>")
        sys.exit(1)
    dataset_name = sys.argv[1]
    zips_to_generate = int(sys.argv[2])
    zip_dataset(dataset_name, zips_to_generate)
