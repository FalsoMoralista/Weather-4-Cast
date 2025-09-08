import os
import shutil
from pathlib import Path


def backup_opera_files():
    suffix = ".rates.crop"
    base_path = "dataset/w4c24"
    backup_path = "dataset/w4c24_backup_opera"

    files = Path(base_path).rglob(
        f"*{suffix}*", case_sensitive=False, recurse_symlinks=True
    )
    print(f"Found {len(list(files))} files to back up.")
    for file in files:
        print(f"Processing file: {file}")
        relative_path = file.relative_to(base_path)
        backup_file_path = Path(backup_path) / relative_path

        if not os.path.exists(backup_file_path):
            print(f"Creating directory: {backup_file_path.parent}")
            os.makedirs(backup_file_path.parent, exist_ok=True)

        try:
            shutil.copy(file, backup_file_path)
            print(f"Copied {file} to {backup_file_path}")
        except Exception as e:
            print(f"Error copying {file} to {backup_file_path}: {e}")


if __name__ == "__main__":
    backup_opera_files()
    print("Backup completed.")
