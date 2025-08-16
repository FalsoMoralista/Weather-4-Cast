import os
from pathlib import Path
import tarfile

if not os.path.exists("packs"):
    os.makedirs("packs")


files = Path("./dataset/w4c24").glob("**/*")
total_size = 0
pack_idx = 0
tar_filename = f"packs/w4c24-{pack_idx}.tar"

tarf = tarfile.TarFile(tar_filename, "a")


gb_2 = 2 * 1024 * 1024 * 1024  # 2GB in bytes

for file in files:
    if file.is_dir():
        continue
    file_size = os.path.getsize(file)
    if file_size > gb_2:
        if total_size > 0:
            tarf.close()
            total_size = 0
        pack_idx += 1
        tar_filename = f"packs/w4c24-{pack_idx}.tar"
        tarf = tarfile.TarFile(tar_filename, "a")
        tarf.add(file, arcname=file.relative_to("./dataset/w4c24"), recursive=False)
        tarf.close()
        print(f"File {file} is larger than 2GB, packed separately.")
        pack_idx += 1
        tar_filename = f"packs/w4c24-{pack_idx}.tar"
        tarf = tarfile.TarFile(tar_filename, "a")
        continue
    if total_size + file_size > gb_2:
        if total_size > 0:
            total_size = 0
            tarf.close()
        print(f"Pack {pack_idx} reached 2GB, starting a new pack.")
        total_size = 0
        pack_idx += 1
        tar_filename = f"packs/w4c24-{pack_idx}.tar"
        tarf = tarfile.TarFile(tar_filename, "a")
    tarf.add(file, arcname=file.relative_to("./dataset/w4c24"), recursive=False)
    total_size += file_size
tarf.close()
