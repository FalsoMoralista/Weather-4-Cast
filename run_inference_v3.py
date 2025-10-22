import torch
import torch.nn as nn
import logging
import numpy as np
import sys
import os
import pandas as pd
from torch.nn import functional as F

from src.models.vision_transformer import VisionTransformer
from src.models.model_wrapper import ModelWrapper
from src.datasets.InferenceDatasetv2 import InferenceDatasetV2, worker_init_fn

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


if not torch.cuda.is_available():
    print("Cuda not available")
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

# Load checkpoint
dino_path = (
    "/home/lucianodourado/dinov3-weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
)

dinov3 = torch.hub.load(
    "../dinov3", "dinov3_vitl16", source="local", weights=dino_path
).to(device)
dinov3 = torch.compile(dinov3, mode="reduce-overhead")

for epoch in range(5, 6, 1):
    vjepa = VisionTransformer(
        img_size=(224, 224),
        patch_size=16,
        mlp_ratio=4,
        num_frames=4,
        use_rope=True,
        embed_dim=1024,
        num_heads=16,
        depth=16,
        tubelet_size=1,
        ignore_patches=True,
        use_activation_checkpointing=False,
    )
    vjepa.patch_embed = nn.Identity()

    model = ModelWrapper(
        backbone=dinov3,
        vjepa=vjepa,
        patch_size=16,
        dim_out=384,
        num_heads=16,
        num_decoder_layers=8,
        num_target_channels=16,
        vjepa_size_in=14,
        vjepa_size_out=18,
        num_frames=4,
    )

    # TODO: move below into helper.py
    try:
        BASE_DIR = "/home/lucianodourado/weather-4-cast/"
        tag = "constrained_dinepa"
        r_path = "dinepa_v2/{}-ep{}.pth.tar".format(tag, epoch)
        checkpoint = torch.load(r_path, map_location=torch.device("cpu"))

        for idx, key in enumerate(["downsample", "vjepa"]):
            state_dict = {
                k.replace(key + ".", ""): v
                for k, v in checkpoint["model"].items()
                if k.startswith(key)
            }
            if idx == 0:
                msg = model.downsample.load_state_dict(state_dict)
            else:
                msg = model.vjepa.load_state_dict(state_dict)
            logger.info(f"loaded layer {key} with msg: {msg}")

        key = "vit_decoder"
        state_dict = {}
        for k, v in checkpoint["model"].items():
            if k.startswith(key):
                new_key = k.replace("vit_decoder.", "", 1)
                if any(
                    name in new_key
                    for name in [
                        "patch_embed",
                        "blocks",
                        "norm",
                        "time_expansion",
                        "conv_regression",
                    ]
                ):
                    state_dict[new_key] = v

        msg = model.vit_decoder.load_state_dict(state_dict)
        logger.info(f"loaded layer {key} with msg: {msg}")

        model.backbone = dinov3
        model.to(device)
        model.eval()
        del checkpoint
    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    task = "cum1"  # TODO colocar em .yaml
    years = ["19", "20"]
    filenames = ["roxi_0008", "roxi_0009", "roxi_0010"]

    for year in years:
        predictions = {}
        for name in filenames:
            print("Year:", year)
            type = name + "." + task + "test" + year

            prediction_key = f"{name}.{task}test{year}"
            dictionary_dir = BASE_DIR + "/submissions"

            dict_path = os.path.join(
                dictionary_dir, f"{name}.{task}test_dictionary.csv"
            )
            meta_df = pd.read_csv(dict_path)
            year_full = int(f"20{year}")
            year_specific_df = meta_df[meta_df["year"] == year_full].reset_index(
                drop=True
            )

            dataset = InferenceDatasetV2(
                InferenceDatasetV2.ROOT, type=type, input_size=(252, 252)
            )

            print("Dataset length:", len(dataset))

            dist_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=dataset, num_replicas=1, rank=0
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                sampler=dist_sampler,
                batch_size=len(year_specific_df),
                drop_last=False,
                pin_memory=True,
                num_workers=8,
                persistent_workers=False,
                worker_init_fn=worker_init_fn,
            )
            print("Dataloader length:", len(loader))

            @torch.no_grad()
            def evaluate():
                for idx, x in enumerate(loader):
                    print("Iteration:", idx, "(should run for a single iteration only)")

                    images = x.to(device, non_blocking=True, dtype=torch.float32)
                    print("Images shape", images.size())

                    images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
                    images = torch.clamp_min(images, 0)

                    resized_images = []
                    for i, row in year_specific_df.iterrows():
                        case_id = row["Case-id"]

                        x_top_left, x_bottom_right = (
                            row["x-top-left"] // 6,
                            row["x-bottom-right"] // 6,
                        )
                        y_top_left, y_bottom_right = (
                            row["y-top-left"] // 6,
                            row["y-bottom-right"] // 6,
                        )

                        slot_start, slot_end = row["slot-start"], row["slot-end"]
                        placeholder = images[slot_start // 4][
                            :, :, y_top_left:y_bottom_right, x_top_left:x_bottom_right
                        ]

                        resized_img = F.interpolate(
                            placeholder,
                            size=(224, 224),
                            mode="bicubic",
                        )
                        resized_images.append(resized_img)
                    images = torch.stack(resized_images)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        with torch.inference_mode():
                            reconstructed_matrix = model(images).squeeze(2)
                            print(
                                "Reconstructed matrix:",
                                reconstructed_matrix.size(),
                                flush=True,
                            )

                # predictions[type] = torch.cat(predictions[type], dim=0)

                submission_results = []
                for i, row in year_specific_df.iterrows():
                    case_id = row["Case-id"]
                    slot_start, _ = row["slot-start"], row["slot-end"]
                    print("Sum first", reconstructed_matrix[i][0:4].mean(), flush=True)
                    print("Sum second", reconstructed_matrix[i][4:8].mean(), flush=True)
                    print("Sum third", reconstructed_matrix[i][8:12].mean(), flush=True)
                    print(
                        "Sum fourth", reconstructed_matrix[i][12:16].mean(), flush=True
                    )
                    print(
                        "------------------------------------------------------------",
                        flush=True,
                    )
                    prediction_value = reconstructed_matrix[i].mean() * 4
                    prediction_value = max(0, prediction_value.item())
                    submission_results.append([case_id, prediction_value, 1])

                output_df = pd.DataFrame(submission_results, columns=None)
                output_dir = BASE_DIR + "submission_files"
                output_filename = os.path.join(
                    output_dir, f"{tag}/20{year}/{name}.test.cum4h.csv"
                )
                output_df.to_csv(output_filename, index=False, header=False)
                print(f"Successfully generated submission file: {output_filename}")

            evaluate()
        torch.save(
            predictions, "predictions/predictions_{}.pth".format(task + "test" + year)
        )
        print("Predictions:", predictions)
        print("Predictions length:", len(predictions))
