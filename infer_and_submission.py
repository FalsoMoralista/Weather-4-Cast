import os
import pandas as pd
from argparse import ArgumentParser

import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

from src.models.vision_transformer import VisionTransformer, vit_small
from src.models.model_wrapper import ModelWrapper
from src.models.model_v2 import ModelWrapperV2
from src.transforms import CenterSuperResCrop, DeterministicCrop


from src.datasets.InferenceDatasetv2 import InferenceDatasetV2, worker_init_fn

deterministic_crop = DeterministicCrop(
    input_patch_size=32,
    output_patch_size=32,
)


if not torch.cuda.is_available():
    print("Cuda not available")
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


def dinepa_infer_transform(sample):
    return ()


def parse_args():
    parser = ArgumentParser(description="Inference and Submission Script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Epoch number of the model to load.",
    )
    return parser.parse_args()


def load_dinepa(epoch):
    dino_path = "/home/lucianodourado/dinov3-weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
    dinov3 = torch.hub.load(
        "../dinov3", "dinov3_vitl16", source="local", weights=dino_path
    ).to(device)
    dinov3 = torch.compile(dinov3, mode="reduce-overhead")

    tag = "constrained_dinepa"

    vjepa = VisionTransformer(
        img_size=(224, 224),
        patch_size=16,
        mlp_ratio=4,
        num_frames=4,
        use_rope=True,
        embed_dim=1024,
        num_heads=16,
        depth=12,
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
        num_decoder_layers=6,
        num_target_channels=16,
        vjepa_size_in=14,
        vjepa_size_out=18,
        num_frames=4,
    )

    r_path = "dinepa_v2/{}-ep{}.pth.tar".format(tag, epoch)
    print("Loading checkpoint from:", r_path)
    checkpoint = torch.load(r_path, map_location=torch.device("cpu"))
    print("checkpoint keys:", checkpoint["model"].keys())
    for idx, key in enumerate(["downsample", "vjepa"]):
        state_dict = {
            k.replace(key + ".", ""): v
            for k, v in checkpoint["model"].items()
            if k.startswith(key)
        }
        if idx == 0:
            msg = model.downsample.load_state_dict(state_dict)
        else:
            if tag == "constrained_dinepa":
                msg = model.vjepa.load_state_dict(state_dict)
        print(f"Loaded layer {key} with msg: {msg}")

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
                        "conv_bins",
                        "squeeze_1",
                        "squeeze_2",
                    ]
                ):
                    state_dict[new_key] = v

        msg = model.vit_decoder.load_state_dict(state_dict)
        print(f"Loaded layer {key} with msg: {msg}")

    model.backbone = dinov3
    print("Model:", model)
    model.to(device)
    model.eval()
    del checkpoint
    return model


def load_vanilla_crps(epoch):
    vjepa = vit_small(
        img_size=(32, 32),
        in_chans=11,
        patch_size=2,
        num_frames=4,
        tubelet_size=1,
        use_activation_checkpointing=False,
    )
    model = ModelWrapperV2(
        vjepa=vjepa,
        patch_size=2,
        dim_out=384,
        num_heads=32,
        num_decoder_layers=8,
        num_target_channels=16,
        vjepa_size_in=16,
        num_frames=4,
        image_size=32,
        n_bins=513,
    )

    tag = "vjepa_2"
    r_path = "vanilla_vjepa_crps/{}-ep{}.pth.tar".format(tag, epoch)
    print("Loading checkpoint from:", r_path)
    checkpoint = torch.load(r_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])

    return model


def load_vanilla_emd(epoch):
    vjepa = vit_small(
        img_size=(32, 32),
        in_chans=11,
        patch_size=2,
        num_frames=4,
        tubelet_size=1,
        use_activation_checkpointing=False,
    )
    model = ModelWrapperV2(
        vjepa=vjepa,
        patch_size=2,
        dim_out=384,
        num_heads=32,
        num_decoder_layers=8,
        num_target_channels=16,
        vjepa_size_in=16,
        num_frames=4,
        image_size=32,
        n_bins=513,
    )

    tag = "vjepa_2"
    r_path = "vanilla_vjepa_emd/{}-ep{}.pth.tar".format(tag, epoch)
    print("Loading checkpoint from:", r_path)
    checkpoint = torch.load(r_path, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model"])

    return model


args = parse_args()


model_map = {
    "dinepa_v2": load_dinepa,
    "vanilla_vjepa_crps": load_vanilla_crps,
    "vanilla_vjepa_emd": load_vanilla_emd,
}

model = model_map[args.model](args.epoch).to(device)

BASE_DIR = "/home/lucianodourado/weather-4-cast/"

task = "cum1"
years = ["19", "20"]
filenames = ["roxi_0008", "roxi_0009", "roxi_0010"]

for year in years:
    predictions = {}
    for name in filenames:
        type = name + "." + task + "test" + year

        prediction_key = f"{name}.{task}test{year}"
        dictionary_dir = BASE_DIR + "/submissions"

        dict_path = os.path.join(dictionary_dir, f"{name}.{task}test_dictionary.csv")
        meta_df = pd.read_csv(dict_path)
        year_full = int(f"20{year}")
        year_specific_df = meta_df[meta_df["year"] == year_full].reset_index(drop=True)

        dataset = InferenceDatasetV2(
            InferenceDatasetV2.ROOT,
            type=type,
            input_size=(252, 252) if model_map == "dinepa_v2" else None,
        )

        print("Dataset length:", len(dataset))

        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=1,
            rank=0,
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

        for idx, x in enumerate(loader):
            images = x.to(device, non_blocking=True, dtype=torch.float32)
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
            print("Images shape after cropping and resizing", images.size())

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                with torch.inference_mode():
                    model_prediction = model(images)
                    print(
                        "Model prediction:",
                        model_prediction.size(),
                        model_prediction,
                        flush=True,
                    )
            submission_results = []
            for i, row in year_specific_df.iterrows():
                case_id = row["Case-id"]
                slot_start = row["slot-start"]

                slot_result = model_prediction[i]
                print("Slot result shape:", slot_result.size(), flush=True)
                slot_result = F.softmax(slot_result, dim=0)
                print("Softmax applied to slot result.", flush=True)
                ecdf_per_timestep = torch.cumsum(slot_result, dim=-1)
                print("ECDF per timestep shape:", ecdf_per_timestep.size(), flush=True)

                for bin_index in range(ecdf_per_timestep.size(0)):
                    submission_results.append(
                        [
                            case_id,
                            bin_index * 0.25,
                            ecdf_per_timestep[bin_index].item(),
                        ]
                    )

            print("Total submission results:", len(submission_results), flush=True)
            output_df = pd.DataFrame(submission_results, columns=None)
            output_dir = BASE_DIR + "submission_files"
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(
                output_dir, f"{args.model}/20{year}/{name}.test.cum4h.csv"
            )
            output_df.to_csv(
                output_filename, index=False, header=False, float_format="%.20f"
            )
