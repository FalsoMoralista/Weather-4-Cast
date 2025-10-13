import torch
import logging
import numpy as np
import sys

from src.models.model_v2 import ModelWrapperV2
from src.datasets.InferenceDataset import InferenceDataset, worker_init_fn

from torchvision import transforms
from src.models.vision_transformer import vit_large_rope


def permute(x):
    return x.permute(1, 0, 2, 3)


def composed():
    return transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def transform_inner(x):
    # t = composed()
    # x = t(x)
    return permute(x)


def make_transforms():
    return transform_inner


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
tag = "vjepa2"
epoch_to_load = 5
model_path = "./logs/" + f"{tag}" + f"-ep{epoch_to_load}.pth.tar"

checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
logger.info(f"Loaded checkpoint from {model_path} at epoch {epoch_to_load}")

model_checkpoint = checkpoint["model"]

vjepa = vit_large_rope(
    patch_size=16,
    img_size=(224, 224),
    num_frames=4,
    tubelet_size=1,
    use_activation_checkpointing=False,
    in_chans=11,
)

model = ModelWrapperV2(
    vjepa=vjepa,
    patch_size=16,
    dim_out=1024,
    num_heads=16,
    num_decoder_layers=8,
    num_target_channels=16,
    vjepa_size_in=14,
    vjepa_size_out=18,
    num_frames=4,
)

msg = model.load_state_dict(model_checkpoint)
print("Loading model state dict", msg)
model = model.to(device)
model.eval()

task = "cum1"  # TODO colocar em .yaml
years = ["19", "20"]
filenames = ["roxi_0008", "roxi_0009", "roxi_0010"]

transform = make_transforms()

for year in years:
    predictions = {}
    for name in filenames:
        print("Year:", year)
        type = name + "." + task + "test" + year
        dataset = InferenceDataset(InferenceDataset.ROOT, type=type, transform=transform)

        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=1,
            rank=0,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=dist_sampler,
            batch_size=64,
            drop_last=False,
            pin_memory=True,
            num_workers=8,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        @torch.no_grad()
        def evaluate():
            for idx, x in enumerate(loader):
                images = x.to(device, non_blocking=True, dtype=torch.float32)

                images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
                images = torch.clamp_min(images, 0)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    with torch.inference_mode():
                        reconstructed_matrix = model(images).squeeze(2)
                        if type not in predictions:
                            predictions[type] = []
                        predictions[type].append(reconstructed_matrix)
            predictions[type] = torch.cat(predictions[type], dim=0)
            print(f"Predictions shape {predictions[type].size()}")

        evaluate()
    torch.save(
        predictions, "predictions/predictions_{}.pth".format("vjepa2_" + task + "test" + year)
    )
    print("Predictions:", predictions)
    print("Predictions length:", len(predictions))
