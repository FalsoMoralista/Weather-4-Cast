# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter  # , grad_logger

from functools import partial

from src.datasets.SatDataset import make_sat_dataset

from src.helper import load_DC_checkpoint, init_model, init_vjepa_opt

from src.models.vision_transformer import VisionTransformer

# from src.transforms import make_transforms
import time

# --BROUGHT fRoM MAE
# from timm.data.mixup import Mixup
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from timm.utils import accuracy

from sklearn.metrics import accuracy_score


from src.models.model_wrapper import ModelWrapper


from torchvision import transforms
from PIL import Image
import random


import h5py

# --
log_timings = True
log_freq = 128
checkpoint_freq = 10
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args["meta"]["use_bfloat16"]
    model_name = args["meta"]["model_name"]
    load_model = args["meta"]["load_checkpoint"] or resume_preempt
    r_file = args["meta"]["read_checkpoint"]
    copy_data = args["meta"]["copy_data"]
    pred_depth = args["meta"]["pred_depth"]
    pred_emb_dim = args["meta"]["pred_emb_dim"]
    if not torch.cuda.is_available():
        print("Cuda not available")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args["data"]["use_gaussian_blur"]
    use_horizontal_flip = args["data"]["use_horizontal_flip"]
    use_color_distortion = args["data"]["use_color_distortion"]
    color_jitter = args["data"]["color_jitter_strength"]

    drop_path = args["data"]["drop_path"]
    mixup = args["data"]["mixup"]
    cutmix = args["data"]["cutmix"]
    reprob = args["data"]["reprob"]
    nb_classes = args["data"]["nb_classes"]

    # -- K-means
    K_range = args["k_means"]["K_range"]
    reinitialize_centroids = args["k_means"]["reinitialize_centroids"]

    # -- VICReg
    alpha = args["vicreg"]["alpha"]
    beta = args["vicreg"]["beta"]
    gamma = args["vicreg"]["gamma"]

    # -- # Gradient accumulation
    accum_iter = 1  # batch_size = accum_iter * batch_size

    # --
    batch_size = args["data"]["batch_size"]
    pin_mem = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]
    root_path = args["data"]["root_path"]
    image_folder = args["data"]["image_folder"]
    crop_size = args["data"]["crop_size"]
    crop_scale = args["data"]["crop_scale"]
    resume_epoch = args["data"]["resume_epoch"]
    cache_path = args["data"]["cache_path"]

    # -- MASK
    allow_overlap = args["mask"][
        "allow_overlap"
    ]  # whether to allow overlap b/w context and target blocks
    patch_size = args["mask"]["patch_size"]  # patch-size for model training
    num_enc_masks = args["mask"]["num_enc_masks"]  # number of context blocks
    min_keep = args["mask"]["min_keep"]  # min number of patches in context block
    enc_mask_scale = args["mask"]["enc_mask_scale"]  # scale of context blocks
    num_pred_masks = args["mask"]["num_pred_masks"]  # number of target blocks
    pred_mask_scale = args["mask"]["pred_mask_scale"]  # scale of target blocks
    aspect_ratio = args["mask"]["aspect_ratio"]  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args["optimization"]["ema"]
    ipe_scale = args["optimization"]["ipe_scale"]  # scheduler scale factor (def: 1.0)
    wd = float(args["optimization"]["weight_decay"])
    final_wd = float(args["optimization"]["final_weight_decay"])
    num_epochs = args["optimization"]["epochs"]
    warmup = args["optimization"]["warmup"]
    start_lr = args["optimization"]["start_lr"]
    lr = args["optimization"]["lr"]
    final_lr = args["optimization"]["final_lr"]
    smoothing = args["optimization"]["label_smoothing"]

    # -- LOGGING
    folder = args["logging"]["folder"]
    tag = args["logging"]["write_tag"]

    dump = os.path.join(folder, "params-ijepa.yaml")
    with open(dump, "w") as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    save_path = os.path.join(folder, f"{tag}" + "-ep{epoch}.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")

    load_path = None

    if load_model:
        load_path = "/home/lucianodourado/dinov3-weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth"

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "Train loss"),
        ("%.5f", "Test loss"),
        ("%.3f", "Test - Mean Average Error"),
        ("%d", "Test time (ms)"),
        ("%d", "time (ms)"),
    )

    stats_logger = CSVLogger(
        folder + "/experiment_log.csv",
        ("%d", "epoch"),
        ("%.5f", "vjepa lr"),
        ("%.5f", "train loss"),
        ("%.3f", "Test - Mean Average Error"),
        ("%d", "time (ms)"),
    )

    # -- init data-loaders/samplers
    train_dataset, supervised_loader_train, supervised_sampler_train = make_sat_dataset(
        transform=None,
        batch_size=batch_size,
        collator=None,
        pin_mem=True,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        copy_data=copy_data,
        drop_last=False,
    )

    val_dataset, supervised_loader_val, supervised_sampler_val = make_sat_dataset(
        transform=None,
        batch_size=batch_size,  # TODO: double it up
        collator=None,
        pin_mem=True,
        training=False,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        copy_data=copy_data,
        drop_last=False,
    )

    ipe = len(supervised_loader_train)
    print("Training dataset, length:", ipe * batch_size)

    vjepa = VisionTransformer(
        img_size=(224, 224),
        patch_size=16,
        mlp_ratio=4,
        num_frames=4,
        use_rope=True,
        embed_dim=4096,
        num_heads=32,
        depth=8,
        tubelet_size=1,
        ignore_patches=True,
    )
    vjepa.patch_embed = nn.Identity()

    vjepa = vjepa.to(device)
    vjepa = torch.compile(vjepa, mode="reduce-overhead")

    total_params = sum(p.numel() for p in vjepa.parameters() if p.requires_grad)
    print(f"V-jepa Total parameters: {total_params / 1.0e9} B")

    dinov3 = torch.hub.load(
        "../dinov3", "dinov3_vit7b16", source="local", weights=load_path
    ).to(device, dtype=torch.bfloat16)

    dinov3 = torch.compile(dinov3, mode="reduce-overhead")

    # for p in dinov3.parameters():
    #    p.requires_grad = False

    # print("Dinov3 Model:", dinov3)

    model = ModelWrapper(
        backbone=dinov3,
        vjepa=vjepa,
        patch_size=16,
        dim_in=4096,
        dim_out=2048,
        num_heads=16,
        num_layers=1,
        num_target_channels=16,
        vjepa_size_in=14,
        vjepa_size_out=18,
        last_linear_dimension=324,
        batch_size=batch_size,
        num_frames=4,
    ).to(device)

    model.vision_decoder = torch.compile(model.vision_decoder, mode="reduce-overhead")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Total parameters: {total_params / 1.0e9} B")

    allocated_bytes = torch.cuda.memory_allocated()
    allocated_gb = allocated_bytes / (1024**3)
    print("allocated mem from model setup:", allocated_gb)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_vjepa_opt(
        encoder=model,
        wd=wd,  # TODO
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe if accum_iter == 1 else ipe // accum_iter,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )

    model = DistributedDataParallel(model, static_graph=True)

    def save_checkpoint(epoch):
        save_dict = {
            "model": model.module.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f"{epoch + 1}"))
            elif epoch + 1 == 1:
                torch.save(save_dict, save_path.format(epoch=f"{epoch + 1}"))

    print("Batch Size:", batch_size)
    logger.info(model)

    # TODO: ADJUST THIS later!
    if resume_epoch != 0:
        target_encoder, optimizer, scaler, start_epoch = load_DC_checkpoint(
            device=device,
            r_path=load_path,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(resume_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    start_epoch = resume_epoch

    # -- TRAINING LOOP
    total_test_loss_meter = AverageMeter()
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        total_loss_meter = AverageMeter()
        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        supervised_sampler_train.set_epoch(epoch)

        for itr, (image, label) in enumerate(supervised_loader_train):

            def load_imgs():
                img = image.to(device, non_blocking=True, dtype=torch.float32)
                target = label.to(device, non_blocking=True, dtype=torch.float32)
                return (img, target)

            def train_step():
                x, y = load_imgs()

                print("X:", x.size())
                print("y:", y.size())

                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=use_bfloat16
                ):
                    vjepa_embeddings = model(x)

                allocated_bytes = torch.cuda.max_memory_allocated()
                allocated_gb = allocated_bytes / (1024**3)
                print("Max allocated mem from feature extract:", allocated_gb)

                loss = F.smooth_l1_loss(vjepa_embeddings, y)
                loss_val = loss.item()

                # Clear embedding tensors after loss computation
                # del (projector_embeddings, positive_embeddings)
                del vjepa_embeddings
                torch.cuda.empty_cache()

                loss_meter.update(loss_val)

                if accum_iter > 1:
                    loss = loss / accum_iter

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    update_grad = (itr + 1) % accum_iter == 0
                    if update_grad:
                        scaler.step(optimizer)
                        scaler.update()
                        _new_lr = scheduler.step()
                        _new_wd = wd_scheduler.step()
                        # momentum update of target encoder
                    else:
                        _new_lr = scheduler.get_last_lr()[0]
                        _new_wd = wd_scheduler.get_last_value()

                else:  # not used
                    loss.backward()
                    optimizer.step()

                # grad_stats = grad_logger(model.module.named_parameters())

                if (itr + 1) % accum_iter == 0:
                    optimizer.zero_grad()

                del loss
                torch.cuda.empty_cache()

                return (loss_val, _new_lr, _new_wd)

            (loss, _new_lr, _new_wd), etime = gpu_timer(train_step)

            total_loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %d/%d] - Train Loss: [%.4f] -"
                        "| [wd: %.2e] [lr: %.2e] | "
                        "[mem: %.2f] "
                        "(%.1f ms)"
                        % (
                            epoch + 1,
                            (itr // accum_iter),
                            (ipe // accum_iter),
                            total_loss_meter.avg,
                            _new_wd,
                            _new_lr,
                            (torch.cuda.max_memory_allocated() / 1024**3),
                            time_meter.avg,
                        )
                    )

                    # if grad_stats is not None:
                    #    logger.info(
                    #        "[%d, %d] grad_stats: [%.2e %.2e] (%.2e, %.2e)"
                    #        % (
                    #            epoch + 1,
                    #            itr,
                    #            grad_stats.first_layer,
                    #            grad_stats.last_layer,
                    #            grad_stats.min,
                    #            grad_stats.max,
                    #        )
                    #    )

            log_stats()

        testAcc1 = AverageMeter()
        testAcc5 = AverageMeter()
        test_loss = AverageMeter()

        # Warning: Enabling distributed evaluation with an eval dataset not divisible by process number
        # will slightly alter validation results as extra duplicate entries are added to achieve equal
        # num of samples per-process.

        @torch.no_grad()
        def evaluate():
            model.module.eval()
            # -- Enable shuffling to reduce monitor bias
            supervised_sampler_val.set_epoch(epoch)

            test_mae = AverageMeter()
            MAE = nn.L1Loss()

            for _, (samples, targets) in enumerate(supervised_loader_val):
                images = samples.to(device, non_blocking=True)
                labels = targets.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    with torch.inference_mode():
                        reconstructed_matrix = model(images)

                mae = MAE(reconstructed_matrix, labels)
                test_mae.update(mae)

            total_test_loss_meter.update(test_mae)

            logger.info(f"Average accuracy over evaluation dataset: {test_mae.avg:.3f}")
            logger.info(
                "Mean Average error across epochs: %.3f" % total_test_loss_meter.avg
            )

        vtime = gpu_timer(evaluate)

        model.module.train(True)
        model.module.backbone.eval()
        model.module.backbone.requires_grad_(False)

        params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        print(f"Model Total parameters: {params / 1.0e9} == {total_params / 1.0e9}? ")

        stats_logger.log(
            epoch + 1,
            lr,
            loss_meter.avg,
            total_test_loss_meter.avg,
            time_meter.avg,
        )
        # -- Save Checkpoint after every epoch
        logger.info("avg. train_loss %.3f" % total_loss_meter.avg)

        save_checkpoint(epoch + 1)
        assert not np.isnan(loss), "loss is nan"
        logger.info("Loss %.4f" % loss)


if __name__ == "__main__":
    main()
