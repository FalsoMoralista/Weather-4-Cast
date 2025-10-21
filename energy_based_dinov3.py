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
    # TODO: testing op below
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
except Exception:
    pass


import copy
import logging
import sys
import yaml

import numpy as np

import torch
# torch.autograd.set_detect_anomaly(True)

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

from src.helper import load_DC_checkpoint, init_model, init_vjepa_opt, reload_checkpoint

from src.models.vision_transformer import VisionTransformer

# from src.transforms import make_transforms
import time

# --BROUGHT fRoM MAE
# from timm.data.mixup import Mixup
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from timm.utils import accuracy


from src.models.model_wrapper import ModelWrapper


from torchvision import transforms
from PIL import Image
import random

from src.transforms import RandomSuperResCrop, CenterSuperResCrop

# --
log_timings = True
log_freq = 64
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def dino_train_transform(sample):
    resize = transforms.Resize((224, 224))
    crop = RandomSuperResCrop(
        input_patch_size=32,
        output_patch_size=32,
        scale_factor=6,
        rain_sampling_p=0.75,
        rain_sampling_threshold=0.2,
    )
    x, _ = crop(sample)
    x = resize(x)
    return (x, _)


def dino_val_transform(sample):
    resize = transforms.Resize((224, 224))
    crop = CenterSuperResCrop(32, 32, 6, 16)
    x, _ = crop(sample)
    x = resize(x)
    return (x, _)


def make_val_transform():
    return dino_val_transform


def make_train_transform():
    return dino_train_transform


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

    # -- # Gradient accumulation
    accum_iter = 64  # batch_size = accum_iter * batch_size

    # --
    batch_size = args["data"]["batch_size"]
    pin_mem = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]
    root_path = args["data"]["root_path"]
    image_folder = args["data"]["image_folder"]
    resume_epoch = args["data"]["resume_epoch"]

    # --

    # -- OPTIMIZATION
    ipe_scale = args["optimization"]["ipe_scale"]  # scheduler scale factor (def: 1.0)
    wd = float(args["optimization"]["weight_decay"])
    final_wd = float(args["optimization"]["final_weight_decay"])
    num_epochs = args["optimization"]["epochs"]
    warmup = args["optimization"]["warmup"]
    start_lr = args["optimization"]["start_lr"]
    lr = args["optimization"]["lr"]
    final_lr = args["optimization"]["final_lr"]

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
        load_path = "/home/lucianodourado/dinov3-weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

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

    train_transformer = make_train_transform()
    val_transformer = make_val_transform()

    # -- init data-loaders/samplers
    train_dataset, supervised_loader_train, supervised_sampler_train = make_sat_dataset(
        transform=train_transformer,
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
        transform=val_transformer,
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
        embed_dim=1024,
        num_heads=32,
        depth=12,
        tubelet_size=1,
        ignore_patches=True,
        use_activation_checkpointing=False,
    )
    vjepa.patch_embed = nn.Identity()

    vjepa = vjepa.to(device)

    total_params = sum(p.numel() for p in vjepa.parameters() if p.requires_grad)
    print(f"V-jepa Total parameters: {total_params / 1.0e9} B")

    dinov3 = torch.hub.load(
        "../dinov3", "dinov3_vitl16", source="local", weights=load_path
    ).to(device)

    for p in dinov3.parameters():
        p.requires_grad = False

    dinov3 = torch.compile(dinov3, mode="reduce-overhead")

    # print("Dinov3 Model:", dinov3)

    model = ModelWrapper(
        backbone=dinov3,
        vjepa=vjepa,
        patch_size=16,
        dim_out=384,
        num_heads=32,
        num_decoder_layers=8,
        num_target_channels=16,
        vjepa_size_in=14,
        vjepa_size_out=18,
        num_frames=4,
    ).to(device)

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

    # model = DistributedDataParallel(model, static_graph=True)

    def save_checkpoint(epoch):
        model_state_dict = {
            k: v
            for k, v in model.state_dict().items()
            if not k.startswith(
                "backbone"
            )  # Remove pre-trained backbone from checkpoint
        }

        save_dict = {
            "model": model_state_dict,
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
            if epoch % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f"{epoch}"))

    print("Batch Size:", batch_size)
    logger.info(model)

    # TODO: ADJUST THIS later!
    if resume_epoch != 0:
        model = reload_checkpoint(model, resume_epoch, device)
        num_steps_to_advance = resume_epoch * (ipe // accum_iter)
        for _ in range(num_steps_to_advance):
            new_lr = scheduler.step()
            new_wd = wd_scheduler.step()
        logger.info("Resuming LR %f" % (new_lr))
        logger.info("Resuming WD %f" % (new_wd))
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
                img[~torch.isfinite(img)] = 0
                target = label.to(device, non_blocking=True, dtype=torch.float32)
                # isFinite = torch.isfinite(img)
                # if not torch.all(isFinite):
                #    torch.where(isFinite, img, torch.tensor(0.0, dtype=torch.float32, device=device), out=img)
                return (img, target)

            def crps_discrete_from_probs(probs, y_true_mm, bins):
                """
                probs: [B, K]  (probabilidades por bin; soma=1)
                y_true_mm: [B] (alvo em mm, acumulado 4h)
                bins: [K]      (valores y_k crescentes, em mm)
                """
                # CDF prevista
                probs = probs / probs.sum(dim=-1, keepdim=True)
                F_pred = probs.cumsum(dim=-1)  # [B, K]

                # CDF-verdade (degrau em x): T_k = 1{ y_k >= x }
                T = (
                    y_true_mm.unsqueeze(1).ge(bins.unsqueeze(0))
                ).float()  # (bins.unsqueeze(0) <= y_true_mm.unsqueeze(1)).float()  # [B, K]

                # Weights Δ_k (larguras)
                delta = torch.diff(
                    bins, prepend=bins[:1]
                )  # useless as bins have uniform width

                crps = ((F_pred - T) ** 2 * delta.unsqueeze(0)).sum(dim=-1).mean()
                return crps

            def train_step():
                x, y = load_imgs()
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=use_bfloat16
                ):
                    vjepa_logits = model(x)

                probs = torch.softmax(vjepa_logits, dim=-1)
                m = y.mean(dim=(2, 3))  # [B,16] média espacial por slot (mm/h)
                y_true_mm = m.sum(dim=1) / 4.0  # [B]  acum. 4h em mm
                loss = crps_discrete_from_probs(
                    probs,
                    y_true_mm,
                    bins=torch.arange(0.0, 512.0 + 4, 4.0, device=device),
                )

                loss_val = loss.item()

                loss_meter.update(loss_val)

                if accum_iter > 1:
                    loss = loss / accum_iter

                # Backward pass
                if use_bfloat16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                update_grad = (itr + 1) % accum_iter == 0
                if update_grad:
                    if use_bfloat16:
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    _new_lr = scheduler.step()
                    _new_wd = wd_scheduler.step()
                else:
                    _new_lr = scheduler.get_last_lr()[0]
                    _new_wd = wd_scheduler.get_last_value()

                if update_grad:
                    optimizer.zero_grad()

                # grad_stats = grad_logger(model.module.named_parameters())

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
        # End of epoch

        # Warning: Enabling distributed evaluation with an eval dataset not divisible by process number
        # will slightly alter validation results as extra duplicate entries are added to achieve equal
        # num of samples per-process.

        @torch.no_grad()
        def evaluate():
            model.eval()
            # -- Enable shuffling to reduce monitor bias
            supervised_sampler_val.set_epoch(epoch)

            test_mae = AverageMeter()

            for _, (samples, targets) in enumerate(supervised_loader_val):
                images = samples.to(device, non_blocking=True, dtype=torch.float32)
                labels = targets.to(device, non_blocking=True, dtype=torch.float32)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    with torch.inference_mode():
                        model_logits = model(images)

                probs = torch.softmax(model_logits, dim=-1)
                m = labels.mean(dim=(2, 3))
                y_true_mm = m.sum(dim=1) / 4.0  # [B]  acum. 4h em mm
                test_loss = crps_discrete_from_probs(
                    probs,
                    y_true_mm,
                    bins=torch.arange(0.0, 512.0 + 4, 4.0, device=device),
                )

            total_test_loss_meter.update(test_loss)

            logger.info(f"Average accuracy over evaluation dataset: {test_loss:.3f}")
            logger.info(
                "Mean Average error across epochs: %.3f" % total_test_loss_meter.avg
            )

        vtime = gpu_timer(evaluate)

        model.train(True)
        model.backbone.eval()
        model.backbone.requires_grad_(False)

        if epoch + 1 == 1:
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"Model Total parameters: {params / 1.0e9} == {total_params / 1.0e9}? "
            )

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
