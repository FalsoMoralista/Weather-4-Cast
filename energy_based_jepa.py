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

import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from src.utils.distributed import init_distributed
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter

from src.datasets.SatDataset import make_sat_dataset
from utils.checkpoint import remove_prefix, remove_with_name

from src.helper import load_DC_checkpoint, init_vjepa_opt

from src.models.model_v2 import ModelWrapperV2
from src.models.vision_transformer import vit_large_rope

from src.transforms import RandomSuperResCrop, CenterSuperResCrop


def vjepa_train_transform(sample):
    crop = RandomSuperResCrop(32, 32, 6)
    x, _ = crop(sample)
    x = x.permute(1, 0, 2, 3)
    return (x, _)


def vjepa_val_transform(sample):
    crop = CenterSuperResCrop(32, 32, 6, 16)
    x, _ = crop(sample)
    x = x.permute(1, 0, 2, 3)
    return (x, _)


def make_val_transform():
    return vjepa_val_transform


def make_train_transform():
    return vjepa_train_transform


# --
log_timings = True
log_freq = 128
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
    accum_iter = 128  # batch_size = accum_iter * batch_size

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

    loss_fn = args["optimization"]["loss_function"]

    logger.info("Configured loss function: %s" % loss_fn)

    loss_fn_map = {
        "mse": F.mse_loss,
        "l1": F.l1_loss,
        "smooth_l1": F.smooth_l1_loss,
    }

    loss_function = loss_fn_map.get(loss_fn, loss_fn_map["smooth_l1"])
    logger.info("Using loss function: %s" % loss_function)

    # -- LOGGING
    folder = args["logging"]["folder"]
    tag = args["logging"]["write_tag"]
    checkpoint_freq = args["logging"]["checkpoint_freq"]

    dump = os.path.join(folder, "params-vjepa.yaml")
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

    train_transform = make_train_transform()
    val_transform = make_val_transform()

    # -- init data-loaders/samplers
    train_dataset, supervised_loader_train, supervised_sampler_train = make_sat_dataset(
        transform=train_transform,
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
        transform=val_transform,
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

    vjepa = vit_large_rope(
        patch_size=2,
        img_size=(32, 32),
        # mlp_ratio=4,
        num_frames=4,
        # use_rope=True,
        # embed_dim=1280,
        # num_heads=16,
        # depth=16,
        tubelet_size=1,
        # ignore_patches=True,
        use_activation_checkpointing=False,
        in_chans=11,
    )
    jepa_checkpoint_path = (
        "/home/lucianodourado/weather-4-cast/jepa_checkpoints/vjepa_vitl.pt"
    )
    vjepa_checkpoint = torch.load(jepa_checkpoint_path)
    encoder_checkpoint = remove_prefix(vjepa_checkpoint["encoder"], "module.backbone.")
    encoder_checkpoint = remove_with_name(encoder_checkpoint, "patch_embed")
    msg = vjepa.load_state_dict(encoder_checkpoint, strict=False)
    print("Loading checkpoint with message:", msg)
    # vjepa.patch_embed = PatchEmbed3D(
    #     patch_size=16,
    #     tubelet_size=1,
    #     in_chans=11,
    #     embed_dim=1280,
    # )
    # for name, p in vjepa.named_parameters():
    #     if "patch_embed" in name:
    #         continue
    #     p.requires_grad = False
    vjepa = vjepa.to(device)
    total_params = sum(p.numel() for p in vjepa.parameters() if p.requires_grad)
    print(f"V-jepa Total parameters: {total_params / 1.0e9} B")

    # dinov3 = torch.hub.load(
    #     "../dinov3", "dinov3_vitl16", source="local", weights=load_path
    # ).to(device)

    # for p in dinov3.parameters():
    #     p.requires_grad = False

    # dinov3 = torch.compile(dinov3, mode="reduce-overhead")

    # print("Dinov3 Model:", dinov3)

    # model = ModelWrapper(
    #     backbone=dinov3,
    #     vjepa=vjepa,
    #     patch_size=16,
    #     dim_out=1024,
    #     num_heads=16,
    #     num_decoder_layers=8,
    #     num_target_channels=16,
    #     vjepa_size_in=14,
    #     vjepa_size_out=18,
    #     num_frames=4,
    # ).to(device)

    model = ModelWrapperV2(
        vjepa=vjepa,
        patch_size=2,
        dim_out=1024,
        num_heads=1,
        num_decoder_layers=1,
        num_target_channels=16,
        vjepa_size_in=16,
        num_frames=4,
        image_size=32,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Total parameters: {total_params / 1.0e9} B")

    allocated_bytes = torch.cuda.memory_allocated()
    allocated_gb = allocated_bytes / (1024**3)
    print("allocated mem from model setup:", allocated_gb, "gb")

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
            elif epoch == 1:
                torch.save(save_dict, save_path.format(epoch=f"{epoch}"))

    print("Batch Size:", batch_size)
    logger.info(model)

    # TODO: ADJUST THIS later!
    if False:
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
                img[~torch.isfinite(img)] = 0
                target = label.to(device, non_blocking=True, dtype=torch.float32)
                # isFinite = torch.isfinite(img)
                # if not torch.all(isFinite):
                #    torch.where(isFinite, img, torch.tensor(0.0, dtype=torch.float32, device=device), out=img)
                return (img, target)

            def train_step():
                x, y = load_imgs()

                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=use_bfloat16,
                ):
                    vjepa_embeddings = model(x)
                    loss = loss_function(vjepa_embeddings, y)
                # loss = F.smooth_l1_loss(vjepa_embeddings, y)

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

                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=use_bfloat16,
                ):
                    with torch.inference_mode():
                        reconstructed_matrix = model(images)
                    mae = loss_function(reconstructed_matrix, labels)
                # mae = F.smooth_l1_loss(
                #     reconstructed_matrix, labels
                # )
                # MAE(reconstructed_matrix, labels)
                test_mae.update(mae)

            total_test_loss_meter.update(test_mae.avg)

            logger.info(f"Average accuracy over evaluation dataset: {test_mae.avg:.3f}")
            logger.info(
                "Mean Average error across epochs: %.3f" % total_test_loss_meter.avg
            )

        vtime = gpu_timer(evaluate)

        model.train(True)
        # model.backbone.eval()
        # model.backbone.requires_grad_(False)

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
