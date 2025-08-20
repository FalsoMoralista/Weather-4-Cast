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
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter

from functools import partial

#from src.datasets.FineTuningDataset import make_GenericDataset
#from src.datasets.triplet_dataset import make_TripletDataset

#from src.datasets.paired_batch_dataset import make_paired_batch_dataset

from sklearn.neighbors import NearestNeighbors

from src.utils.schedulers import WarmupCosineSchedule

from src.helper import (
    load_checkpoint,
    load_DC_checkpoint,
    init_model,
    init_optimizer,
    init_opt,
    init_DC_opt,
    build_cache_v2
)

from src.models.joint_model import JointFTModel
from src.models.vision_transformer import CustomVisionTransformer

#from src.transforms import make_transforms
import time

# --BROUGHT fRoM MAE
#from timm.data.mixup import Mixup
#from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
#from timm.utils import accuracy

from sklearn.metrics import accuracy_score


from src.models.model_wrapper import ModelWrapper


import pickle

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
        load_path = (
            "/home/lucianodourado/dinov3-weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
        )
    
    model = torch.hub.load(
        '../dinov3',
        'dinov3_vit7b16',
        source='local',
        weights=load_path
    ).to(device,  dtype=torch.bfloat16)
    
    model.eval()

    print('Dinov3 Model:', model)
    allocated_bytes = torch.cuda.memory_allocated()
    model = ModelWrapper(model).to(device)

    # Convert bytes to gigabytes
    allocated_gb = allocated_bytes / (1024**3)
    print('allocated mem from model loading:', allocated_gb)

    f = h5py.File("dataset/w4c24/2020/HRIT/boxi_0015.train.reflbt0.ns.h5", "r")
    data = f["REFL-BT"]
    print(data.shape)

    tensor = torch.from_numpy(f["REFL-BT"][:4]).to(device, dtype=torch.bfloat16)
    tensor2 = torch.from_numpy(f["REFL-BT"][4:8]).to(device, dtype=torch.bfloat16)
    tensor = torch.stack((tensor,tensor2), dim=0)
    B, T, C, H, W = tensor.shape  # (2, 4, 11, 252, 252)
    
    tensor = tensor.view(B * T, C, H, W) # [8, 11, 252, 252]
    print("tensor size:", tensor.size())
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        features = model(tensor)
        features = features.view(B, T, -1, features.size(-1))
        print('feature shape', features.size())
        allocated_gb = allocated_bytes / (1024**3)
        print('allocated mem from feature extraction:', allocated_gb)
        print('max mem allocated:', (torch.cuda.max_memory_allocated() / 1024**3))

    print('Destroying process')
    dist.destroy_process_group()
    return 0
    


    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "Train loss"),
        ("%.5f", "Test loss"),
        ("%.3f", "Test - Acc@1"),
        ("%.3f", "Test - Acc@5"),
        ("%d", "Test time (ms)"),
        ("%d", "time (ms)"),
    )

    stats_logger = CSVLogger(
        folder + "/experiment_log.csv",
        ("%d", "epoch"),
        ("%.5f", "backbone lr"),
        ("%.5f", "autoencoder lr"),
        ("%.5f", "total train loss"),
        ("%.5f", "orignal label train loss"),
        ("%.5f", "original label test loss"),
        ("%.5f", "pseudo-label loss"),
        ("%.5f", "Reconstruction loss"),
        ("%.5f", "K-Means loss"),
        ("%.5f", "Consistency loss"),
        ("%.5f", "VICReg loss"),
        ("%.3f", "Test - Acc@1"),
        ("%.3f", "Test - Acc@5"),
        ("%f", "avg_empty_clusters_per_class"),
        ("%d", "time (ms)"),
    )

    if args["dinov2"] and args["dinov2_meta"]["model_name"] == "vit_large":
        proj_embed_dim = 1024
    elif args["dinov2"] and args["dinov2_meta"]["model_name"] == "vit_giant":
        proj_embed_dim = 1536
    else:
        proj_embed_dim = 1280

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    target_encoder = copy.deepcopy(encoder)

    training_transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        supervised=True,
        validation=False,
        color_jitter=color_jitter,
    )

    val_transform = make_transforms( 
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        supervised=True,
        validation=True,
        color_jitter=color_jitter)


    # -- init data-loaders/samplers
    _, supervised_loader_train, _ = make_GenericDataset(
        feature_extraction=True,
        transform=training_transform,
        batch_size=batch_size,
        collator=None,
        pin_mem=False,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        copy_data=copy_data,
        drop_last=False,
    )

    _, supervised_loader_val, supervised_sampler_val = make_GenericDataset(
            transform=val_transform,
            batch_size=128,
            collator= None,
            pin_mem=False,
            training=False,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=False)

    dataset, dataloader, _ = make_paired_batch_dataset(
        ssl_transform=None,
        batch_size=batch_size,
        collator=None,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        copy_data=copy_data,
        drop_last=False,
    )


    ipe = len(dataloader)
    print("Training dataset, length:", ipe * batch_size)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
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
    encoder = DistributedDataParallel(encoder, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)

    # -- Load weights
    if resume_epoch == 0:
        if args["dinov2"] and args["dinov2_meta"]["model_name"] == "vit_large":
            target_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            target_encoder.to(device)
            cache_path = cache_path + "/dinov2_vit_large"
            logger.info(f"Dino target encoder: {target_encoder}")
        elif args["dinov2"] and args["dinov2_meta"]["model_name"] == "vit_giant":
            target_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
            target_encoder.to(device)
            cache_path = cache_path + "/dinov2_vit_giant"
            logger.info(f"Dino target encoder: {target_encoder}")
        else:
            cache_path = cache_path + "/ijepa_vit_huge"
            encoder, predictor, target_encoder, optimizer, scaler, start_epoch = (
                load_checkpoint(
                    device=device,
                    r_path=load_path,
                    encoder=encoder,
                    predictor=predictor,
                    target_encoder=target_encoder,
                    opt=optimizer,
                    scaler=scaler,
                )
            )
    del predictor

    def save_checkpoint(epoch):
        save_dict = {
            "model": model.module.state_dict(),
            "opt_A": optimizer_A.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "model_B": encoder.module.state_dict(),
            "opt_B": optimizer_B.state_dict(),
            "epoch": epoch,
            "loss": total_loss_meter.avg,
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


    narrow_vit = CustomVisionTransformer(
        embed_dim=proj_embed_dim,
        depth=8,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        qkv_bias=True,
    )

    print("Batch Size:", batch_size)
    model = JointFTModel(
        backbone=target_encoder.module, projector=narrow_vit, embed_dim=proj_embed_dim
    ).to(device)
    logger.info(model)

    # -- Override previously loaded optimization configs.
    # TODO: adjust regularization parameters e.g., weight decay
    # TODO: reset optimizer settings after projector pre-training?
    optimizer_A, scaler, scheduler_A, wd_scheduler_A = init_optimizer(
        projector=model.projector,
        wd=wd,
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

    optimizer_B, _, scheduler_B, wd_scheduler_B = init_optimizer(
        encoder=encoder,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=1e-3,
        final_lr=final_lr,
        iterations_per_epoch=ipe if accum_iter == 1 else ipe // accum_iter,
        warmup=2*warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )

    model = DistributedDataParallel(
        model, static_graph=True, find_unused_parameters=False
    )
    logger.info(encoder)


    # -- momentum schedule
    total_updates = (ipe // accum_iter) * num_epochs * ipe_scale
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / total_updates
        for i in range(int(total_updates) + 1)
    )

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
    for epoch in range(start_epoch, num_epochs):

        print('verifying treinability')
        for name, param in model.module.backbone.named_parameters():
            if param.requires_grad:  # Only print if True
                print(f"WARNING: {name} requires_grad=True")
        print('pass')

        MODEL_LOSS = []

        logger.info("Epoch %d" % (epoch + 1))

        total_loss_meter = AverageMeter()
        loss_meter = AverageMeter()
        time_meter = AverageMeter()
        
        
        for itr, ((image1, label1), (image2, label2)) in enumerate(dataloader):
            
            assert (label1 == label2).all(), " anchor labels different than positive labels"
            
            def load_imgs():
                a = image1.to(device, non_blocking=True)
                p = image2.to(device, non_blocking=True)
                return (a, p)

            def train_step():

                anchor, positive  = load_imgs()   
                
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):

                    projector_embeddings = model(anchor)
                    positive_embeddings = encoder(positive)

                    loss = F.smooth_l1_loss(projector_embeddings, positive_embeddings) 
                
                loss_val = loss.item()

                # Clear embedding tensors after loss computation
                del (projector_embeddings, positive_embeddings)
                torch.cuda.empty_cache()

                loss_meter.update(loss_val)

                if accum_iter > 1:
                    loss = loss / accum_iter

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    update_grad = (itr + 1) % accum_iter == 0
                    if update_grad:
                        scaler.step(optimizer_A)
                        scaler.step(optimizer_B)
                        scaler.update()
                        _new_lr_A = scheduler_A.step()
                        _new_wd_A = wd_scheduler_A.step()
                        _new_lr_B = scheduler_B.step()
                        _new_wd_B = wd_scheduler_B.step()
                        # momentum update of target encoder
                        with torch.no_grad():
                            m = next(momentum_scheduler)
                            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)                        
                    else:
                        _new_lr_A = scheduler_A.get_last_lr()[0]
                        _new_lr_B = scheduler_B.get_last_lr()[0]
                        _new_wd_A = wd_scheduler_A.get_last_value()
                        _new_wd_B = wd_scheduler_B.get_last_value()
                else:  # not used
                    loss.backward()
                    optimizer.step()

                grad_stats = grad_logger(model.module.named_parameters())
                
                if (itr + 1) % accum_iter == 0:
                    optimizer_A.zero_grad()
                    optimizer_B.zero_grad()

                del loss
                torch.cuda.empty_cache()

                return (loss_val, _new_lr_A,_new_wd_A,_new_lr_B,_new_wd_B, grad_stats)

            (loss, _new_lr_A, _new_wd_A, _new_lr_B, _new_wd_B, grad_stats), etime = (gpu_timer(train_step))

            total_loss_meter.update(loss)
            time_meter.update(etime)

            MODEL_LOSS.append(loss_meter.avg)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %d/%d] - Train Loss: [%.4f] -"
                        "| Model A: [wd: %.2e] [lr: %.2e] | "
                        "| Model B: [wd: %.2e] [lr: %.2e] | "
                        "[mem: %.2f] "
                        "(%.1f ms)"
                        % (
                            epoch + 1,
                            (itr // accum_iter),
                            (ipe // accum_iter),
                            total_loss_meter.avg,
                            _new_wd_A,
                            _new_lr_A,
                            _new_wd_B,
                            _new_lr_B,
                            (torch.cuda.max_memory_allocated() / 1024**3),
                            time_meter.avg,
                        )
                    )

                    if grad_stats is not None:
                        logger.info(
                            "[%d, %d] grad_stats: [%.2e %.2e] (%.2e, %.2e)"
                            % (
                                epoch + 1,
                                itr,
                                grad_stats.first_layer,
                                grad_stats.last_layer,
                                grad_stats.min,
                                grad_stats.max,
                            )
                        )

            log_stats()

            def plot_losses():
                plt.figure(figsize=(20, 10), dpi=100)
                x = list(range(start_epoch, num_epochs))
                sns.lineplot(x=x, y=MODEL_LOSS, label="Model Loss", color="blue")

                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training Loss")
                plt.legend()
                plt.savefig(f"plots/losses/losses_epoch_{epoch + 1}.png")
                plt.close()
            #plot_losses()

        testAcc1 = AverageMeter()
        testAcc5 = AverageMeter()
        test_loss = AverageMeter()
        
        # Warning: Enabling distributed evaluation with an eval dataset not divisible by process number
        # will slightly alter validation results as extra duplicate entries are added to achieve equal 
        # num of samples per-process.

        @torch.no_grad()
        def evaluate():
            logger.info(f"Building epoch's {epoch} cache...")
            cached_features = build_cache_v2(
                data_loader=supervised_loader_train,
                device=device,
                target_encoder=model,
                joint_embedding=False,
                proj_embed_dim=proj_embed_dim,
                dinov2=args["dinov2"],
                epoch=epoch,
                path=cache_path,
            )

            centroid_labels = {}
            centroids = []
            quantities = {}
            for key in cached_features.keys():
                c = torch.mean(torch.stack(cached_features[key]), dim=0) # take the average representation for each class
                centroids.append(c)  
                centroid_labels[len(centroids)-1] = key
                quantities[key] = len(cached_features[key])

            centroid_labels = np.array([centroid_labels[i] for i in range(len(centroid_labels))])

            centroids = torch.stack(centroids)

            knn = NearestNeighbors(n_neighbors=min(nb_classes, 2), metric="euclidean")
            knn.fit(centroids)

            # -- Enable shuffling to reduce monitor bias
            supervised_sampler_val.set_epoch(epoch) 

            test_accuracy = AverageMeter()

            total_correct, total_samples = 0, 0
            for _, (samples, targets) in enumerate(supervised_loader_val):
                images = samples.to(device, non_blocking=True)
                _ = targets.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    with torch.inference_mode():
                        projector_embeddings = model(images)

                projector_embeddings = torch.mean(projector_embeddings, dim=1).squeeze(1)
                _, indices = knn.kneighbors(projector_embeddings.cpu(), n_neighbors=2)

                preds = centroid_labels[indices[:, 1]]

                correct = (torch.from_numpy(preds) == targets.cpu()).sum().item()

                total_correct += correct
                total_samples += images.size(0)

                acc1 = correct / images.size(0)
                test_accuracy.update(acc1)

            total_acc = total_correct / total_samples
            logger.info("accuracy: %.3f" % total_acc)

            logger.info(
                f"Average accuracy over evaluation dataset: {test_accuracy.avg:.3f}"
            )
            
        vtime = gpu_timer(evaluate)
        
        model.module.backbone.train(False)
        model.module.projector.train(True)

        # TODO: FIXME
        stats_logger.log(
            epoch + 1,
            lr,
            loss_meter.avg,
            0,
            0,
            0,
            0,
            0,
            time_meter.avg,
        )
        # -- Save Checkpoint after every epoch
        logger.info("avg. train_loss %.3f" % total_loss_meter.avg)
        save_checkpoint(epoch + 1)
        assert not np.isnan(loss), "loss is nan"
        logger.info("Loss %.4f" % loss)


if __name__ == "__main__":
    main()
