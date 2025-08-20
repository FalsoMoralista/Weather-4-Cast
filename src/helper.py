# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys
import copy

import torch

import src.models.vision_transformer as vit
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_
import torch.nn  as nn
import torch.nn.functional as F
from torch import inf 

import src.models.autoencoder as AE

import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']
        
        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch

def load_DC_checkpoint(
    device,
    r_path,
    target_encoder,
    hierarchical_classifier,
    autoencoder,
    opt,
    AE_optimizer,
    scaler,
    scaler2=None
):

    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')
        if hierarchical_classifier is not None:
            pretrained_dict = checkpoint['classification_head']
            msg = hierarchical_classifier.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')
        if autoencoder is not None:
            pretrained_dict = checkpoint['autoencoder']
            msg = autoencoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained autoencoder from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if AE_optimizer is not None:
            AE_optimizer.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return target_encoder, hierarchical_classifier, None, opt, None, scaler, epoch


class VICReg(nn.Module):
    def __init__(self, args, num_features, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.args = args
        self.num_features = num_features
        
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x, y):
        batch_size = x.size(0)
        
        repr_loss = 0
        if self.sim_coeff > 0:
            repr_loss = F.mse_loss(x, y)

        # Center embeddings
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        epsilon = 1e-4
        
        # Per feature variance across the batch
        std_x = torch.sqrt(x.var(dim=0) + epsilon) 
        std_y = torch.sqrt(y.var(dim=0) + epsilon)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + self.off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss    

# Borrowed from MAE.
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, retain_graph=None, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    
    return encoder, predictor


def build_cache_v2(data_loader, device, target_encoder, path, epoch=0, proj_embed_dim=1280, joint_embedding = False, dinov2=False, hierarchical_classifier=None):   
    target_encoder.eval()
    if not hierarchical_classifier is None: 
        hierarchical_classifier.eval()

    items = []
    def forward_inputs():
        for itr, (sample, target) in enumerate(data_loader):
            def load_imgs():
                samples = sample.to(device, non_blocking=True)
                targets = target.to(device, non_blocking=True)
                return (samples, targets)
            imgs, _ = load_imgs()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):            
                # TODO: review normalization below
                with torch.inference_mode():
                    if joint_embedding:
                        _, h = target_encoder(imgs)
                    else:                     
                        h = target_encoder(imgs)
                if not dinov2:
                    #h = F.layer_norm(h, (h.size(-1),)) # Normalize over feature-dim 
                    h = torch.mean(h, dim=1) # Mean over patch-level representation and squeeze
                    h = torch.squeeze(h, dim=1)

                h = h.to(device=torch.device('cpu'), dtype=torch.float32)
                items.append((h, target))

    def build_feature_cache():
        cache = {}        
        for output, target in items:
            for x, y in zip(output, target):
                class_id = y.item()
                if not class_id in cache:
                    cache[class_id] = []                    
                cache[class_id].append(x)
        return cache
    if not os.path.exists(path + f'/cached_features_{proj_embed_dim}_epoch_{epoch}.pt'):
        logger.info(f'Cached features not detected for epoch: {epoch}, building cache')        
        forward_inputs()
        cache = build_feature_cache()
        torch.save(cache, path + f'/cached_features_{proj_embed_dim}_epoch_{epoch}.pt')
    else:
        logger.info(f'Loading cached features at {path}')        
        cache = torch.load(path + f'/cached_features_{proj_embed_dim}_epoch_{epoch}.pt')        
    target_encoder.train(True)
    
    if not hierarchical_classifier is None: 
        hierarchical_classifier.train(True)

    return cache

def build_cache(data_loader, device, target_encoder, hierarchical_classifier, autoencoder, path):   

    target_encoder.eval()
    autoencoder.eval()
    hierarchical_classifier.eval()

    items = []
    def forward_inputs():
        with torch.no_grad():
            for itr, (sample, target) in enumerate(data_loader):
                def load_imgs():
                    samples = sample.to(device, non_blocking=True)
                    targets = target.to(device, non_blocking=True)
                    return (samples, targets)
                imgs, _ = load_imgs()            
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):            
                    h = target_encoder(imgs)
                    h = torch.mean(h, dim=1) # Mean over patch-level representation and squeeze
                    h = torch.squeeze(h, dim=1) 
                    h = F.layer_norm(h, (h.size(-1),)) # Normalize over feature-dim 
                    #_, _, _, child_proj_emb = hierarchical_classifier(h, device)                    
                    # We performed this modification such that the clustering feature s 
                    _, bottleneck_output = autoencoder(h, device) 
                    items.append((bottleneck_output, target))
    def build_cache():
        cache = {}        
        for bottleneck_output, target in items:
            bottleneck_output = bottleneck_output.to(device=torch.device('cpu'), dtype=torch.float32)
            for x, y in zip(bottleneck_output, target):
                class_id = y.item()
                if not class_id in cache:
                    cache[class_id] = []                    
                cache[class_id].append(x)
        return cache
    if not os.path.exists(path + '/cached_features_epoch_0.pt'):
        forward_inputs()
        cache = build_cache()
        torch.save(cache, path + '/cached_features_epoch_0.pt')
    else:
        cache = torch.load(path + '/cached_features_epoch_0.pt')        
    autoencoder.train(True)
    target_encoder.train(True)
    hierarchical_classifier.train(True)
    return cache

def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0 
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def init_optimizer(
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    ipe_scale=1.25,
    backbone=None,
    projector=None,
    encoder=None,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False
    ):
    
    if backbone is not None:
        param_groups = [
            {
                'params': (p for n, p in backbone.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in projector.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in backbone.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0 
            }, {
                'params': (p for n, p in projector.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]
    elif encoder is not None:       
        param_groups = [
            {
                'params': (p for n, p in encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0 
            }
        ]
    else:
        print('Building projector optimizer')
        param_groups = [
            {
                'params': (p for n, p in projector.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in projector.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]        

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def init_DC_opt(
    model,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(model.parameters())
    
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = NativeScalerWithGradNormCount() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler 