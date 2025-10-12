import torch
import torch.nn as nn
import logging
import numpy as np 
import sys
from src.models.vision_transformer import VisionTransformer
from src.models.model_wrapper import ModelWrapper
from src.datasets.InferenceDataset import InferenceDataset, worker_init_fn

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
dino_path = "/home/lucianodourado/dinov3-weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" 

dinov3 = torch.hub.load(
    "../dinov3", "dinov3_vitl16", source="local", weights=dino_path
).to(device)
dinov3 = torch.compile(dinov3, mode="reduce-overhead")

for epoch in range(22, 23, 1):

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
        dim_out=1024,
        num_heads=16,
        num_decoder_layers=8,
        num_target_channels=16,
        vjepa_size_in=14,
        vjepa_size_out=18,
        num_frames=4,
    )

    # TODO: move below into helper.py  
    try:
        r_path = '../../rtcalumby/adam/luciano/LifeCLEFPlant2022/logs/PlantNet300k_exp80/jepa-ep{}.pth.tar'.format(epoch)
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))

        for idx, key in enumerate(['downsample', 'vjepa']):

            state_dict = {
                k.replace(key+'.', ''): v for k, v in checkpoint['model'].items()
                if k.startswith(key)
            }
            if idx == 0:
                msg = model.downsample.load_state_dict(state_dict)
            else:
                msg = model.vjepa.load_state_dict(state_dict)  
            logger.info(f'loaded layer {key} with msg: {msg}')

        key = 'vit_decoder'
        state_dict = {}
        for k, v in checkpoint['model'].items():
            if k.startswith(key):
                new_key = k.replace('vit_decoder.', '', 1)
                if any(name in new_key for name in ['patch_embed', 'blocks', 'norm', 'time_expansion', 'conv_regression']):
                    state_dict[new_key] = v

        msg = model.vit_decoder.load_state_dict(state_dict)
        logger.info(f'loaded layer {key} with msg: {msg}')
        
        model.backbone = dinov3
        model.to(device)
        model.eval()
        del checkpoint
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    task = 'cum1' # TODO colocar em .yaml
    years = ['19', '20']
    filenames = ['roxi_0008', 'roxi_0009', 'roxi_0010']
    
    for year in years:
        predictions = {}
        for name in filenames:
            print('Year:', year)
            type = name + '.' + task + "test" + year
            dataset = InferenceDataset(InferenceDataset.ROOT, type=type)

            dist_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=dataset, num_replicas=1, rank=0
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
                            if not type in predictions:
                                predictions[type] = []
                            predictions[type].append(reconstructed_matrix)
                predictions[type] = torch.cat(predictions[type], dim=0)
                print(f'Predictions shape {predictions[type].size()}')    
            evaluate()
        torch.save(predictions, 'predictions_{}.pth'.format(task+'test'+year))
        print('Predictions:', predictions)
        print('Predictions length:', len(predictions))
        

                    
