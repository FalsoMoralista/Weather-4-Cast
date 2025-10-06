import torch
import torch.nn as nn
import logging
import numpy as np 
import sys
from src.models.vision_transformer import VisionTransformer
from src.models.model_wrapper import ModelWrapper

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

for epoch in range(2, 3, 1):

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

    # TODO: Load dataset
    f_names = ['roxi_0008.cum1test20.reflbt0.ns.h5','roxi_0009.cum1test20.reflbt0.ns.h5', 'roxi_0010.cum1test20.reflbt0.ns.h5']

    ROOT: str = "/home/lucianodourado/weather-4-cast/dataset/w4c24" # TODO colocar em .yaml
    task = 'cum1' # TODO colocar em .yaml (futuramente)
    
    root = Path(self.dataset_path)
    years = root.glob("20*")
    self.paths = [root / str(year) for year in self.years]
    hrit_path = [p / "HRIT" for p in self.paths]
    year = year.replace('20', '')
    if task == "cum1":
        for p in self.hrit_path:
            files.extend(p.glob(f"*cum1{year}"))    


    @torch.no_grad()
    def evaluate():
        for _, x in enumerate(supervised_loader_val):
            images = x.to(device, non_blocking=True, dtype=torch.float32)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                with torch.inference_mode():
                    reconstructed_matrix = model(images)
                    
