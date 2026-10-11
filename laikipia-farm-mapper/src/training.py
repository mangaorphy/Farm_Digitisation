"""Supervised three-class U-Net training entry point.
Training data must contain aligned image GeoTIFFs and raster masks (0 background, 1 interior, 2 boundary).
"""
from __future__ import annotations
from pathlib import Path
from .common import UserFacingError, require

def train(config: dict):
    torch=require("torch", "pip install torch segmentation-models-pytorch rasterio")
    smp=require("segmentation_models_pytorch", "pip install segmentation-models-pytorch")
    rasterio=require("rasterio", "pip install rasterio")
    root=Path(config.get("training_dataset") or "")
    images=sorted((root/"images").glob("*.tif")) if root else []
    if not images: raise UserFacingError("No labelled training tiles. Set training_dataset with images/*.tif and masks/*.tif.")
    pairs=[(p,root/"masks"/p.name) for p in images]
    if any(not m.exists() for _,m in pairs): raise UserFacingError("Every training image must have an equally named mask GeoTIFF.")
    device="cuda" if torch.cuda.is_available() else "cpu"; model=smp.Unet(encoder_name=config["model"]["unet_encoder"],in_channels=4,classes=3).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4); ce=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.,2.,5.],device=device))
    model.train()
    for image,mask in pairs:
        with rasterio.open(image) as s: x=s.read().astype("float32")
        with rasterio.open(mask) as s: y=s.read(1).astype("int64")
        if x.shape[1:] != y.shape or x.shape[0] != 4 or y.min()<0 or y.max()>2: raise UserFacingError(f"Invalid aligned training pair: {image.name}")
        logits=model(torch.from_numpy(x)[None].to(device)); target=torch.from_numpy(y)[None].to(device)
        loss=ce(logits,target); opt.zero_grad(); loss.backward(); opt.step()
    ckpt=Path(config["model"]["checkpoint"]); ckpt.parent.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),ckpt)
    return ckpt
