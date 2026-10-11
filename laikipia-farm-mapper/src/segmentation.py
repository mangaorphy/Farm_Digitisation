"""SAM baseline and U-Net inference. Both outputs are preliminary unless independently validated."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .common import UserFacingError, require, write_json

def predict_unet(tile_paths, config: dict, out_dir="outputs/predictions"):
    torch=require("torch", "pip install torch segmentation-models-pytorch rasterio")
    smp=require("segmentation_models_pytorch", "pip install segmentation-models-pytorch")
    rasterio=require("rasterio", "pip install rasterio")
    ckpt=Path(config["model"]["checkpoint"])
    if not ckpt.exists(): raise UserFacingError(f"U-Net checkpoint missing: {ckpt}. Train first or set model.checkpoint.")
    device="cuda" if config["model"].get("device")=="auto" and torch.cuda.is_available() else "cpu"
    model=smp.Unet(encoder_name=config["model"]["unet_encoder"], in_channels=4, classes=3).to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device)); model.eval(); dest=Path(out_dir); dest.mkdir(parents=True,exist_ok=True); results=[]
    for tile in tile_paths:
        with rasterio.open(tile) as src:
            x=src.read().astype("float32"); profile=src.profile.copy()
        with torch.no_grad(): probs=torch.softmax(model(torch.from_numpy(x)[None].to(device)),1)[0].cpu().numpy()
        profile.update(count=3,dtype="float32",nodata=0); output=dest/(Path(tile).stem+"_prob.tif")
        with rasterio.open(output,"w",**profile) as d: d.write(probs)
        results.append(output)
    write_json(dest/"model_metadata.json",{"method":"unet","checkpoint":str(ckpt),"device":device,"validation_status":"unverified"}); return results

def predict_samgeo(tile_paths, config: dict, out_dir="outputs/predictions"):
    try: from samgeo import SamGeo
    except ImportError as exc: raise UserFacingError("SamGeo is optional and not installed. Install `pip install samgeo` and configure weights, or use U-Net.") from exc
    dest=Path(out_dir); dest.mkdir(parents=True,exist_ok=True); outputs=[]
    for tile in tile_paths:
        output=dest/(Path(tile).stem+"_sam.tif")
        sam=SamGeo(model_type="vit_h", checkpoint=config["model"].get("samgeo_model"), automatic=True)
        sam.generate(str(tile), output=str(output), foreground=True)
        outputs.append(output)
    write_json(dest/"model_metadata.json",{"method":"samgeo automatic masks","warning":"SAM segments visual regions, not verified farms.","validation_status":"unverified"}); return outputs
