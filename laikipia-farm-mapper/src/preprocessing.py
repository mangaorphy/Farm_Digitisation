from __future__ import annotations
from pathlib import Path
import numpy as np
from .common import UserFacingError, require, write_json

def normalize(array: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    result = array.astype("float32", copy=True)
    for b in range(result.shape[0]):
        valid = ~nodata_mask
        if valid.any():
            lo, hi = np.percentile(result[b][valid], (2, 98)); result[b] = np.clip((result[b]-lo)/max(hi-lo,1e-6), 0, 1)
    result[:, nodata_mask] = 0
    return result

def tile_raster(path: str, config: dict, tiles_dir="data/tiles") -> list[Path]:
    rasterio = require("rasterio", "pip install rasterio")
    from rasterio.windows import Window
    src_path = Path(path); dest = Path(tiles_dir); dest.mkdir(parents=True, exist_ok=True)
    size, overlap = int(config["tile_size"]), int(config["tile_overlap"])
    if overlap >= size: raise UserFacingError("tile_overlap must be less than tile_size.")
    step = size-overlap; outputs=[]; manifest=[]
    with rasterio.open(src_path) as src:
        for row in range(0, src.height, step):
            for col in range(0, src.width, step):
                h,w=min(size,src.height-row),min(size,src.width-col)
                window=Window(col,row,w,h); raw=src.read(window=window, masked=True); valid=~np.all(raw.mask,axis=0) if np.ma.isMaskedArray(raw) else np.ones((h,w),bool)
                if not valid.any(): continue
                profile=src.profile.copy(); profile.update(height=h,width=w,transform=src.window_transform(window),nodata=0,dtype="float32")
                data=normalize(np.asarray(raw.filled(0)), ~valid)
                output=dest/f"tile_r{row}_c{col}.tif"
                with rasterio.open(output,"w",**profile) as out: out.write(data)
                outputs.append(output); manifest.append({"file":str(output),"row":row,"col":col,"bounds":list(rasterio.windows.bounds(window,src.transform))})
    write_json(dest/"manifest.json",manifest); return outputs

def rgb_preview(path: str, output="outputs/rgb_preview.png"):
    rasterio=require("rasterio", "pip install rasterio matplotlib"); plt=require("matplotlib.pyplot", "pip install matplotlib")
    with rasterio.open(path) as src:
        a=src.read([3,2,1],masked=True); a=normalize(a,np.all(a.mask,axis=0) if np.ma.isMaskedArray(a) else np.zeros(a.shape[1:],bool))
    Path(output).parent.mkdir(parents=True,exist_ok=True); plt.imsave(output,np.moveaxis(a,0,-1)); return output
