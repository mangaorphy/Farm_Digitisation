"""Google Earth Engine: context layers (source `gee`) and Planet NICFI basemaps as an alternative main image.

Setup once on the laptop (free for research / non-commercial use):
    pip install earthengine-api
    earthengine authenticate                     # opens a browser; log in with the account that has Earth Engine
and put your Cloud project id in config.yaml under gee.project.

Pixels are pulled with ee.data.computePixels on the exact working grid (UTM 37N), block by block, so no Drive
exports, no waiting on task queues, and a rerun skips blocks already fetched.

NICFI basemaps (projects/planet-nicfi/assets/basemaps/africa) need a separate free sign-up with Planet's
NICFI programme, and are licensed for non-commercial use only. Check that your project qualifies first.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

EMBEDDING = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
DYNAMIC_WORLD = "GOOGLE/DYNAMICWORLD/V1"
NICFI_AFRICA = "projects/planet-nicfi/assets/basemaps/africa"
DW_CLASSES = ["crops", "trees", "built", "bare"]


def init(cfg):
    import ee
    project = cfg.get("gee", {}).get("project")
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Earth Engine is not set up: run `earthengine authenticate` once, and set gee.project "
                           f"in config.yaml to your Cloud project id ({e})") from e
    return ee


def _grid(wt, shp, crs) -> dict:
    return {"dimensions": {"width": int(shp[1]), "height": int(shp[0])},
            "affineTransform": {"scaleX": wt.a, "shearX": 0, "translateX": wt.c,
                                "shearY": 0, "scaleY": wt.e, "translateY": wt.f},
            "crsCode": crs}


def pixels(ee, image, wt, shp, crs, retries: int = 5) -> np.ndarray:
    """(bands, H, W) array of `image` on the given grid."""
    for attempt in range(retries):
        try:
            arr = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY",
                                         "grid": _grid(wt, shp, crs)})
            return np.stack([arr[n] for n in arr.dtype.names]).astype(np.float32)
        except Exception as e:  # noqa: BLE001 - quota / transient errors: back off and retry
            if attempt == retries - 1:
                raise
            print(f"    Earth Engine: {str(e)[:120]} - retrying")
            time.sleep(5 * (attempt + 1))


def _rect(ee, bnd, crs):
    return ee.Geometry.Rectangle(list(bnd), proj=crs, geodesic=False)


class GEEContext:
    """Block function for context.write_source: embeddings (PCA), Dynamic World, optional NICFI NDVI."""

    def __init__(self, cfg):
        self.cfg, self.ee = cfg, init(cfg)
        ee, g, y = self.ee, cfg.context.gee, int(cfg.context.year)
        self.k = int(g.embedding_components)
        self.crs = cfg.aoi.crs
        # the embedding has no layer for very recent years yet: use the latest one available up to `year`
        coll = ee.ImageCollection(EMBEDDING)
        self.emb_year = next((yr for yr in range(y, 2016, -1)
                              if coll.filterDate(f"{yr}-01-01", f"{yr + 1}-01-01").size().getInfo() > 0), None)
        if self.emb_year is None:
            raise ValueError(f"{EMBEDDING} has no year <= {y}")
        self.emb = coll.filterDate(f"{self.emb_year}-01-01", f"{self.emb_year + 1}-01-01").mosaic()
        # int8 keeps a 512x512x64 request at 16 MB (computePixels limit is 48 MB); -128 marks masked pixels
        self.emb_i8 = self.emb.multiply(127).round().unmask(-128).toInt8()
        self.dw = (ee.ImageCollection(DYNAMIC_WORLD).filterDate(f"{y}-01-01", f"{y + 1}-01-01")
                   .select(DW_CLASSES).mean().unmask(-1))
        self.nicfi_months = list(g.get("nicfi_months") or [])
        self.layers = [f"emb_pc{i + 1}" for i in range(self.k)] + [f"dw_{c}" for c in DW_CLASSES] + \
            [f"nicfi_ndvi_{m}" for m in self.nicfi_months]
        self._fit_pca(int(g.pca_sample_blocks))
        self.tags = {"embedding_year": str(self.emb_year), "dynamic_world_year": str(y),
                     "pca_mean": ",".join(f"{v:.5f}" for v in self.mean),
                     "pca_components": ";".join(",".join(f"{v:.5f}" for v in row) for row in self.components),
                     "pca_std": ",".join(f"{v:.5f}" for v in self.std)}
        print(f"Earth Engine: embedding year {self.emb_year}, Dynamic World {y}, NICFI months {self.nicfi_months}")

    def _embedding(self, wt, shp) -> tuple[np.ndarray, np.ndarray]:
        a = pixels(self.ee, self.emb_i8, wt, shp, self.crs)
        valid = (a != -128).all(axis=0)
        return a.reshape(64, -1).T / 127.0, valid.ravel()

    def _fit_pca(self, n_blocks: int):
        """PCA of the 64-D embedding fitted on a random sample of AOI blocks (fixed seed = reproducible)."""
        from .context import blocks
        cand = list(blocks(self.cfg, 256))
        rng = np.random.default_rng(0)
        pick = rng.choice(len(cand), size=min(n_blocks, len(cand)), replace=False)
        xs = []
        for i in pick:
            _, wt, shp, _ = cand[i]
            x, v = self._embedding(wt, shp)
            if v.any():
                xs.append(x[v][rng.choice(int(v.sum()), size=min(4000, int(v.sum())), replace=False)])
        if not xs:
            raise ValueError("no embedding pixels over the AOI")
        x = np.concatenate(xs)
        self.mean = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.components = vt[:self.k]
        self.std = (s[:self.k] / np.sqrt(len(x) - 1))
        print(f"  embedding PCA on {len(x)} pixels: first {self.k} components explain "
              f"{(s[:self.k] ** 2).sum() / (s ** 2).sum():.0%} of the variance")

    def __call__(self, wt, shp, bnd) -> dict:
        ee, out = self.ee, {}
        x, valid = self._embedding(wt, shp)
        pcs = ((x - self.mean) @ self.components.T) / self.std  # in standard deviations
        pcs[~valid] = np.nan
        for i in range(self.k):
            out[f"emb_pc{i + 1}"] = pcs[:, i].reshape(shp)
        dw = pixels(ee, self.dw.clip(_rect(ee, bnd, self.crs)).unmask(-1), wt, shp, self.crs)
        for c, a in zip(DW_CLASSES, dw):
            out[f"dw_{c}"] = np.where(a >= 0, a, np.nan)
        for m in self.nicfi_months:
            img = nicfi_month(ee, m)
            nd = img.normalizedDifference(["N", "R"]).unmask(-9)
            a = pixels(ee, nd, wt, shp, self.crs)[0]
            out[f"nicfi_ndvi_{m}"] = np.where(a > -2, a, np.nan)
        return out


def nicfi_month(ee, month: str):
    """NICFI monthly basemap mosaic for 'YYYY-MM' (bands B, G, R, N)."""
    start = ee.Date(f"{month}-01")
    return ee.ImageCollection(NICFI_AFRICA).filterDate(start, start.advance(1, "month")).mosaic()


def export_nicfi(cfg, month: str, out_dir: Path | None = None, gsd: float = 4.77, block_px: int = 1024) -> Path:
    """One NICFI monthly basemap over the AOI as a single 4-band GeoTIFF (blue, green, red, nir; uint16) in the
    raw imagery folder of the NICFI config, so `ingest` treats it like any other scene (one scene = no seams)."""
    import math

    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window
    from shapely.geometry import box

    from .context import aoi_geometry
    from .labels import aoi_bbox
    ee = init(cfg)
    out_dir = Path(out_dir or cfg.paths.imagery_raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"nicfi_{month.replace('-', '')}01.tif"
    b = transform_bounds("EPSG:4326", cfg.aoi.crs, *aoi_bbox(cfg))
    x0, y1 = math.floor(b[0] / gsd) * gsd, math.ceil(b[3] / gsd) * gsd
    W, H = int(math.ceil((b[2] - x0) / gsd)), int(math.ceil((y1 - b[1]) / gsd))
    transform = from_origin(x0, y1, gsd, gsd)
    aoi = aoi_geometry(cfg)
    img = nicfi_month(ee, month).select(["B", "G", "R", "N"]).unmask(0).toUint16()
    if not path.exists():
        prof = dict(driver="GTiff", width=W, height=H, count=4, dtype="uint16", nodata=0, crs=cfg.aoi.crs,
                    transform=transform, tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                    predictor=2, BIGTIFF="IF_SAFER")
        with rasterio.open(path, "w", **prof) as d:
            d.descriptions = ("blue", "green", "red", "nir")
            d.update_tags(source=NICFI_AFRICA, month=month, licence="NICFI non-commercial")
    done_path = path.with_suffix(".done.txt")
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    jobs = []
    for r in range(0, H, block_px):
        for c in range(0, W, block_px):
            win = Window(c, r, min(block_px, W - c), min(block_px, H - r))
            if f"{r}_{c}" not in done and aoi.intersects(box(*rasterio.windows.bounds(win, transform))):
                jobs.append(win)
    print(f"NICFI {month}: {len(jobs)} block(s) -> {path}")
    with rasterio.open(path, "r+") as d:
        for i, win in enumerate(jobs, 1):
            wt = rasterio.windows.transform(win, transform)
            a = pixels(ee, img, wt, (int(win.height), int(win.width)), cfg.aoi.crs)
            d.write(a.astype(np.uint16), window=win)
            done.add(f"{win.row_off}_{win.col_off}")
            done_path.write_text("\n".join(sorted(done)))
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")
    return path
