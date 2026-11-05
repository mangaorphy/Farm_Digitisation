"""Turn raw SDX downloads (Vantor/Maxar-style GeoTIFF or NITF, PAN + MS or already pansharpened) into
aligned 8-bit scenes on one UTM grid, plus a VRT mosaic.

Per scene, processed block by block so a 50k x 50k pixel file never has to fit in RAM:
  1. warp to cfg.aoi.crs at cfg.imagery.gsd_m on a grid aligned to multiples of the pixel size
     (files with only RPCs are orthorectified with a DEM or a constant height);
  2. if a PAN file pairs with the MS file, ratio (Brovey) pansharpen with a per-scene PAN→intensity match;
  3. keep R, G, B (+ NIR) and contrast-stretch each band to 1..255 with per-scene percentiles (0 = nodata).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window

RASTER_EXT = {".tif", ".tiff", ".ntf", ".nitf", ".jp2", ".img"}
PAN_TOKENS = ("P1BS", "-P", "_P", "PAN")
MS_TOKENS = ("M1BS", "-M", "_M", "MUL", "MS")
BLOCK = 2048


@dataclass
class Scene:
    id: str
    ms: Path | None = None
    pan: Path | None = None
    date: str = ""
    notes: list[str] = field(default_factory=list)


def _georef(src) -> dict:
    """Kwargs for reproject(): a CRS+transform, or RPCs for un-orthorectified products."""
    if src.crs is not None and not src.transform.is_identity:
        return {"src_crs": src.crs, "src_transform": src.transform}
    if src.rpcs is not None:
        return {"rpcs": src.rpcs}
    if src.gcps[0]:
        return {"gcps": src.gcps[0], "src_crs": src.gcps[1]}
    raise ValueError(f"{src.name}: no CRS, RPCs or GCPs - cannot place it on the map")


def _scene_key(path: Path) -> str:
    """Pair PAN/MS files: Maxar names differ only in P1BS/M1BS (or -P/-M) tokens."""
    stem = path.stem.upper()
    stem = re.sub(r"(P1BS|M1BS|PAN|MUL)", "X", stem)
    stem = re.sub(r"(?<=[-_])[PM](?=[-_.]|$)", "X", stem)
    return stem


def _date_from_name(path: Path) -> str:
    m = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", path.name)
    if m:
        return "".join(m.groups())
    m = re.search(r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})", path.name.upper())
    if m:  # Maxar catalog style, e.g. 23JAN15 = 2023-01-15
        return datetime.strptime(m.group(0), "%y%b%d").strftime("%Y%m%d")
    return ""


def discover(raw_dir: Path) -> list[Scene]:
    """Group raster files into scenes (MS with optional PAN). Single-band files without a partner are skipped."""
    files = sorted(p for p in Path(raw_dir).rglob("*") if p.suffix.lower() in RASTER_EXT and p.is_file())
    groups: dict[str, Scene] = {}
    for f in files:
        try:
            with rasterio.open(f) as src:
                count = src.count
        except Exception as e:  # noqa: BLE001 - report and carry on with other files
            print(f"  skip {f.name}: {e}")
            continue
        key = _scene_key(f)
        sc = groups.setdefault(key, Scene(id=re.sub(r"[^A-Za-z0-9_-]+", "_", key)[:80], date=_date_from_name(f)))
        if count == 1:
            sc.pan = f
        elif sc.ms is None:
            sc.ms = f
        else:
            sc.notes.append(f"extra multiband file ignored: {f.name}")
    scenes = []
    for sc in groups.values():
        if sc.ms is None:
            print(f"  skip {sc.id}: PAN only (no multispectral partner)")
            continue
        scenes.append(sc)
    return scenes


def band_indices(src, cfg) -> dict[str, int]:
    """1-based band numbers for R, G, B (and N) in a multispectral file."""
    names = ["R", "G", "B", "N"]
    if cfg.imagery.bands != "auto":
        return dict(zip(names, cfg.imagery.bands))
    desc = [(d or "").lower() for d in src.descriptions]
    found = {}
    for i, d in enumerate(desc, start=1):
        if "red" in d and "edge" not in d and "R" not in found:
            found["R"] = i
        elif "green" in d:
            found.setdefault("G", i)
        elif "blue" in d:
            found.setdefault("B", i)
        elif ("nir" in d or "near" in d) and "N" not in found:
            found["N"] = i
    if {"R", "G", "B"} <= found.keys():
        return found
    # band-count conventions: WorldView/Legion 8-band, 4-band BGRN, 3-band RGB
    return {8: {"R": 5, "G": 3, "B": 2, "N": 7},
            4: {"R": 3, "G": 2, "B": 1, "N": 4},
            3: {"R": 1, "G": 2, "B": 3}}.get(src.count) or _fail(src)


def _fail(src):
    raise ValueError(f"{src.name}: {src.count} bands with no descriptions - set imagery.bands in config.yaml")


def target_grid(src, cfg) -> tuple[rasterio.Affine, int, int]:
    """Output grid in cfg.aoi.crs covering the file ∩ AOI, aligned to multiples of gsd."""
    gsd = float(cfg.imagery.gsd_m)
    if "rpcs" in _georef(src):
        lons = [src.rpcs.long_off - src.rpcs.long_scale, src.rpcs.long_off + src.rpcs.long_scale]
        lats = [src.rpcs.lat_off - src.rpcs.lat_scale, src.rpcs.lat_off + src.rpcs.lat_scale]
        b = transform_bounds("EPSG:4326", cfg.aoi.crs, min(lons), min(lats), max(lons), max(lats))
    else:
        b = transform_bounds(src.crs, cfg.aoi.crs, *src.bounds)
    from .labels import aoi_bbox
    if aoi_bbox(cfg):
        a = transform_bounds("EPSG:4326", cfg.aoi.crs, *aoi_bbox(cfg))
        b = (max(b[0], a[0]), max(b[1], a[1]), min(b[2], a[2]), min(b[3], a[3]))
    if b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("scene does not overlap the AOI")
    x0, y1 = math.floor(b[0] / gsd) * gsd, math.ceil(b[3] / gsd) * gsd
    x1, y0 = math.ceil(b[2] / gsd) * gsd, math.floor(b[1] / gsd) * gsd
    return from_origin(x0, y1, gsd, gsd), int(round((x1 - x0) / gsd)), int(round((y1 - y0) / gsd))


def _warp(src, bands, transform, shape, cfg, resampling) -> np.ndarray:
    out = np.zeros((len(bands), *shape), dtype=np.float32)
    nodata = src.nodata if src.nodata is not None else 0
    extra = {}
    if "rpcs" in _georef(src):
        extra = {"RPC_DEM": str(cfg.imagery.dem)} if cfg.imagery.get("dem") else {"RPC_HEIGHT": cfg.imagery.rpc_height_m}
    for k, b in enumerate(bands):
        reproject(rasterio.band(src, b), out[k], dst_transform=transform, dst_crs=cfg.aoi.crs,
                  src_nodata=nodata, dst_nodata=0, resampling=resampling, num_threads=4, **_georef(src), **extra)
    return out


def _sample(src, bands, max_px=1500) -> np.ndarray:
    """Decimated read for statistics (no georeferencing needed)."""
    f = max(1, int(max(src.width, src.height) / max_px))
    arr = src.read(bands, out_shape=(len(bands), src.height // f, src.width // f),
                   resampling=Resampling.average).astype(np.float32)
    nodata = src.nodata if src.nodata is not None else 0
    arr[arr == nodata] = np.nan
    return arr


def ingest_scene(sc: Scene, cfg, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sc.id}.tif"
    with rasterio.open(sc.ms) as ms:
        idx = band_indices(ms, cfg)
        names = ["R", "G", "B"] + (["N"] if cfg.imagery.use_nir and "N" in idx else [])
        bands = [idx[n] for n in names]
        transform, width, height = target_grid(ms, cfg)

        stats = _sample(ms, bands)
        lo_p, hi_p = cfg.imagery.stretch_percentiles
        lo = np.nanpercentile(stats.reshape(len(bands), -1), lo_p, axis=1)
        hi = np.nanpercentile(stats.reshape(len(bands), -1), hi_p, axis=1)

        pan = rasterio.open(sc.pan) if (sc.pan and cfg.imagery.pansharpen in ("auto", True)) else None
        if pan is not None:
            ps = _sample(pan, [1])[0]
            intensity = np.nanmean(stats[:3], axis=0)
            # PAN and MS DNs are on different scales: match PAN mean/std to MS intensity (per scene, no seams)
            gain = np.nanstd(intensity) / max(np.nanstd(ps), 1e-6)
            offset = np.nanmean(intensity) - gain * np.nanmean(ps)

        profile = dict(driver="GTiff", width=width, height=height, count=len(bands), dtype="uint8", nodata=0,
                       crs=cfg.aoi.crs, transform=transform, tiled=True, blockxsize=512, blockysize=512,
                       compress="deflate", predictor=2, BIGTIFF="IF_SAFER")
        valid_px = 0
        with rasterio.open(out_path, "w", **profile) as dst:
            for r in range(0, height, BLOCK):
                for c in range(0, width, BLOCK):
                    win = Window(c, r, min(BLOCK, width - c), min(BLOCK, height - r))
                    wt = rasterio.windows.transform(win, transform)
                    shape = (int(win.height), int(win.width))
                    arr = _warp(ms, bands, wt, shape, cfg, Resampling.bilinear)
                    valid = (arr > 0).all(axis=0)
                    if pan is not None and valid.any():
                        p = _warp(pan, [1], wt, shape, cfg, Resampling.average)[0]
                        valid &= p > 0
                        inten = arr[:3].mean(axis=0)
                        ratio = np.where(inten > 0, (gain * p + offset) / np.maximum(inten, 1e-6), 1.0)
                        arr *= np.clip(ratio, 0.2, 5.0)[None]
                    if not valid.any():
                        continue
                    scaled = (arr - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None]
                    out = np.clip(np.round(1 + scaled * 254), 1, 255).astype(np.uint8)
                    out[:, ~valid] = 0
                    dst.write(out, window=win)
                    valid_px += int(valid.sum())
            dst.descriptions = tuple(names)
            dst.update_tags(source_ms=sc.ms.name, source_pan=sc.pan.name if pan is not None else "",
                            pansharpened=str(pan is not None), date=sc.date, gsd_m=str(cfg.imagery.gsd_m))
        if pan is not None:
            pan.close()
    if valid_px == 0:
        out_path.unlink(missing_ok=True)
        raise ValueError("no valid pixels inside the AOI")
    return {"id": sc.id, "path": str(out_path), "date": sc.date, "bands": names, "pansharpened": pan is not None,
            "valid_km2": round(valid_px * float(cfg.imagery.gsd_m) ** 2 / 1e6, 2),
            "ms": str(sc.ms), "pan": str(sc.pan) if sc.pan else None, "notes": sc.notes}


def write_vrt(scene_paths: list[Path], vrt_path: Path) -> Path:
    """Stitch aligned scenes into one VRT; later entries are drawn on top (nodata 0 is transparent)."""
    metas = []
    for p in scene_paths:
        with rasterio.open(p) as s:
            metas.append((p, s.transform, s.width, s.height, s.count, s.crs, s.descriptions))
    counts = {m[4] for m in metas}
    if len(counts) != 1:
        raise ValueError(f"scenes have different band counts {counts}; set imagery.use_nir consistently")
    gsd = metas[0][1].a
    x0 = min(m[1].c for m in metas)
    y1 = max(m[1].f for m in metas)
    x1 = max(m[1].c + m[2] * gsd for m in metas)
    y0 = min(m[1].f - m[3] * gsd for m in metas)
    W, H = int(round((x1 - x0) / gsd)), int(round((y1 - y0) / gsd))
    n = metas[0][4]
    lines = [f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">',
             f"  <SRS>{escape(metas[0][5].to_wkt())}</SRS>",
             f"  <GeoTransform>{x0}, {gsd}, 0, {y1}, 0, {-gsd}</GeoTransform>"]
    for b in range(1, n + 1):
        lines.append(f'  <VRTRasterBand dataType="Byte" band="{b}">')
        lines.append("    <NoDataValue>0</NoDataValue>")
        lines.append(f"    <Description>{escape(metas[0][6][b - 1] or '')}</Description>")
        for p, t, w, h, *_ in metas:
            xoff, yoff = int(round((t.c - x0) / gsd)), int(round((y1 - t.f) / gsd))
            rel = Path(p).resolve()
            lines += ["    <ComplexSource>",
                      f'      <SourceFilename relativeToVRT="0">{escape(str(rel))}</SourceFilename>',
                      f"      <SourceBand>{b}</SourceBand>",
                      f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>',
                      f'      <DstRect xOff="{xoff}" yOff="{yoff}" xSize="{w}" ySize="{h}"/>',
                      "      <NODATA>0</NODATA>",
                      "    </ComplexSource>"]
        lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    vrt_path.write_text("\n".join(lines))
    return vrt_path


def ingest(cfg, only: list[str] | None = None) -> list[dict]:
    raw, out_dir = Path(cfg.paths.imagery_raw), Path(cfg.paths.scenes)
    scenes = discover(raw)
    if only:
        scenes = [s for s in scenes if s.id in only]
    if not scenes:
        raise FileNotFoundError(f"No usable imagery in {raw}. Copy the SDX downloads there (any sub-folders).")
    print(f"{len(scenes)} scene(s) found in {raw}")
    index_path = out_dir / "scenes.json"
    done = {d["id"]: d for d in json.loads(index_path.read_text())} if index_path.exists() else {}
    for sc in scenes:
        if sc.id in done and Path(done[sc.id]["path"]).exists():
            print(f"  {sc.id}: already ingested")
            continue
        print(f"  {sc.id}: MS={sc.ms.name} PAN={sc.pan.name if sc.pan else '-'}")
        try:
            done[sc.id] = ingest_scene(sc, cfg, out_dir)
            print(f"    -> {done[sc.id]['valid_km2']} km² inside AOI")
        except ValueError as e:
            print(f"    skipped: {e}")
    rows = sorted(done.values(), key=lambda d: d["date"], reverse=not cfg.imagery.newest_on_top)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(rows, indent=2))
    write_vrt([Path(r["path"]) for r in rows], Path(cfg.paths.mosaic))
    print(f"Mosaic: {cfg.paths.mosaic} ({len(rows)} scene(s), newest on top={cfg.imagery.newest_on_top})")
    return rows
