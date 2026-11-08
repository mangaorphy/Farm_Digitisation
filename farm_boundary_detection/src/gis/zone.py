"""Map one digitising zone (e.g. a corridor path) from every image that covers it.

1. every scene over the zone is predicted (only the zone's extent is read);
2. the scenes' probabilities are averaged on one grid (several dates = fewer errors from haze, shadow or crop
   stage; pixels that are nodata or cloudy in one scene are filled by the others);
3. farm polygons are built once from the fused probabilities;
4. the farms already digitised by hand are kept as they are, and model farms are added only where they do not
   duplicate them (a model polygon mostly covered by a digitised farm is dropped, a small overlap is cut off),
   so the result has no overlaps;
5. the result is written per work tile: GPKG / GeoJSON (QGIS) and a KMZ with a folder per tile (Earth Pro).

The zone file is a KML/KMZ (or any vector file) holding the zone polygon (its name contains "zone") and,
optionally, work tiles named "T01 ...", "T02 ...".
"""
from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

log = logging.getLogger(__name__)


def read_zone(path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """-> (zone: one polygon, tiles: name + geometry; empty if none), both EPSG:4326."""
    from ..data.ingestion import read_vector_file

    g, _ = read_vector_file(Path(path))
    g = g.to_crs("EPSG:4326")
    name = g["name"].fillna("").astype(str) if "name" in g else pd.Series([""] * len(g), index=g.index)
    is_tile = name.str.match(r"^\s*T\d+")
    is_zone = name.str.contains("zone", case=False) & ~is_tile
    if is_zone.any():
        zone_geom = unary_union(g[is_zone].geometry)
    elif is_tile.any():
        zone_geom = unary_union(g[is_tile].geometry)
    else:
        zone_geom = unary_union(g.geometry)
    tiles = g[is_tile].copy()
    tiles["tile"] = name[is_tile].str.extract(r"^\s*(T\d+)", expand=False).values
    zone = gpd.GeoDataFrame({"name": ["zone"]}, geometry=[zone_geom], crs="EPSG:4326")
    return zone, tiles[["tile", "name", "geometry"]].reset_index(drop=True)


def fuse_probabilities(probs: list[Path], out_path: Path, crs, bounds: tuple, res: float = 3.0) -> Path:
    """Average probability rasters (uint8 1..255, 0 = nodata; same band meaning) on one grid. A pixel is the
    mean over the rasters that have data there; band count = the smallest of the inputs."""
    x0, y0, x1, y1 = bounds
    W, H = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    transform = rasterio.transform.from_origin(x0, y1, res, res)
    n_bands = min(rasterio.open(p).count for p in probs)
    total = np.zeros((n_bands, H, W), np.float32)
    count = np.zeros((H, W), np.uint8)
    for p in probs:
        with rasterio.open(p) as src:
            buf = np.zeros((n_bands, H, W), np.uint8)
            for b in range(n_bands):
                reproject(rasterio.band(src, b + 1), buf[b], dst_transform=transform, dst_crs=crs,
                          src_nodata=0, dst_nodata=0, resampling=Resampling.bilinear)
        ok = buf.min(0) > 0
        total[:, ok] += buf[:, ok]
        count += ok
    out = np.zeros((n_bands, H, W), np.uint8)
    has = count > 0
    out[:, has] = np.clip(np.round(total[:, has] / count[has]), 1, 255).astype(np.uint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", driver="GTiff", width=W, height=H, count=n_bands, dtype="uint8", crs=crs,
                       transform=transform, nodata=0, tiled=True, blockxsize=512, blockysize=512,
                       compress="deflate") as dst:
        dst.write(out)
        dst.descriptions = ("interior_probability", "boundary_probability", "distance")[:n_bands]
        dst.update_tags(n_images=len(probs), images=";".join(Path(p).name for p in probs))
    log.info("fused %d probability rasters -> %s (%.0f%% of the grid covered, %.1f images per pixel)",
             len(probs), out_path, 100 * has.mean(), count[has].mean() if has.any() else 0)
    return out_path


def _largest(geom):
    parts = [p for p in getattr(geom, "geoms", [geom]) if isinstance(p, Polygon) and not p.is_empty]
    return max(parts, key=lambda p: p.area) if parts else Polygon()


def zone_product(model: gpd.GeoDataFrame, digitised: gpd.GeoDataFrame, zone: gpd.GeoDataFrame,
                 tiles: gpd.GeoDataFrame, duplicate_share: float = 0.3, min_area_m2: float = 200.0,
                 id_prefix: str = "AUTO", max_area_ha: float | None = None,
                 min_compact: float = 0.0) -> gpd.GeoDataFrame:
    """Digitised farms (kept as they are) + model farms that are not already digitised; nothing overlaps.

    A model polygon is dropped when more than ``duplicate_share`` of it lies on digitised farms (it is one of
    them, drawn differently); otherwise the digitised area is cut off it. Model polygons are kept if their
    centroid is in the zone; digitised farms if they touch it. Model polygons larger than ``max_area_ha`` or less
    compact than ``min_compact`` (4*pi*area/perimeter^2) are dropped: in bush the model tends to produce large
    irregular blobs that are not farms. Digitised farms are never filtered."""
    utm = zone.estimate_utm_crs()
    z = zone.to_crs(utm).geometry.iloc[0]
    d = digitised.to_crs(utm)
    d = d[d.intersects(z)].copy()
    d_union = unary_union(d.geometry) if len(d) else Polygon()
    m = model.to_crs(utm)
    m = m[m.geometry.centroid.within(z)].copy()
    if len(m) and not d_union.is_empty:
        share = np.array([g.intersection(d_union).area / max(g.area, 1e-9) for g in m.geometry])
        m = m[share <= duplicate_share].copy()
        m["geometry"] = [_largest(g.difference(d_union).buffer(0)) for g in m.geometry]
    m = m[m.area >= min_area_m2].copy()
    m["compact"] = (4 * np.pi * m.area / m.length.clip(lower=1e-9) ** 2).round(3)
    n0 = len(m)
    if max_area_ha:
        m = m[m.area <= max_area_ha * 1e4]
    m = m[m["compact"] >= min_compact].copy()
    if len(m) < n0:
        log.info("dropped %d model polygons larger than %s ha or less compact than %s", n0 - len(m), max_area_ha,
                 min_compact)
    m["review"] = np.where((m["score"] >= 0.8) & (m["compact"] >= 0.3), "likely farm", "check")
    m["source"] = "Model"
    m["farm_id"] = [f"{id_prefix}-{i + 1:05d}" for i in range(len(m))]
    id_col = next((c for c in ("FARM_ID", "farm_uid", "farm_id") if c in d), None)
    d["farm_id"] = d[id_col].astype(str) if id_col else [f"DIG-{i + 1:05d}" for i in range(len(d))]
    d["source"] = "Digitized"
    d["review"] = "existing"
    d["score"] = np.nan
    keep = ["farm_id", "source", "review", "score", "geometry"]
    out = gpd.GeoDataFrame(pd.concat([d[keep], m[keep]], ignore_index=True), crs=utm)
    out["area_ha"] = (out.area / 1e4).round(4)
    out["score"] = out["score"].round(3)
    t = tiles.to_crs(utm) if len(tiles) else tiles
    cent = gpd.GeoDataFrame(geometry=out.geometry.representative_point(), crs=utm)
    if len(t):
        j = gpd.sjoin(cent, t[["tile", "geometry"]], how="left", predicate="within")
        out["tile"] = j[~j.index.duplicated()]["tile"].fillna("outside tiles").values
    else:
        out["tile"] = "zone"
    return out


STYLES = ('<Style id="existing"><LineStyle><color>ffff00ff</color><width>1.5</width></LineStyle>'
          '<PolyStyle><color>33ff00ff</color></PolyStyle></Style>'
          '<Style id="likely"><LineStyle><color>ff00ff00</color><width>1.5</width></LineStyle>'
          '<PolyStyle><color>2200ff00</color></PolyStyle></Style>'
          '<Style id="check"><LineStyle><color>ff0080ff</color><width>1.5</width></LineStyle>'
          '<PolyStyle><color>220080ff</color></PolyStyle></Style>'
          '<Style id="tile"><LineStyle><color>ffffffff</color><width>2</width></LineStyle>'
          '<PolyStyle><fill>0</fill></PolyStyle></Style>')


def write_zone_kmz(farms: gpd.GeoDataFrame, tiles: gpd.GeoDataFrame, out_path: Path, title: str) -> Path:
    """KMZ: work tiles, then per tile: digitised (magenta), model 'likely farm' (green), model 'check' (orange)."""
    from .kmz import _polygon_kml

    f = farms.to_crs("EPSG:4326")
    t = tiles.to_crs("EPSG:4326") if len(tiles) else tiles

    def mark(name, desc, style, geom, data=""):
        return (f"<Placemark><name>{escape(str(name))}</name><description>{escape(desc)}</description>"
                f"<styleUrl>#{style}</styleUrl>{data}{_polygon_kml(_largest(geom))}</Placemark>")

    def ext(row):
        return "<ExtendedData>" + "".join(
            f'<Data name="{c}"><value>{escape(str(row[c]))}</value></Data>'
            for c in ("farm_id", "source", "review", "score", "area_ha", "tile")) + "</ExtendedData>"

    body = ["<Folder><name>Work tiles</name>" +
            "".join(mark(r.tile, r.name, "tile", r.geometry) for r in t.itertuples()) + "</Folder>"] if len(t) else []
    groups = [("Digitized", "existing", "Digitized (existing)", "existing"),
              ("Model", "likely farm", "Model - likely farms", "likely"),
              ("Model", "check", "Model - check", "check")]
    for tile in sorted(f["tile"].unique(), key=lambda s: (not re.match(r"T\d+", s), s)):
        ft = f[f["tile"] == tile]
        sub = []
        for source, review, label, style in groups:
            g = ft[(ft.source == source) & (ft.review == review)]
            if len(g):
                sub.append(f"<Folder><name>{label} ({len(g)})</name>" + "".join(
                    mark(r["farm_id"], f"{r['area_ha']} ha" + (f", score {r['score']}" if source == "Model" else ""),
                         style, r.geometry, ext(r)) for _, r in g.iterrows()) + "</Folder>")
        body.append(f"<Folder><name>{tile} ({len(ft)} farms)</name>{''.join(sub)}</Folder>")
    kml = ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           f"<name>{escape(title)}</name>{STYLES}" + "".join(body) + "</Document></kml>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
    return out_path


def run_zone(cfg, model_paths, zone_path: Path, images: list[Path], out_dir: Path, name: str = "zone",
             margin_m: float = 250.0, digitised: gpd.GeoDataFrame | None = None,
             only_tiles: list[str] | None = None, max_area_ha: float | None = None, min_compact: float = 0.0) -> dict:
    """Predict + fuse + polygonize + merge with the digitised farms for one zone. ``cfg.postprocess`` and
    ``cfg.inference`` are used as given (set the tuned values first). ``only_tiles`` (e.g. ["T02"]) maps just
    those work tiles. Returns paths and per-tile counts."""
    from ..data.ingestion import load_farm_polygons
    from ..inference.predictor import Predictor
    from .polygon_reconstruction import export_polygons, polygonize, remove_overlaps

    out_dir = Path(out_dir)
    zone, tiles = read_zone(zone_path)
    if only_tiles:
        tiles = tiles[tiles.tile.isin(only_tiles)].reset_index(drop=True)
        if tiles.empty:
            raise ValueError(f"none of the tiles {only_tiles} is in {zone_path}")
        zone = gpd.GeoDataFrame({"name": ["zone"]}, geometry=[unary_union(tiles.geometry)], crs="EPSG:4326")
    utm = zone.estimate_utm_crs()
    aoi = zone.to_crs(utm).buffer(margin_m).iloc[0]
    used, probs = [], []
    predictor = Predictor(model_paths, cfg)
    for img in images:
        with rasterio.open(img) as src:
            if src.crs is None:
                continue
            b = transform_bounds(utm, src.crs, *aoi.bounds)
            if not box(*src.bounds).intersects(box(*b)):
                continue
        prob = predictor.predict_raster(img, out_dir / "prediction_masks" / f"{name}__{Path(img).stem}_prob.tif",
                                        bounds=b)
        probs.append(prob)
        used.append(Path(img).name)
    if not probs:
        raise ValueError(f"no image covers the zone in {zone_path}")
    with rasterio.open(probs[0]) as src:
        res = abs(src.transform.a)                       # the imagery's own pixel size
    fused = fuse_probabilities(probs, out_dir / "prediction_masks" / f"{name}_fused_prob.tif", utm, aoi.bounds, res)
    model = polygonize(cfg, fused, source_name=f"{name} ({len(probs)} images)")
    if len(model):
        model = remove_overlaps(model.to_crs(utm), 0.3, float(cfg.postprocess.min_area_m2))
    if digitised is None:
        polys = load_farm_polygons(cfg)
        digitised = polys[0] if isinstance(polys, tuple) else polys
    product = zone_product(model, digitised, zone, tiles, min_area_m2=float(cfg.postprocess.min_area_m2),
                           max_area_ha=max_area_ha, min_compact=min_compact)
    paths = export_polygons(product, out_dir, f"{name}_farms")
    paths["kmz"] = write_zone_kmz(product, tiles, out_dir / "kmz" / f"{name}_farms.kmz",
                                  f"{name}: digitised + model farms")
    counts = product.groupby(["tile", "source", "review"]).size().unstack(["source", "review"], fill_value=0)
    log.info("%s: %d farms (%d digitised, %d model) from %d images", name, len(product),
             int((product.source == "Digitized").sum()), int((product.source == "Model").sum()), len(probs))
    return {"paths": paths, "fused": fused, "images": used, "counts": counts, "farms": product}
