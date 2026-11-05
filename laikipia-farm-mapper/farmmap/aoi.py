"""Working area = fence corridor (fence lines buffered by aoi.fence_buffer_m) ∪ the area of your training parcels.

SDX only searches by bounding box (an uploaded GeoJSON is reduced to its full extent), and a box around a
~200 km fence corridor would be mostly empty. So the area is also cut into small search boxes
(aoi.search_box_km) that touch it; search SDX box by box and download only scenes that cover the corridor.
"""
from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from .export import to_kmz
from .labels import load_labels, matches

# rough Legion volume: PAN 0.3 m + 8-band MS 1.2 m, 16-bit, before compression (~33 MB/km²);
# scenes extend past the AOI and overlap, so the real download is larger
MB_PER_KM2 = 33


def fence_lines(cfg) -> gpd.GeoDataFrame:
    parts = []
    for layer in cfg.aoi.fence_layers:
        path = (cfg.root / layer["path"]).resolve()
        g = gpd.read_file(path).to_crs(cfg.aoi.crs)
        g = g[matches(g, layer.get("where"))]
        g["layer"] = path.stem
        parts.append(g[["layer", "geometry"]])
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=cfg.aoi.crs)


def build(cfg, out_dir: Path | None = None) -> gpd.GeoDataFrame:
    out_dir = Path(out_dir or Path(cfg.paths.aoi).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    fences = fence_lines(cfg)
    corridor = fences.union_all().buffer(float(cfg.aoi.fence_buffer_m))
    pieces = [("fence corridor", corridor)]

    inc = cfg.aoi.get("include_labels")
    lab = None
    if inc:
        lab = load_labels(cfg, bbox=None, apply_shifts=False)
        lab = lab[matches(lab, inc) & lab["train"]]
        # clusters of training parcels; small gaps closed so one cluster = one area
        b = float(cfg.aoi.label_buffer_m)
        clusters = lab.geometry.buffer(b).union_all().buffer(-b / 2).simplify(50)
        pieces.append(("training parcels", clusters))
        print(f"{len(lab)} training parcels define the label area")

    area = gpd.GeoDataFrame({"part": [p for p, _ in pieces]}, geometry=[g for _, g in pieces], crs=cfg.aoi.crs)
    union = area.union_all()
    aoi = gpd.GeoDataFrame({"name": ["AOI"], "area_km2": [round(union.area / 1e6, 1)]}, geometry=[union],
                           crs=cfg.aoi.crs)

    km = float(cfg.aoi.search_box_km) * 1000
    x0, y0, x1, y1 = union.bounds
    boxes = []
    for i in range(math.floor(x0 / km), math.ceil(x1 / km)):
        for j in range(math.floor(y0 / km), math.ceil(y1 / km)):
            cell = box(i * km, j * km, (i + 1) * km, (j + 1) * km)
            inside = cell.intersection(union)
            if inside.area > 0.02 * cell.area:
                parts = [p for p, g in pieces if g.intersects(cell)]
                n_lab = int(lab.representative_point().within(cell).sum()) if lab is not None else 0
                boxes.append({"covers": " + ".join(parts), "aoi_km2_inside": round(inside.area / 1e6, 1),
                              "training_parcels": n_lab, "geometry": cell})
    boxes = gpd.GeoDataFrame(boxes, crs=cfg.aoi.crs)
    # priority 1 = holds traced parcels (needed to train and score the pilot), 2 = corridor only
    boxes["priority"] = [1 if n >= 20 else 2 for n in boxes.training_parcels]
    boxes = boxes.sort_values(["priority", "training_parcels", "aoi_km2_inside"], ascending=[True, False, False])
    boxes.insert(0, "box", [f"B{i + 1:02d}" for i in range(len(boxes))])
    boxes = boxes.reset_index(drop=True)
    ll = boxes.to_crs("EPSG:4326")
    b = ll.geometry.bounds.round(5)
    boxes["west"], boxes["south"], boxes["east"], boxes["north"] = b.minx, b.miny, b.maxx, b.maxy
    # in the order of SDX's "Edit AOI" form: Northeast longitude, latitude; Southwest longitude, latitude
    boxes["NE_longitude"], boxes["NE_latitude"] = boxes.east, boxes.north
    boxes["SW_longitude"], boxes["SW_latitude"] = boxes.west, boxes.south

    aoi.to_crs("EPSG:4326").to_file(cfg.paths.aoi, driver="GeoJSON")
    area.to_crs("EPSG:4326").to_file(out_dir / "aoi_parts.geojson", driver="GeoJSON")
    fences.to_crs("EPSG:4326").to_file(out_dir / "fence_lines.geojson", driver="GeoJSON")
    boxes.to_crs("EPSG:4326").to_file(out_dir / "sdx_search_boxes.geojson", driver="GeoJSON")
    boxes.drop(columns="geometry").to_csv(out_dir / "sdx_search_boxes.csv", index=False)
    to_kmz(boxes[["box", "covers", "aoi_km2_inside", "geometry"]].rename(columns={"box": "farm_id"}),
           out_dir / "sdx_search_boxes.kmz", name="SDX search boxes")
    to_kmz(aoi.rename(columns={"name": "farm_id"}), out_dir / "aoi.kmz", name="AOI")

    km2 = union.area / 1e6
    print(f"AOI {km2:,.0f} km² (corridor {corridor.area / 1e6:,.0f} km²) -> {cfg.paths.aoi}")
    print(f"{len(boxes)} SDX search boxes of {cfg.aoi.search_box_km} km -> {out_dir / 'sdx_search_boxes.csv'}")
    print(f"Rough Legion volume for the AOI itself: ~{km2 * MB_PER_KM2 / 1000:,.0f} GB uncompressed "
          f"(whole scenes and overlaps make the download larger; check the quota counter in SDX)")
    return boxes
