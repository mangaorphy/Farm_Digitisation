"""Task 1: geospatial ingestion and alignment.

* Farm polygons: every KML / KMZ (and GPKG / SHP / GeoJSON) under ``paths.labels_dir``. KML is parsed here in
  one pass (Google Earth Pro exports have dozens of folders, and GDAL's KML driver re-reads the whole file per
  folder); ExtendedData attributes (SchemaData / Data) are kept. Geometries are validated, repaired where
  possible, de-duplicated, and every farm gets a unique ``farm_uid`` and integer ``instance_id``.
* Rasters: every PlanetScope GeoTIFF under ``paths.imagery_dir`` (UDM2 masks are paired, not treated as
  images). Metadata (CRS, transform, resolution, size, nodata) is read without loading pixels.
* Alignment: polygons are reprojected into each raster's own CRS and tested against its extent.
"""
from __future__ import annotations

import fnmatch
import io
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.validation import make_valid

log = logging.getLogger(__name__)

VECTOR_EXT = {".kml", ".kmz", ".gpkg", ".shp", ".geojson", ".json"}


class IngestionError(RuntimeError):
    """Raised when inputs are missing or unusable."""


# ============================================================== KML / KMZ parsing
def _local(tag: str) -> str:
    """Tag name without its XML namespace (KML 2.1, 2.2 and gx files all work)."""
    return tag.rsplit("}", 1)[-1]


def _child(el: ET.Element, name: str) -> ET.Element | None:
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _coords(text: str | None) -> list[tuple[float, float]]:
    """KML 'lon,lat[,alt] lon,lat[,alt] ...' -> 2D coordinates (altitude dropped)."""
    pts = []
    for tok in (text or "").split():
        parts = tok.split(",")
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pts


def _ring(boundary_el: ET.Element | None) -> list[tuple[float, float]]:
    if boundary_el is None:
        return []
    for el in boundary_el.iter():
        if _local(el.tag) == "coordinates":
            return _coords(el.text)
    return []


def _kml_polygon(el: ET.Element) -> Polygon | None:
    shell = _ring(_child(el, "outerBoundaryIs"))
    if len(shell) < 3:
        return None
    holes = [_ring(c) for c in el if _local(c.tag) == "innerBoundaryIs"]
    try:
        return Polygon(shell, [h for h in holes if len(h) >= 3])
    except (ValueError, TypeError):
        return None


def read_kml_bytes(data: bytes, source_file: str) -> tuple[gpd.GeoDataFrame, int]:
    """Placemarks with polygons -> GeoDataFrame (EPSG:4326). Returns (frame, number of non-polygon placemarks)."""
    try:
        root = ET.parse(io.BytesIO(data)).getroot()
    except ET.ParseError as e:
        raise IngestionError(f"{source_file}: not valid KML ({e})") from e
    rows: list[dict[str, Any]] = []
    skipped = 0

    def walk(node: ET.Element, folder: str | None) -> None:
        nonlocal skipped
        for c in node:
            tag = _local(c.tag)
            if tag in ("Document", "Folder"):
                name_el = _child(c, "name")
                walk(c, (name_el.text or folder) if (tag == "Folder" and name_el is not None) else folder)
            elif tag == "Placemark":
                polys = [p for p in (_kml_polygon(e) for e in c.iter() if _local(e.tag) == "Polygon") if p]
                if not polys:
                    skipped += 1  # points / lines are not farms
                    continue
                attrs: dict[str, Any] = {}
                for e in c.iter():
                    t = _local(e.tag)
                    if t == "SimpleData" and e.get("name"):
                        attrs[e.get("name")] = (e.text or "").strip()
                    elif t == "Data" and e.get("name"):
                        v = _child(e, "value")
                        attrs[e.get("name")] = ((v.text if v is not None else "") or "").strip()
                name_el = _child(c, "name")
                attrs["name"] = (name_el.text or "").strip() if name_el is not None else ""
                attrs["folder"] = folder
                attrs["geometry"] = polys[0] if len(polys) == 1 else MultiPolygon(polys)
                rows.append(attrs)

    walk(root, None)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326") if rows else \
        gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    return gdf, skipped


def read_vector_file(path: Path, assume_crs: str = "EPSG:4326") -> tuple[gpd.GeoDataFrame, int]:
    """One label file -> (GeoDataFrame in its own CRS, number of skipped non-polygon features)."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".kmz":
            with zipfile.ZipFile(path) as z:
                kmls = [n for n in z.namelist() if n.lower().endswith(".kml")]
                if not kmls:
                    raise IngestionError(f"{path.name}: KMZ contains no .kml file")
                parts = [read_kml_bytes(z.read(n), path.name) for n in kmls]
            frames = [p[0] for p in parts if len(p[0])]
            skipped = sum(p[1] for p in parts)
            gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326") \
                if frames else parts[0][0]
            return gdf, skipped
        if suffix == ".kml":
            return read_kml_bytes(path.read_bytes(), path.name)
        gdf = gpd.read_file(path)
    except zipfile.BadZipFile as e:
        raise IngestionError(f"{path.name}: corrupted KMZ ({e})") from e
    except IngestionError:
        raise
    except Exception as e:  # driver errors, unreadable files
        raise IngestionError(f"{path.name}: cannot read ({e})") from e
    if gdf.crs is None:
        log.warning("%s has no CRS: assuming %s", path.name, assume_crs)
        gdf = gdf.set_crs(assume_crs)
    is_poly = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]) | gdf.geometry.isna()
    return gdf[is_poly].copy(), int((~is_poly).sum())


def discover_label_files(labels_dir: Path) -> list[Path]:
    labels_dir = Path(labels_dir)
    if labels_dir.is_file():
        return [labels_dir]
    if not labels_dir.exists():
        raise IngestionError(f"labels folder not found: {labels_dir}")
    files = sorted(p for p in labels_dir.rglob("*") if p.suffix.lower() in VECTOR_EXT and not p.name.startswith("."))
    if not files:
        raise IngestionError(f"no KML/KMZ/GPKG/SHP/GeoJSON files in {labels_dir}")
    return files


# ============================================================== geometry validation
def clean_geometry(geom) -> tuple[Any, str]:
    """-> (polygonal geometry or None, status) with status in ok / repaired / empty / unrepairable."""
    if geom is None or geom.is_empty:
        return None, "empty"
    geom = shapely.force_2d(geom)
    status = "ok"
    if not geom.is_valid:
        status = "repaired"
        try:
            geom = make_valid(geom)
        except Exception:  # noqa: BLE001 - shapely raises GEOSException subclasses
            geom = geom.buffer(0)
    if geom.geom_type == "GeometryCollection":
        parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        polys = [p for g in parts for p in getattr(g, "geoms", [g])]
        geom = MultiPolygon(polys) if len(polys) > 1 else (polys[0] if polys else None)
    if geom is None or geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon") or geom.area == 0:
        return None, "unrepairable" if status == "repaired" else "empty"
    if not geom.is_valid:
        geom = geom.buffer(0)
        if geom.is_empty or not geom.is_valid:
            return None, "unrepairable"
        status = "repaired"
    return geom, status


def _matches(gdf: pd.DataFrame, filters: dict | None) -> np.ndarray:
    """Rows whose columns take one of the allowed values (missing columns do not filter)."""
    keep = np.ones(len(gdf), dtype=bool)
    for col, allowed in (filters or {}).items():
        if col in gdf.columns:
            keep &= gdf[col].astype(str).isin([str(a) for a in allowed]).to_numpy()
    return keep


def _matches_any(gdf: pd.DataFrame, filters: dict | None) -> np.ndarray:
    hit = np.zeros(len(gdf), dtype=bool)
    for col, values in (filters or {}).items():
        if col in gdf.columns:
            hit |= gdf[col].astype(str).isin([str(v) for v in values]).to_numpy()
    return hit


def _remove_duplicates(gdf: gpd.GeoDataFrame, iou_threshold: float) -> tuple[gpd.GeoDataFrame, int, int]:
    """Exact duplicates (same geometry to 1 cm) and near duplicates (IoU > threshold). Kept-for-training
    polygons win over ignored ones; otherwise the first one read wins. ``gdf`` must be in a metric CRS."""
    order = np.lexsort((np.arange(len(gdf)), ~gdf["train"].to_numpy()))
    gdf = gdf.iloc[order]
    key = shapely.to_wkb(shapely.normalize(shapely.set_precision(gdf.geometry.values, 0.01)))
    exact = pd.Series(key, index=gdf.index).duplicated()
    gdf = gdf[~exact.to_numpy()]
    n_exact = int(exact.sum())
    drop: set[int] = set()
    if iou_threshold and iou_threshold < 1 and len(gdf):
        geoms = gdf.geometry.values
        a_idx, b_idx = gdf.sindex.query(geoms, predicate="intersects")
        sel = a_idx < b_idx
        a_idx, b_idx = a_idx[sel], b_idx[sel]
        if len(a_idx):
            inter = shapely.area(shapely.intersection(geoms[a_idx], geoms[b_idx]))
            union = shapely.area(geoms[a_idx]) + shapely.area(geoms[b_idx]) - inter
            iou = inter / np.maximum(union, 1e-9)
            for a, b in zip(a_idx[iou > iou_threshold], b_idx[iou > iou_threshold]):
                if a not in drop:
                    drop.add(int(b))
    keep = np.ones(len(gdf), dtype=bool)
    keep[list(drop)] = False
    return gdf[keep], n_exact, len(drop)


def _farm_ids(gdf: pd.DataFrame, id_field: str) -> list[str]:
    """Unique farm ids: '<file stem>/<id>' where <id> is the id attribute, else the placemark name, else a counter.
    Repeated ids get '#2', '#3'..."""
    ids, seen = [], {}
    for i, row in enumerate(gdf.itertuples(index=False)):
        raw = getattr(row, id_field, None) if id_field in gdf.columns else None
        if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == "":
            raw = getattr(row, "name", None) if "name" in gdf.columns else None
        if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == "":
            raw = f"f{i + 1:06d}"
        uid = f"{Path(str(row.source_file)).stem}/{str(raw).strip()}"
        n = seen.get(uid, 0) + 1
        seen[uid] = n
        ids.append(uid if n == 1 else f"{uid}#{n}")
    return ids


def load_farm_polygons(cfg) -> tuple[gpd.GeoDataFrame, dict]:
    """All farm polygons, cleaned, in a metric working CRS (UTM zone of the labels), with columns:

    farm_uid (unique str), instance_id (int >= 1), train (bool: learned as farm; False = pixels ignored),
    source_file, status (ok / repaired) and the original attributes.
    """
    lc = cfg.labels
    files = discover_label_files(cfg.paths.labels_dir)
    frames, report_files = [], []
    skipped_non_polygon = 0
    for f in files:
        try:
            g, skipped = read_vector_file(f, lc.get("assume_crs", "EPSG:4326"))
        except IngestionError as e:
            log.error("%s", e)
            report_files.append({"file": f.name, "error": str(e)})
            continue
        skipped_non_polygon += skipped
        g = g.to_crs("EPSG:4326")  # every file may have its own CRS: bring all to one first
        g["source_file"] = f.name
        report_files.append({"file": f.name, "crs": "EPSG:4326 (KML)" if f.suffix.lower() in (".kml", ".kmz")
                             else str(g.crs), "polygons": len(g), "non_polygon_skipped": skipped})
        frames.append(g)
        log.info("read %s: %d polygons", f.name, len(g))
    if not frames:
        raise IngestionError(f"no readable label files in {cfg.paths.labels_dir}")
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    n_source = len(gdf)

    cleaned = [clean_geometry(g) for g in gdf.geometry]
    gdf["geometry"] = [c[0] for c in cleaned]
    gdf["status"] = [c[1] for c in cleaned]
    status_counts = gdf["status"].value_counts().to_dict()
    gdf = gdf[gdf.geometry.notna()].copy()

    for col, value in (lc.get("defaults") or {}).items():
        if col not in gdf.columns:
            gdf[col] = value
        else:
            gdf[col] = gdf[col].replace("", np.nan).fillna(value)
    gdf["farm_id_original"] = gdf[lc.id_field] if lc.id_field in gdf.columns else gdf.get("name", "")
    gdf["farm_uid"] = _farm_ids(gdf, lc.id_field)
    gdf["train"] = _matches(gdf, lc.get("keep")) & ~_matches_any(gdf, lc.get("drop"))

    work_crs = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(work_crs)
    # reprojection can make a nearly-degenerate polygon invalid again
    re_cleaned = [clean_geometry(g) for g in gdf.geometry]
    gdf["geometry"] = [c[0] for c in re_cleaned]
    gdf = gdf[gdf.geometry.notna()].copy()
    small = gdf.area < float(lc.get("min_area_m2", 0))
    gdf = gdf[~small].copy()
    gdf, n_exact, n_near = _remove_duplicates(gdf, float(lc.get("duplicate_iou", 1.0)))
    gdf = gdf.sort_values(["source_file", "farm_uid"]).reset_index(drop=True)
    gdf["instance_id"] = np.arange(1, len(gdf) + 1, dtype=np.int64)

    report = {
        "label_files": report_files,
        "source_polygons": n_source,
        "non_polygon_features_skipped": skipped_non_polygon,
        "valid_as_read": int(status_counts.get("ok", 0)),
        "repaired": int(status_counts.get("repaired", 0)),
        "removed_empty": int(status_counts.get("empty", 0)),
        "removed_unrepairable": int(status_counts.get("unrepairable", 0)),
        "removed_too_small": int(small.sum()),
        "removed_exact_duplicates": n_exact,
        "removed_near_duplicates": n_near,
        "valid_polygons": len(gdf),
        "train_polygons": int(gdf["train"].sum()),
        "ignored_polygons": int((~gdf["train"]).sum()),
        "working_crs": str(work_crs),
    }
    log.info("polygons: %d read -> %d valid (%d repaired, %d duplicates removed), %d for training, %d ignored",
             n_source, len(gdf), report["repaired"], n_exact + n_near, report["train_polygons"],
             report["ignored_polygons"])
    return gdf, report


# ============================================================== rasters
@dataclass
class RasterInfo:
    path: Path
    scene_id: str
    crs: str
    transform: tuple
    res: tuple
    width: int
    height: int
    count: int
    dtype: str
    nodata: float | None
    bounds: tuple
    descriptions: list = field(default_factory=list)
    udm2_path: Path | None = None

    @property
    def affine(self) -> rasterio.Affine:
        return rasterio.Affine(*self.transform[:6])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"], d["udm2_path"] = str(self.path), (str(self.udm2_path) if self.udm2_path else None)
        return d


_PLANET_SUFFIX = re.compile(r"(_3B|_BGRN|_AnalyticMS|_Analytic|_SR|_8b|_4b|_harmonized|_clip|_file_format|_udm2)"
                            r".*$", re.IGNORECASE)


def scene_key(path: Path) -> str:
    """Planet scene id from a file name, e.g. 20250615_073012_12_24a3_3B_AnalyticMS_SR_8b_clip.tif ->
    20250615_073012_12_24a3 (its UDM2 file, ..._3B_udm2_clip.tif, has the same key)."""
    key = _PLANET_SUFFIX.sub("", Path(path).stem)
    return key or Path(path).stem


def _matches_patterns(name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns or [])


def discover_rasters(cfg) -> tuple[list[RasterInfo], list[dict]]:
    """All imagery GeoTIFFs with their metadata. Unreadable files are reported, not fatal."""
    ic = cfg.imagery
    root = Path(cfg.paths.imagery_dir)
    if not root.exists():
        raise IngestionError(f"imagery folder not found: {root}")
    all_tifs = sorted(p for p in root.rglob("*") if p.is_file() and _matches_patterns(p.name, ic.include))
    udm2 = {scene_key(p): p for p in all_tifs if "udm2" in p.name.lower()}
    images = [p for p in all_tifs if not _matches_patterns(p.name, ic.exclude)]
    infos, problems = [], []
    seen_ids: dict[str, int] = {}
    for p in images:
        try:
            with rasterio.open(p) as src:
                if src.crs is None:
                    raise IngestionError("no CRS")
                if src.count < 1 or src.width < 1 or src.height < 1:
                    raise IngestionError("empty raster")
                if ic.get("require_projected_crs", True) and not CRS.from_user_input(src.crs).is_projected:
                    raise IngestionError(f"CRS {src.crs} is not projected (metres needed): reproject it, e.g. "
                                         "gdalwarp -t_srs EPSG:32637")
                # read one small block to catch truncated / corrupted files early
                src.read(1, window=rasterio.windows.Window(0, 0, min(64, src.width), min(64, src.height)))
                key = scene_key(p)
                n = seen_ids.get(key, 0) + 1
                seen_ids[key] = n
                info = RasterInfo(
                    path=p, scene_id=key if n == 1 else f"{key}_{n}", crs=src.crs.to_string(),
                    transform=tuple(src.transform)[:6], res=tuple(src.res), width=src.width, height=src.height,
                    count=src.count, dtype=src.dtypes[0], nodata=src.nodata, bounds=tuple(src.bounds),
                    descriptions=[d or "" for d in src.descriptions],
                    udm2_path=udm2.get(key) if ic.get("use_udm2", True) else None)
        except Exception as e:  # noqa: BLE001 - any read problem means "skip this file"
            log.error("skipping %s: %s", p.name, e)
            problems.append({"file": str(p), "error": str(e)})
            continue
        if info.udm2_path:
            with rasterio.open(info.udm2_path) as u:
                if (u.width, u.height) != (info.width, info.height) or u.crs != rasterio.crs.CRS.from_string(info.crs):
                    log.warning("%s: UDM2 grid differs from the image; not used", p.name)
                    info.udm2_path = None
        infos.append(info)
        log.info("raster %s: %dx%d px, %d bands %s, %.2f m, %s, nodata=%s, udm2=%s", info.scene_id, info.width,
                 info.height, info.count, info.dtype, info.res[0], info.crs, info.nodata, bool(info.udm2_path))
    if not infos:
        raise IngestionError(f"no usable GeoTIFF imagery in {root}" + (f" ({len(problems)} unreadable)" if problems
                                                                       else ""))
    return infos, problems


def select_bands(info: RasterInfo, bands: list[int] | None) -> list[int]:
    """1-based band numbers to read from this raster (``imagery.bands`` or all)."""
    if not bands:
        return list(range(1, info.count + 1))
    bad = [b for b in bands if b < 1 or b > info.count]
    if bad:
        raise IngestionError(f"{info.path.name}: bands {bad} do not exist (file has {info.count})")
    return [int(b) for b in bands]


def nodata_value(info: RasterInfo, cfg) -> float:
    v = cfg.imagery.get("nodata")
    return float(v) if v is not None else (float(info.nodata) if info.nodata is not None else 0.0)


# ============================================================== alignment
def check_crs_roundtrip(gdf: gpd.GeoDataFrame, dst_crs: str, n: int = 25, tol_m: float = 0.05) -> float:
    """Transform sample points to ``dst_crs`` and back: the error (m) must be ~0 if the CRSs are well defined."""
    if gdf.empty:
        return 0.0
    pts = gdf.geometry.representative_point().iloc[:: max(1, len(gdf) // n)]
    fwd = Transformer.from_crs(gdf.crs, dst_crs, always_xy=True)
    back = Transformer.from_crs(dst_crs, gdf.crs, always_xy=True)
    x, y = fwd.transform(pts.x.to_numpy(), pts.y.to_numpy())
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise IngestionError(f"polygons cannot be projected into {dst_crs}: non-finite coordinates")
    x2, y2 = back.transform(x, y)
    err = float(np.max(np.hypot(x2 - pts.x.to_numpy(), y2 - pts.y.to_numpy())))
    if err > tol_m:
        raise IngestionError(f"CRS round trip to {dst_crs} is off by {err:.3f} m")
    return err


def polygons_for_raster(gdf: gpd.GeoDataFrame, info: RasterInfo) -> tuple[gpd.GeoDataFrame, dict]:
    """Polygons reprojected into the raster's exact CRS, restricted to those intersecting its extent."""
    check_crs_roundtrip(gdf, info.crs)
    g = gdf.to_crs(info.crs)
    g["geometry"] = [clean_geometry(x)[0] for x in g.geometry]
    g = g[g.geometry.notna()]
    extent = box(*info.bounds)
    hit = g.intersects(extent)
    inside = g[hit].copy()
    stats = {"scene_id": info.scene_id, "polygons_intersecting": int(hit.sum()),
             "polygons_outside": int((~hit).sum()),
             "train_polygons_intersecting": int(inside["train"].sum()),
             "fully_inside": int(inside.within(extent).sum())}
    if hit.sum() == 0:
        log.warning("%s: no farm polygon overlaps this image", info.scene_id)
    return inside, stats


def run_ingestion(cfg, write_report: bool = True) -> tuple[gpd.GeoDataFrame, list[RasterInfo], dict]:
    """Labels + rasters + per-raster overlap, with the preprocessing report written to
    ``processed_dir/metadata/preprocessing_report.json``."""
    from ..utils.helpers import write_json

    labels, label_report = load_farm_polygons(cfg)
    rasters, problems = discover_rasters(cfg)
    per_raster = []
    for info in rasters:
        _, stats = polygons_for_raster(labels, info)
        stats.update({"file": info.path.name, "crs": info.crs, "size": [info.width, info.height],
                      "resolution_m": list(info.res), "bands": info.count, "dtype": info.dtype,
                      "nodata": info.nodata, "udm2": info.udm2_path.name if info.udm2_path else None})
        per_raster.append(stats)
    counts = {info.count for info in rasters}
    band_sel = cfg.imagery.get("bands")
    if len(counts) > 1 and not band_sel:
        raise IngestionError(f"images have different band counts {sorted(counts)}: set imagery.bands so every "
                             "image gives the same bands (or keep 4-band and 8-band products in separate runs)")
    report = {**label_report, "rasters": per_raster, "unreadable_rasters": problems,
              "rasters_with_farms": sum(r["polygons_intersecting"] > 0 for r in per_raster)}
    if report["rasters_with_farms"] == 0:
        raise IngestionError("no farm polygon overlaps any image: wrong area, or wrong CRS?")
    if write_report:
        out = Path(cfg.paths.processed_dir) / "metadata" / "preprocessing_report.json"
        write_json(report, out)
        log.info("preprocessing report -> %s", out)
    return labels, rasters, report
