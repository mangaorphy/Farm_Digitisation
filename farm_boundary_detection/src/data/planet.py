"""PlanetScope imagery for the Education & Research plan: AOI, scene search, order (clipped), download.

Quota: the E&R Basic plan allows 3,000 km2 per month, counted on the CLIPPED area of each ordered item (overlap
between scenes counts twice). So imagery is ordered clipped to an AOI built around the digitised farms (only
places with labels are useful for training), the scene selection covers the AOI with as few clear scenes as
possible, and ``planet-order`` refuses anything above ``planet.max_order_km2``.

Authentication: the API key from ``planet.env_file`` (``api_key=...``) or the environment variable
``PL_API_KEY``. Nothing here prints the key.

Steps (CLI)::

    python -m src aoi              # data/aoi/training_aoi.geojson from the farm polygons
    python -m src planet-search    # candidate + selected scenes, estimated km2 (no quota used)
    python -m src planet-order --confirm      # places the order (uses quota)
    python -m src planet-download --order-id <id>
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import mapping, shape

from ..utils.helpers import read_env_file, write_json

log = logging.getLogger(__name__)
DATA_API = "https://api.planet.com/data/v1"
ORDERS_API = "https://api.planet.com/compute/ops/orders/v2"


class PlanetError(RuntimeError):
    pass


# ============================================================== AOI
def build_aoi(cfg, labels: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    """Buffered union of the training farms, simplified to Planet's clip-vertex limit, in EPSG:4326."""
    ac = cfg.aoi
    t = labels[labels["train"]]
    if t.empty:
        raise PlanetError("no training polygons to build an AOI from")
    union = t.geometry.buffer(float(ac.buffer_m)).union_all()
    parts = [p for p in getattr(union, "geoms", [union]) if p.area >= float(ac.min_part_km2) * 1e6]
    if not parts:
        raise PlanetError("every AOI part is smaller than aoi.min_part_km2")
    geom = shapely.MultiPolygon([shapely.Polygon(p.exterior) for p in parts])  # no holes
    tol = float(ac.simplify_m)
    simple = geom.simplify(tol)
    while shapely.get_num_coordinates(simple) > int(ac.max_vertices):
        tol *= 1.5
        simple = geom.simplify(tol).buffer(0)
    gdf = gpd.GeoDataFrame({"name": ["training_aoi"]}, geometry=[simple], crs=labels.crs)
    info = {"area_km2": round(simple.area / 1e6, 1), "parts": len(getattr(simple, "geoms", [simple])),
            "vertices": int(shapely.get_num_coordinates(simple)), "simplify_m": round(tol, 1),
            "farms_covered": int(t.representative_point().within(simple).sum()), "farms": len(t)}
    out = Path(cfg.paths.aoi)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_crs("EPSG:4326").to_file(out, driver="GeoJSON")
    from ..gis.kmz import to_kmz
    to_kmz(gdf.assign(farm_id="training AOI"), out.with_suffix(".kmz"), name="Training AOI")
    log.info("AOI -> %s: %.1f km2 in %d parts, %d vertices; covers %d of %d training farms", out, info["area_km2"],
             info["parts"], info["vertices"], info["farms_covered"], info["farms"])
    if info["area_km2"] > 3000:
        log.warning("the AOI (%.0f km2) exceeds the monthly quota of 3,000 km2: order it in parts or reduce "
                    "aoi.buffer_m", info["area_km2"])
    return gdf.to_crs("EPSG:4326"), info


def load_aoi(cfg) -> gpd.GeoDataFrame:
    p = Path(cfg.paths.aoi)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `python -m src aoi` first")
    return gpd.read_file(p).to_crs("EPSG:4326")


# ============================================================== HTTP
def api_key(cfg) -> str:
    key = os.environ.get("PL_API_KEY")
    if not key:
        env = read_env_file((Path(cfg.root) / cfg.planet.env_file).resolve())
        key = env.get("PL_API_KEY") or env.get("api_key")
    if not key:
        raise PlanetError("no Planet API key: set PL_API_KEY or put api_key=... in planet.env_file")
    return key


class PlanetClient:
    def __init__(self, key: str, timeout: int = 120):
        self.auth = "Basic " + base64.b64encode(f"{key}:".encode()).decode()
        self.timeout = timeout

    def request(self, url: str, body: dict | None = None, method: str | None = None, retries: int = 5) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(retries):
            req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                                         headers={"Authorization": self.auth, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:500]
                if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(2 ** attempt * 2)
                    continue
                if e.code == 401:
                    raise PlanetError("Planet rejected the API key (HTTP 401)") from e
                raise PlanetError(f"Planet API HTTP {e.code}: {msg}") from e
            except urllib.error.URLError as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise PlanetError(f"cannot reach Planet: {e}") from e
        raise PlanetError("unreachable")

    def download(self, url: str, dest: Path) -> Path:
        """Result links carry their own token: fetch without the Authorization header (cloud storage rejects
        two auth methods); fall back to it if the link asks for authentication."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        for headers in ({}, {"Authorization": self.auth}):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as r, \
                        open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
                return dest
            except urllib.error.HTTPError as e:
                if e.code in (401, 403) and not headers:
                    continue
                raise PlanetError(f"download failed (HTTP {e.code}): {dest.name}") from e
        raise PlanetError(f"download failed: {dest.name}")


# ============================================================== search and selection
def search_filter(cfg, aoi_geom) -> dict:
    pc = cfg.planet
    return {"type": "AndFilter", "config": [
        {"type": "GeometryFilter", "field_name": "geometry", "config": mapping(aoi_geom)},
        {"type": "DateRangeFilter", "field_name": "acquired",
         "config": {"gte": f"{pc.date_start}T00:00:00Z", "lte": f"{pc.date_end}T23:59:59Z"}},
        {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": float(pc.max_cloud_cover)}},
        {"type": "RangeFilter", "field_name": "clear_percent", "config": {"gte": float(pc.min_clear_percent)}},
        {"type": "PermissionFilter", "config": ["assets:download"]},
        {"type": "AssetFilter", "config": [pc.asset]},
    ]}


def search_scenes(cfg, client: PlanetClient, aoi: gpd.GeoDataFrame, max_items: int = 5000) -> gpd.GeoDataFrame:
    """Every downloadable scene of the configured type over the AOI (paged)."""
    geom = aoi.union_all()
    body = {"item_types": [cfg.planet.item_type], "filter": search_filter(cfg, geom)}
    url, feats = f"{DATA_API}/quick-search?_page_size=250&_sort=acquired%20desc", []
    first = True
    while url and len(feats) < max_items:
        page = client.request(url, body if first else None, "POST" if first else "GET")
        first = False
        feats += page.get("features", [])
        url = page.get("_links", {}).get("_next")
    if not feats:
        return gpd.GeoDataFrame({"id": []}, geometry=[], crs="EPSG:4326")
    rows = [{"id": f["id"], "acquired": f["properties"].get("acquired"),
             "cloud_cover": f["properties"].get("cloud_cover"), "clear_percent": f["properties"].get("clear_percent"),
             "instrument": f["properties"].get("instrument"), "gsd": f["properties"].get("pixel_resolution"),
             "geometry": shape(f["geometry"])} for f in feats]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    log.info("%d downloadable %s scenes (%s) over the AOI, %s .. %s", len(gdf), cfg.planet.item_type,
             cfg.planet.asset, gdf.acquired.min()[:10], gdf.acquired.max()[:10])
    return gdf


def select_scenes(scenes: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame, coverage_target: float = 0.98,
                  max_scenes: int = 60, max_candidates: int = 800) -> tuple[gpd.GeoDataFrame, dict]:
    """Greedy set cover: repeatedly take the scene that adds the most still-uncovered AOI area, weighted by its
    clear share; stop at ``coverage_target`` or ``max_scenes``. Returns (selected, summary with km2)."""
    if scenes.empty:
        raise PlanetError("no scenes to select from: widen the dates or relax the cloud limits")
    utm = aoi.estimate_utm_crs()
    a = aoi.to_crs(utm).union_all()
    s = scenes.to_crs(utm).copy()
    s["quality"] = (s["clear_percent"].fillna(100) / 100.0) * (1 - s["cloud_cover"].fillna(0))
    s = s.sort_values(["quality", "acquired"], ascending=[False, False]).head(max_candidates)
    s["clip"] = s.geometry.intersection(a)
    s = s[~s["clip"].is_empty]
    uncovered, chosen = a, []
    total = a.area
    while len(chosen) < max_scenes and uncovered.area > (1 - coverage_target) * total and len(s):
        gain = s["clip"].intersection(uncovered).area * s["quality"]
        best = gain.idxmax()
        if gain[best] < 1e-4 * total:
            break
        chosen.append(best)
        uncovered = uncovered.difference(s.at[best, "clip"])
        s_best = s.loc[best]
        s = s.drop(index=best)
        log.debug("picked %s (%s), coverage %.1f%%", s_best["id"], s_best["acquired"][:10],
                  100 * (1 - uncovered.area / total))
    sel = scenes.loc[chosen].copy()
    clip_area = sel.to_crs(utm).geometry.intersection(a).area
    sel["clip_km2"] = (clip_area / 1e6).round(2).values
    summary = {"aoi_km2": round(total / 1e6, 1), "scenes": len(sel),
               "coverage": round(1 - uncovered.area / total, 4), "clipped_km2_quota": round(float(sel.clip_km2.sum()), 1),
               "dates": sorted({d[:10] for d in sel["acquired"]})}
    return sel, summary


def order_payload(cfg, item_ids: list[str], aoi: gpd.GeoDataFrame, name: str) -> dict:
    pc = cfg.planet
    tools: list[dict] = [{"clip": {"aoi": mapping(aoi.union_all())}}]
    if pc.get("harmonize"):
        tools.append({"harmonize": {"target_sensor": "Sentinel-2"}})
    return {"name": name, "products": [{"item_ids": list(item_ids), "item_type": pc.item_type,
                                        "product_bundle": pc.product_bundle}], "tools": tools}


def plan_path(cfg) -> Path:
    return Path(cfg.paths.imagery_dir).parent / "orders" / "selected_scenes.geojson"


def plan(cfg) -> tuple[gpd.GeoDataFrame, dict]:
    """Search + select, written to data/imagery/orders/ for inspection in QGIS. Uses no quota."""
    client = PlanetClient(api_key(cfg))
    aoi = load_aoi(cfg)
    scenes = search_scenes(cfg, client, aoi)
    sel, summary = select_scenes(scenes, aoi, float(cfg.planet.coverage_target), int(cfg.planet.max_scenes))
    out = plan_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(scenes):
        scenes.to_file(out.with_name("candidate_scenes.geojson"), driver="GeoJSON")
    sel.to_file(out, driver="GeoJSON")
    write_json(summary, out.with_name("selection_summary.json"))
    log.info("selected %d scenes covering %.1f%% of the %.1f km2 AOI; clipped area (quota) %.1f km2 -> %s",
             summary["scenes"], 100 * summary["coverage"], summary["aoi_km2"], summary["clipped_km2_quota"], out)
    return sel, summary


def place_order(cfg, confirm: bool, force: bool = False, name: str | None = None) -> str | None:
    """Order the scenes in the saved selection, clipped to the AOI. Without ``confirm`` only shows the payload."""
    sel_path = plan_path(cfg)
    if not sel_path.exists():
        raise PlanetError(f"{sel_path} not found: run `python -m src planet-search` first")
    sel = gpd.read_file(sel_path)
    km2 = float(sel["clip_km2"].sum())
    limit = float(cfg.planet.max_order_km2)
    if km2 > limit and not force:
        raise PlanetError(f"the order would use ~{km2:.0f} km2 of quota, above planet.max_order_km2={limit:.0f}. "
                          "Reduce the AOI / max_scenes, or pass --force")
    name = name or f"farm-boundaries-{time.strftime('%Y%m%d-%H%M')}"
    payload = order_payload(cfg, sel["id"].tolist(), load_aoi(cfg), name)
    orders_dir = sel_path.parent
    write_json(payload, orders_dir / f"{name}_payload.json")
    if not confirm:
        log.info("DRY RUN: would order %d scenes (~%.1f km2 of quota). Payload -> %s. Re-run with --confirm to "
                 "place it.", len(sel), km2, orders_dir / f"{name}_payload.json")
        return None
    resp = PlanetClient(api_key(cfg)).request(ORDERS_API, payload, "POST")
    oid = resp.get("id")
    write_json(resp, orders_dir / f"{name}_order.json")
    log.info("order %s placed (%d scenes, ~%.1f km2). Download with: python -m src planet-download --order-id %s",
             oid, len(sel), km2, oid)
    return oid


def download_order(cfg, order_id: str, poll_s: int = 30, timeout_h: float = 6.0) -> list[Path]:
    """Wait for the order to finish, then download every result file to paths.imagery_dir/<order_id>/."""
    client = PlanetClient(api_key(cfg))
    t0 = time.time()
    while True:
        order = client.request(f"{ORDERS_API}/{order_id}")
        state = order.get("state")
        if state in ("success", "partial"):
            break
        if state in ("failed", "cancelled"):
            raise PlanetError(f"order {order_id} {state}: {order.get('last_message')}")
        if time.time() - t0 > timeout_h * 3600:
            raise PlanetError(f"order {order_id} still {state} after {timeout_h} h")
        log.info("order %s: %s ... (checking again in %d s)", order_id, state, poll_s)
        time.sleep(poll_s)
    if state == "partial":
        log.warning("order %s finished PARTIALLY: some scenes failed (%s)", order_id, order.get("last_message"))
    out_dir = Path(cfg.paths.imagery_dir) / order_id
    files = []
    for res in order.get("_links", {}).get("results", []):
        rel = Path(res["name"])
        dest = out_dir / rel.name if rel.suffix.lower() in (".tif", ".tiff") else out_dir / "meta" / rel.name
        if dest.exists():
            files.append(dest)
            continue
        log.info("downloading %s", rel.name)
        files.append(client.download(res["location"], dest))
    tifs = [f for f in files if f.suffix.lower() in (".tif", ".tiff")]
    log.info("order %s: %d files (%d GeoTIFFs) in %s", order_id, len(files), len(tifs), out_dir)
    return files


def scenes_table(sel: gpd.GeoDataFrame) -> pd.DataFrame:
    return pd.DataFrame(sel.drop(columns="geometry"))
