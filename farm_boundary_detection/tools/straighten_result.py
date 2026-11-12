"""Give an existing zone result straight edges and separated outlines, without retraining.

    python tools/straighten_result.py <zone>_farms.gpkg data/zones/<zone>.kmz <out_dir> [name]

Model polygons are regularised (see src/gis/regularize.py and postprocess.regularize in configs/config.yaml);
digitised farms are kept exactly as they are. Writes <out_dir>/<name>.kmz (one folder per tile) and .gpkg.
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from src.gis.regularize import regularize  # noqa: E402
from src.gis.zone import read_zone, write_zone_kmz  # noqa: E402
from src.utils.helpers import load_config  # noqa: E402

src, zone_file, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
name = sys.argv[4] if len(sys.argv) > 4 else src.stem + "_straight"
rc = dict(load_config(Path(__file__).resolve().parents[1] / "configs" / "config.yaml").postprocess.regularize)
g = gpd.read_file(src)
utm = g.estimate_utm_crs()
g = g.to_crs(utm)
dig, mod = g[g.source == "Digitized"].copy(), g[g.source == "Model"].copy()
t0 = time.time()
reg = regularize(mod, rc, taken=unary_union(dig.geometry.buffer(0).values) if len(dig) else None, reference=dig)
out = gpd.GeoDataFrame(np.concatenate([dig.values, reg[dig.columns].values]), columns=dig.columns, crs=utm) \
    .set_geometry("geometry")
out["area_ha"] = (out.area / 1e4).round(4)
_, tiles = read_zone(zone_file)
out_dir.mkdir(parents=True, exist_ok=True)
out.to_crs(4326).to_file(out_dir / f"{name}.gpkg", driver="GPKG")
write_zone_kmz(out, tiles, out_dir / f"{name}.kmz", f"{name}: digitised + model farms (straight, separated)")
m = out[out.source == "Model"]


def nverts(geoms):
    return int(np.median([sum(len(q.exterior.coords) for q in getattr(p, "geoms", [p])) for p in geoms]))


a, b = out.sindex.query(out.geometry, predicate="intersects")
touch = sum(1 for i, j in zip(a, b) if i < j and "Model" in (out.source.iloc[i], out.source.iloc[j]))
print(f"{len(mod)} model polygons -> {len(m)} kept in {time.time() - t0:.0f} s | median vertices "
      f"{nverts(mod.geometry)} -> {nverts(m.geometry)} | "
      f"rectangles {int(sum(p.geom_type == 'Polygon' and len(p.exterior.coords) == 5 for p in m.geometry))} | "
      f"model pairs touching or overlapping: {touch} | -> {out_dir / (name + '.kmz')}")
