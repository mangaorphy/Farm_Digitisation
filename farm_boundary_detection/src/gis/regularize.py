"""Straight-edged, separated farm polygons.

Polygons traced from a 3 m probability map have rounded corners and staircase edges, and neighbouring farms
share their outlines (in Google Earth Pro two touching outlines look like one line, so farms seem to merge).
Smallholder plots are mostly rectangles laid out on a local grid, and digitised farms have straight sides. This
step, applied to model polygons only, makes each farm look like a hand-digitised one:

1. **direction** - each farm takes the dominant direction of the farms around it (``orientation_radius_m``),
   digitised farms counting most, so neighbouring plots line up instead of each being tilted on its own; a farm
   in an area without a clear direction keeps its own;
2. **straight edges** - in that direction, the polygon becomes a rectangle if a rectangle of the same area fits it
   (IoU >= ``rect_min_iou``); otherwise the outline is simplified, its edges snapped to the direction or the
   perpendicular (within ``ortho_angle_deg``), short clipped corners removed (``chamfer_m``) and the corners
   rebuilt where the straight edges meet, spikes thinner than ``spike_m`` trimmed; a shape that no longer matches the traced one (IoU < ``min_iou``)
   falls back to the plainly simplified outline;
3. **no overlaps** - most confident polygon first; a straightened polygon gives up any area already taken;
4. **a gap between neighbours** (``gap_m``) - every farm has its own outline.

Digitised farms are never changed.
"""
from __future__ import annotations

import logging
import math

import geopandas as gpd
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box

log = logging.getLogger(__name__)

DEFAULTS = {"enabled": True, "orientation_radius_m": 150.0, "rect_min_iou": 0.72, "ortho_angle_deg": 20.0,
            "simplify_m": 3.0, "chamfer_m": 7.0, "spike_m": 3.0, "gap_m": 1.0, "min_iou": 0.70, "min_area_m2": 100.0}


def _largest(geom) -> Polygon:
    parts = [p for p in getattr(geom, "geoms", [geom]) if isinstance(p, Polygon) and not p.is_empty]
    return max(parts, key=lambda p: p.area) if parts else Polygon()


def _open(geom, width: float) -> Polygon:
    """Morphological opening with square joins: removes spikes and slivers thinner than ``width``, keeps right
    angles. Returns the largest remaining part (the input itself if opening would erase it)."""
    if geom.is_empty or width <= 0:
        return _largest(geom)
    d = width / 2
    # mitre joins can re-grow a sharp corner past the original outline: clip back to it, so opening never
    # takes area from a neighbour or from the gap
    out = _largest(geom.buffer(-d, join_style="mitre").buffer(d, join_style="mitre").intersection(geom))
    out = out if not out.is_empty else _largest(geom)
    return _largest(out.simplify(0.001, preserve_topology=True))     # drop in-line vertices left by clipping


def _iou(a, b) -> float:
    u = a.union(b).area
    return a.intersection(b).area / u if u > 0 else 0.0


def own_direction(poly: Polygon) -> tuple[float, float]:
    """(angle of the longest side of the minimum rotated rectangle in [0, 90) degrees, elongation >= 1)."""
    c = list(poly.minimum_rotated_rectangle.exterior.coords)
    edges = [(c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1]) for i in range(min(4, len(c) - 1))]
    if not edges:
        return 0.0, 1.0
    lens = [math.hypot(*e) for e in edges]
    k = int(np.argmax(lens))
    ang = math.degrees(math.atan2(edges[k][1], edges[k][0])) % 90.0
    short = max(min(lens), 1e-9)
    return ang, max(lens) / short


def local_directions(polys: list[Polygon], refs: list[Polygon], radius: float) -> np.ndarray:
    """Dominant direction (degrees, period 90) around each polygon: weighted circular mean over neighbouring model
    polygons and reference (digitised) farms within ``radius``; weights favour large, elongated shapes and digitised
    farms. Where neighbours disagree (resultant < 0.5) the polygon's own direction is used."""
    shapes = list(polys) + list(refs)
    if not shapes:
        return np.zeros(0)
    dirs = [own_direction(p) for p in shapes]
    ang = np.radians(np.array([d[0] for d in dirs]) * 4.0)          # period 90 deg -> full circle
    elong = np.array([d[1] for d in dirs])
    area = np.array([p.area for p in shapes])
    w = np.sqrt(area) * np.clip(elong - 1.0, 0.05, 2.0)
    w[len(polys):] *= 3.0                                           # digitised farms: hand-drawn, trusted
    cent = gpd.GeoSeries([p.centroid for p in shapes])
    sidx = cent.sindex
    out = np.empty(len(polys))
    for i, p in enumerate(polys):
        c = p.centroid
        idx = sidx.query(box(c.x - radius, c.y - radius, c.x + radius, c.y + radius))
        sx, sy, sw = (w[idx] * np.cos(ang[idx])).sum(), (w[idx] * np.sin(ang[idx])).sum(), w[idx].sum()
        if sw > 0 and math.hypot(sx, sy) / sw >= 0.5:
            out[i] = (math.degrees(math.atan2(sy, sx)) / 4.0) % 90.0
        else:
            out[i] = dirs[i][0]
    return out


def _intersect(l1, l2):
    (p, r), (q, s) = l1, l2
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / den
    return p + t * r


def _rectangle(rot: Polygon) -> Polygon:
    """Axis-aligned rectangle of the same area and centre as ``rot`` (already rotated into the farm frame), with
    the proportions of its bounding box."""
    x0, y0, x1, y1 = rot.bounds
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return Polygon()
    k = math.sqrt(rot.area / (w * h))
    c = rot.centroid
    return box(c.x - k * w / 2, c.y - k * h / 2, c.x + k * w / 2, c.y + k * h / 2)


def _orthogonal(rot: Polygon, simplify_m: float, angle_deg: float, chamfer_m: float) -> Polygon:
    """Snap a farm-frame outline to horizontal / vertical edges; drop short diagonal (clipped-corner) edges."""
    p = rot.simplify(simplify_m, preserve_topology=True)
    pts = list(p.exterior.coords)[:-1]
    if len(pts) < 4:
        return p
    tan = math.tan(math.radians(angle_deg))
    edges = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        kind = "H" if abs(dy) <= tan * abs(dx) else ("V" if abs(dx) <= tan * abs(dy) else "D")
        edges.append([kind, a, b, math.hypot(dx, dy)])
    if sum(e[0] != "D" for e in edges) >= 3:                      # clipped corners: short diagonals go
        edges = [e for e in edges if not (e[0] == "D" and e[3] < chamfer_m)]
    merged = []
    for e in edges:                                               # consecutive edges of one kind -> one edge
        if merged and e[0] != "D" and merged[-1][0] == e[0]:
            m = merged[-1]
            m[1] = ((m[1][0] * m[3] + e[1][0] * e[3]) / (m[3] + e[3]), (m[1][1] * m[3] + e[1][1] * e[3]) / (m[3] + e[3]))
            m[2], m[3] = e[2], m[3] + e[3]
        else:
            merged.append(list(e))
    if len(merged) > 1 and merged[0][0] != "D" and merged[0][0] == merged[-1][0]:
        merged[0][3] += merged[-1][3]
        merged.pop()
    if len(merged) < 3:
        return p
    lines = []
    for kind, a, b, _ in merged:
        if kind == "H":
            lines.append((np.array([a[0], (a[1] + b[1]) / 2]), np.array([1.0, 0.0])))
        elif kind == "V":
            lines.append((np.array([(a[0] + b[0]) / 2, a[1]]), np.array([0.0, 1.0])))
        else:
            lines.append((np.array(a, float), np.array(b, float) - np.array(a, float)))
    corners = []
    for i in range(len(lines)):
        q = _intersect(lines[i - 1], lines[i])
        corners.append(tuple(q) if q is not None else tuple(merged[i][1]))
    out = Polygon(corners)
    return out if out.is_valid else _largest(out.buffer(0))


def straighten(poly: Polygon, theta: float, rc: dict) -> Polygon:
    """One polygon -> straight-edged version in direction ``theta`` (degrees)."""
    if poly.is_empty or poly.area <= 0:
        return poly
    poly = _largest(poly.buffer(0))
    origin = poly.centroid
    rot = affinity.rotate(poly, -theta, origin=origin)
    best = None
    for cand in (_rectangle(rot), _orthogonal(rot, float(rc["simplify_m"]), float(rc["ortho_angle_deg"]),
                                                float(rc["chamfer_m"]))):
        if cand.is_empty or not cand.is_valid:
            continue
        if len(cand.exterior.coords) > 5:   # remove thin spikes where two snapped edges meet at a sharp angle
            cand = _open(cand, float(rc["spike_m"]))
        iou = _iou(cand, rot)
        need = float(rc["rect_min_iou"]) if len(cand.exterior.coords) == 5 else float(rc["min_iou"])
        if iou >= need:
            best = cand
            break
    if best is None:
        return _open(poly.simplify(float(rc["simplify_m"]), preserve_topology=True), float(rc["spike_m"]))
    return affinity.rotate(best, theta, origin=origin)


def regularize(model: gpd.GeoDataFrame, rc: dict | None = None, taken=None,
               reference: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    """Straighten, de-overlap and separate model polygons (metric CRS; needs a ``score`` column).

    ``taken``: geometry already occupied (e.g. the digitised farms): model polygons keep ``gap_m`` away from it.
    ``reference``: farms whose directions guide the local grid (e.g. the digitised farms).
    Returns the kept polygons with areas recomputed."""
    rc = {**DEFAULTS, **(rc or {})}
    if len(model) == 0 or not rc.get("enabled", True):
        return model
    g = model.sort_values("score", ascending=False).reset_index(drop=True)
    polys = [_largest(p.buffer(0)) for p in g.geometry]
    refs = [] if reference is None or len(reference) == 0 else \
        [_largest(p.buffer(0)) for p in reference.to_crs(g.crs).geometry]
    theta = local_directions(polys, refs, float(rc["orientation_radius_m"]))
    straight = [straighten(p, t, rc) for p, t in zip(polys, theta)]
    gap = float(rc["gap_m"])
    sidx = gpd.GeoSeries(straight, crs=g.crs).sindex
    kept: list = [None] * len(g)
    blocked = taken.buffer(gap, join_style="mitre") if taken is not None and not taken.is_empty else None
    for i, geom in enumerate(straight):
        if geom.is_empty:
            continue
        geom = geom.buffer(-gap / 2, join_style="mitre")               # half the gap on each side
        if blocked is not None and geom.intersects(blocked):
            geom = geom.difference(blocked)
        for j in sidx.query(straight[i], predicate="intersects"):
            if j < i and kept[j] is not None:
                ring = kept[j].buffer(gap, join_style="mitre")
                if geom.intersects(ring):
                    geom = geom.difference(ring)
        if not geom.is_empty:   # cutting a neighbour out can leave notches and slivers
            geom = _open(geom.buffer(0), float(rc["spike_m"]))
        if not geom.is_empty and geom.area >= float(rc["min_area_m2"]):
            kept[i] = geom
    out = g[[k is not None for k in kept]].copy()
    out["geometry"] = [k for k in kept if k is not None]
    out = out.set_geometry("geometry")
    if "area_m2" in out:
        out["area_m2"] = out.area.round(1)
    if "area_ha" in out:
        out["area_ha"] = (out.area / 1e4).round(4)
    if "perimeter_m" in out:
        out["perimeter_m"] = out.length.round(1)
    n_rect = int(sum(len(p.exterior.coords) == 5 for p in out.geometry))
    log.info("regularised %d model polygons -> %d kept (%d rectangles), gap %.1f m between neighbours",
             len(g), len(out), n_rect, gap)
    return out
