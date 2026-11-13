"""Geographic train / validation / test split, decided BEFORE patches are cut.

The ground is divided into square blocks (``split.block_size_m``) on one grid in one projected CRS, shared by
all images. Each block belongs to one split. A patch is kept only if its whole footprint lies in blocks of a
single split, so overlapping patches, neighbouring patches and the same place seen in two different images
can never end up in different splits.

Blocks holding training farms are assigned greedily (largest first, to the split furthest below its target
share of farms), so each split gets its share of farms rather than of empty land. Blocks without farms get
a split from a seeded hash. The assignment is saved and re-used: adding labels or images later never moves a
block (and its patches) from test into train.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import box

from ..utils.helpers import read_json, stable_seed, write_json

log = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")


class GeoSplit:
    def __init__(self, crs: str, block_size_m: float, fractions: dict, seed: int,
                 assignment: dict[tuple[int, int], str] | None = None):
        if not CRS.from_user_input(crs).is_projected:
            raise ValueError(f"split.crs must be projected (metres), got {crs}")
        total = sum(float(fractions[s]) for s in SPLITS)
        if abs(total - 1) > 1e-6:
            raise ValueError(f"split.fractions must sum to 1, got {total}")
        self.crs = str(crs)
        self.block = float(block_size_m)
        self.fractions = {s: float(fractions[s]) for s in SPLITS}
        self.seed = int(seed)
        self.assignment: dict[tuple[int, int], str] = dict(assignment or {})

    # ------------------------------------------------------------------ blocks
    def block_of(self, x: float, y: float) -> tuple[int, int]:
        return math.floor(x / self.block), math.floor(y / self.block)

    def split_of_block(self, key: tuple[int, int]) -> str:
        if key in self.assignment:
            return self.assignment[key]
        u = stable_seed(self.seed, key[0], key[1]) / 0xFFFFFFFF  # blocks without farms
        f = self.fractions
        return "train" if u < f["train"] else ("val" if u < f["train"] + f["val"] else "test")

    def assign_from_labels(self, labels: gpd.GeoDataFrame) -> None:
        """Greedy, farm-count-balanced assignment of blocks that contain training farms (new blocks only)."""
        t = labels[labels["train"]] if "train" in labels else labels
        pts = t.to_crs(self.crs).geometry.representative_point()
        keys = list(zip(np.floor(pts.x / self.block).astype(int), np.floor(pts.y / self.block).astype(int)))
        counts: dict[tuple[int, int], int] = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        assigned = {s: 0.0 for s in SPLITS}
        for k, s in self.assignment.items():
            assigned[s] += counts.get(k, 0)
        new = [k for k in counts if k not in self.assignment]
        rng = np.random.default_rng(self.seed)
        rng.shuffle(new)
        new.sort(key=lambda k: -counts[k])  # stable: ties keep the seeded random order
        total = float(sum(counts.values()))
        for k in new:
            deficit = {s: self.fractions[s] * total - assigned[s] for s in SPLITS}
            s = max(SPLITS, key=lambda name: (deficit[name], -SPLITS.index(name)))
            self.assignment[k] = s
            assigned[s] += counts[k]
        # with few blocks, make sure no split is empty: move the smallest block of the biggest split
        farm_blocks = [k for k in counts]
        if len(farm_blocks) >= 3:
            for s in SPLITS:
                if self.fractions[s] <= 0:      # e.g. cross-validation: no test split wanted
                    continue
                if not any(self.assignment[k] == s for k in farm_blocks):
                    donor = max(SPLITS, key=lambda n: sum(1 for k in farm_blocks if self.assignment[k] == n))
                    smallest = min((k for k in farm_blocks if self.assignment[k] == donor), key=lambda k: counts[k])
                    self.assignment[smallest] = s
        share = {s: sum(counts[k] for k in counts if self.assignment[k] == s) / max(total, 1) for s in SPLITS}
        log.info("split blocks (%.0f m): %d with farms; farm share train %.2f / val %.2f / test %.2f",
                 self.block, len(counts), share["train"], share["val"], share["test"])

    def split_for_footprint(self, footprint) -> str | None:
        """Split of a footprint (in ``self.crs``) if it lies within one split, else None (straddles)."""
        x0, y0, x1, y1 = footprint.bounds
        eps = 1e-6 * self.block
        bx0, by0 = self.block_of(x0 + eps, y0 + eps)
        bx1, by1 = self.block_of(x1 - eps, y1 - eps)
        splits = set()
        for bx in range(bx0, bx1 + 1):
            for by in range(by0, by1 + 1):
                cell = box(bx * self.block, by * self.block, (bx + 1) * self.block, (by + 1) * self.block)
                if cell.intersection(footprint).area > 1.0:
                    splits.add(self.split_of_block((bx, by)))
        return splits.pop() if len(splits) == 1 else None

    def blocks_gdf(self, bounds) -> gpd.GeoDataFrame:
        """All blocks covering ``bounds`` (in ``self.crs``) with their split."""
        x0, y0, x1, y1 = bounds
        rows = []
        for bx in range(math.floor(x0 / self.block), math.ceil(x1 / self.block)):
            for by in range(math.floor(y0 / self.block), math.ceil(y1 / self.block)):
                rows.append({"block": f"{bx}_{by}", "split": self.split_of_block((bx, by)),
                             "geometry": box(bx * self.block, by * self.block, (bx + 1) * self.block,
                                             (by + 1) * self.block)})
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=self.crs)

    def region(self, split: str, bounds) -> object:
        """Union of the blocks of one split inside ``bounds``."""
        b = self.blocks_gdf(bounds)
        return b[b["split"] == split].union_all()

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> Path:
        return write_json({"crs": self.crs, "block_size_m": self.block, "fractions": self.fractions, "seed": self.seed,
                           "assignment": [[k[0], k[1], s] for k, s in sorted(self.assignment.items())]}, path)

    @classmethod
    def load(cls, path: Path) -> "GeoSplit":
        d = read_json(path)
        return cls(d["crs"], d["block_size_m"], d["fractions"], d["seed"],
                   {(int(a), int(b)): s for a, b, s in d["assignment"]})


def split_path(cfg) -> Path:
    return Path(cfg.paths.processed_dir) / "metadata" / "split_blocks.json"


def build_split(cfg, labels: gpd.GeoDataFrame) -> GeoSplit:
    """Load the saved split (if its grid matches the config) and add any new farm blocks; else build one."""
    sc = cfg.split
    crs = sc.get("crs") or str(labels.crs)
    path = split_path(cfg)
    if path.exists():
        old = GeoSplit.load(path)
        if CRS.from_user_input(old.crs) == CRS.from_user_input(crs) and old.block == float(sc.block_size_m) \
                and old.seed == int(sc.seed):
            log.info("re-using split %s (%d blocks assigned)", path, len(old.assignment))
            old.fractions = {s: float(sc.fractions[s]) for s in SPLITS}
            old.assign_from_labels(labels)
            old.save(path)
            return old
        log.warning("split settings changed: building a new split (old one replaced). Models trained on the old "
                    "split must not be evaluated on the new test set.")
    split = GeoSplit(crs, sc.block_size_m, sc.fractions, sc.seed)
    split.assign_from_labels(labels)
    split.save(path)
    return split
