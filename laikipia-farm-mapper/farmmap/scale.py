"""Raster value scaling shared by every module.

Sentinel-2 composites made by this project are uint16 reflectance x 10000. Your own imagery may be
8-bit (typical RGB drone/aerial) or 16-bit, or float 0-1. Everything is brought to roughly 0-1 here;
the model then standardises per band, so the exact scale does not matter, only that it is consistent.
"""
from __future__ import annotations

import numpy as np


def to_unit(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        scale = 255.0
    elif a.dtype.kind == "f":
        scale = 1.0
    else:  # uint16 / int16 reflectance x 10000
        scale = 10000.0
    return a.astype(np.float32) / scale
