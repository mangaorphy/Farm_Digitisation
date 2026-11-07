"""Shared helpers: configuration, logging, seeding, devices, small I/O utilities."""
from __future__ import annotations

import copy
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

logger = logging.getLogger("farmseg")


class Config(dict):
    """A dict with attribute access (``cfg.train.epochs``) that stays YAML/JSON serialisable."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"config has no key {key!r}") from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return _wrap(copy.deepcopy(to_plain(self), memo))


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def to_plain(obj: Any) -> Any:
    """Config / Path / numpy values -> plain Python (for checkpoints and JSON)."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path: Path) -> dict:
    """YAML file; a top-level ``base: other.yaml`` is loaded first and overridden by this file."""
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    return _merge(_read_yaml((path.parent / base).resolve()), raw) if base else raw


def load_config(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> Config:
    """Read the YAML config, apply ``a.b.c=value`` overrides (values parsed as YAML) and resolve paths.

    ``paths.*`` are made absolute relative to the project folder (the parent of ``configs/``).
    """
    path = Path(path or DEFAULT_CONFIG).resolve()
    raw = _read_yaml(path)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, _, value = item.partition("=")
        node = raw
        *parents, leaf = key.strip().split(".")
        for p in parents:
            if not isinstance(node.setdefault(p, {}), dict):
                raise ValueError(f"cannot override {key}: {p} is not a section")
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    cfg = _wrap(raw)
    cfg.config_path = str(path)
    cfg.root = str(root)
    cfg.paths = Config({k: (None if v is None else (root / v).resolve()) for k, v in raw.get("paths", {}).items()})
    return cfg


def setup_logging(log_dir: str | Path | None = None, level: str = "INFO", name: str = "run") -> Path | None:
    """Log to the console and (if ``log_dir``) to ``log_dir/<name>_<timestamp>.log``."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for h in list(root.handlers):
        if getattr(h, "_farmseg", False):
            root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console._farmseg = True  # type: ignore[attr-defined]
    root.addHandler(console)
    log_file = None
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        fh._farmseg = True  # type: ignore[attr-defined]
        root.addHandler(fh)
    for noisy in ("rasterio", "fiona", "pyogrio", "matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def get_device(prefer: str | None = None):
    """CUDA if available, else Apple MPS, else CPU (``prefer`` forces one)."""
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain(obj), indent=2, default=str))
    return path


def read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text())


def stable_seed(*parts: Any) -> int:
    """A seed derived from strings/numbers that is the same in every Python process (unlike hash())."""
    import hashlib

    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def read_env_file(path: str | Path) -> dict[str, str]:
    """KEY=value lines (quotes stripped, # comments ignored)."""
    env: dict[str, str] = {}
    path = Path(path)
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip("'\"")
    return env


def rgb_indices(n_bands: int, descriptions: Iterable[str | None] | None = None,
                override: list[int] | None = None) -> list[int]:
    """0-based indices of the red, green and blue bands among ``n_bands`` input bands.

    ``override`` is 1-based (config ``imagery.rgb_bands``). Otherwise band descriptions are used when they name
    the colours, else Planet's band orders: 4-band = B, G, R, NIR; 8-band SuperDove = coastal blue, blue,
    green I, green, yellow, red, red edge, NIR; 3-band = R, G, B.
    """
    if override:
        idx = [int(b) - 1 for b in override]
        if len(idx) != 3 or min(idx) < 0 or max(idx) >= n_bands:
            raise ValueError(f"imagery.rgb_bands={override} is invalid for {n_bands} bands")
        return idx
    desc = [str(d or "").lower() for d in (descriptions or [])]
    if len(desc) == n_bands:
        def find(*names: str) -> int | None:
            for i, d in enumerate(desc):
                if d in names:
                    return i
            return None
        r, g, b = find("red"), find("green", "green_ii"), find("blue")
        if None not in (r, g, b):
            return [r, g, b]  # type: ignore[list-item]
    if n_bands == 4:
        return [2, 1, 0]
    if n_bands == 8:
        return [5, 3, 1]
    if n_bands >= 3:
        return [0, 1, 2]
    return [0] * 3
