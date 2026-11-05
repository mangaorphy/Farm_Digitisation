"""Load config.yaml into a dot-accessible object; resolve paths relative to the config file."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


class Cfg(dict):
    """dict with attribute access (cfg.train.epochs) that stays YAML/JSON serialisable."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


def _wrap(obj):
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: Path) -> dict:
    """YAML file; a top-level `base: other.yaml` loads that file first and overrides only the keys given here."""
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    return _merge(_read((path.parent / base).resolve()), raw) if base else raw


def load(path: str | Path = "config.yaml", overrides: list[str] | None = None) -> Cfg:
    """Read YAML, apply `a.b.c=value` overrides, and turn paths.* into absolute Paths."""
    path = Path(path).resolve()
    raw = _read(path)
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    cfg = _wrap(raw)
    cfg.root = path.parent
    cfg.paths = Cfg({k: (path.parent / v).resolve() for k, v in raw["paths"].items()})
    return cfg


def to_plain(cfg) -> dict:
    """Copy without Path objects (for saving inside model checkpoints)."""
    out = copy.deepcopy(dict(cfg))
    out.pop("root", None)
    out["paths"] = {k: str(v) for k, v in cfg.paths.items()}
    return out
