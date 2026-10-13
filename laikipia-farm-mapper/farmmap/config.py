"""Configuration: one YAML file, attribute access, paths resolved relative to it."""
from __future__ import annotations

from pathlib import Path

import yaml


class Cfg(dict):
    """dict with attribute access (cfg.train.epochs); nested dicts are wrapped."""

    def __init__(self, data=None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = Cfg(value) if isinstance(value, dict) else value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        # cfg.imagery.bbox = None must change the same value that cfg.imagery.get("bbox") reads
        self[key] = Cfg(value) if isinstance(value, dict) else value

    def path(self, value) -> Path:
        """Resolve a config path relative to the config file's folder."""
        p = Path(value)
        return p if p.is_absolute() else self.root / p


def load_config(path="config.yaml") -> Cfg:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = Cfg(yaml.safe_load(path.read_text()))
    cfg.root = path.parent
    return cfg
