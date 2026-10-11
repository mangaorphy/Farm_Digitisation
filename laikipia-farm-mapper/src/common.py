from __future__ import annotations
import json
from pathlib import Path
from typing import Any

class UserFacingError(RuntimeError):
    pass

def require(name: str, extra: str = ""):
    try:
        return __import__(name)
    except ImportError as e:
        suffix = f" Install with: {extra}" if extra else ""
        raise UserFacingError(f"Missing optional dependency '{name}'.{suffix}") from e

def ensure_dirs(config: dict) -> None:
    Path(config.get("output_directory", "outputs")).mkdir(parents=True, exist_ok=True)
    for d in ("data/aoi", "data/imagery", "data/labels", "data/tiles", "data/models", "reports"):
        Path(d).mkdir(parents=True, exist_ok=True)

def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
