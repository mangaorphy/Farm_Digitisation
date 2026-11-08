"""A single self-contained HTML report (tables + embedded figures) that travels with the results."""
from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path


def _img(path: Path, width: str = "100%") -> str:
    path = Path(path)
    if not path.exists():
        return f"<p><em>missing figure: {html.escape(path.name)}</em></p>"
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:{width}">'


def build_report(out_path: Path, title: str, sections: list[tuple[str, list]]) -> Path:
    """``sections`` = [(heading, [items])]; an item is a str (paragraph HTML), a pandas DataFrame (table) or a
    Path to a PNG (embedded)."""
    import pandas as pd

    parts = [f"<h1>{html.escape(title)}</h1><p class='muted'>generated {datetime.now():%Y-%m-%d %H:%M}</p>"]
    for heading, items in sections:
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        for it in items:
            if isinstance(it, pd.DataFrame):
                parts.append(it.to_html(float_format=lambda v: f"{v:.3f}", border=0, classes="t"))
            elif isinstance(it, Path):
                parts.append(_img(it))
            else:
                parts.append(f"<p>{it}</p>")
    css = ("body{font-family:system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;color:#222}"
           "table.t{border-collapse:collapse;margin:8px 0 18px;font-size:13px}"
           "table.t td,table.t th{padding:4px 10px;border-bottom:1px solid #ddd;text-align:right}"
           "table.t th{background:#f3f3f3}.muted{color:#777}h2{margin-top:34px;border-bottom:2px solid #eee}"
           "img{display:block;margin:8px 0 16px}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
                        f"<style>{css}</style></head><body>{''.join(parts)}</body></html>")
    return out_path
