from __future__ import annotations
from pathlib import Path
import csv, zipfile
from .common import require

def validate(frame, config: dict, kmz_path=None):
    gpd=require("geopandas", "pip install geopandas shapely")
    problems=[]
    if frame.crs is None: problems.append("Missing CRS")
    if not frame.geometry.is_valid.all(): problems.append("Invalid geometries")
    if frame.farm_id.duplicated().any(): problems.append("Duplicate farm IDs")
    overlaps=0
    for i,g in enumerate(frame.geometry):
        for other in frame.geometry.iloc[i+1:]:
            if g.intersection(other).area > 1: overlaps += 1
    if overlaps: problems.append(f"{overlaps} overlapping polygon pairs")
    if kmz_path:
        try:
            with zipfile.ZipFile(kmz_path) as z:
                valid="doc.kml" in z.namelist() and b"<kml" in z.read("doc.kml")
            if not valid: problems.append("KMZ lacks valid doc.kml")
        except Exception: problems.append("KMZ archive is unreadable")
    summary={"polygon_count":len(frame),"total_area_ha":float(frame.area_ha.sum()) if len(frame) else 0,"crs":str(frame.crs),"overlap_pairs":overlaps,"issues":problems,"status":"PASS" if not problems else "REVIEW"}
    Path("reports").mkdir(exist_ok=True)
    with open("reports/validation.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=summary.keys()); w.writeheader(); w.writerow(summary)
    Path("reports/validation.md").write_text("# Validation report\n\n"+"\n".join(f"- **{k}**: {v}" for k,v in summary.items())+"\n\nCandidate boundaries require visual/field verification.\n")
    return summary
