"""Offline tests on a synthetic scene. They verify the code path, not real-world accuracy."""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from farmmap.config import load_config  # noqa: E402
from farmmap.synthetic import demo_config, make_demo_data  # noqa: E402


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    make_demo_data(out, size=1024, n_fields=14)
    return demo_config(load_config(ROOT / "config.yaml"), out, epochs=3), out


def test_config_loads_and_county_boundary_is_valid():
    from farmmap.county import load_county

    cfg = load_config(ROOT / "config.yaml")
    county = load_county(cfg)  # uses the shipped data/aoi/laikipia_county.geojson (downloads only if missing)
    assert county.crs.to_string() == "EPSG:32637"
    area_km2 = county.area.iloc[0] / 1e6
    assert 8000 < area_km2 < 11000  # Laikipia County is ~9,500 km2


def test_missing_reference_gives_helpful_error():
    from farmmap.reference import load_reference

    cfg = load_config(ROOT / "config.yaml")
    cfg.reference.extra_dirs = ["data/labels/_does_not_exist"]
    with pytest.raises(ValueError, match="data/labels/manual"):
        load_reference(cfg)


def test_estimate_skips_chunks_outside_county():
    from farmmap.imagery import estimate

    cfg = load_config(ROOT / "config.yaml")
    cfg.imagery.bbox = None
    est = estimate(cfg)
    assert est["chunks_to_download"] <= est["chunks_total"] and 8000 < est["area_km2"] < 11000


def test_normalisation_keeps_nodata_zero():
    from farmmap.data import to_model_input

    norm = {"mean": [0.1, 0.2], "std": [0.05, 0.05], "red": 0, "nir": 1}
    arr = np.zeros((2, 4, 4), np.uint16)
    arr[:, 1:, 1:] = 2000
    x = to_model_input(arr, norm)
    assert x.shape == (3, 4, 4) and np.all(x[:, 0, :] == 0) and np.isfinite(x).all()


def test_labels_have_three_classes(demo):
    import rasterio

    from farmmap.labels import build_labels

    cfg, out = demo
    lab = build_labels(cfg, out / "demo_imagery.tif", progress=lambda *_: None)
    with rasterio.open(lab) as s:
        vals = set(np.unique(s.read(1)).tolist())
    assert {0, 1, 2} <= vals or {1, 2} <= vals


def test_train_predict_polygonize_export(demo):
    from farmmap.evaluate import evaluate
    from farmmap.pipeline import predict_area, prepare_training_data
    from farmmap.train import train

    cfg, out = demo
    scenes = prepare_training_data(cfg, progress=lambda *_: None)
    assert len(scenes) == 1
    res = train(cfg, scenes, progress=lambda *_: None)
    assert Path(res["checkpoint"]).exists() and res["history"]

    # retraining from the saved checkpoint must work and keep the same split
    split_before = json.loads(Path(cfg.split.file).read_text())["scenes"]
    res2 = train(cfg, scenes, init_checkpoint=res["checkpoint"], epochs=1,
                 out_checkpoint=out / "finetuned.pt", progress=lambda *_: None)
    assert Path(res2["checkpoint"]).exists()
    assert json.loads(Path(cfg.split.file).read_text())["scenes"] == split_before

    metrics = evaluate(cfg, scenes, progress=lambda *_: None)
    assert 0 <= metrics["iou_interior"] <= 1 and metrics["n_reference_farms"] >= 1

    pred = predict_area(cfg, scenes[0]["imagery"], "demo", progress=lambda *_: None)
    assert len(pred["frame"]) >= 1
    assert pred["frame"].geometry.is_valid.all()
    with zipfile.ZipFile(pred["kmz"]) as z:
        assert b"<Polygon>" in z.read("doc.kml")


def test_kmz_roundtrips_as_reference(tmp_path, demo):
    """A KMZ exported for Google Earth must be readable back as reference polygons."""
    import geopandas as gpd
    from shapely.geometry import box

    from farmmap.export import write_kmz
    from farmmap.reference import read_vector

    frame = gpd.GeoDataFrame({"farm_id": [1]}, geometry=[box(300000, 10000, 301000, 11000)], crs="EPSG:32637")
    kmz = write_kmz(frame, tmp_path / "x.kmz")
    back = read_vector(kmz)
    assert len(back) == 1 and back.geometry.iloc[0].geom_type == "Polygon"


KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<Folder><name>Existing farms</name>
 <Placemark><name>A</name><ExtendedData><SchemaData schemaUrl="#farms">
   <SimpleData name="FARM_ID">LKP-1</SimpleData><SimpleData name="SOURCE">Digitized</SimpleData>
   <SimpleData name="QA_FLAG">OK</SimpleData></SchemaData></ExtendedData>
  <Polygon><outerBoundaryIs><LinearRing><coordinates>
   36.60,0.10,0 36.61,0.10,0 36.61,0.11,0 36.60,0.11,0 36.60,0.10,0</coordinates></LinearRing></outerBoundaryIs>
   <innerBoundaryIs><LinearRing><coordinates>
   36.603,0.103,0 36.604,0.103,0 36.604,0.104,0 36.603,0.104,0 36.603,0.103,0</coordinates></LinearRing></innerBoundaryIs>
  </Polygon></Placemark>
 <Placemark><name>B</name><ExtendedData><SchemaData schemaUrl="#farms">
   <SimpleData name="FARM_ID">LKP-2</SimpleData><SimpleData name="SOURCE">Scheme layer</SimpleData>
   <SimpleData name="QA_FLAG">OVERLAP</SimpleData></SchemaData></ExtendedData>
  <Polygon><outerBoundaryIs><LinearRing><coordinates>
   36.62,0.10,0 36.63,0.10,0 36.63,0.11,0 36.62,0.11,0 36.62,0.10,0</coordinates></LinearRing></outerBoundaryIs></Polygon>
 </Placemark>
</Folder>
<Folder><name>Fence</name><Placemark><name>f</name><LineString><coordinates>36.6,0.1,0 36.7,0.2,0</coordinates></LineString></Placemark></Folder>
</Document></kml>"""


def test_kml_with_attributes_holes_lines_and_qa_filter(tmp_path):
    from farmmap.config import Cfg
    from farmmap.reference import load_reference, read_vector

    f = tmp_path / "farms.kml"
    f.write_text(KML)
    df = read_vector(f)
    assert len(df) == 2 and list(df.FARM_ID) == ["LKP-1", "LKP-2"]  # the fence line is skipped
    assert len(df.geometry.iloc[0].interiors) == 1  # hole preserved

    cfg = Cfg({"project": {"crs": "EPSG:32637"},
               "reference": {"polygons": [str(f)], "extra_dirs": [], "name_field": "FARM_ID",
                             "filters": {"QA_FLAG": ["OK"]}}})
    cfg.root = tmp_path
    ref = load_reference(cfg)
    assert len(ref) == 2 and list(ref.ref_use) == [True, False]  # flagged parcel kept but not used
    assert list(ref.ref_name) == ["LKP-1", "LKP-2"]
