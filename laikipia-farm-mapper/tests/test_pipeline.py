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


# ---------------------------------------------------------------- scoring + Delineate Anything wrapper


def test_scorer_matches_and_stratifies_by_size():
    import geopandas as gpd
    from shapely.geometry import box

    from farmmap.config import Cfg
    from farmmap.scoring import Scorer

    cfg = Cfg({"reference": {"coverage_buffer_m": 100}})
    # reference: a 1.0 ha farm (100x100 m) and a 0.04 ha farm (20x20 m)
    ref = gpd.GeoDataFrame({"ref_use": [True, True]}, crs="EPSG:32637",
                           geometry=[box(0, 0, 100, 100), box(200, 0, 220, 20)])
    pred = gpd.GeoDataFrame(crs="EPSG:32637", geometry=[box(5, 5, 100, 100),     # IoU ~0.9: found
                                                        box(190, 0, 260, 60)])   # IoU ~0.1: not found
    s = Scorer(cfg)
    s.add(pred, ref, box(-50, -50, 400, 200))
    out = s.summary()
    assert out["n_reference"] == 2 and out["matched_iou50"] == 1
    assert out["recall"] == 0.5 and out["precision"] == 0.5
    assert out["recall_0-0.25ha"] == 0.0 and out["recall_1-4ha"] == 1.0


def test_delineate_maps_pixel_polygons_to_map_coordinates(tmp_path):
    """A stand-in detector returns a known pixel square; it must land at the right map position."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    from farmmap.config import Cfg
    from farmmap.delineate import delineate

    tr = from_origin(250000, 30000, 10, 10)
    p = tmp_path / "img.tif"
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint16",
                       crs="EPSG:32637", transform=tr) as d:
        d.write(np.random.default_rng(0).integers(500, 3000, (3, 256, 256), dtype=np.uint16))
    cfg = Cfg({"project": {"crs": "EPSG:32637"},
               "imagery": {"source": "local", "local_bands": ["R", "G", "B"]},
               "polygonize": {"min_area_ha": 0.0},
               "delineate": {"upsample": 1, "tile_px": 128, "overlap_px": 32, "imgsz": 640, "conf": 0.2,
                             "iou": 0.5, "stretch_percentiles": [2, 98]}})
    cfg.root = tmp_path
    calls = []

    def fake(img):  # always "finds" a square at pixels 40..60 of the padded tile
        calls.append(img.shape)
        return [(np.array([[40, 40], [60, 40], [60, 60], [40, 60]], float), 0.9)]

    out = delineate(cfg, p, regions=[Window(0, 0, 128, 128)], predictor=fake, progress=lambda *_: None)
    assert calls and calls[0] == (160, 160, 3)  # 128 core + 2*16 context pad (overlap_px=32)
    # pad=16 => square at native px 24..44 (core-relative); its centre (34, 34) is inside the 128 core
    assert len(out) == 1
    minx, miny, maxx, maxy = out.geometry.iloc[0].bounds
    assert abs(minx - (250000 + 24 * 10)) < 2 and abs(maxx - (250000 + 44 * 10)) < 2
    assert abs(maxy - (30000 - 24 * 10)) < 2 and abs(miny - (30000 - 44 * 10)) < 2


# ---------------------------------------------------------------- your own imagery (8-bit, alpha, lon/lat)


def test_uint8_rgb_with_alpha_in_lonlat_is_prepared_and_accepted(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from farmmap.data import imagery_problems, to_model_input
    from farmmap.prepare import prepare_imagery
    from farmmap.scale import to_unit

    # 8-bit RGBA in geographic coordinates, like a drone orthomosaic exported from a photogrammetry tool
    src = tmp_path / "ortho.tif"
    rng = np.random.default_rng(0)
    data = rng.integers(20, 230, (4, 120, 120), dtype=np.uint8)
    data[3] = 255  # alpha
    with rasterio.open(src, "w", driver="GTiff", width=120, height=120, count=4, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(36.70, 0.20, 0.00002, 0.00002)) as d:
        d.write(data)
    out = prepare_imagery(src, tmp_path / "ortho_utm.tif", crs="EPSG:32637", resolution=2.0, bands=[1, 2, 3],
                          progress=lambda *_: None)
    with rasterio.open(out) as r:
        assert r.crs.to_string() == "EPSG:32637" and r.count == 3 and r.dtypes[0] == "uint8"
        assert abs(r.transform.a - 2.0) < 1e-6
        arr = r.read()
    # 8-bit data must not be rejected as "constant" and must standardise to a sane range
    refl = to_unit(arr)[:, (arr > 0).any(0)]
    assert imagery_problems(refl, 1.0) == []
    norm = {"mean": refl.mean(1).tolist(), "std": (refl.std(1) + 1e-6).tolist(), "red": None, "nir": None}
    x = to_model_input(arr, norm)
    assert np.isfinite(x).all() and abs(float(x[:, (arr > 0).any(0)].mean())) < 0.5


def test_highres_config_is_complete_and_consistent():
    from farmmap.delineate import rgb_positions
    from farmmap.imagery import band_positions
    from farmmap.pipeline import main_imagery

    cfg = load_config(ROOT / "config_highres.yaml")
    assert cfg.imagery.source == "local" and main_imagery(cfg).name == "ortho.tif"
    assert band_positions(cfg) == (None, None)          # RGB only: no NDVI channel
    assert rgb_positions(cfg) == [0, 1, 2]
    # nothing is shared with the Sentinel-2 project, so models/labels never mix
    base = load_config(ROOT / "config.yaml")
    for a, b in ((cfg.labels.dir, base.labels.dir), (cfg.model.checkpoint, base.model.checkpoint),
                 (cfg.split.file, base.split.file), (cfg.imagery.dir, base.imagery.dir),
                 (cfg.outputs.dir, base.outputs.dir)):
        assert a != b
