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


def test_real_config_loads_and_reference_is_readable():
    from farmmap.reference import load_reference

    cfg = load_config(ROOT / "config.yaml")
    ref = load_reference(cfg)
    assert len(ref) > 100 and ref.crs.to_string() == "EPSG:32637"
    assert ref.geometry.is_valid.all() and not ref.geometry.has_z.any()


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
