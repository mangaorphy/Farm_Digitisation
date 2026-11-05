"""Offline tests on a synthetic SDX-style scene. Run: pytest -q  (about 2-3 minutes on a laptop CPU)."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from farmmap import config, evaluate, imagery, labels, polygonize, predict, register, synthetic, tiles, train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("farmmap")
    raw, lab = synthetic.make_scene(d, size_m=500, n_fields=30, seed=1)
    cfg = config.load(ROOT / "config.yaml", [
        f"paths.imagery_raw={raw}", f"paths.labels={lab}", f"paths.scenes={d / 'scenes'}",
        f"paths.mosaic={d / 'mosaic.vrt'}", f"paths.tiles={d / 'tiles'}", f"paths.models={d / 'models'}",
        f"paths.outputs={d / 'out'}", f"paths.aoi={d / 'no_aoi.geojson'}", "aoi.bbox=null",
        "tiles.size_px=128", "tiles.block_m=128", "train.batch_size=4", "train.num_workers=0", "model.encoder=resnet18", "model.encoder_weights=null",
        "predict.tile_px=128", "predict.overlap_px=32", "predict.chunk_px=512",
        "polygonize.block_px=512", "polygonize.halo_px=256"])
    imagery.ingest(cfg)
    return cfg


def test_scene_pairing():
    a = imagery._scene_key(Path("23JAN15081234-P1BS-0123_01_P001.TIF"))
    b = imagery._scene_key(Path("23JAN15081234-M1BS-0123_01_P001.TIF"))
    assert a == b
    assert imagery._date_from_name(Path("23JAN15081234-P1BS-x.TIF")) == "20230115"


def test_ingest_pansharpens_to_grid(work):
    with rasterio.open(work.paths.mosaic) as s:
        assert s.count == 4 and abs(s.transform.a - 0.5) < 1e-9
        assert s.transform.c % 0.5 == 0 and s.transform.f % 0.5 == 0  # grid aligned to pixel multiples
        img = s.read()
    assert (img > 0).mean() > 0.95


def test_kmz_roundtrip_and_filters(work):
    g = labels.load_labels(work)
    assert len(g) > 20 and g.crs.to_epsg() == 32637
    assert (~g.train).sum() == (g.QA_FLAG != "OK").sum()  # QA-flagged parcels are ignored, not dropped


def test_tiles_masks(work):
    idx = tiles.make_tiles(work)
    assert set(idx.split) >= {"train", "val"} or len(idx) < 6
    with rasterio.open(Path(work.paths.tiles) / "masks" / f"{idx.id[0]}.tif") as s:
        m = s.read(1)
    assert set(np.unique(m)) <= {0, 1, 2, 255} and (m == 1).any() and (m == 2).any()


def test_registration_detects_shift(work):
    base = labels.load_labels

    def shifted(cfg, bbox="aoi", apply_shifts=True):
        return labels.apply_label_shifts(base(cfg, bbox, False), "SOURCE", {"Scheme layer": [5, -3]})

    register.load_labels = shifted
    try:
        df = register.check(work, max_shift_m=12, per_group=30)
    finally:
        register.load_labels = base
    row = df.set_index("group").loc["SOURCE=Scheme layer"]
    assert row.verdict == "SHIFTED"
    assert abs(row.best_dx_m + 5) <= 1 and abs(row.best_dy_m - 3) <= 1
    assert df.set_index("group").loc["SOURCE=Digitized"].offset_m <= 1


def test_train_predict_polygonize_evaluate(work):
    if not (Path(work.paths.tiles) / "index.csv").exists():
        tiles.make_tiles(work)
    best = train.train(work, epochs=2)
    prob = predict.predict(work, best)
    with rasterio.open(prob) as s:
        assert s.count == 3
    farms = polygonize.polygonize(work, prob)
    assert isinstance(farms, gpd.GeoDataFrame)
    res = evaluate.evaluate(work, Path(work.paths.outputs) / "farms.gpkg", split="all")
    assert 0 <= res["precision"] <= 1 and res["parcels"] > 0


def test_polygonize_blocks_consistent(tmp_path):
    """Block-wise polygonize must give the same farms as one window."""
    from shapely.geometry import box
    from rasterio.transform import from_origin
    from rasterio.features import rasterize

    rng = np.random.default_rng(0)
    farms = [box(x, y, x + rng.uniform(20, 45), y + rng.uniform(20, 45))
             for x in range(0, 400, 50) for y in range(0, 400, 50)]
    t = from_origin(0, 400, 0.5, 0.5)
    inside = rasterize([(f, 1) for f in farms], out_shape=(800, 800), transform=t).astype(bool)
    edge = rasterize([(f.boundary.buffer(0.75), 1) for f in farms], out_shape=(800, 800), transform=t).astype(bool)
    p_int = np.where(inside & ~edge, 0.95, 0.03)
    p_bnd = np.where(edge, 0.9, 0.02)
    bands = np.stack([1 - p_int - p_bnd, p_int, p_bnd]).clip(0, 1)
    path = tmp_path / "prob.tif"
    with rasterio.open(path, "w", driver="GTiff", width=800, height=800, count=3, dtype="uint8",
                       crs="EPSG:32637", transform=t, nodata=0) as d:
        d.write((bands * 254 + 1).astype(np.uint8))
    out = {}
    for b, h in [(4096, 0), (300, 128)]:
        cfg = config.load(ROOT / "config.yaml", [f"paths.outputs={tmp_path}", f"polygonize.block_px={b}",
                                                 f"polygonize.halo_px={h}"])
        out[b] = polygonize.polygonize(cfg, path, tmp_path / f"f{b}.gpkg")
    assert len(out[4096]) == len(farms)
    m = evaluate.match(out[300], out[4096], 0.9)
    assert m.matched.sum() == len(farms)


def test_new_farm_defaults_and_cli_set(tmp_path):
    """Polygons drawn in Earth Pro's "New farms" folder (no SOURCE/QA_FLAG) must become trainable parcels."""
    from shapely.geometry import box
    from farmmap.export import to_kmz
    from farmmap import cli

    lab = tmp_path / "labels"
    lab.mkdir()
    to_kmz(gpd.GeoDataFrame({"farm_id": ["NEW-0001", "NEW-0002"]},
                            geometry=[box(36.60, 0.10, 36.601, 0.101), box(36.602, 0.10, 36.603, 0.101)],
                            crs="EPSG:4326"), lab / "new.kmz")
    cfg = config.load(ROOT / "config.yaml", [f"paths.labels={lab}", f"paths.aoi={tmp_path / 'none.geojson'}"])
    g = labels.load_labels(cfg)
    assert len(g) == 2 and g.train.all() and set(g.SOURCE) == {"Digitized"}
    # options after the command must not swallow it
    cli.main(["labels", "--set", f"paths.labels={lab}", f"paths.aoi={tmp_path / 'none.geojson'}"])


def _fake_context(cfg, path, layers=("crop_prob", "ndvi_h1")):
    """A 10 m context file over the synthetic scene, as `farmmap context` would write it."""
    from farmmap import context as CTX
    from rasterio.transform import from_origin
    with rasterio.open(cfg.paths.mosaic) as s:
        L, B, R, T = s.bounds
    gsd = 10.0
    x0, y1 = np.floor(L / gsd) * gsd - 50, np.ceil(T / gsd) * gsd + 50
    W, H = int((R - x0) / gsd) + 10, int((y1 - B) / gsd) + 10
    yy, xx = np.mgrid[0:H, 0:W]
    vals = {"crop_prob": 100.0 * xx / W, "ndvi_h1": np.full((H, W), 0.5)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=len(layers), dtype="uint8", nodata=0,
                       crs=cfg.aoi.crs, transform=from_origin(x0, y1, gsd, gsd)) as d:
        d.write(np.stack([CTX.encode(vals[k], k) for k in layers]))
        d.descriptions = tuple(f"ctx:dea:{k}" for k in layers)
    return path


def test_context_encode_read_roundtrip(work, tmp_path):
    from farmmap import context as CTX
    v = np.array([-0.2, 0.0, 0.45, 0.9, np.nan])
    back = CTX.decode(CTX.encode(v, "ndvi_h1"), "ndvi_h1")
    assert np.allclose(back[:4], v[:4], atol=0.005) and np.isnan(back[4])
    f = _fake_context(work, tmp_path / "ctx" / "dea.tif")
    with rasterio.open(work.paths.mosaic) as s:
        t = s.transform
    arr = CTX.read([f], work.aoi.crs, t, (64, 64))
    assert arr.shape == (2, 64, 64) and (arr > 0).all()
    assert abs(CTX.decode(arr[1], "ndvi_h1").mean() - 0.5) < 0.01


def test_pipeline_with_context_bands(work, tmp_path):
    """Context bands become extra chip channels, the model records them, predict re-attaches them,
    and a different context.use is refused."""
    from farmmap import config as C
    _fake_context(work, tmp_path / "ctx" / "dea.tif")
    cfg = C.Cfg(work)
    cfg.paths = C.Cfg({**work.paths, "context": tmp_path / "ctx", "tiles": tmp_path / "tiles",
                       "models": tmp_path / "models", "outputs": tmp_path / "out"})
    cfg.context = C.Cfg({**work.context, "use": ["dea"]})
    idx = tiles.make_tiles(cfg)
    with rasterio.open(Path(cfg.paths.tiles) / "images" / f"{idx.id[0]}.tif") as s:
        assert s.count == 6 and s.descriptions[4:] == ("ctx:dea:crop_prob", "ctx:dea:ndvi_h1")
        assert (s.read()[4:] > 0).all()
    best = train.train(cfg, epochs=1)
    import torch
    meta = torch.load(best, map_location="cpu")["meta"]
    assert meta["in_channels"] == 6 and meta["image_bands"] == 4 and len(meta["context_bands"]) == 2
    prob = predict.predict(cfg, best)
    with rasterio.open(prob) as s:
        assert s.count == 3 and (s.read(1) > 0).mean() > 0.9
    # predict takes the context sources from the model, not from context.use
    cfg.context = C.Cfg({**work.context, "use": []})
    predict.predict(cfg, best, out_path=tmp_path / "p2.tif")
    _fake_context(work, tmp_path / "ctx" / "dea.tif", layers=("crop_prob",))  # a different stack is refused
    with pytest.raises(ValueError, match="context bands"):
        predict.predict(cfg, best, out_path=tmp_path / "p3.tif")
    _fake_context(work, tmp_path / "ctx" / "dea.tif")
    # optional post-filter: crop_prob rises west->east, so a high minimum drops the western farms
    cfg.polygonize = C.Cfg({**work.polygonize, "context_filter": {"band": "ctx:dea:crop_prob", "min": 50}})
    farms = polygonize.polygonize(cfg, prob, tmp_path / "out" / "f.gpkg")
    if len(farms):
        assert (farms.ctx_value >= 50).all()


def test_config_base_inheritance():
    cfg = config.load(ROOT / "config_nicfi.yaml")
    assert cfg.imagery.gsd_m == 4.77 and cfg.tiles.size_px == 256
    assert cfg.labels.keep.QA_FLAG == ["OK"]          # inherited from config.yaml
    assert cfg.paths.tiles.name == "tiles_nicfi" and cfg.paths.labels.name == "manual"


def test_infer_chunk_smaller_than_tile():
    """Edge chunks narrower than one tile (e.g. 192 px with 256 px tiles) are padded, predicted and cropped."""
    import torch
    net = torch.nn.Conv2d(4, 3, 1).eval()
    seen = []
    net.register_forward_hook(lambda m, i, o: seen.append(tuple(i[0].shape[-2:])))
    for shape in [(256, 192), (100, 300), (50, 60)]:
        img = np.random.default_rng(0).integers(1, 255, (4, *shape), dtype=np.uint8)
        out = predict._infer_chunk(net, img, tile=256, overlap=64, batch=4, dev="cpu", n_image=4)
        assert out.shape == (3, *shape)
        assert np.allclose(out.sum(axis=0), 1, atol=1e-4)  # probabilities everywhere
    assert set(seen) == {(256, 256)}  # the model only ever sees full tiles
