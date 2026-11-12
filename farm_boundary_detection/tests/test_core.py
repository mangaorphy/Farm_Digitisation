"""Unit tests for the critical pieces. Run from the project folder: pytest -q"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from shapely.geometry import box

from src.data.ingestion import clean_geometry, read_kml_bytes, scene_key
from src.data.patch_extraction import window_starts
from src.data.rasterization import BoundarySpec, boundary_shapes
from src.data.splits import GeoSplit
from src.gis.polygon_reconstruction import segment
from src.models.multi_task_unet import MultiTaskResUNet, adapt_first_conv
from src.training.losses import MultiTaskLoss
from src.training.metrics import SegmentationMetrics, instance_scores
from src.utils.helpers import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs" / "config.yaml")


def test_planet_scene_key_pairs_image_and_udm2():
    a = scene_key(Path("20260615_073012_12_24a3_3B_AnalyticMS_SR_8b_clip.tif"))
    b = scene_key(Path("20260615_073012_12_24a3_3B_udm2_clip.tif"))
    assert a == b == "20260615_073012_12_24a3"


def test_kml_parser_keeps_attributes_and_skips_lines():
    kml = b"""<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Folder><name>F1</name>
      <Placemark><name>A</name><ExtendedData><SchemaData><SimpleData name="FARM_ID">X1</SimpleData>
      </SchemaData></ExtendedData><Polygon><outerBoundaryIs><LinearRing><coordinates>
      36,0,0 36.001,0,0 36.001,0.001,0 36,0,0</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
      <Placemark><LineString><coordinates>36,0 36.1,0.1</coordinates></LineString></Placemark>
      </Folder></Document></kml>"""
    g, skipped = read_kml_bytes(kml, "t.kml")
    assert len(g) == 1 and skipped == 1 and g.FARM_ID[0] == "X1" and g.folder[0] == "F1"


def test_invalid_polygon_is_repaired():
    from shapely.geometry import Polygon
    geom, status = clean_geometry(Polygon([(0, 0), (1, 1), (1, 0), (0, 1)]))  # bow-tie
    assert status == "repaired" and geom.is_valid and geom.area > 0


def test_boundary_width_radius_and_shared_edge():
    """Two farms sharing an edge: the shared boundary is present, interiors are not eroded."""
    from rasterio.features import rasterize
    gsd = 3.0
    t = from_origin(0, 60, gsd, gsd)
    a, b = box(0, 0, 30, 60), box(30, 0, 60, 60)
    for width, expect_mode in ((3.0, "line"), (6.0, "buffer")):
        spec = BoundarySpec.from_config(width, gsd)
        assert spec.mode == expect_mode and spec.buffer_radius_m == width / 2
        bnd = rasterize([(g, 1) for g in boundary_shapes([a, b], spec)], out_shape=(20, 20), transform=t,
                        all_touched=spec.mode == "line")
        interior = rasterize([(a, 1), (b, 1)], out_shape=(20, 20), transform=t)
        assert interior.sum() == 400  # nothing erased
        mid_row = bnd[10]
        assert mid_row[9] or mid_row[10]  # the shared edge at x = 30 m (between columns 9 and 10)
        width_px = mid_row[7:13].sum()
        assert 1 <= width_px <= 2 + (width >= 6)


def test_window_starts_cover_whole_length():
    assert window_starts(1000, 512, 256) == [0, 256, 488]
    assert window_starts(300, 512, 256) == [0]


def test_split_patches_never_straddle():
    s = GeoSplit("EPSG:32637", 1000, {"train": 0.6, "val": 0.2, "test": 0.2}, 1,
                 {(0, 0): "train", (1, 0): "val", (0, 1): "train"})
    assert s.split_for_footprint(box(100, 100, 400, 400)) == "train"
    assert s.split_for_footprint(box(900, 100, 1200, 400)) is None          # train / val
    assert s.split_for_footprint(box(100, 900, 400, 1200)) == "train"       # train / train
    assert s.split_for_footprint(box(0, 0, 1000, 1000)) == "train"          # touching edges only


def test_segment_separates_adjacent_farms_by_boundary():
    """Two touching farms with a 1-px boundary line between them become two instances; plain connected
    components of the interior would give one."""
    from scipy import ndimage as ndi
    p_int = np.zeros((40, 40), np.float32)
    p_int[5:35, 5:35] = 0.95
    p_bnd = np.full((40, 40), 0.02, np.float32)
    p_bnd[5:35, 20] = 0.9
    pc = load_config(ROOT / "configs" / "config.yaml").postprocess
    lab = segment(p_int, p_bnd, pc, 3.0)
    assert len(np.unique(lab[lab > 0])) == 2
    assert ndi.label(p_int > 0.5)[1] == 1


def test_loss_ignores_invalid_pixels(cfg):
    torch.manual_seed(0)
    loss_fn = MultiTaskLoss(cfg)
    b = {"interior_mask": (torch.rand(2, 1, 32, 32) > 0.5).float(), "boundary_mask": (torch.rand(2, 1, 32, 32) > 0.9).float(),
         "valid_mask": torch.ones(2, 1, 32, 32), "weight_map": torch.ones(2, 1, 32, 32)}
    b["valid_mask"][:, :, :, 16:] = 0
    il, bl = torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)
    l1, _ = loss_fn(il, bl, b)
    il2, bl2 = il.clone(), bl.clone()
    il2[..., 16:] = 50.0
    bl2[..., 16:] = -50.0
    l2, _ = loss_fn(il2, bl2, b)
    assert torch.allclose(l1, l2)


def test_boundary_tolerance_metric():
    gt = torch.zeros(1, 1, 20, 20)
    gt[..., 10] = 1
    pred_logits = torch.full((1, 1, 20, 20), -10.0)
    pred_logits[..., 11] = 10.0  # one pixel off
    batch = {"interior_mask": torch.zeros_like(gt), "boundary_mask": gt, "valid_mask": torch.ones_like(gt)}
    m = SegmentationMetrics(0.5, tolerance=1)
    m.update(torch.zeros_like(gt), pred_logits, batch)
    r = m.compute()
    assert r["boundary_f1"] == pytest.approx(1.0) and r["boundary_iou"] == 0.0


def test_instance_scores_counts_merge():
    gt = np.zeros((10, 30), np.int32)
    gt[:, :10], gt[:, 10:20], gt[:, 20:] = 1, 2, 3
    merged = (gt > 0).astype(np.int32)  # one region over three farms: IoU 1/3 with each
    s = instance_scores(gt, merged, np.ones_like(gt, bool))
    assert s["gt"] == 3 and s["matched"] == 0 and s["merged_predictions"] == 1
    s2 = instance_scores(gt, gt, np.ones_like(gt, bool))
    assert s2["matched"] == 3


def test_model_shapes_and_first_conv_adaptation():
    conv = torch.nn.Conv2d(3, 64, 7, 2, 3, bias=False)
    new = adapt_first_conv(conv, 8, [5, 3, 1])
    assert new.weight.shape == (64, 8, 7, 7)
    assert torch.allclose(new.weight[:, 5], conv.weight[:, 0] * 3 / 8)  # red filter on the red band
    net = MultiTaskResUNet(8, "resnet18", pretrained=False, decoder_channels=(64, 32, 16, 16)).eval()
    with torch.no_grad():
        i, b = net(torch.randn(2, 8, 70, 90))
    assert i.shape == b.shape == (2, 1, 70, 90)


def test_dataset_augmentation_keeps_alignment_and_binary(tmp_path):
    from src.data.dataset import FarmPatchDataset, Normalizer
    for sub in ("images", "interior_masks", "boundary_masks", "valid_masks"):
        (tmp_path / sub).mkdir()
    m = np.zeros((64, 64), np.uint8)
    m[10:30, 5:50] = 1
    img = np.stack([m * 1000 + 100] * 4).astype(np.uint16)
    t = from_origin(0, 64, 1, 1)
    def w(sub, arr, dtype):
        arr = arr if arr.ndim == 3 else arr[None]
        with rasterio.open(tmp_path / sub / "p.tif", "w", driver="GTiff", width=64, height=64, count=arr.shape[0],
                           dtype=dtype, crs="EPSG:32637", transform=t) as d:
            d.write(arr.astype(dtype))
    w("images", img, "uint16")
    w("interior_masks", m, "uint8")
    w("boundary_masks", m, "uint8")
    w("valid_masks", np.ones_like(m), "uint8")
    norm = Normalizer({"clip_low": [0] * 4, "clip_high": [2000] * 4, "mean": [500] * 4, "std": [500] * 4,
                       "nodata": 0})
    aug = {"hflip": 0.5, "vflip": 0.5, "rot90": True, "rotate_prob": 1.0, "rotate_max_deg": 20,
           "brightness": 0.0, "contrast": 0.0, "noise_std": 0.0}
    ds = FarmPatchDataset(tmp_path, ["p"], norm, aug, {"w0": 4, "sigma_px": 2})
    for _ in range(5):
        it = ds[0]
        mask = it["interior_mask"][0].numpy()
        assert set(np.unique(mask)) <= {0.0, 1.0}
        bright = it["image"][0].numpy() > 0.5
        valid = it["valid_mask"][0].numpy() > 0
        agree = (bright == (mask > 0))[valid & (np.abs(it["image"][0].numpy()) > 0.01)].mean()
        assert agree > 0.97


def test_instance_distance_peaks_inside_each_farm():
    from src.data.dataset import instance_distance
    inst = np.zeros((20, 40), np.int32)
    inst[2:18, 2:20], inst[2:18, 20:38] = 1, 2  # two touching farms
    d = instance_distance(inst)
    assert d.max() == pytest.approx(1.0) and d[inst == 0].max() == 0
    assert d[10, 19] < 0.3 and d[10, 20] < 0.3  # low on the shared edge
    assert d[10, 10] > 0.8 and d[10, 29] > 0.8  # high at each centre


def test_segment_with_distance_seeds_and_min_area():
    pc = load_config(ROOT / "configs" / "config.yaml").postprocess
    pc["seed_source"], pc["min_area_m2"] = "distance", 0
    p_int = np.zeros((40, 40), np.float32)
    p_int[5:35, 5:35] = 0.95
    p_bnd = np.full((40, 40), 0.1, np.float32)  # boundary invisible...
    from src.data.dataset import instance_distance
    inst = np.zeros((40, 40), np.int32)
    inst[5:35, 5:20], inst[5:35, 20:35] = 1, 2
    p_dist = instance_distance(inst)            # ...but the distance map still separates the two farms
    lab = segment(p_int, p_bnd, pc, 3.0, p_dist)
    assert len(np.unique(lab[lab > 0])) == 2
    pc["min_area_m2"] = 10000                   # 1 ha: both 0.4 ha farms removed
    assert segment(p_int, p_bnd, pc, 3.0, p_dist).max() == 0


def test_model_with_distance_head_returns_three_outputs():
    net = MultiTaskResUNet(4, "resnet18", pretrained=False, decoder_channels=(64, 32, 16, 16), distance_head=True).eval()
    with torch.no_grad():
        out = net(torch.randn(1, 4, 64, 64))
    assert len(out) == 3 and all(o.shape == (1, 1, 64, 64) for o in out)


def test_remove_overlaps():
    """Duplicates are dropped, small overlaps are trimmed off the lower-scoring farm, nothing overlaps after."""
    import geopandas as gpd
    from shapely.geometry import box
    from src.gis.polygon_reconstruction import remove_overlaps

    g = gpd.GeoDataFrame({"score": [0.9, 0.8, 0.7], "area_m2": [0, 0, 0]},
                         geometry=[box(0, 0, 100, 100), box(95, 0, 195, 100), box(1, 1, 99, 99)], crs="EPSG:32737")
    out = remove_overlaps(g, overlap=0.3, min_area_m2=200)
    assert len(out) == 2                                   # the 0.7 copy of farm 1 is dropped
    assert abs(out.area.iloc[1] - 9500) < 1                # 5 m strip cut off the 0.8 farm
    assert out.geometry.iloc[0].intersection(out.geometry.iloc[1]).area < 1e-6
    assert (out.area_m2 == out.area.round(1)).all()


def test_input_upsample_keeps_output_size():
    from src.models.multi_task_unet import MultiTaskResUNet
    import torch

    m = MultiTaskResUNet(8, "resnet18", pretrained=False, decoder_channels=(32, 16, 16, 8), input_upsample=2).eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 8, 64, 48))
    assert [tuple(o.shape) for o in out] == [(1, 1, 64, 48)] * 2


def test_fuse_probabilities_averages_where_data(tmp_path):
    """Two rasters on the same grid: overlap = mean, single coverage = that raster, nowhere = nodata."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from src.gis.zone import fuse_probabilities

    tr = from_origin(500000, 9990000, 3, 3)
    a = np.zeros((2, 10, 10), np.uint8); a[:, :, :6] = 101
    b = np.zeros((2, 10, 10), np.uint8); b[:, :, 4:8] = 201
    paths = []
    for name, arr in (("a", a), ("b", b)):
        p = tmp_path / f"{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", width=10, height=10, count=2, dtype="uint8", crs="EPSG:32737",
                           transform=tr, nodata=0) as d:
            d.write(arr)
        paths.append(p)
    out = fuse_probabilities(paths, tmp_path / "f.tif", "EPSG:32737", (500000, 9989970, 500030, 9990000), 3)
    f = rasterio.open(out).read()
    assert (f[:, :, :4] == 101).all() and (f[:, :, 4:6] == 151).all() and (f[:, :, 6:8] == 201).all()
    assert (f[:, :, 8:] == 0).all()


def test_zone_product_keeps_digitised_and_adds_only_new():
    import geopandas as gpd
    from shapely.geometry import box as _box

    def box(x0, y0, x1, y1):  # metres near Laikipia (UTM 37S)
        return _box(x0 + 260000, y0 + 9990000, x1 + 260000, y1 + 9990000)
    from src.gis.zone import zone_product

    crs = "EPSG:32737"
    zone = gpd.GeoDataFrame({"name": ["zone"]}, geometry=[box(0, 0, 1000, 1000)], crs=crs).to_crs(4326)
    tiles = gpd.GeoDataFrame({"tile": ["T01", "T02"], "name": ["a", "b"]},
                             geometry=[box(0, 0, 500, 1000), box(500, 0, 1000, 1000)], crs=crs).to_crs(4326)
    dig = gpd.GeoDataFrame({"FARM_ID": ["LKP-1"]}, geometry=[box(100, 100, 200, 200)], crs=crs)
    model = gpd.GeoDataFrame({"score": [0.9, 0.9, 0.5]}, geometry=[
        box(105, 105, 195, 195),          # the digitised farm again -> dropped
        box(190, 100, 290, 200),          # small overlap -> trimmed to x >= 200
        box(600, 600, 700, 700)], crs=crs)  # new
    out = zone_product(model, dig, zone, tiles)
    out = out.to_crs(crs)
    assert sorted(out.source) == ["Digitized", "Model", "Model"]
    assert out.loc[out.source == "Digitized", "farm_id"].iloc[0] == "LKP-1"
    trimmed = out[(out.source == "Model") & (out.tile == "T01")].geometry.iloc[0]
    assert abs(trimmed.bounds[0] - 260200) < 1e-3
    assert set(out[out.source == "Model"].review) == {"likely farm", "check"}
    assert out[out.source == "Model"].tile.tolist().count("T02") == 1


def test_read_zone_real_file():
    from pathlib import Path
    from src.gis.zone import read_zone

    p = Path(__file__).resolve().parents[1] / "data" / "zones" / "mutara_path.kmz"
    if not p.exists():
        return
    zone, tiles = read_zone(p)
    assert len(zone) == 1 and len(tiles) == 11 and tiles.tile.iloc[0] == "T01"
    assert 90e6 < zone.to_crs(zone.estimate_utm_crs()).area.iloc[0] < 97e6


def test_zone_product_drops_blobs_not_digitised():
    import geopandas as gpd
    from shapely.geometry import box as _box
    from src.gis.zone import zone_product

    def box(x0, y0, x1, y1):
        return _box(x0 + 260000, y0 + 9990000, x1 + 260000, y1 + 9990000)

    crs = "EPSG:32737"
    zone = gpd.GeoDataFrame({"name": ["zone"]}, geometry=[box(0, 0, 3000, 3000)], crs=crs)
    tiles = gpd.GeoDataFrame({"tile": [], "name": []}, geometry=[], crs=crs)
    dig = gpd.GeoDataFrame({"FARM_ID": ["BIG"]}, geometry=[box(2000, 2000, 2900, 2900)], crs=crs)   # 81 ha, kept
    model = gpd.GeoDataFrame({"score": [0.9, 0.9, 0.9]}, geometry=[
        box(0, 0, 1000, 1000),            # 100 ha blob -> dropped
        box(1200, 0, 1210, 900),          # 10 x 900 m sliver, compactness ~0.03 -> dropped
        box(1500, 1500, 1600, 1600)], crs=crs)   # 1 ha farm -> kept
    out = zone_product(model, dig, zone, tiles, max_area_ha=10, min_compact=0.1).to_crs(crs)
    assert sorted(out.farm_id.str[:3]) == ["AUT", "BIG"]
    assert abs(out[out.source == "Model"].area.iloc[0] - 1e4) < 1


def test_ftp_unet_interface_and_probabilities():
    """The 3-class network is exposed as (farm, boundary) logits: P(farm) = P(interior) + P(boundary)."""
    import pytest
    import torch
    pytest.importorskip("segmentation_models_pytorch")
    from src.models.ftp_unet import FTPUnet, bgrn_indices

    assert bgrn_indices(8) == [1, 3, 5, 7] and bgrn_indices(4) == [0, 1, 2, 3]
    assert bgrn_indices(5, ["nir", "x", "red", "green", "blue"]) == [4, 3, 2, 0]
    m = FTPUnet(8, pretrained=False).eval()
    x = torch.rand(2, 8, 70, 50) * 0.3
    with torch.no_grad():
        farm, bnd = m(x)
        x4 = torch.nn.functional.pad(x[:, [1, 3, 5, 7]], (0, 14, 0, 26), mode="reflect")
        p = m.net(torch.cat([x4, x4], 1))[..., :70, :50].softmax(1)
    assert farm.shape == bnd.shape == (2, 1, 70, 50)
    assert torch.allclose(torch.sigmoid(farm)[:, 0], p[:, 1] + p[:, 2], atol=1e-5)
    assert torch.allclose(torch.sigmoid(bnd)[:, 0], p[:, 2], atol=1e-5)


def test_edt_seeds_split_two_farms_joined_by_a_gap():
    """Two squares whose boundary has a 2-px gap: plain seeds give one farm, edt seeds give two."""
    import numpy as np
    from src.gis.polygon_reconstruction import segment
    from src.utils.helpers import Config

    p_int = np.zeros((40, 70), np.float32); p_int[5:35, 5:65] = 0.95
    p_bnd = np.zeros((40, 70), np.float32); p_bnd[5:35, 35] = 0.95; p_bnd[19:21, 35] = 0.0   # gap in the fence
    base = dict(interior_threshold=0.5, seed_threshold=0.5, boundary_threshold=0.5, min_seed_px=10, min_area_m2=0)
    one = segment(p_int, p_bnd, Config({**base, "seed_source": "boundary"}), 3.0)
    two = segment(p_int, p_bnd, Config({**base, "seed_source": "edt", "edt_h": 2.0}), 3.0)
    assert one.max() == 1 and two.max() == 2
    assert two[20, 10] != two[20, 60]


def test_simplify_together_neighbours_never_cross():
    """Staircase edge shared by two farms: simplified together they still do not overlap; separately they do."""
    import geopandas as gpd
    import numpy as np
    import shapely
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from shapely.geometry import shape
    from src.gis.polygon_reconstruction import simplify_together
    from src.utils.helpers import Config

    if not hasattr(shapely, "coverage_simplify"):
        return
    lab = np.zeros((60, 60), np.int32)
    yy, xx = np.mgrid[0:60, 0:60]
    lab[(yy > 5) & (yy < 55) & (xx > 5) & (xx < 55)] = 1
    lab[(lab == 1) & (xx > 10 + 0.7 * yy)] = 2                    # diagonal (staircase) shared edge
    tr = from_origin(260000, 9990180, 3, 3)
    geoms = [shape(g) for g, v in shapes(lab, mask=lab > 0, transform=tr, connectivity=4)]
    gdf = gpd.GeoDataFrame({"score": [0.9, 0.8]}, geometry=geoms, crs="EPSG:32737")
    pc = Config({"coverage_simplify": True, "coverage_tolerance_m": 4.0, "simplify_m": 3.0})
    out = simplify_together(gdf.copy(), pc, 3.0)
    assert out.geometry.iloc[0].intersection(out.geometry.iloc[1]).area < 1e-6
    assert sum(len(g.exterior.coords) for g in out.geometry) < 0.5 * sum(len(g.exterior.coords) for g in geoms)
    assert abs(out.area.sum() - gdf.area.sum()) < 0.02 * gdf.area.sum()
    sep = gdf.geometry.simplify(3.0)
    assert sep.iloc[0].intersection(sep.iloc[1]).area > 1.0        # the old way: neighbours cross
