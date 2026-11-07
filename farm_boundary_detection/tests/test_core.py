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
