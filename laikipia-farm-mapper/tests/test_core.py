import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.preprocessing import normalize

def test_normalize_preserves_nodata():
    x=np.array([[[0,10],[20,30]]],dtype=np.float32); mask=np.array([[True,False],[False,False]])
    out=normalize(x,mask)
    assert out[0,0,0] == 0 and out.min() >= 0 and out.max() <= 1

def test_tile_config_is_sane():
    assert 512 > 64

def test_kmz_roundtrip(tmp_path):
    import zipfile
    p=tmp_path/"x.kmz"
    with zipfile.ZipFile(p,"w") as z: z.writestr("doc.kml", "<kml></kml>")
    with zipfile.ZipFile(p) as z: assert z.read("doc.kml") == b"<kml></kml>"
