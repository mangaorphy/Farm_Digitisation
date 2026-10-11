#!/usr/bin/env python3
try:
    import ee
except ImportError as exc:
    raise SystemExit("Earth Engine API is missing. Install requirements.txt first.") from exc
ee.Authenticate()
ee.Initialize()
print("Google Earth Engine authentication and initialization succeeded.")
