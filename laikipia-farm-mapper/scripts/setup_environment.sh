#!/usr/bin/env bash
set -euo pipefail

# Rasterio/GDAL are native libraries. Conda-forge supplies compatible macOS
# binaries; a plain pip install may otherwise try (and fail) to compile GDAL.
if command -v conda >/dev/null 2>&1; then
  conda env create --file environment.yml 2>/dev/null || conda env update --file environment.yml --prune
  echo "Environment ready. Run: conda activate laikipia-farm-mapper && python scripts/authenticate_earth_engine.py"
else
  echo "Conda was not found. Install Miniforge, then re-run this script." >&2
  echo "https://github.com/conda-forge/miniforge" >&2
  exit 1
fi
