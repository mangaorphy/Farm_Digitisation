# No-farm / complete areas (optional, strongly recommended)

The farm KMZ files only say where farms ARE. Away from digitised farms the model is never told "this is not a
farm", which is why it draws loose shapes over bush, riverine vegetation and ranch land.

To fix that, draw a few large polygons in Google Earth Pro over land where there are **no farms at all** (bush,
forest, ranch, conservancy), or over blocks where **every** farm is already digitised. Save them as one file,
`no_farm_areas.kmz`, in this folder, commit and push. `notebooks/kaggle_zone.ipynb` picks it up automatically
(it sets `paths.complete_areas`).

Ten to twenty polygons of 0.5-3 km across, spread over the area, are enough. They must overlap the Planet imagery
(the area being mapped). Do not include any undigitised farm.
