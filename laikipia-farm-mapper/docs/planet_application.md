# Automated Farm Boundary Mapping in Laikipia County, Kenya

**Applicant:** [Your name], [degree / programme], African Leadership University
**Supervisor / host organisation:** [name, organisation]
**Project repository:** https://github.com/mangaorphy/Farm_Digitisation
**Use:** non-commercial academic research

## Summary

Farm boundaries in Laikipia County, Kenya, are currently digitised by hand in Google Earth Pro, one polygon at
a time. This project builds an open-source pipeline that maps every farm automatically. Satellite imagery goes
into a deep-learning segmentation model (U-Net), and the model's output is converted into one polygon per farm.
The results are delivered as GIS layers (GeoPackage and KMZ) for review and correction. The work focuses on the
smallholder farming belt along the West Laikipia Fence, where up-to-date farm maps are needed for land-use and
human–wildlife coexistence planning.

## Why PlanetScope imagery is needed

The pipeline currently runs on free 10 m Sentinel-2 composites. That resolution is too coarse for the farms in
this area. From 3,536 hand-traced farms in our reference data:

- the median farm is **0.35 ha** and **45 m wide**, only 4–5 Sentinel-2 pixels across;
- **61 %** of farms are smaller than 0.5 ha;
- **80 %** share a boundary with a neighbouring farm, so at 10 m adjacent farms merge into one block.

At 3 m, a median farm is about 15 pixels across and most field edges become visible. PlanetScope's near-daily
revisit also lets us choose dates in different seasons, when field boundaries contrast most with the land
around them.

## Objectives

1. Train and evaluate a farm-boundary model on PlanetScope imagery, using more than 14,000 existing parcel
   outlines as training labels.
2. Measure how much accuracy improves over the 10 m Sentinel-2 baseline, using object-level precision, recall
   and F1 on held-out test areas, reported by farm size.
3. Produce a reviewed farm-boundary layer for the study area.
4. Publish the code and methods openly so they can be reused in other smallholder regions.

## Study area and imagery requested

| Item | Detail |
|---|---|
| Area | West Laikipia Fence corridor (+5 km buffer) and training sites, Laikipia County, Kenya (lon 36.2–37.0 E, lat 0.15 S – 0.85 N) |
| Size | about 3,200 km² |
| Product | PSScene, 4- or 8-band surface reflectance (orthorectified) |
| Period | 2025–2026: 2–4 cloud-free dates covering the long rains, short rains and dry season |
| Volume | downloads kept within the program's monthly quota, starting with the training sites and test areas |

## Method

1. **Imagery:** download, orthorectify and mosaic the PlanetScope scenes onto one grid (UTM 37N).
2. **Labels:** check that the existing parcel outlines line up with visible edges in the imagery, and correct
   any offsets.
3. **Model:** train a U-Net with a pretrained encoder to classify each pixel as farm interior, farm boundary
   or background. Multi-date Sentinel-2 layers from Digital Earth Africa are added as extra inputs.
4. **Polygons:** split farm interiors into individual farms (watershed), then simplify and clean the polygons.
5. **Evaluation:** score the polygons on 2 km test blocks never used in training, with objects matched at
   IoU ≥ 0.5, and compare against the Sentinel-2 baseline.

## Current progress

- The pipeline is complete and tested end to end: imagery preparation, registration check, training,
  prediction, polygons and evaluation.
- A 10 m Sentinel-2 composite of the full study area (6,450 km²) has been downloaded and processed. It is the
  baseline that PlanetScope results will be compared against.
- Training labels are ready: 14,822 usable parcel outlines, including 3,536 hand-traced reference farms.

## Expected outputs and data use

- A farm-boundary dataset for the study area, with an accuracy report by farm size.
- Open-source code and a written report or manuscript describing the method and results.
- Planet imagery will be used only for this non-commercial research. Raw imagery will not be redistributed.
  Only derived products (farm polygons, accuracy statistics, figures) will be shared, with Planet credited as
  the imagery source.
