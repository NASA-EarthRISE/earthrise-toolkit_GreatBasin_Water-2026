# Bear Lake Hydrometry Phase 2 QGIS layers

Phase 2 is a validation-gated Bear Lake prototype. It combines the official
USGS surveyed topobathymetric DEM (2002 bathymetry and 2016/2018 lidar), Bear
Lake observations in NAVD88, and the official elevation--area--volume table.
It does not add a QGIS plugin tab or change classification behavior.

## Layer inventory

The completed numerical gate selected the 5 m quantitative grid. Load its 15
GeoTIFFs from
`data/hydrometry/phase2/bear_lake/` and the named layers in
`bear_lake_phase2.gpkg`. Reference rasters are
`bear_lake_topobathy_working.tif`, `bear_lake_basin_mask.tif`, and
`bear_lake_bathymetry_coverage.tif`. State rasters include current, recent-low,
recent-high, year-ago, and historical-median depth and inundation. Change
rasters compare latest with the real observation nearest one year earlier.

`bear_lake_basin_mask.tif` retains a stable compatibility filename, but it is
the validated **quantitative processing-domain mask**, not an independently
surveyed or official basin boundary. No independent official basin mask is
available in Phase 2, so its included domain currently coincides with valid
topobathymetry coverage. Its codes are 1 inside the processing domain, 0
outside that domain where source coverage exists, and 255 unsupported/nodata.
The separate coverage raster uses 1 for supported source coverage and 0 for
unsupported coverage (255 remains its declared nodata value).

State inundation rasters use 0 for supported dry cells, 1 for connected water,
and 255 for unsupported/nodata cells. Depth rasters use zero for supported dry
cells and their declared floating-point nodata value for unsupported cells.
Water-surface, shoreline, and contour vectors are derived only within this
supported processing domain.

The GeoPackage layers are `shoreline_current`, `shorelines_selected`,
`shoreline_change_latest_vs_year_ago`, `depth_contours_current`,
`water_surface_current`, and `water_surfaces_selected`. Quantitative geometries
retain holes/islands and unsimplified raster boundaries.

Hydraulic connectivity remains the configured 8-neighbour raster rule. At
diagonal pixel contacts, its raw polygon rings can be invalid under OGC vector
topology; those rings are repaired with Shapely `make_valid`, which represents
point-touching water as valid multipart geometry. Every repaired geometry must
still match the source water-pixel area within the documented numeric check.

## CRS, datum, and styling

All quantitative outputs use the source projected horizontal CRS,
NAD83(2011) / UTM zone 12N (EPSG:6341), with metre horizontal units. Elevation,
stage, and depth use metres relative to NAVD88; this is not an ellipsoidal
height. Do not assign a different vertical datum merely because another layer
uses the same numeric unit.

In QGIS 3.44:

1. Add the GeoTIFFs with **Layer > Add Layer > Add Raster Layer** and the
   GeoPackage with **Add Vector Layer**.
2. Style depth as a sequential blue ramp beginning at zero; retain the raster
   nodata transparency and use one common scale when comparing states.
3. Style inundation change categorically: 0 unchanged dry (transparent or pale
   gray), 1 persistent water (blue), 2 newly inundated (green-blue), 3 exposed
   or lost water (orange), and 255 nodata/unsupported (transparent).
4. Style `depth_contours_current` by `depth_m`, using subdued thin lines and
   labels only at useful intervals. Contours do not imply better vertical
   precision than the surveyed/interpolated source.
5. Use `bear_lake_topobathy_working.tif` as the QGIS 3D raster terrain. Start
   with terrain tile resolution near the quantitative grid size and a modest
   vertical exaggeration (about 5--10×); disclose the final value in exports.
6. Render `water_surface_current` at the absolute `stage_m_navd88` field, not a
   constant height in a different datum. A PolygonZ may be available when the
   local vector writer supports it; otherwise use data-defined extrusion or Z
   translation from that field.
7. Configure temporal properties on `start_datetime`/`end_datetime`; use
   `frame_order` to keep the selected states deterministic. These attributes
   prepare manual QGIS animation and are not plugin integration.
8. Export a 3D scene image from the 3D Map View only after matching extent,
   camera, exaggeration, and layer ranges across states.

A QGIS camera animation moves the viewpoint through a static scene. A temporal
water-level animation holds the camera fixed while changing connected
inundation, shoreline, depth, area, and storage. Do not substitute the former
for the latter.

## Quantitative boundary and warnings

The source grid is the unchanged 0.5 m official raster; scalar validation is
windowed at source resolution. The quantitative spatial grid selected by the
completed 5/10/30 m benchmark is 5 m. A further-decimated, approximately 60 m
rendering grid is only for 3D/GIF/HTML display and must not be used for reported
area or volume.

The bathymetry is based on 2002 sonar and interpolation, while shoreline lidar
is from 2016 and 2018. The source reports no formal product-wide uncertainty
bound; interpolation and survey age therefore remain material limitations.
The workflow uses `numerically_consistent`, rather than `validated`, when only
source-method consistency and grid convergence are established. The official
FGDC entity description also contains a retained “north arm of Bear Lake”
wording that appears inconsistent with the release title and whole-lake extent.
It is not silently reinterpreted.

The policy keeps exact numeric reproduction separate from source-method
consistency. The completed source-method gate passed, but exact numeric
reproduction did not. Its one area-precision exception is `table_grid_005` at
1749.43 m NAVD88: official area 1,242,850 m² versus raster area 1,242,700 m².
The 150 m² difference exceeds the exact numeric precision envelope; storage
remains inside its envelope. The stage's 0.037957 m equal-area-radius difference
and the 0.058642 m maximum across all stages are below the 0.707107 m native-cell
diagonal used by the source-method gate. Equal-area-radius displacement is an
area-equivalent comparison scale, not a measured shoreline offset or a bound on
local shoreline-position error.

The exact numeric exception, strict storage result, gate rule, and scientific
warning remain in the hashed `validation_policy.json`. The labeled
current-depth figure, static 3D views, GIF frames, and interactive HTML visibly
repeat the source-method status and exact numeric exception count. Figure notes
summarize that evidence; the full rule and warning remain in the hashed policy
and provenance. Text-free figure variants intentionally retain geometry while
omitting all labels.

If `validation_policy.json` reports `failed`, spatial layers are diagnostic
only and polished 3D, GIF, and interactive products must not be produced.
