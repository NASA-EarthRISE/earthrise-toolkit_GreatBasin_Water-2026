# Hydrometry Phase 2: Bear Lake topobathymetry

Phase 2 is a Bear Lake-only, validation-gated prototype. It combines the
official USGS surveyed topobathymetric elevation model, the Phase 1 Bear Lake
NAVD88 observations, and the official elevation--area--volume (E-A-V) table.
It does not change the QGIS plug-in, classification thresholds, or Phase 1
hydrometry products.

## Scientific boundary

The workflow keeps three grids distinct:

- **Source grid:** the unchanged official 0.5 m raster, processed by windows
  for scalar area and storage checks.
- **Quantitative spatial grid:** a selected 5, 10, or 30 m grid used for depth,
  connected inundation, shoreline, and change products.
- **Rendering grid:** a further-decimated display grid used only for 3D, GIF,
  and HTML output.

Rendering never supplies authoritative area or storage. Polished rendering is
blocked unless source and working-grid results pass the persisted consistency
policy against the official E-A-V relation. Because the release does not
publish a product-wide uncertainty model, an evidence-supported source pass is
called source-method consistent, not exact reproduction or a formal
uncertainty-bounded validation. The underlying policy model retains its
`numerically_consistent` status label for compatibility; the policy JSON also
records the more specific source-method status and exact-numeric diagnostics.

## Inputs and provenance

The configured official source is DOI `10.5066/P14KZS2D`, ScienceBase item
`698226f4b66b018d00102e42`. Phase 1 remains authoritative for the Bear Lake
stage record and E-A-V parsing/interpolation. Source binaries and generated
outputs remain under ignored `data/hydrometry/phase2/` and `outputs/` paths.

The official metadata describe:

- 2002 interpolated bathymetry;
- 2016 and 2018 Quality Level 1 lidar;
- 2026 shoreline-gap interpolation/production, which is not a new field
  survey;
- NAD83(2011) / UTM zone 12N (EPSG:6341), metre horizontal units;
- NAVD88/GEOID12B elevations in metres;
- a 0.5 m advertised source grid; and
- CC0 licensing.

The FGDC record contains a retained “north arm of Bear Lake” phrase. The
release title, thumbnail, geographic bounds, and spatial coverage metadata
indicate whole-lake coverage, so the phrase is reported as an apparent metadata
inconsistency rather than silently reinterpreted.

## Staged commands

Use the repository environment explicitly in PowerShell:

```powershell
& "C:\Code\Envs\lake-scaling\python.exe" `
  -m regional_analysis.hydrometry.phase2_bear_lake `
  --config "configs\hydrometry_pilots.yaml" `
  --metadata-only
```

Metadata-only mode never requests the 1.34 GB raster archive. An explicit
zero-network replay is:

```powershell
& "C:\Code\Envs\lake-scaling\python.exe" `
  -m regional_analysis.hydrometry.phase2_bear_lake `
  --config "configs\hydrometry_pilots.yaml" `
  --offline `
  --metadata-only
```

After reviewing the metadata and disk gate, the subsequent stages are:

```powershell
# One bounded anonymous download from the official ScienceBase Catalog file
# service. No login, API token, or temporary URL is required.
& "C:\Code\Envs\lake-scaling\python.exe" `
  -m regional_analysis.hydrometry.phase2_bear_lake `
  --config "configs\hydrometry_pilots.yaml" `
  --live `
  --download-only

# Every later calculation is local/offline.
& "C:\Code\Envs\lake-scaling\python.exe" `
  -m regional_analysis.hydrometry.phase2_bear_lake `
  --config "configs\hydrometry_pilots.yaml" `
  --offline `
  --extract-reference

& "C:\Code\Envs\lake-scaling\python.exe" `
  -m regional_analysis.hydrometry.phase2_bear_lake `
  --config "configs\hydrometry_pilots.yaml" `
  --offline `
  --validate-raster-model

& "C:\Code\Envs\lake-scaling\python.exe" `
  -m regional_analysis.hydrometry.phase2_bear_lake `
  --config "configs\hydrometry_pilots.yaml" `
  --offline `
  --build-spatial-products

& "C:\Code\Envs\lake-scaling\python.exe" `
  -m regional_analysis.hydrometry.phase2_bear_lake `
  --config "configs\hydrometry_pilots.yaml" `
  --offline `
  --render-only
```

The stable ScienceBase `manager/download/...` address is an HTML application,
not the ZIP response. The workflow instead constructs ScienceBase's official
anonymous Catalog file URL from the validated item ID and exact advertised
filename. That response currently omits a content length and range validators,
so the downloader uses the item metadata's byte count as a hard limit, requires
an exact final count, computes SHA-256, and streams every ZIP member through CRC
validation before atomic promotion. An interrupted Catalog transfer safely
restarts from byte zero rather than attempting an unverifiable range resume.

`--all` replays extraction through rendering offline. It does not download a
missing archive. `--render-only` verifies quantitative hashes before and after
rendering and cannot rewrite quantitative products.

Spatial provenance retains two complementary hash maps. The
`quantitative_artifact_sha256` map hashes exact file bytes and remains the
before/after render tamper guard for rasters, vectors, selected stages,
validation results, and the scientific policy. The
`quantitative_scientific_content_sha256` map is the offline replay comparison.
It canonicalizes only an explicit runtime allowlist: `inspection_seconds` in
the source-raster inspection, `created_at_utc` in the extraction manifest,
`source_processing_seconds` and `total_validation_seconds` in the validation
summary, and the `processing_seconds` column in the source-scalar,
working-scalar, and resolution-benchmark CSVs. All other JSON fields, CSV
columns and rows, and binary artifacts are hashed as scientific content.
Missing allowlisted fields fail closed. Runtime values remain truthful in the
files and remain covered by the raw-byte hash map even though they do not make
equivalent offline science appear different.

## Safety and numerical gates

The binary subsystem streams into a `.part` file, validates byte length and
SHA-256, and atomically installs only a complete archive. Resume is accepted
only when server validators and range behavior are compatible. ZIP paths,
symlinks, unsupported special members, ambiguity, and insufficient free space
fail closed. Only the selected source raster is extracted.

The raster gate requires a projected metre CRS, established NAVD88/metre
vertical reference, one band, explicit nodata or mask, no unexplained
scale/offset, finite values, the documented approximate resolution, and the
expected Bear Lake extent. The source is never rewritten and is never loaded
as one full NumPy array.

Scalar calculations align each requested stage to the source raster's floating
dtype and use `bed elevation <= effective stage` semantics. Both requested and
effective stages and the plane dtype are retained in the calculation evidence.
This implements an inclusive convention consistent with the documented
methodology: cells exactly on the plane are wet with zero depth, while islands
with bed elevations above the plane remain dry. The delivered official table
and TIFF are not bit-for-bit reproducible by local thresholding at every tested
stage, so exact numeric exceptions remain visible rather than being described
as reproduction. Accumulation uses float64 and block-window processing.
Multi-stage threshold events retain exact elevation sums; selected direct
windowed sums independently check that optimization.

Every local raster action also caps GDAL's process-global block cache at
128 MiB. The retained `peak_memory_estimate_bytes` value describes bounded
algorithmic NumPy working arrays; it is not a total-process peak and excludes
the GDAL block cache, Python/runtime memory, and operating-system file cache.
The cap is recorded separately in the validation summary, policy, and spatial
provenance.

Source validation deliberately keeps exact numeric reproduction separate from
source-method consistency. The stage-specific one-ULP/published-precision
envelopes remain in `eav_raster_validation.csv`; an area outside that envelope
is retained as a numeric area exception rather than erased or widened. The
method-consistency area gate converts official and independently calculated
areas to equal-area-circle radii and requires their radial difference to be no
greater than the diagonal of one native source cell. The diagonal is calculated
from the inspected raster pixel width and height, not from a configured or
working-grid resolution. Storage receives no spatial-resolution exception and
must pass its original stage-specific numeric precision envelope at every
stage.

Equal-area-radius displacement is a scalar, area-equivalent comparison scale.
It is not a measured shoreline offset and does not bound maximum local
shoreline-position error, especially along a complex shoreline. The CSV and
`validation_policy.json` therefore retain the exact numeric flags, numeric area
exception count and stage IDs, native-cell diagonal, maximum equal-area-radius
displacement, gate rule, and this warning. A two-ULP or half-table-step
tolerance is not substituted: an extra ULP does not reproduce the official
area and storage jointly, while half of the 0.01 m table interval is not an
uncertainty for a table-aligned official stage.

Working-resolution convergence is assessed with a rule declared independently
of the Bear Lake residuals. For each finer/coarser pair, the finer grid must
improve the median and 95th percentile of absolute relative error over the full
validation-stage grid, and the maximum absolute relative error over the fixed
observed-stage subset, for both area and storage. Individual stage regressions
are retained in `resolution_convergence.csv` as diagnostics, but a single
discrete shoreline-cell reversal does not override improvement of all six
distributional measures. No residual-derived percentage or widened tolerance
is used.

Quantitative grids require complete source coverage. Connected inundation uses
the configured interior seed and 8-neighbour connectivity; disconnected low
depressions are excluded and dry islands remain holes. Its gate is independent
of the resolution gate: connected-seed and all-valid inundation are compared on
the same selected working grid at the fixed observed stages. The difference
between their equal-area-circle radii must be no more than one selected-grid
cell diagonal, and the excluded storage fraction must be no greater than the
excluded area fraction. The latter requires excluded components to be no deeper
on average than the full inundated basin. Connected area and storage must also
increase monotonically with stage. Errors against the source grid remain in the
outputs as diagnostics, but do not conflate resampling with basin-method
selection.

Raster hydraulics retain 8-neighbour connectivity. If diagonal pixel contacts
produce self-touching rings under OGC vector topology, vectorization applies
Shapely `make_valid` and then rechecks polygon area against the exact raster
water-cell area; the quantitative raster mask itself is not altered.

If `validation_policy.json` reports `failed`, spatial products are not labeled
quantitative and static 3D, animation, and HTML are not produced.

## Products

The completed passing run wrote QGIS-ready tiled/DEFLATE GeoTIFFs, one
GeoPackage with shoreline/contour/water-surface layers, the selected-stage and
validation tables, 2D PNG/SVG/no-text figures, three fixed-camera 3D PNGs, an
observed-stage GIF, and a size-gated standalone Plotly HTML product.

Manual QGIS 3.44 loading and styling are documented in
[`hydrometry_phase2_qgis_layers.md`](hydrometry_phase2_qgis_layers.md).

## Current execution status

The real official archive was downloaded in one anonymous request (1,339,051,329
bytes in 150.807 seconds) and passed byte-count, SHA-256, and ZIP CRC checks. Its
SHA-256 is
`0d826349381dc8fbb06e68feed8d48c0cb8820959e35d96e5427d73c0f203e43`.
The archive inventory found one raster candidate and 1,422,112,114 total
uncompressed bytes. The selected TIFF is 1,153,021,924 bytes with SHA-256
`a34452308f0560982903eac0ffb8ef629ad4a65efaeed974ae336c5d38ee922c`.

Raster inspection passed for the 31,269 by 67,999, single-band float32 grid.
It has 0.5 m cells, 1,210,810,709 valid cells, NAD83(2011) / UTM zone 12N
(EPSG:6341) horizontal coordinates, and NAVD88 (EPSG:5703) / GEOID12B metre
elevations. Its measured valid elevation range is 1747.397583 to 1863.439941 m.

The source sweep evaluated 159 validation stages. The persisted policy status
is `numerically_consistent`, with the more precise scientific interpretation
`source_method_consistent_with_numeric_area_warning`. Source-method consistency
and strict storage-precision gates passed, but exact numeric reproduction did
not. The one retained area-precision exception is `table_grid_005` at 1749.43 m
NAVD88: official area 1,242,850 m² versus raster area 1,242,700 m², a 150 m²
difference. Storage at that stage differs by 34.339668 m³ and remains inside
its 151.725088 m³ precision envelope. The stage's equal-area-radius difference
is 0.037957 m; the maximum over all validation stages is 0.058642 m, below the
0.707107 m native-cell diagonal. These radius values are area-equivalent scalar
diagnostics, not measured local shoreline errors.

The independently declared resolution-convergence and connected-basin gates
both passed, selecting the 5 m quantitative grid over the next-coarser 10 m
candidate. Rendering was therefore allowed. The completed spatial inventory is
15 GeoTIFFs plus a GeoPackage containing six named vector layers. The completed
render inventory includes four labeled PNG/SVG figure pairs, four matching
no-text pairs, three fixed-camera 3D PNGs, an 18-frame GIF, a standalone
6,486,563-byte Plotly HTML scene, and figure notes. The associated animation
frame table brings the provenance render-product inventory to 23 files.
Rendering used a separate approximately 60 m display grid and 8× vertical
exaggeration; none of its values are authoritative for area or storage.

The final provenance status is `render_products_complete`. It records 35 raw
quantitative artifact hashes, 35 replay-stable scientific-content hashes, and
23 render-product hashes. The hashed validation policy and input-set digests
are respectively
`e0de959166b804868e45d468166dc2337884fcfc5c40b0598b9e8ee268b24f0d` and
`e35d0825f4e45ac0baa8208a996ab7ce8d01c6d0296fc9c04f7428a3c7f02b0d`.

A second complete `--offline --all` replay under the
`phase2-scientific-content-sha256-v2` contract made zero network requests and
finished in approximately 719.8 seconds. All 35 scientific-content hashes, all
23 render-product hashes, the logical validation-policy hash, and the bound
input-set digest were identical to the immediately preceding v2 baseline. Six
raw hashes changed only in the explicitly declared timing-bearing artifacts;
their scientific-content hashes remained identical.
