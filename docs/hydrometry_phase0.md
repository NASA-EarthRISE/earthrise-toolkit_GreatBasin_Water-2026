# Hydrometry Phase 0

Phase 0 is a research-only capability inventory for lake level, storage, and
bathymetry sources. It establishes typed records, verifies public source
metadata, and records the vertical-reference work that must precede any storage
or three-dimensional calculation. It does **not** calculate volume and it does
not change the QGIS plug-in.

## Pilot scope

The registry contains exactly five pilots:

| Profile | HydroLAKES ID | Purpose |
|---|---:|---|
| Great Salt | 67 | Separate north- and south-arm levels plus official topobathymetry |
| Lake Abert | 776 | Short NAVD88 lake-level record and a known bathymetry gap |
| Bear Lake | 778 | NAVD88 lake level plus official elevation-area-volume metadata |
| Lake Powell | 802 | RISE operational-source discovery plus official topobathymetry and capacity metadata |
| Flaming Gorge | 784 | RISE elevation, area, bank-storage metadata, discoverable total storage, and ACAP survey metadata |

HydroLAKES identifiers identify the lake profile only. They are never treated
as USGS monitoring-location IDs, RISE catalog IDs, or SWOT Prior Lake Database
(PLD) IDs. The pilot list is independent of both the QGIS threshold registry and
the regional threshold-transfer lake selection.

## Capability and evidence scales

Capability levels describe products, not source quality:

1. surface extent;
2. water level;
3. relative storage change;
4. absolute storage;
5. quantitative 3D representation.

Evidence classes are tracked separately:

| Class | Evidence |
|---|---|
| A | Direct authoritative storage |
| B | Stage plus an official capacity relation |
| C | Stage plus surveyed bathymetry |
| D | Empirical area-stage hypsometry |
| E | Satellite relative storage change |
| F | Modelled or coarse fallback |

HydroLAKES `Vol_total`, `Depth_avg`, and related global attributes are useful
context but can never promote a capability to Level 4. Capability decisions are
conservative: a configured identifier is not considered confirmed until its
returned title, location, variable, units, and datum are consistent with the
profile.

## Normalized records

Provider payloads remain inside adapters. The shared time-series record stores
the profile and lake identity, provider and record/item identifiers, normalized
variable, value and unit, UTC timestamp, quality/provisional/approval state,
height type and vertical reference, retrieval timestamp, source version and
public URL, raw parameter/statistic codes, and non-secret raw metadata.

The vertical-reference record separately carries:

- height type;
- source datum and units;
- datum epoch;
- gage-datum offset and its source;
- intended target datum;
- transformation method and uncertainty;
- normalization status and notes.

Allowed height types distinguish local gage height, project-datum elevation,
orthometric height, ellipsoidal height, geoid-referenced height, and unknown
height. Phase 0 retains source values. It can report that normalization was not
attempted, is blocked, or appears possible but is not implemented; it does not
perform a datum transformation. Equal units do not imply equal datums, a local
gage height cannot be normalized without a published offset, and a valid
negative elevation must not be mistaken for missing data.

## Source priorities

Source priority is explicit within each profile and there is no automatic
fallback. USGS probes use the modern Water Data for the Nation OGC API at
`https://api.waterdata.usgs.gov/ogcapi/`. They first resolve monitoring-location
metadata and discover time-series metadata before retrieving a bounded daily or
continuous series. Parameter codes such as `62614` and `62615` are assertions
against discovered metadata, not blind selectors. Gage height is not treated as
absolute lake elevation.

Reclamation probes use the RISE JSON:API at
`https://data.usbr.gov/rise/api`, including the required
`Accept: application/vnd.api+json` header. Catalog records are resolved before
items are inventoried, and elevation, area, and storage remain distinct
variables. The agency disclaimer and provisional status are retained.

The SWOT/Hydrocron adapter is deliberately bounded. Every pilot begins with a
null PLD ID and `requires_crosswalk`. HydroLAKES and SWOT PLD identifiers are
separate systems, no satellite availability is confirmed without a successful
ID/spatial crosswalk, and direct LakeSP and Hydrocron product versions can
differ. If a PLD ID is supplied later, the collection/product version must be
stored with the observations. Phase 0 does not add Earthdata authentication or
download LakeSP archives.

Official bathymetry and capacity entries are metadata-only records. The catalog
resolves public DOI/RISE landing pages and records title, provider, product type,
survey dates, coordinate and vertical references, resolution and published
elevation range, raster/table availability, license, evidence-class candidate,
and warnings. It does not download rasters or tables.

## Live, offline, and cache behavior

Choose network behavior explicitly:

```powershell
python -m regional_analysis.hydrometry.capability_inventory `
    --config configs\hydrometry_pilots.yaml `
    --live --days 90

python -m regional_analysis.hydrometry.capability_inventory `
    --config configs\hydrometry_pilots.yaml `
    --offline
```

`--live` permits bounded public requests and reuses valid cached responses;
`--refresh` forces fresh requests. `--offline` makes no network request. A
missing cache is recorded explicitly as a failed probe while the inventory
continues conservatively for the other sources. `WDFN_API_KEY` is optional
and, when present, is sent only as a request header/parameter supported by the
adapter; it is removed from public URLs, cache metadata, normalized records,
and reports.

Raw responses are stored below `data/hydrometry/cache/`; the normalized preview
is written to `data/hydrometry/pilot_timeseries_preview.csv`. Both locations are
ignored by Git. Each response has request
parameters with secrets removed, a public URL, retrieval time, status, content
type, SHA-256 digest, schema/version information where available, and raw and
normalized paths. Writes use a temporary file followed by replacement, so a
failed refresh cannot overwrite the last valid response. Retries are bounded
and apply only to HTTP 429 and transient 5xx responses.

## Pilot cautions

- Great Salt north and south arms are separate measurements and must not be
  merged as interchangeable observations. The configured gauges are NGVD29,
  whereas the official topobathymetry is NAVD88; arm handling and datum
  reconciliation block absolute storage.
- Lake Abert can support lake level but has no confirmed authoritative
  bathymetry/capacity relation, so Phase 0 keeps it below Level 4.
- Bear Lake can reach potential Level 4/5 only if both the NAVD88 level series
  and the official NAVD88 elevation-area-volume/topobathymetry metadata validate.
- Flaming Gorge can reach current Level 4 only through a validated direct RISE
  storage item. Future storage recomputation from ACAP is a separate check.
- Lake Powell record 2362 is an item-discovery starting point, not a license to
  guess item IDs. Level 4 remains blocked until operational elevation/storage
  items and their relationship to the NAVD88 capacity product are documented.

## Known limitations and next phase

Phase 0 does not implement stage-capacity interpolation, raster integration,
empirical hypsometry, uncertainty propagation, a SWOT PLD crosswalk, ICESat-2,
LakeSP downloads, 3D products, custom-AOI hydrometry, or any Earth Engine work.
The existing QGIS interface remains unchanged because source selection,
vertical-datum review, asynchronous network behavior, and uncertainty display
need a separate design and validation pass.

A defensible Phase 1 should first resolve the blocking vertical relationships
and operational series identity for each pilot, then implement one independently
testable storage pathway at a time with uncertainty and cross-source validation.
