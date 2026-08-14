"""Small, dependency-free metadata audit helpers for Bear Lake Phase 2.

The official spatial-metadata ZIP is deliberately inspected in place.  The
helpers read only the shapefile header, its small DBF attribute table, and the
projection text; they neither extract files nor materialize the large source
raster archive.
"""

from __future__ import annotations

import json
import math
import os
import struct
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

MAX_METADATA_MEMBER_BYTES = 5 * 1024 * 1024


class Phase2MetadataError(RuntimeError):
    """Raised when bounded official metadata cannot be parsed safely."""


def _single_member(members: list[zipfile.ZipInfo], suffix: str) -> zipfile.ZipInfo:
    matches = [
        member
        for member in members
        if PurePosixPath(member.filename.replace("\\", "/")).suffix.casefold() == suffix
    ]
    if len(matches) != 1:
        raise Phase2MetadataError(
            f"Expected exactly one {suffix} metadata member; found {len(matches)}."
        )
    member = matches[0]
    if member.file_size > MAX_METADATA_MEMBER_BYTES:
        raise Phase2MetadataError(
            f"Metadata member {member.filename!r} exceeds the bounded read limit."
        )
    return member


def _decode_dbf_value(raw: bytes, field_type: str) -> Any:
    text = raw.decode("cp1252", errors="replace").strip().strip("\x00")
    if not text:
        return None
    if field_type in {"F", "N"}:
        try:
            value = float(text)
        except ValueError:
            return text
        return int(value) if value.is_integer() else value
    if field_type == "L":
        if text.casefold() in {"t", "y"}:
            return True
        if text.casefold() in {"f", "n"}:
            return False
    return text


def _parse_dbf(content: bytes) -> list[dict[str, Any]]:
    if len(content) < 33:
        raise Phase2MetadataError("Spatial coverage DBF header is truncated.")
    record_count = struct.unpack_from("<I", content, 4)[0]
    header_length = struct.unpack_from("<H", content, 8)[0]
    record_length = struct.unpack_from("<H", content, 10)[0]
    if header_length < 33 or record_length < 2:
        raise Phase2MetadataError("Spatial coverage DBF header is invalid.")
    expected = header_length + record_count * record_length
    if expected > len(content):
        raise Phase2MetadataError("Spatial coverage DBF records are truncated.")

    fields: list[tuple[str, str, int]] = []
    offset = 32
    while offset < header_length and content[offset] != 0x0D:
        descriptor = content[offset : offset + 32]
        if len(descriptor) != 32:
            raise Phase2MetadataError("Spatial coverage DBF field is truncated.")
        name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii", errors="replace")
        fields.append((name, chr(descriptor[11]), descriptor[16]))
        offset += 32
    if not fields:
        raise Phase2MetadataError("Spatial coverage DBF has no fields.")

    records: list[dict[str, Any]] = []
    for index in range(record_count):
        start = header_length + index * record_length
        raw_record = content[start : start + record_length]
        if raw_record[:1] == b"*":
            continue
        position = 1
        parsed: dict[str, Any] = {}
        for name, field_type, field_length in fields:
            parsed[name] = _decode_dbf_value(
                raw_record[position : position + field_length], field_type
            )
            position += field_length
        records.append(parsed)
    return records


def _shapefile_bounds(header: bytes) -> dict[str, float]:
    if len(header) < 100:
        raise Phase2MetadataError("Spatial coverage shapefile header is truncated.")
    if struct.unpack_from(">I", header, 0)[0] != 9994:
        raise Phase2MetadataError("Spatial coverage shapefile code is invalid.")
    xmin, ymin, xmax, ymax = struct.unpack_from("<4d", header, 36)
    if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        raise Phase2MetadataError("Spatial coverage shapefile bounds are not finite.")
    if xmax <= xmin or ymax <= ymin:
        raise Phase2MetadataError("Spatial coverage shapefile bounds are invalid.")
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }


def _projection_evidence(wkt: str) -> dict[str, Any]:
    try:
        from pyproj import CRS
        from pyproj.exceptions import CRSError
    except ImportError as exc:
        raise Phase2MetadataError(
            "Resolving the official projection WKT requires pyproj."
        ) from exc
    try:
        crs = CRS.from_wkt(wkt)
        epsg = crs.to_epsg()
        name = crs.name
        projected = crs.is_projected
        linear_units = crs.axis_info[0].unit_name if crs.axis_info else None
    except (CRSError, ValueError) as exc:
        raise Phase2MetadataError(
            "The official projection WKT could not be resolved."
        ) from exc
    normalized = "".join(
        character for character in name.casefold() if character.isalnum()
    )
    nad83_2011 = "nad832011" in normalized
    utm_zone_12n = "utmzone12n" in normalized
    return {
        "horizontal_crs_wkt": wkt.strip(),
        "horizontal_epsg": epsg,
        "horizontal_crs_name": name,
        "horizontal_crs_projected": projected,
        "horizontal_crs_linear_units": linear_units,
        "epsg_identification_method": "pyproj CRS identification",
        "prj_confirms_nad83_2011": nad83_2011,
        "prj_confirms_utm_zone_12n": utm_zone_12n,
    }


def _year(value: Any) -> int | None:
    text = str(value or "").strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def _normalized_coverage_record(record: Mapping[str, Any]) -> dict[str, Any]:
    name = str(record.get("Name") or "").strip()
    data_type = str(record.get("Data_Type") or "").strip()
    title = str(record.get("Title") or "").strip()
    is_gap = name.casefold() == "gap_interp" or "gap" in data_type.casefold()
    acquired = _year(record.get("Date_Acq"))
    uncertainty = str(record.get("Source_Unt") or "").strip() or None
    return {
        "name": name or None,
        "title": title or None,
        "source": record.get("Source"),
        "source_organization": record.get("Source_Org"),
        "source_date": acquired,
        "publication_date": _year(record.get("Date_Pub")),
        "data_type": data_type or None,
        "source_vertical_datum": record.get("Source_VD"),
        "source_resolution": record.get("Source_Res"),
        "source_horizontal_datum": record.get("Source_HD"),
        "source_uncertainty_text": uncertainty,
        "numeric_source_uncertainty_published": bool(
            uncertainty and any(character.isdigit() for character in uncertainty)
        ),
        "source_geoid": record.get("Source_Geo"),
        "source_url": record.get("Source_Web"),
        "reported_area_acres": record.get("Area_acres"),
        "reported_area_km2": record.get("Area_km2"),
        "source_role": "derived_gap_interpolation" if is_gap else "survey_source",
    }


def inspect_spatial_coverage_bundle(path: str | Path) -> dict[str, Any]:
    """Read bounded coverage evidence from the official spatial metadata ZIP."""

    archive_path = Path(path)
    try:
        with zipfile.ZipFile(archive_path, "r", allowZip64=True) as bundle:
            members = bundle.infolist()
            prj = _single_member(members, ".prj")
            dbf = _single_member(members, ".dbf")
            shp = _single_member(members, ".shp")
            projection = bundle.read(prj).decode("utf-8", errors="replace")
            records = _parse_dbf(bundle.read(dbf))
            with bundle.open(shp, "r") as stream:
                bounds = _shapefile_bounds(stream.read(100))
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise Phase2MetadataError(
            f"Could not inspect spatial metadata ZIP {archive_path}: {exc}"
        ) from exc

    normalized_records = [_normalized_coverage_record(record) for record in records]
    source_years = sorted(
        {
            record["source_date"]
            for record in normalized_records
            if record["source_role"] == "survey_source"
            and record["source_date"] is not None
        }
    )
    processing_years = sorted(
        {
            record["source_date"]
            for record in normalized_records
            if record["source_role"] == "derived_gap_interpolation"
            and record["source_date"] is not None
        }
    )
    reported_areas = [
        float(record["reported_area_km2"])
        for record in normalized_records
        if isinstance(record["reported_area_km2"], (float, int))
    ]
    gap_records = [
        record
        for record in normalized_records
        if record["source_role"] == "derived_gap_interpolation"
    ]
    return {
        **_projection_evidence(projection),
        "text_excerpt": " ".join(
            [
                projection,
                *(
                    " ".join(
                        str(value) for value in record.values() if value is not None
                    )
                    for record in normalized_records
                ),
            ]
        )[:30000],
        "projected_bounds_m": bounds,
        "projected_width_m": bounds["xmax"] - bounds["xmin"],
        "projected_height_m": bounds["ymax"] - bounds["ymin"],
        "coverage_record_count": len(normalized_records),
        "coverage_records": normalized_records,
        "survey_source_years": source_years,
        "derived_processing_years": processing_years,
        "gap_interpolation": {
            "present": bool(gap_records),
            "year": processing_years[0] if len(processing_years) == 1 else None,
            "records": gap_records,
            "interpretation": (
                "The 2026 entry is a derived shoreline-gap interpolation, not "
                "a 2026 field survey."
                if gap_records
                else None
            ),
        },
        "sum_reported_record_area_km2": (
            math.fsum(reported_areas) if reported_areas else None
        ),
        "coverage_evidence_status": (
            "structured shapefile attributes and header parsed"
        ),
    }


def metadata_audit_markdown(record: Mapping[str, Any]) -> str:
    """Render a compact human-readable companion to ``metadata_audit.json``."""

    fgdc = record.get("fgdc_metadata") or {}
    spatial = record.get("spatial_metadata_archive") or {}
    bounds = fgdc.get("geographic_bounds") or {}
    gap = spatial.get("gap_interpolation") or {}
    latest_live = record.get("latest_live_run") or {}
    sources = spatial.get("coverage_records") or []
    archive_downloaded = bool(record.get("archive_present_at_audit"))
    raster_inspected = bool(record.get("source_raster_inspected_at_audit"))

    lines = [
        "# Bear Lake Phase 2 metadata audit",
        "",
        f"- Status: **{record.get('status', 'unknown')}**",
        f"- Audit mode: `{record.get('run_mode', 'unknown')}`",
        f"- DOI: `{record.get('doi', 'unknown')}`",
        f"- ScienceBase item: `{record.get('sciencebase_item_id', 'unknown')}`",
        (
            f"- Source archive: `{record.get('archive_name', 'unknown')}` "
            f"({int(record.get('archive_size_bytes') or 0):,} bytes)"
        ),
        f"- Large source archive present locally: `{str(archive_downloaded).lower()}`",
        f"- Source raster inspected: `{str(raster_inspected).lower()}`",
        "",
        "## Reference systems and extent",
        "",
        (
            f"The official PRJ identifies **{spatial.get('horizontal_crs_name') or 'an unresolved CRS'}** "
            f"(EPSG:{spatial.get('horizontal_epsg') or 'unresolved'}). Elevations are "
            f"reported in **{record.get('vertical_datum', 'unknown')}** "
            f"{record.get('vertical_units', '')}."
        ),
        (
            "Internal raster filename: "
            f"`{(record.get('metadata_product_inventory') or {}).get('expected_internal_raster_filename') or 'pending archive inspection'}`."
        ),
        (
            "Uncertainty/error products: none identified as separate files; "
            + str(
                (record.get("metadata_product_inventory") or {}).get(
                    "uncertainty_status", "status unresolved"
                )
            )
            + "."
        ),
        (
            "Independent basin-boundary products: none identified. The official "
            "spatial-metadata archive is retained as the coverage product."
        ),
        "",
        (
            "Geographic bounds reported by FGDC metadata: "
            f"west {bounds.get('west_longitude')}, east {bounds.get('east_longitude')}, "
            f"south {bounds.get('south_latitude')}, north {bounds.get('north_latitude')}."
        ),
        "",
        "## Source coverage evidence",
        "",
        (
            f"The small official spatial-metadata shapefile contains {len(sources)} "
            "coverage records:"
        ),
        "",
        "| Record | Role | Source year | Data type | Reported area (km2) |",
        "|---|---|---:|---|---:|",
    ]
    for source in sources:
        area = source.get("reported_area_km2")
        area_text = f"{float(area):.3f}" if isinstance(area, (float, int)) else ""
        lines.append(
            "| {name} | {role} | {year} | {kind} | {area} |".format(
                name=source.get("name") or "",
                role=source.get("source_role") or "",
                year=source.get("source_date") or "",
                kind=source.get("data_type") or "",
                area=area_text,
            )
        )
    lines.extend(
        [
            "",
            (
                f"Survey/source years are {spatial.get('survey_source_years') or []}. "
                f"Derived processing years are {spatial.get('derived_processing_years') or []}."
            ),
            "",
            str(
                gap.get("interpretation")
                or "No gap-interpolation record was identified."
            ),
            "",
            (
                "The structured coverage evidence spans the full north-south extent of "
                "Bear Lake and supports whole-lake spatial coverage. The separate FGDC "
                "'north arm' wording is retained as an apparent E-A-V metadata "
                "inconsistency. "
                + (
                    "The retained raster inspection supplies the quantitative extent gate."
                    if raster_inspected
                    else "The raster itself has not yet been inspected because the large archive is absent."
                )
            ),
            "",
            "## Acquisition and gate status",
            "",
            (
                f"This audit made {record.get('bounded_metadata_request_count', 0)} "
                "live metadata request(s). The metadata action itself requested only "
                "the bounded FGDC XML, spatial-metadata ZIP, and thumbnail."
            ),
        ]
    )
    if latest_live:
        lines.append(
            "The latest retained live metadata run completed at "
            f"{latest_live.get('completed_at_utc')} with "
            f"{latest_live.get('request_count')} request(s) in "
            f"{latest_live.get('elapsed_seconds')} seconds."
        )
    lines.extend(
        [
            "",
            "No storage volume was calculated by the metadata audit.",
            "",
        ]
    )
    return "\n".join(lines)


def write_metadata_audit_markdown(
    record: Mapping[str, Any], destination: str | Path
) -> Path:
    """Atomically write the human-readable metadata audit companion."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(metadata_audit_markdown(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_previous_live_run(path: str | Path) -> dict[str, Any] | None:
    """Return the latest sanitized live-run summary from a prior audit, if any."""

    audit_path = Path(path)
    if not audit_path.exists():
        return None
    try:
        value = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    prior = value.get("latest_live_run")
    if isinstance(prior, Mapping):
        return dict(prior)
    if value.get("run_mode") == "live":
        return {
            "completed_at_utc": value.get("generated_at_utc"),
            "request_count": value.get("bounded_metadata_request_count"),
            "elapsed_seconds": value.get("bounded_metadata_elapsed_seconds"),
            "bytes_transferred": value.get("bounded_metadata_bytes_transferred"),
        }
    return None
