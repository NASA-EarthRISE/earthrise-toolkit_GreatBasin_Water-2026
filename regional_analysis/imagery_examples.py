"""Create no-export Landsat/reference thumbnails and local change GIFs."""

from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

from .config import ConfigError, load_config
from .discover_lakes import DiscoveryError, initialize_earth_engine
from .expanded_lake_selection import DEFAULT_MANIFEST_PATH, selected_ids
from .expanded_transfer import expanded_output_paths
from .quick_examples import DEFAULT_CANDIDATE_PATH, DEFAULT_CONFIG_PATH
from .quick_sampling import (
    LakeSpec,
    QuickSamplingError,
    QuickSamplingSettings,
    build_buffered_lake_aoi,
    build_labelled_collection,
    build_landsat_collection,
    load_candidate_inventory,
    select_lake_specs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY_CACHE = REPOSITORY_ROOT / "data" / "regional" / "quick_transfer_cache"
DEFAULT_EXPANDED_CACHE = (
    REPOSITORY_ROOT / "data" / "regional" / "expanded_transfer_cache"
)
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data" / "regional"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "imagery_examples"

STATIC_PLAN = (
    (776, "OPERA"),
    (789, "OPERA"),
    (780, "JRC"),
    (798, "JRC"),
)
GIF_PLAN = ((67, "JRC"), (798, "JRC"), (780, "JRC"))
POINT_MASS_PLAN = ((67, "OPERA"), (798, "JRC"))
STATIC_ROLES = {
    (776, "OPERA"): "strong-transfer example",
    (789, "OPERA"): "weak OPERA-transfer example",
    (780, "JRC"): "weak JRC-transfer example",
    (798, "JRC"): "original-six threshold-location example",
}
PRESENTATION_SENSOR_PREFERENCE = {(798, "JRC"): "L8"}


class ImageryError(RuntimeError):
    """Raised when local planning or thumbnail retrieval is incomplete."""


@dataclass(frozen=True)
class SceneCollections:
    aoi: Any
    landsat: Any
    labelled: Any


def load_sample_cache(
    cache_dirs: tuple[Path, ...], reference: str, lake_id: int
) -> pd.DataFrame:
    """Load one unique local sample realization from prioritized cache roots."""

    candidates = [
        root / f"{reference.lower()}_{lake_id}.csv"
        for root in cache_dirs
        if (root / f"{reference.lower()}_{lake_id}.csv").is_file()
    ]
    if not candidates:
        raise ImageryError(f"No local {reference} sample cache for Hylak_id {lake_id}.")
    frames = [pd.read_csv(path) for path in candidates]
    required = {
        "Hylak_id",
        "reference",
        "acquisition_time",
        "sensor",
        "index_value",
        "label",
        "split",
        "block_id",
        "lon",
        "lat",
    }
    for path, frame in zip(candidates, frames, strict=True):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ImageryError(f"{path} is missing: {', '.join(missing)}.")
    first = (
        frames[0]
        .sort_values(["acquisition_time", "lon", "lat"], kind="mergesort")
        .reset_index(drop=True)
    )
    for path, other in zip(candidates[1:], frames[1:], strict=True):
        other = other.sort_values(
            ["acquisition_time", "lon", "lat"], kind="mergesort"
        ).reset_index(drop=True)
        compare = [
            "Hylak_id",
            "reference",
            "acquisition_time",
            "sensor",
            "index_value",
            "label",
            "lon",
            "lat",
        ]
        if len(other) != len(first) or not other[compare].equals(first[compare]):
            raise ImageryError(
                f"Conflicting cache realization for {reference} {lake_id}: {path}."
            )
    first["acquisition_date"] = pd.to_datetime(
        first["acquisition_time"], unit="ms", utc=True
    )
    return first


def select_representative_acquisitions(
    cache: pd.DataFrame,
    *,
    n_frames: int = 5,
    one_per_year: bool = True,
    months: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Select deterministic high-coverage experiment acquisitions."""

    if n_frames <= 0:
        raise ImageryError("n_frames must be positive.")
    required = {"acquisition_time", "acquisition_date", "sensor", "block_id"}
    missing = sorted(required.difference(cache.columns))
    if missing:
        raise ImageryError(
            "Cache is missing scene-selection columns: " + ", ".join(missing)
        )
    grouped = cache.groupby(["acquisition_time", "sensor"], as_index=False).agg(
        n_sample_rows=("index_value", "size"),
        n_blocks=("block_id", "nunique"),
        point_mass_rows=("index_value", lambda values: int(values.eq(1.0).sum())),
    )
    grouped["acquisition_date"] = pd.to_datetime(
        grouped["acquisition_time"], unit="ms", utc=True
    )
    grouped["year"] = grouped["acquisition_date"].dt.year
    if months is not None:
        if not months or any(month < 1 or month > 12 for month in months):
            raise ImageryError("months must contain calendar month numbers 1..12.")
        grouped = grouped[grouped["acquisition_date"].dt.month.isin(months)]
        if grouped.empty:
            raise ImageryError(
                "No cached acquisitions fall inside the requested seasonal window."
            )
    grouped = grouped.sort_values(
        ["n_sample_rows", "n_blocks", "acquisition_time"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    if one_per_year:
        grouped = grouped.drop_duplicates("year", keep="first")
        grouped = grouped.sort_values("year", kind="mergesort")
        if len(grouped) > n_frames:
            positions = np.linspace(0, len(grouped) - 1, n_frames).round().astype(int)
            grouped = grouped.iloc[np.unique(positions)]
    else:
        grouped = grouped.head(n_frames).sort_values("acquisition_time")
    return grouped.reset_index(drop=True)


def presentation_cache(
    cache: pd.DataFrame, lake_id: int, reference: str
) -> pd.DataFrame:
    """Apply an explicit sensor preference only to presentation scene choice."""

    sensor = PRESENTATION_SENSOR_PREFERENCE.get((lake_id, reference))
    if sensor is None:
        return cache
    preferred = cache[cache["sensor"].eq(sensor)].copy()
    if preferred.empty:
        raise ImageryError(
            f"No cached {sensor} presentation scenes for {reference} {lake_id}."
        )
    return preferred


def rank_point_mass_acquisitions(
    cache: pd.DataFrame, n_scenes: int = 1
) -> pd.DataFrame:
    """Rank actual experiment acquisitions by exact endpoint-row count."""

    if n_scenes <= 0:
        raise ImageryError("n_scenes must be positive.")
    endpoint = cache[cache["index_value"].eq(1.0)]
    if endpoint.empty:
        raise ImageryError("Cache contains no exact NDWI=1 acquisitions.")
    ranked = (
        endpoint.groupby(["acquisition_time", "sensor"], as_index=False)
        .agg(point_mass_rows=("index_value", "size"), n_blocks=("block_id", "nunique"))
        .sort_values(
            ["point_mass_rows", "n_blocks", "acquisition_time"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(n_scenes)
    )
    ranked["acquisition_date"] = pd.to_datetime(
        ranked["acquisition_time"], unit="ms", utc=True
    )
    return ranked.reset_index(drop=True)


def load_regional_threshold(
    data_dir: str | Path, reference: str, tier: str = "tier1"
) -> float:
    """Read the full-precision expanded pooled threshold without refitting."""

    path = expanded_output_paths(data_dir, reference, tier).regional
    if not path.is_file():
        raise ImageryError(f"Missing expanded regional table: {path}")
    table = pd.read_csv(path)
    if "regional_threshold" not in table.columns:
        raise ImageryError(f"{path} lacks regional_threshold.")
    values = (
        pd.to_numeric(table["regional_threshold"], errors="coerce").dropna().unique()
    )
    if len(values) != 1 or not np.isfinite(values[0]):
        raise ImageryError(f"{path} does not contain one stable regional threshold.")
    return float(values[0])


def build_scene_collections(
    ee_module: Any,
    hydrolakes_asset: str,
    lake: LakeSpec,
    reference: str,
    settings: QuickSamplingSettings,
) -> SceneCollections:
    """Build the same harmonized/matched collections used by sampling."""

    _feature, aoi = build_buffered_lake_aoi(
        ee_module, hydrolakes_asset, lake, settings.buffer_km
    )
    start, end = settings.reference_window(reference)
    landsat = build_landsat_collection(
        ee_module,
        aoi,
        start,
        end,
        cloud_cover_max=settings.cloud_cover_max,
    )
    labelled = build_labelled_collection(ee_module, aoi, reference, settings)
    return SceneCollections(aoi=aoi, landsat=landsat, labelled=labelled)


def scene_products(
    ee_module: Any,
    collections: SceneCollections,
    acquisition_time: int,
    sensor: str,
    threshold: float,
) -> dict[str, Any]:
    """Select one exact experiment scene and create presentation images."""

    time_filter = ee_module.Filter.eq("system:time_start", int(acquisition_time))
    sensor_filter = ee_module.Filter.eq("sensor", str(sensor))
    landsat = ee_module.Image(
        collections.landsat.filter(time_filter).filter(sensor_filter).first()
    )
    labelled = ee_module.Image(
        collections.labelled.filter(time_filter).filter(sensor_filter).first()
    )
    true_color = landsat.select(["red", "green", "blue"]).visualize(
        min=0.02,
        max=0.30,
        gamma=1.2,
    )
    label = labelled.select("label")
    reference_mask = label.unmask(2).visualize(
        min=0,
        max=2,
        palette=["E8E8E8", "0072B2", "FFFFFF"],
    )
    threshold_mask = (
        labelled.select("index_value")
        .gte(float(threshold))
        .updateMask(label.mask())
        .unmask(2)
        .visualize(min=0, max=2, palette=["E8E8E8", "009E73", "FFFFFF"])
    )
    ndwi = labelled.select("index_value").visualize(
        min=-0.35,
        max=1.0,
        palette=["6E3B1F", "E6C676", "F4F4F4", "56B4E9", "004488"],
    )
    return {
        "true_color": true_color.clip(collections.aoi),
        "reference": reference_mask.clip(collections.aoi),
        "threshold": threshold_mask.clip(collections.aoi),
        "ndwi": ndwi.clip(collections.aoi),
    }


def fetch_thumbnail(
    image: Any,
    region: Any,
    *,
    dimensions: int = 640,
    retries: int = 3,
    timeout: float = 120,
    session: Any = requests,
) -> Image.Image:
    """Download one ephemeral EE thumbnail URL with bounded retries."""

    if dimensions <= 0 or retries <= 0:
        raise ImageryError("dimensions and retries must be positive.")
    url = image.getThumbURL(
        {
            "region": region.bounds(),
            "dimensions": int(dimensions),
            "crs": "EPSG:3857",
            "format": "png",
        }
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"transient HTTP {response.status_code}", response=response
                )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if content_type and not (
                content_type.startswith("image/") or "octet-stream" in content_type
            ):
                raise ImageryError(
                    f"Thumbnail returned unexpected content type {content_type!r}."
                )
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except (OSError, requests.RequestException, ImageryError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise ImageryError(
        f"Thumbnail download failed after {retries} attempts: {last_error}"
    )


def _normalized_images(images: list[Image.Image]) -> list[Image.Image]:
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    return [
        image.resize((width, height), Image.Resampling.LANCZOS)
        if image.size != (width, height)
        else image
        for image in images
    ]


def save_static_panel(
    images: list[Image.Image],
    titles: list[str],
    output_path: str | Path,
    *,
    heading: str,
) -> Path:
    """Assemble aligned raster panels with restrained labels."""

    if len(images) != len(titles) or not images:
        raise ImageryError("Panel images and titles must be non-empty and aligned.")
    normalized = _normalized_images(images)
    fig, axes = plt.subplots(
        1,
        len(normalized),
        figsize=(4.2 * len(normalized), 4.6),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, image, title in zip(axes[0], normalized, titles, strict=True):
        axis.imshow(image)
        axis.set_title(title, fontsize=9.5)
        axis.set_axis_off()
    fig.suptitle(heading, fontsize=11)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def compose_gif_frame(
    left: Image.Image,
    right: Image.Image,
    *,
    heading: str,
    left_label: str = "Landsat true color",
    right_label: str = "Regional-threshold water mask",
) -> Image.Image:
    """Create a consistent two-panel RGB GIF frame."""

    left, right = _normalized_images([left, right])
    header = 54
    canvas = Image.new("RGB", (left.width + right.width, left.height + header), "white")
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    draw.text((10, 6), heading, fill="#111111", font=font)
    draw.text((10, 31), left_label, fill="#333333", font=small)
    draw.text((left.width + 10, 31), right_label, fill="#333333", font=small)
    return canvas


def save_gif(
    frames: list[Image.Image],
    output_path: str | Path,
    *,
    duration_ms: int = 900,
) -> Path:
    """Write an optimized, looping local GIF with no remote export."""

    if len(frames) < 2:
        raise ImageryError("A change GIF requires at least two frames.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalized_images([frame.convert("RGB") for frame in frames])
    normalized[0].save(
        path,
        save_all=True,
        append_images=normalized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return path


def _slug(name: str) -> str:
    return "_".join(
        part
        for part in "".join(
            char.lower() if char.isalnum() else " " for char in name
        ).split()
    )


def expected_imagery_paths(
    lakes_by_id: dict[int, LakeSpec],
    cache_dirs: tuple[Path, ...],
    output_dir: Path,
) -> tuple[Path, ...]:
    """Resolve the deterministic output inventory without touching EE."""

    expected: list[Path] = []
    for lake_id, reference in STATIC_PLAN:
        lake = lakes_by_id[lake_id]
        cache = presentation_cache(
            load_sample_cache(cache_dirs, reference, lake_id), lake_id, reference
        )
        scene = select_representative_acquisitions(
            cache,
            n_frames=1,
            one_per_year=False,
            months=(6, 7, 8, 9),
        ).iloc[0]
        date = scene["acquisition_date"].date().isoformat()
        expected.append(
            output_dir
            / f"static_{_slug(lake.lake_name)}_{reference.lower()}_{date}.png"
        )
    expected.extend(
        output_dir
        / f"change_{_slug(lakes_by_id[lake_id].lake_name)}_{reference.lower()}.gif"
        for lake_id, reference in GIF_PLAN
    )
    for lake_id, reference in POINT_MASS_PLAN:
        lake = lakes_by_id[lake_id]
        cache = presentation_cache(
            load_sample_cache(cache_dirs, reference, lake_id), lake_id, reference
        )
        scene = rank_point_mass_acquisitions(cache).iloc[0]
        date = scene["acquisition_date"].date().isoformat()
        expected.append(
            output_dir
            / f"point_mass_scene_{_slug(lake.lake_name)}_{reference.lower()}_{date}.png"
        )
    return tuple(expected)


def finalize_existing_imagery(
    lakes_by_id: dict[int, LakeSpec],
    cache_dirs: tuple[Path, ...],
    data_dir: Path,
    output_dir: Path,
    *,
    tier: str = "tier1",
) -> tuple[Path, ...]:
    """Validate a completed thumbnail set, write notes, and prune stale products."""

    expected = expected_imagery_paths(lakes_by_id, cache_dirs, output_dir)
    missing = [
        str(path) for path in expected if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise ImageryError(
            "Expected imagery products are missing: " + ", ".join(missing)
        )
    for path in expected:
        if path.suffix.lower() == ".gif":
            with Image.open(path) as image:
                if not 2 <= image.n_frames <= 10:
                    raise ImageryError(
                        f"{path} has an invalid frame count: {image.n_frames}."
                    )

    thresholds = {
        reference: load_regional_threshold(data_dir, reference, tier)
        for reference in ("OPERA", "JRC")
    }
    lines = [
        "# Imagery example notes",
        "",
        (
            "All scenes use the HydroLAKES polygon plus 5 km. These are "
            "qualitative, 8-bit Earth Engine thumbnail products selected from "
            "actual cached experiment acquisitions; they are not quantitative "
            "exports."
        ),
        "",
        "## Static analytical examples",
        "",
    ]
    for lake_id, reference in STATIC_PLAN:
        lake = lakes_by_id[lake_id]
        cache = presentation_cache(
            load_sample_cache(cache_dirs, reference, lake_id), lake_id, reference
        )
        scene = select_representative_acquisitions(
            cache,
            n_frames=1,
            one_per_year=False,
            months=(6, 7, 8, 9),
        ).iloc[0]
        date = scene["acquisition_date"].date().isoformat()
        filename = f"static_{_slug(lake.lake_name)}_{reference.lower()}_{date}.png"
        lines.append(
            f"- `{filename}`: **{STATIC_ROLES[(lake_id, reference)]}**; "
            f"{lake.lake_name} (Hylak_id {lake_id}), {scene['sensor']} on {date}, "
            f"{reference}, pooled Tier 1 threshold {thresholds[reference]:+.8f}. "
            "Panels show Landsat true color, matched reference water, and the "
            "threshold result restricted to reference-valid pixels."
        )
    lines.extend(["", "## Change GIFs", ""])
    for lake_id, reference in GIF_PLAN:
        lake = lakes_by_id[lake_id]
        cache = presentation_cache(
            load_sample_cache(cache_dirs, reference, lake_id), lake_id, reference
        )
        scenes = select_representative_acquisitions(
            cache,
            n_frames=5,
            one_per_year=True,
            months=(6, 7, 8, 9),
        )
        dates = ", ".join(
            scene.date().isoformat() for scene in scenes["acquisition_date"]
        )
        filename = f"change_{_slug(lake.lake_name)}_{reference.lower()}.gif"
        lines.append(
            f"- `{filename}`: **dynamic example**; {lake.lake_name} (Hylak_id "
            f"{lake_id}), {reference}, pooled threshold "
            f"{thresholds[reference]:+.8f}; one deterministic high-coverage "
            f"June-September cached acquisition per year ({dates})."
        )
    lines.extend(["", "## NDWI = 1 context", ""])
    for lake_id, reference in POINT_MASS_PLAN:
        lake = lakes_by_id[lake_id]
        cache = load_sample_cache(cache_dirs, reference, lake_id)
        scene = rank_point_mass_acquisitions(cache).iloc[0]
        date = scene["acquisition_date"].date().isoformat()
        filename = (
            f"point_mass_scene_{_slug(lake.lake_name)}_{reference.lower()}_{date}.png"
        )
        lines.append(
            f"- `{filename}`: {lake.lake_name} (Hylak_id {lake_id}), "
            f"{reference}, {date}; scene with {int(scene['point_mass_rows'])} "
            "cached exact-NDWI=1 samples. The true color, NDWI, reference, and "
            "prediction panels provide plausible water-scene context, not a "
            "pixel-location or causal map."
        )
    lines.extend(
        [
            "",
            "## Display and interpretation limits",
            "",
            (
                "True color uses fixed scaled-reflectance limits 0.02-0.30 and "
                "gamma 1.2. In masks, green is threshold-positive water, pale "
                "gray is valid non-water, white is reference-invalid or "
                "uncovered, and black is outside the AOI or Landsat footprint. "
                "OPERA and JRC retain different temporal windows."
            ),
            (
                "GIFs use five June-September scenes, not annual composites or "
                "water-area estimates. No Earth Engine batch export, Drive, GCS, "
                "or remote output asset was used."
            ),
        ]
    )
    notes = output_dir / "imagery_notes.md"
    notes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    retained = {path.resolve() for path in (*expected, notes)}
    for pattern in ("static_*.png", "point_mass_scene_*.png", "change_*.gif"):
        for stale in output_dir.glob(pattern):
            if stale.resolve() not in retained:
                stale.unlink()
    return (*expected, notes)


def run_imagery_plan(
    ee_module: Any,
    hydrolakes_asset: str,
    lakes_by_id: dict[int, LakeSpec],
    settings: QuickSamplingSettings,
    cache_dirs: tuple[Path, ...],
    data_dir: Path,
    output_dir: Path,
    *,
    tier: str = "tier1",
    dimensions: int = 560,
) -> tuple[Path, ...]:
    """Create the fixed small static/GIF/endpoint presentation set."""

    thresholds = {
        reference: load_regional_threshold(data_dir, reference, tier)
        for reference in ("OPERA", "JRC")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_cache: dict[tuple[int, str], SceneCollections] = {}
    sample_cache: dict[tuple[int, str], pd.DataFrame] = {}

    def context(
        lake_id: int, reference: str
    ) -> tuple[LakeSpec, pd.DataFrame, SceneCollections]:
        key = (lake_id, reference)
        lake = lakes_by_id[lake_id]
        if key not in sample_cache:
            sample_cache[key] = load_sample_cache(cache_dirs, reference, lake_id)
        if key not in collection_cache:
            collection_cache[key] = build_scene_collections(
                ee_module, hydrolakes_asset, lake, reference, settings
            )
        return lake, sample_cache[key], collection_cache[key]

    saved: list[Path] = []
    records: list[str] = [
        "# Imagery example notes",
        "",
        (
            "All scenes use the HydroLAKES polygon plus 5 km. Static panels and "
            "GIFs are qualitative thumbnail products selected from actual cached "
            "experiment acquisitions; thresholds are the exact expanded Tier 1 "
            "pooled values."
        ),
        "",
    ]
    for lake_id, reference in STATIC_PLAN:
        lake, cache, collections = context(lake_id, reference)
        cache = presentation_cache(cache, lake_id, reference)
        scene = select_representative_acquisitions(
            cache,
            n_frames=1,
            one_per_year=False,
            months=(6, 7, 8, 9),
        ).iloc[0]
        products = scene_products(
            ee_module,
            collections,
            int(scene["acquisition_time"]),
            str(scene["sensor"]),
            thresholds[reference],
        )
        images = [
            fetch_thumbnail(products[name], collections.aoi, dimensions=dimensions)
            for name in ("true_color", "reference", "threshold")
        ]
        date = scene["acquisition_date"].date().isoformat()
        path = (
            output_dir
            / f"static_{_slug(lake.lake_name)}_{reference.lower()}_{date}.png"
        )
        saved.append(
            save_static_panel(
                images,
                [
                    "Landsat true color",
                    f"{reference} reference",
                    "Regional NDWI mask\n(reference-valid pixels)",
                ],
                path,
                heading=(
                    f"{lake.lake_name} — {date} — {reference} "
                    f"threshold {thresholds[reference]:+.4f}"
                ),
            )
        )
        records.append(
            f"- `{path.name}`: **{STATIC_ROLES[(lake_id, reference)]}**, Hylak_id "
            f"{lake.hylak_id}; exact {scene['sensor']!s} experiment acquisition "
            f"on {date}; true color, matched {reference} binary reference, and the "
            f"expanded-{tier} pooled NDWI mask at threshold "
            f"{thresholds[reference]:+.8f}."
        )

    for lake_id, reference in GIF_PLAN:
        lake, cache, collections = context(lake_id, reference)
        cache = presentation_cache(cache, lake_id, reference)
        scenes = select_representative_acquisitions(
            cache,
            n_frames=5,
            one_per_year=True,
            months=(6, 7, 8, 9),
        )
        frames: list[Image.Image] = []
        dates: list[str] = []
        for scene in scenes.itertuples(index=False):
            products = scene_products(
                ee_module,
                collections,
                int(scene.acquisition_time),
                str(scene.sensor),
                thresholds[reference],
            )
            true_color = fetch_thumbnail(
                products["true_color"], collections.aoi, dimensions=dimensions
            )
            threshold = fetch_thumbnail(
                products["threshold"], collections.aoi, dimensions=dimensions
            )
            date = scene.acquisition_date.date().isoformat()
            dates.append(date)
            frames.append(
                compose_gif_frame(
                    true_color,
                    threshold,
                    heading=f"{lake.lake_name} - {date} - {scene.sensor}",
                    right_label=f"Threshold ({reference}-valid pixels)",
                )
            )
        path = output_dir / f"change_{_slug(lake.lake_name)}_{reference.lower()}.gif"
        saved.append(save_gif(frames, path))
        records.append(
            f"- `{path.name}`: **dynamic example**, Hylak_id {lake.hylak_id}; one "
            f"high-coverage cached {reference} experiment "
            f"June-September acquisition per year ({', '.join(dates)}), shown as "
            "true color and "
            f"expanded-{tier} pooled-threshold mask at "
            f"{thresholds[reference]:+.8f}."
        )

    for lake_id, reference in POINT_MASS_PLAN:
        lake, cache, collections = context(lake_id, reference)
        scene = rank_point_mass_acquisitions(cache).iloc[0]
        products = scene_products(
            ee_module,
            collections,
            int(scene["acquisition_time"]),
            str(scene["sensor"]),
            thresholds[reference],
        )
        images = [
            fetch_thumbnail(products[name], collections.aoi, dimensions=dimensions)
            for name in ("true_color", "ndwi", "reference", "threshold")
        ]
        date = scene["acquisition_date"].date().isoformat()
        path = (
            output_dir
            / f"point_mass_scene_{_slug(lake.lake_name)}_{reference.lower()}_{date}.png"
        )
        saved.append(
            save_static_panel(
                images,
                [
                    "Landsat true color",
                    "NDWI",
                    f"{reference} reference",
                    "Regional mask",
                ],
                path,
                heading=(
                    f"{lake.lake_name} — {date} — "
                    f"{int(scene['point_mass_rows'])} sampled NDWI=1 rows"
                ),
            )
        )
        records.append(
            f"- `{path.name}`: **NDWI=1 context**, Hylak_id {lake.hylak_id}; "
            "acquisition with the largest cached exact-NDWI=1 "
            f"count for this lake/reference ({int(scene['point_mass_rows'])} sampled "
            f"rows), using {reference} threshold {thresholds[reference]:+.8f}. "
            "This is scene context, not pixel-level causal proof."
        )

    records.extend(
        [
            "",
            "## Limitations",
            "",
            (
                "All AOIs are HydroLAKES polygons plus 5 km. Thumbnails are "
                "8-bit, resampled presentation products requested with "
                "`getThumbURL`; they are not quantitative exports. In the mask "
                "panels, green is threshold-positive water, pale gray is valid "
                "non-water, white is reference-invalid/uncovered, and black is "
                "outside the AOI or Landsat footprint. True color "
                "uses fixed scaled-reflectance display limits (0.02–0.30, gamma "
                "1.2). Reference and threshold panels use exact matched scenes; "
                "OPERA and JRC therefore retain different temporal windows."
            ),
            (
                "The GIFs show one deterministic high-coverage June-September "
                "cached acquisition per available year, not annual composites. "
                "They are exploratory "
                "change examples and should not be interpreted as complete annual "
                "water-area estimates. No Earth Engine batch exports, Drive, GCS, "
                "or remote output assets were used."
            ),
        ]
    )
    notes = output_dir / "imagery_notes.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    notes.write_text("\n".join(records) + "\n", encoding="utf-8")
    saved.append(notes)
    retained = {path.resolve() for path in saved}
    for pattern in ("static_*.png", "point_mass_scene_*.png", "change_*.gif"):
        for stale in output_dir.glob(pattern):
            if stale.resolve() not in retained:
                stale.unlink()
    return tuple(saved)


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a small no-export static/GIF imagery set with EE thumbnails "
            "and local Pillow assembly."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / DEFAULT_CONFIG_PATH
    )
    parser.add_argument(
        "--candidate-csv", type=Path, default=REPOSITORY_ROOT / DEFAULT_CANDIDATE_PATH
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--primary-cache-dir", type=Path, default=DEFAULT_PRIMARY_CACHE)
    parser.add_argument(
        "--expanded-cache-dir", type=Path, default=DEFAULT_EXPANDED_CACHE
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dimensions", type=_positive_int, default=560)
    parser.add_argument("--tier", choices=("tier1",), default="tier1")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Validate existing expected files, rewrite notes, and skip EE.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        inventory = load_candidate_inventory(args.candidate_csv)
        lake_ids = selected_ids(args.tier)
        lakes = select_lake_specs(inventory, lake_ids)
        lakes_by_id = {lake.hylak_id: lake for lake in lakes}
        required_ids = {
            lake_id for lake_id, _ in (*STATIC_PLAN, *GIF_PLAN, *POINT_MASS_PLAN)
        }
        missing = sorted(required_ids.difference(lakes_by_id))
        if missing:
            raise ImageryError(
                f"Imagery plan lakes are absent from {args.tier}: {missing}."
            )
        # Complete local validation before any remote initialization.
        for lake_id, reference in {*STATIC_PLAN, *GIF_PLAN, *POINT_MASS_PLAN}:
            load_sample_cache(
                (args.expanded_cache_dir, args.primary_cache_dir), reference, lake_id
            )
        for reference in ("OPERA", "JRC"):
            load_regional_threshold(args.data_dir, reference, args.tier)
        if args.finalize_existing:
            saved = finalize_existing_imagery(
                lakes_by_id,
                (args.expanded_cache_dir, args.primary_cache_dir),
                args.data_dir,
                args.output_dir,
                tier=args.tier,
            )
            print(f"Finalized {len(saved)} imagery products under {args.output_dir}")
            return 0
        ee_module, project = initialize_earth_engine()
        print(f"Earth Engine project: {project}")
        saved = run_imagery_plan(
            ee_module,
            config.earth_engine.hydrolakes_asset,
            lakes_by_id,
            QuickSamplingSettings(),
            (args.expanded_cache_dir, args.primary_cache_dir),
            args.data_dir,
            args.output_dir,
            tier=args.tier,
            dimensions=args.dimensions,
        )
        print(f"Wrote {len(saved)} imagery products under {args.output_dir}")
        return 0
    except (
        ConfigError,
        DiscoveryError,
        ImageryError,
        OSError,
        QuickSamplingError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
