#!/usr/bin/env python3
"""
Voxelize a DEM (GeoTIFF) into a compact height-column grid for fast rendering in Three.js.

Outputs (into --outdir):
- heights.bin        int16 array of shape (rows, cols) storing number of vertical voxels (column height)
- header.json        metadata (grid size, origin, voxel sizes, CRS, nodata, min/max)
- preview.png        quick shaded preview for sanity check

Examples:
  python voxelize_dem.py /path/to/dem.tif --outdir out --h-voxel 25 --v-voxel 10 --target-res 25
  python voxelize_dem.py dem.tif --bbox 8.80 45.80 9.10 46.05 --crs EPSG:4326 --outdir out_lugano
"""
import argparse
import json
import sys
from pathlib import Path
import requests
from io import BytesIO
import tempfile
from datetime import datetime

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject
from rasterio.crs import CRS
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt


def fetch_dem_from_opentopo(bbox, api_key, demtype="COP30"):
    """
    Fetch DEM data from OpenTopography API.

    Args:
        bbox: (minx, miny, maxx, maxy) in EPSG:4326 coordinates
        api_key: OpenTopography API key
        demtype: DEM type (COP30, SRTM, etc.)

    Returns:
        tuple: (rasterio dataset, file_path) - file_path is for cleanup
    """
    minx, miny, maxx, maxy = bbox

    # Validate bbox size (prevent huge requests)
    if (maxx - minx) > 5.0 or (maxy - miny) > 5.0:
        raise ValueError("Bounding box too large. Maximum 5 degrees per side.")

    # Build API URL
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": demtype,
        "south": miny,
        "north": maxy,
        "west": minx,
        "east": maxx,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }

    print(f"Fetching DEM data from OpenTopography...")
    print(f"  Bbox: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}")
    print(f"  DEM Type: {demtype}")

    # Make API request
    response = requests.get(url, params=params, timeout=120)

    if response.status_code != 200:
        raise ValueError(
            f"OpenTopography API error: {response.status_code} - {response.text}"
        )

    # Check if response is actually a GeoTIFF
    if not response.content.startswith(b"II*\x00") and not response.content.startswith(
        b"MM\x00*"
    ):
        raise ValueError(f"API returned invalid GeoTIFF data: {response.text[:200]}")

    print(f"Downloaded {len(response.content):,} bytes")

    # Save the fetched file to disk - this avoids MemoryFile corruption issues
    # and allows for inspection when problems occur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_filename = f"fetched_dem_{demtype}_{timestamp}.tif"
    debug_path = Path("debug_downloads") / debug_filename
    debug_path.parent.mkdir(exist_ok=True)

    with open(debug_path, "wb") as f:
        f.write(response.content)
    print(f"Saved DEM to: {debug_path}")

    # Open GeoTIFF from disk (more reliable than MemoryFile for some corrupted files)
    return rasterio.open(debug_path), debug_path


def determine_target_crs(src_crs, dst_crs=None, target_res=None):
    """Determine the final target CRS for processing."""
    if dst_crs is not None:
        dst_crs_final = rasterio.crs.CRS.from_string(dst_crs)
    else:
        dst_crs_final = src_crs

    # Force metric CRS if target_res is specified but CRS is geographic
    if target_res is not None and dst_crs_final.is_geographic:
        print(
            "Target resolution provided in meters but CRS is geographic; reprojecting to EPSG:3857 for metric resampling."
        )
        dst_crs_final = CRS.from_epsg(3857)

    # Auto-select metric CRS if source is geographic and no CRS specified
    elif dst_crs is None and src_crs.is_geographic:
        print(
            f"Source CRS is geographic ({src_crs}). Reprojecting to EPSG:3857 for metric voxel processing."
        )
        dst_crs_final = CRS.from_epsg(3857)

    return dst_crs_final


def crop_to_bbox(data, transform, bbox):
    """Crop data array to bounding box and update transform."""
    if bbox is None:
        return data, transform

    print(f"Cropping to bbox: {bbox}")
    minx, miny, maxx, maxy = bbox

    # Compute window indices
    col0 = int((minx - transform.c) / transform.a)
    col1 = int((maxx - transform.c) / transform.a)
    row0 = int((transform.f - maxy) / abs(transform.e))
    row1 = int((transform.f - miny) / abs(transform.e))

    # Clip to bounds
    col0 = np.clip(col0, 0, data.shape[2])
    col1 = np.clip(col1, 0, data.shape[2])
    row0 = np.clip(row0, 0, data.shape[1])
    row1 = np.clip(row1, 0, data.shape[1])

    # Crop data
    data = data[:, row0:row1, col0:col1]

    # Recompute transform origin
    new_origin_x = transform.c + col0 * transform.a
    new_origin_y = transform.f - row0 * abs(transform.e)
    new_transform = from_origin(
        new_origin_x, new_origin_y, transform.a, abs(transform.e)
    )

    return data, new_transform


def reproject_data(data, src_transform, src_crs, dst_crs, target_res=None):
    """Reproject data from source CRS to destination CRS."""
    if src_crs == dst_crs:
        return data, src_transform

    print(f"Reprojecting from {src_crs} to {dst_crs}...")

    # Calculate bounds for the data
    height, width = data.shape[-2:]
    bounds = rasterio.transform.array_bounds(height, width, src_transform)

    # Calculate new transform and dimensions
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, *bounds, resolution=target_res
    )

    # Prepare output array
    dst_data = np.full((data.shape[0], dst_height, dst_width), np.nan, dtype=np.float32)

    print(f"Reprojecting to {dst_width}x{dst_height} grid...")
    reproject(
        source=data[0],
        destination=dst_data[0],
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
        num_threads=2,
    )

    return dst_data, dst_transform


def resample_data(data, transform, current_crs, target_res):
    """Resample data to a different resolution."""
    if target_res is None:
        return data, transform

    print(f"Resampling to {target_res}m resolution...")

    # Calculate new dimensions based on current bounds
    height, width = data.shape[-2:]
    bounds = rasterio.transform.array_bounds(height, width, transform)
    minx, miny, maxx, maxy = bounds

    new_width = int(np.ceil((maxx - minx) / target_res))
    new_height = int(np.ceil((maxy - miny) / target_res))
    new_transform = from_origin(minx, maxy, target_res, target_res)

    # Prepare output array
    out_data = np.full((data.shape[0], new_height, new_width), np.nan, dtype=np.float32)

    reproject(
        source=data[0],
        destination=out_data[0],
        src_transform=transform,
        src_crs=current_crs,
        dst_transform=new_transform,
        dst_crs=current_crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
        num_threads=2,
    )

    return out_data, new_transform


def reproject_and_resample(src, dst_crs=None, target_res=None, bbox=None):
    """Return array, transform, crs after optional reprojection + resampling + crop.
    bbox is (minx, miny, maxx, maxy) expressed in source CRS coordinates.

    Processing order:
    1. Crop to bbox (in source CRS)
    2. Reproject to target CRS
    3. Resample to target resolution
    """
    print(f"Processing DEM: {src.width}x{src.height} pixels, CRS: {src.crs}")
    src_crs = src.crs

    # Determine target CRS
    dst_crs_final = determine_target_crs(src_crs, dst_crs, target_res)

    # Step 1: Load and crop data (in source CRS)
    try:
        data = src.read(1, masked=True).filled(np.nan)[None, ...]
        transform = src.transform
    except Exception as e:
        error_msg = str(e)
        if "LZWDecode" in error_msg and "corrupted" in error_msg:
            raise ValueError(
                "DEM file corruption detected: The downloaded TIFF file has corrupted LZW compression. "
                "This is a known issue with some OpenTopography downloads. "
                "Please try:\n"
                "1. Selecting a different geographic area\n"
                "2. Using a different DEM type (try SRTM instead of COP30, or vice versa)\n"
                "3. Selecting a smaller area\n"
                "4. Retrying the same request (sometimes works on second attempt)"
            )
        elif "TIFFReadEncodedTile" in error_msg:
            raise ValueError(
                "DEM file read error: The TIFF file structure is damaged. "
                "This can happen due to network issues during download. "
                "Please try selecting a different area or DEM type."
            )
        elif "RasterioIOError" in error_msg or "Read failed" in error_msg:
            raise ValueError(
                "DEM file access error: Cannot read the downloaded DEM data. "
                "This may be due to file corruption or format issues. "
                "Please try a different geographic area or DEM type."
            )
        else:
            raise ValueError(f"Unexpected DEM processing error: {error_msg}")

    data, transform = crop_to_bbox(data, transform, bbox)
    current_crs = src_crs

    # Step 2: Reproject if needed
    data, transform = reproject_data(
        data, transform, current_crs, dst_crs_final, target_res
    )
    current_crs = dst_crs_final

    # Step 3: Resample if needed (and not already handled by reprojection)
    if target_res is not None and src_crs == dst_crs_final:
        data, transform = resample_data(data, transform, current_crs, target_res)

    return data[0], transform, dst_crs_final


def write_preview(dem, out_png, clim=None):
    # If the DEM is too small to compute gradients, just write a flat image
    if dem.ndim == 2 and (dem.shape[0] < 2 or dem.shape[1] < 2):
        plt.figure(figsize=(4, 4))
        plt.imshow(np.zeros((2, 2)), cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        plt.close()
        return

    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(dem, cmap=plt.cm.terrain, vert_exag=1.5, blend_mode="soft")
    plt.figure(figsize=(6, 5))
    plt.imshow(rgb, interpolation="nearest")
    plt.title("DEM shaded preview")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "dem", nargs="?", help="Input DEM GeoTIFF path (optional if using --api-key)"
    )
    p.add_argument("--outdir", default="voxel_out", help="Output directory")
    p.add_argument(
        "--h-voxel", type=float, default=25.0, help="Horizontal voxel size in meters"
    )
    p.add_argument(
        "--v-voxel", type=float, default=10.0, help="Vertical voxel size in meters"
    )
    p.add_argument(
        "--target-res",
        type=float,
        default=None,
        help="Optional resampling resolution in meters (pixel size). "
        "If not set, uses DEM native resolution.",
    )
    p.add_argument("--crs", default=None, help="Optional target CRS (e.g., EPSG:2056)")
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=None,
        help="Bbox coordinates (minx miny maxx maxy) in EPSG:4326. Required if using --api-key.",
    )
    p.add_argument(
        "--sea-level",
        type=float,
        default=0.0,
        help="Elevation considered as base (meters)",
    )
    p.add_argument("--api-key", help="OpenTopography API key for fetching DEM data")
    p.add_argument(
        "--demtype",
        default="COP30",
        help="DEM type for OpenTopography API (COP30, SRTM, etc.)",
    )
    args = p.parse_args()

    # Validate arguments
    if args.api_key:
        if not args.bbox:
            print("ERROR: --bbox is required when using --api-key")
            sys.exit(1)
        print("Starting DEM voxelization from OpenTopography API")
    elif args.dem:
        print(f"Starting DEM voxelization: {args.dem}")
    else:
        print("ERROR: Either provide a DEM file or use --api-key with --bbox")
        sys.exit(1)

    print(f"Output directory: {args.outdir}")
    print(f"Voxel sizes: {args.h_voxel}m horizontal, {args.v_voxel}m vertical")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Open DEM source (either local file or API)
    temp_file_to_cleanup = None
    if args.api_key:
        src, temp_file_to_cleanup = fetch_dem_from_opentopo(
            args.bbox, args.api_key, args.demtype
        )
        src_context = src  # Already opened
    else:
        print("Opening DEM file...")
        src_context = rasterio.open(args.dem)

    try:
        with src_context as src:
            print(f"DEM bounds: {src.bounds}")
            print(f"  West: {src.bounds.left:.3f}°, East: {src.bounds.right:.3f}°")
            print(f"  South: {src.bounds.bottom:.3f}°, North: {src.bounds.top:.3f}°")

            if args.bbox:
                minx, miny, maxx, maxy = args.bbox
                print(f"Requested bbox: {args.bbox}")
                print(f"  West: {minx:.3f}°, East: {maxx:.3f}°")
                print(f"  South: {miny:.3f}°, North: {maxy:.3f}°")

                # Check if bbox intersects with DEM bounds
                bbox_intersects = (
                    minx < src.bounds.right
                    and maxx > src.bounds.left
                    and miny < src.bounds.top
                    and maxy > src.bounds.bottom
                )

                if not bbox_intersects:
                    print("ERROR: Bbox does not intersect with DEM bounds!")
                    print(
                        f"DEM covers: {src.bounds.left:.3f}° to {src.bounds.right:.3f}° (lon), {src.bounds.bottom:.3f}° to {src.bounds.top:.3f}° (lat)"
                    )
                    print(
                        f"Bbox requests: {minx:.3f}° to {maxx:.3f}° (lon), {miny:.3f}° to {maxy:.3f}° (lat)"
                    )
                    sys.exit(1)

            dem, transform, crs = reproject_and_resample(
                src, dst_crs=args.crs, target_res=args.target_res, bbox=args.bbox
            )

        dem = np.asarray(dem, dtype=np.float32)
        emin = float(np.nanmin(dem)) if np.isfinite(np.nanmin(dem)) else np.nan
        emax = float(np.nanmax(dem)) if np.isfinite(np.nanmax(dem)) else np.nan
        print(f"DEM shape: {dem.shape}, elevation range: {emin:.1f} to {emax:.1f}m")

        # Replace NaNs with sea level for voxel conversion
        dem = np.nan_to_num(dem, nan=args.sea_level)

        # Abort early if the DEM collapsed to a constant
        if not np.isfinite(emin) or not np.isfinite(emax) or emin == emax:
            print(
                "WARNING: DEM appears constant or empty after reprojection/resampling.\n"
                "Hint: If your source CRS is geographic (EPSG:4326) and you set --target-res in meters,\n"
                "also set --crs to a metric CRS (e.g., EPSG:3857, EPSG:2056 for CH, EPSG:2193 for NZ)."
            )

        print("Converting elevation to voxel heights...")
        # Convert to number of vertical voxels per column
        # dem = the actual elevation data of that point
        col_heights = np.maximum(
            0, np.floor((dem - args.sea_level) / args.v_voxel)
        ).astype(np.int16)
        print(f"Voxel heights range: {col_heights.min()} to {col_heights.max()} voxels")

        # Save binary grid (rows x cols int16)
        print("Writing binary height data...")
        heights_path = outdir / "heights.bin"
        col_heights.tofile(heights_path)

        # Header
        transform_vals = [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ]
        header = {
            "shape": [
                int(col_heights.shape[0]),
                int(col_heights.shape[1]),
            ],  # rows, cols
            "dtype": "int16",
            "endian": "little",
            "horizontal_voxel_m": args.h_voxel,
            "vertical_voxel_m": args.v_voxel,
            "pixel_size_m": [abs(transform.a), abs(transform.e)],
            "origin": [transform.c, transform.f],  # upper-left corner in CRS units
            "crs": str(crs),
            "sea_level_m": args.sea_level,
            "min_height_vox": int(col_heights.min()),
            "max_height_vox": int(col_heights.max()),
            "transform": transform_vals,
        }
        print("Writing metadata header...")
        with open(outdir / "header.json", "w") as f:
            json.dump(header, f, indent=2)

        # Preview
        # print("Generating shaded preview image...")
        print("Generating shaded preview image disabled")
        # write_preview(dem, outdir / "preview.png")

        # Also write a tiny README
        with open(outdir / "README.txt", "w") as f:
            f.write(
                "This folder contains a voxel-ready height grid derived from the DEM.\n"
                "Files:\n"
                "- heights.bin : int16 (rows x cols), number of vertical voxels per column\n"
                "- header.json : metadata (shape, CRS, voxel sizes, transform)\n"
                "- preview.png : quick shaded preview\n\n"
                "To load in JS, fetch heights.bin as an ArrayBuffer and interpret as Int16Array, "
                "then iterate columns to create InstancedMesh boxes up to the height value.\n"
            )

        total_voxels = int(col_heights.sum())
        print(f"✓ Voxelization complete!")
        print(f"✓ Total voxels to render: {total_voxels:,}")
        print(f"✓ Files written in {outdir}:")
        print(f"  - {heights_path.name} ({heights_path.stat().st_size:,} bytes)")
        print(f"  - header.json")
        print(f"  - preview.png")
        print(f"  - README.txt")
    finally:
        # Clean up temporary downloaded file
        if temp_file_to_cleanup and temp_file_to_cleanup.exists():
            try:
                temp_file_to_cleanup.unlink()
                print(f"Cleaned up temporary file: {temp_file_to_cleanup}")
            except Exception:
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()
