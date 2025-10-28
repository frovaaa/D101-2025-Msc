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

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject
from rasterio.crs import CRS
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt


def reproject_and_resample(src, dst_crs=None, target_res=None, bbox=None):
    """Return array, transform, crs after optional reprojection + resampling + crop.
    bbox is (minx, miny, maxx, maxy) expressed in dst_crs if given, else in src.crs.
    """
    print(f"Processing DEM: {src.width}x{src.height} pixels, CRS: {src.crs}")
    src_crs = src.crs
    dst_crs_final = rasterio.crs.CRS.from_string(dst_crs) if dst_crs else src_crs

    # If the user requested a target_res (meters) but the destination CRS is geographic
    # (degrees), force a metric CRS so that the resolution means meters, not degrees.
    if target_res is not None and dst_crs_final.is_geographic:
        print(
            "Target resolution provided in meters but CRS is geographic; reprojecting to EPSG:3857 for metric resampling."
        )
        dst_crs_final = CRS.from_epsg(3857)

    elif dst_crs is None and src_crs.is_geographic or dst_crs_final.is_geographic:
        print(
            f"Source CRS is geographic ({src_crs}). Reprojecting to EPSG:3857 for metric voxel processing."
        )
        dst_crs_final = CRS.from_epsg(3857)

    # Reproject to dst_crs if needed
    if src_crs != dst_crs_final:
        print(f"Reprojecting from {src_crs} to {dst_crs_final}...")
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs_final,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_res,
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs_final,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )
        data = np.full((1, height, width), np.nan, dtype=np.float32)
        print(f"Reprojecting to {width}x{height} grid...")
        reproject(
            source=rasterio.band(src, 1),
            destination=data[0],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs_final,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
            num_threads=2,
        )
    else:
        data = src.read(1, masked=True).filled(np.nan)[None, ...]

    # Resample to target_res if requested and not handled above
    if target_res is not None and src_crs == dst_crs_final:
        print(f"Resampling to {target_res}m resolution...")
        xres, yres = target_res, target_res
        # Compute new size from bounds
        minx, miny, maxx, maxy = src.bounds
        width = int(np.ceil((maxx - minx) / xres))
        height = int(np.ceil((maxy - miny) / yres))
        new_transform = from_origin(minx, maxx, xres, yres)
        out = np.full((1, height, width), np.nan, dtype=np.float32)
        reproject(
            source=data[0],
            destination=out[0],
            src_transform=transform,
            src_crs=dst_crs_final,
            dst_transform=new_transform,
            dst_crs=dst_crs_final,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
            num_threads=2,
        )
        data, transform = out, new_transform

    # Crop to bbox if provided
    if bbox is not None:
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
        data = data[:, row0:row1, col0:col1]
        # Recompute transform origin
        new_origin_x = transform.c + col0 * transform.a
        new_origin_y = transform.f - row0 * abs(transform.e)
        transform = from_origin(
            new_origin_x, new_origin_y, transform.a, abs(transform.e)
        )

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
    p.add_argument("dem", help="Input DEM GeoTIFF path")
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
        help="Optional bbox to crop (minx miny maxx maxy) in target CRS if provided, else DEM CRS.",
    )
    p.add_argument(
        "--sea-level",
        type=float,
        default=0.0,
        help="Elevation considered as base (meters)",
    )
    args = p.parse_args()

    print(f"Starting DEM voxelization: {args.dem}")
    print(f"Output directory: {args.outdir}")
    print(f"Voxel sizes: {args.h_voxel}m horizontal, {args.v_voxel}m vertical")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Opening DEM file...")
    with rasterio.open(args.dem) as src:
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
    col_heights = np.maximum(0, np.floor((dem - args.sea_level) / args.v_voxel)).astype(
        np.int16
    )
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
        "shape": [int(col_heights.shape[0]), int(col_heights.shape[1])],  # rows, cols
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


if __name__ == "__main__":
    main()
