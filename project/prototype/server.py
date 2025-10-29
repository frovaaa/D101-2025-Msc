#!/usr/bin/env python3
"""
FastAPI server for DEM voxelization with real-time processing.
"""
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Optional
import uuid
import shutil
import hashlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn


class ProcessingParams(BaseModel):
    bbox: list[float]  # [minx, miny, maxx, maxy]
    h_voxel: float = 30.0
    v_voxel: float = 30.0
    target_res: Optional[float] = None
    sea_level: float = 0.0
    demtype: str = "COP30"
    api_key: str


app = FastAPI(title="DEM Voxelizer API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, JS, etc.)
app.mount("/static", StaticFiles(directory="."), name="static")

# app.mount("/node_modules", StaticFiles(directory="node_modules"), name="node_modules")

# Persistent results directory for serving generated files
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_job_id(params: ProcessingParams) -> str:
    """Generate deterministic job ID from processing parameters for caching."""
    # Create a string representation of all parameters that affect the output
    param_string = (
        f"{params.bbox}_{params.h_voxel}_{params.v_voxel}_"
        f"{params.target_res}_{params.sea_level}_{params.demtype}"
    )
    # Generate SHA256 hash for consistent, collision-resistant ID
    return hashlib.sha256(param_string.encode()).hexdigest()[:16]


@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    return FileResponse("app.html")


@app.get("/api/results/{job_id}")
async def get_result(job_id: str):
    """Serve existing results by job ID for URL sharing."""
    job_dir = RESULTS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    header_file = job_dir / "header.json"
    if not header_file.exists():
        raise HTTPException(status_code=404, detail="Result header not found")

    # Load header
    with open(header_file) as f:
        header = json.load(f)

    # Build URLs for client to fetch binary safely
    heights_url = f"/results/{job_id}/heights.bin"
    header_url = f"/results/{job_id}/header.json"
    preview_url = (
        f"/results/{job_id}/preview.png" if (job_dir / "preview.png").exists() else None
    )

    return {
        "header": header,
        "heights_url": heights_url,
        "header_url": header_url,
        "preview_url": preview_url,
        "job_id": job_id,
        "cached": True,
    }


@app.post("/process")
async def process_dem(params: ProcessingParams):
    """Process DEM data with given parameters."""
    try:
        # Validate bbox
        if len(params.bbox) != 4:
            raise HTTPException(status_code=400, detail="Bbox must have 4 coordinates")

        minx, miny, maxx, maxy = params.bbox
        if maxx <= minx or maxy <= miny:
            raise HTTPException(status_code=400, detail="Invalid bbox coordinates")

        # Generate deterministic job ID for caching
        job_id = generate_job_id(params)
        job_dir = RESULTS_DIR / job_id

        # Check if results already exist (cache hit)
        if (
            job_dir.exists()
            and (job_dir / "header.json").exists()
            and (job_dir / "heights.bin").exists()
        ):
            print(f"Cache hit! Returning existing results for job {job_id}")

            # Load existing header
            with open(job_dir / "header.json") as f:
                header = json.load(f)

            # Build URLs for existing files
            heights_url = f"/results/{job_id}/heights.bin"
            header_url = f"/results/{job_id}/header.json"
            preview_url = (
                f"/results/{job_id}/preview.png"
                if (job_dir / "preview.png").exists()
                else None
            )

            return {
                "header": header,
                "heights_url": heights_url,
                "header_url": header_url,
                "preview_url": preview_url,
                "job_id": job_id,
                "cached": True,
            }

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "voxel_output"

            # Build command for voxelize_dem.py
            cmd = [
                "python",
                "voxelize_dem.py",
                "--api-key",
                params.api_key,
                "--bbox",
                str(minx),
                str(miny),
                str(maxx),
                str(maxy),
                "--outdir",
                str(output_dir),
                "--h-voxel",
                str(params.h_voxel),
                "--v-voxel",
                str(params.v_voxel),
                "--sea-level",
                str(params.sea_level),
                "--demtype",
                params.demtype,
            ]

            if params.target_res:
                cmd.extend(["--target-res", str(params.target_res)])

            # Run processing
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise HTTPException(
                    status_code=500, detail=f"Processing failed: {result.stderr}"
                )

            # Read results
            header_file = output_dir / "header.json"
            heights_file = output_dir / "heights.bin"
            preview_file = output_dir / "preview.png"

            if not header_file.exists() or not heights_file.exists():
                raise HTTPException(
                    status_code=500,
                    detail="Processing completed but output files missing",
                )

            # Load header
            with open(header_file) as f:
                header = json.load(f)

            # Persist results to the job-specific folder (job_id already generated above)
            # job_dir already created above in cache check

            print(f"DEBUG: Copying {output_dir} to {job_dir}")
            print(f"DEBUG: RESULTS_DIR = {RESULTS_DIR}")
            print(f"DEBUG: Output dir contents: {list(output_dir.glob('*'))}")

            shutil.copytree(output_dir, job_dir)

            print(f"DEBUG: Job dir contents: {list(job_dir.glob('*'))}")

            # Build URLs for client to fetch binary safely
            heights_url = f"/results/{job_id}/heights.bin"
            header_url = f"/results/{job_id}/header.json"
            preview_url = (
                f"/results/{job_id}/preview.png"
                if (job_dir / "preview.png").exists()
                else None
            )

            print(
                f"DEBUG: Generated URLs - heights: {heights_url}, header: {header_url}, preview: {preview_url}"
            )

            return {
                "header": header,  # also include inline for convenience
                "heights_url": heights_url,  # client will fetch as ArrayBuffer
                "header_url": header_url,
                "preview_url": preview_url,
                "processing_log": result.stdout,
                "job_id": job_id,
                "cached": False,
            }

    except Exception as e:
        error_msg = str(e)

        # Add helpful info about debug files when there are corruption errors
        if (
            "DEM file corruption detected" in error_msg
            or "LZWDecode" in error_msg
            or "corrupted" in error_msg
        ):
            debug_dir = Path("debug_downloads")
            if debug_dir.exists():
                debug_files = list(debug_dir.glob("*.tif"))
                if debug_files:
                    latest_file = max(debug_files, key=lambda p: p.stat().st_mtime)
                    error_msg += (
                        f"\n\nDEBUG: The corrupted DEM file was saved to: {latest_file}"
                    )
                    error_msg += f"\nYou can inspect this file with QGIS or other GIS software to verify the corruption."

        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# Mount static files after API routes to avoid conflicts
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


def cleanup_debug_files():
    """Remove debug DEM files older than 24 hours to prevent disk buildup."""
    debug_dir = Path("debug_downloads")
    if not debug_dir.exists():
        return

    import time

    cutoff_time = time.time() - (24 * 3600)  # 24 hours ago

    for file_path in debug_dir.glob("*.tif"):
        if file_path.stat().st_mtime < cutoff_time:
            try:
                file_path.unlink()
                print(f"Cleaned up old debug file: {file_path.name}")
            except Exception:
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    print("Starting DEM Voxelizer API server...")
    print("Open http://localhost:8000 to use the application")

    # Clean up old debug files on startup
    cleanup_debug_files()
    uvicorn.run(app, host="0.0.0.0", port=8000)
