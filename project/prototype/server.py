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
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    return FileResponse("app.html")


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
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "voxel_output"
            
            # Build command for voxelize_dem.py
            cmd = [
                "python", "voxelize_dem.py",
                "--api-key", params.api_key,
                "--bbox", str(minx), str(miny), str(maxx), str(maxy),
                "--outdir", str(output_dir),
                "--h-voxel", str(params.h_voxel),
                "--v-voxel", str(params.v_voxel),
                "--sea-level", str(params.sea_level),
                "--demtype", params.demtype
            ]
            
            if params.target_res:
                cmd.extend(["--target-res", str(params.target_res)])
            
            # Run processing
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Processing failed: {result.stderr}"
                )
            
            # Read results
            header_file = output_dir / "header.json"
            heights_file = output_dir / "heights.bin"
            preview_file = output_dir / "preview.png"

            if not header_file.exists() or not heights_file.exists():
                raise HTTPException(status_code=500, detail="Processing completed but output files missing")

            # Load header
            with open(header_file) as f:
                header = json.load(f)

            # Persist results to a job-specific folder we can serve statically
            job_id = uuid.uuid4().hex
            job_dir = RESULTS_DIR / job_id
            
            print(f"DEBUG: Copying {output_dir} to {job_dir}")
            print(f"DEBUG: RESULTS_DIR = {RESULTS_DIR}")
            print(f"DEBUG: Output dir contents: {list(output_dir.glob('*'))}")
            
            shutil.copytree(output_dir, job_dir)
            
            print(f"DEBUG: Job dir contents: {list(job_dir.glob('*'))}")

            # Build URLs for client to fetch binary safely
            heights_url = f"/results/{job_id}/heights.bin"
            header_url = f"/results/{job_id}/header.json"
            preview_url = f"/results/{job_id}/preview.png" if (job_dir / "preview.png").exists() else None
            
            print(f"DEBUG: Generated URLs - heights: {heights_url}, header: {header_url}, preview: {preview_url}")

            return {
                "header": header,               # also include inline for convenience
                "heights_url": heights_url,     # client will fetch as ArrayBuffer
                "header_url": header_url,
                "preview_url": preview_url,
                "processing_log": result.stdout,
                "job_id": job_id
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    print("Starting DEM Voxelizer API server...")
    print("Open http://localhost:8000 to use the application")
    uvicorn.run(app, host="0.0.0.0", port=8000)