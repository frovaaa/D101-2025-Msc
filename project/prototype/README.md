# Voxel based 3D reconstruction from GeoTiff data

This project is a Voxel based 3D reconstruction visualization tool.

## Data acquisition

The raw GeoTiff data is acquired from [OpenTopography](https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3)

This dataset maps all the Earth's land surface at high resolution.

In particular we are using the "Copernicus GLO-30 Digital Elevation Model (DEM) - Global 30m" dataset.

This dataset contains elevation data with a spatial resolution of 30 meters.

#### Accessing the data

- With no account you are able to download up to 50 million points per cloud each request.

- With a free account you can access up to 250 million points per cloud each request. You are also able to access the [API](https://portal.opentopography.org/apidocs/#/Public/getGlobalDem) to automate data downloads.

- The maximum query areas are limited to 450,000 km2 for DEMs.

- Coordinate System is WGS84 [EPSG: 4326]

The current implementation expects you to download a region of data manually exporting it as GeoTiff from the website.

## Preprocessing

We have developed the script [voxelize_dem.py](voxelize_dem.py) to preprocess the GeoTiff data.

This script takes as input a GeoTiff file and various parameters to control the voxelization process.

The final output is a binary file containing the number of voxels in height for each (x, y) position.

### Usage

```bash
python voxelize_dem.py nuovaZelanda.tif \
  --outdir voxel_out_Lugano10x10x10 \
  --h-voxel 30 --v-voxel 30 \
  --crs EPSG:4326 \
  --target-res 1000 \
  --crs EPSG:3857 \
  --bbox 170.5 -46.9 172.5 -42.8 \
  --sea-level 100
```

### Parameters

- `--outdir`: Directory to save the output binary file.
- `--h-voxel`: Horizontal voxel size in meters.
  - Defines the width and depth of each voxel.
  - Does not impact the final number of voxels generated.
- `--v-voxel`: Vertical voxel size in meters.
  - Defines the height of each voxel. This will impact the final number of voxels generated as all the elevation datapoints lower than the voxel height will be ignored.
- `--target-res`: Target horizontal resolution in meters for the output voxel grid.
  - Uses bilinear interpolation to resample the input data.
- `--crs`: Coordinate Reference System for the output voxel grid.
  - Must be a metric CRS (e.g., EPSG:3857, EPSG:2193).
  - If not provided, or not metric, the script defaults to EPSG:3857 (Web Mercator).
- `--bbox`: Bounding box to crop the input GeoTiff data.
  - In source CRS, meaning degrees for EPSG:4326.
  - If using DEM dataset use degrees as it is in EPSG:4326.
  - Format: `west south east north`
  - The reference point (0,0) is the global one used by the CRS.
    - Example: for EPSG:4326, (0,0) is at the intersection of the Equator and the Prime Meridian.
- `--sea-level`: Elevation threshold to consider as sea level (in meters).
  - All elevation points below this threshold will be ignored.

## 3D Visualization

The output binary file can be visualized using a 3D rendering tool that supports voxel data.

Our implementation available at [index.html](index.html) uses Three.js to render the voxel data in a web browser.

The tool loads the header file `header.json` produced by the preprocessing script to understand the voxel grid dimensions and scaling.
Then it loads the binary voxel data and renders it as a 3D voxel grid.

A cap to the total number of voxels is implemented to prevent out of memory issues in the browser.

A high number of voxels (e.g. more than 10 million) may lead to performance issues or crashes.

A value around or below 5 million voxels is recommended for smooth performance.

To reduce the number of voxels, consider increasing the `--v-voxel` and `--target-res` sizes during preprocessing.

### Running the visualization tool

To run the visualization tool, simply run a local web server in the project directory and open `index.html` in a web browser.

Example using Python's built-in HTTP server:

```bash
python -m http.server 8000
```

### Camera Controls
We use the `OrbitControls` from Three.js to allow interactive camera movement.
- Left mouse button: Rotate around the target.
- Middle mouse button: Zoom in/out.
- Right mouse button: Pan the camera.

## Citations

```
European Space Agency (2024). Copernicus Global Digital Elevation Model. Distributed by OpenTopography. https://doi.org/10.5069/G9028PQB. Accessed 2025-10-28
```
