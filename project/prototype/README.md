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
python voxelize_dem.py switzerland.tif \
  --outdir voxel_out_switzerland \
  --h-voxel 30 --v-voxel 30 \
  --target-res 500 \
  --crs EPSG:3857 \
  --bbox 5.955 10.492 45.817 47.808 \
  --sea-level -10
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
- `--bbox`: Bounding box to crop the input GeoTiff data.
  - In target CRS if provided
- `--sea-level`: Elevation threshold to consider as sea level (in meters).
  - All elevation points below this threshold will be ignored.

## Citations

```
European Space Agency (2024). Copernicus Global Digital Elevation Model. Distributed by OpenTopography. https://doi.org/10.5069/G9028PQB. Accessed 2025-10-28
```
