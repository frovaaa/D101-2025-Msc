# Interactive DEM Voxel Viewer

A web application for visualizing Digital Elevation Models (DEMs) as interactive 3D voxel terrain using OpenTopography API data.

## Features

- 🗺️ **Interactive Map Selection**: Click and draw on a world map to select your area of interest
- 🌐 **Global DEM Data**: Fetch data from OpenTopography API (COP30, SRTM)
- 🎛️ **Real-time Parameters**: Adjust voxel sizes, resolution, and sea level
- 🎮 **3D Visualization**: Navigate the terrain with mouse controls
- ⚡ **Simple Setup**: One script to start everything

## Quick Start

1. **Get an OpenTopography API Key**:
   - Visit [OpenTopography.org](https://portal.opentopography.org/requestAccess)
   - Sign up for free access
   - Note your API key

2. **Run the Application**:
   ```bash
   ./start.sh
   ```

3. **Open your browser**:
   - Go to `http://localhost:8000/static/app.html`

## Usage

### Step 1: Select Area
- Use the map to navigate to your area of interest
- Click the rectangle tool in the top-right
- Draw a rectangle around the terrain you want to visualize
- Keep the area reasonable (< 5° x 5°) to avoid huge downloads

### Step 2: Configure API
- Enter your OpenTopography API key
- Choose DEM type:
  - **COP30**: Copernicus 30m resolution (best quality)
  - **SRTM**: SRTM 90m resolution (global coverage)

### Step 3: Adjust Parameters
- **Horizontal Voxel Size**: Spacing between voxels (10-100m)
- **Vertical Voxel Size**: Height of each voxel layer (5-50m)
- **Target Resolution**: Override DEM resolution (leave Auto for best)
- **Sea Level**: Baseline elevation (adjust for different regions)

### Step 4: Generate
- Click "Generate 3D Terrain"
- Wait for processing (may take 1-2 minutes for larger areas)
- Navigate the 3D terrain with mouse:
  - Left click + drag: Rotate
  - Right click + drag: Pan
  - Scroll: Zoom

## Technical Details

### Architecture
- **Frontend**: Leaflet.js map + Three.js 3D viewer
- **Backend**: FastAPI server + Python geospatial processing
- **Data Source**: OpenTopography Global DEM API

### Processing Pipeline
1. Fetch GeoTIFF data from OpenTopography API
2. Reproject to Web Mercator (EPSG:3857) for metric calculations
3. Resample to target resolution if specified
4. Convert elevation values to voxel heights
5. Generate Three.js InstancedMesh for efficient rendering

### Dependencies
- Python: FastAPI, rasterio, numpy, matplotlib, requests
- JavaScript: Three.js, Leaflet.js with drawing controls

## Troubleshooting

**"Processing failed" Error**:
- Check your API key is valid
- Ensure selected area isn't too large
- Verify internet connection

**"No area selected" Message**:
- Make sure you drew a rectangle on the map
- The rectangle should turn blue when selected

**Slow Performance**:
- Reduce voxel density (increase voxel sizes)
- Select smaller areas
- Lower target resolution

**Empty/Flat Terrain**:
- Check if area has elevation data available
- Try different DEM type (COP30 vs SRTM)
- Adjust sea level if terrain is below/above expected range

## Examples

### Mountain Terrain (Swiss Alps)
- Bbox: `[8.0, 46.0, 8.5, 46.5]`
- Voxel size: 30m horizontal, 20m vertical
- Perfect for dramatic elevation changes

### Coastal Areas
- Adjust sea level to 0m
- Use finer voxel resolution (10-20m)
- COP30 provides best coastal detail

### Large Scale Overview
- Bigger voxel sizes (50-100m)
- Lower target resolution (200-500m)
- SRTM for global coverage

## API Endpoints

- `GET /`: Serves main HTML interface
- `POST /process`: Processes DEM with parameters
- `GET /health`: Health check

## License

MIT License - Feel free to use and modify!