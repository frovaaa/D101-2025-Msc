#!/bin/bash

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Starting DEM Voxelizer API server..."
echo "Open http://localhost:8000/static/app.html to use the application"
python server.py