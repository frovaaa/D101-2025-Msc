#!/bin/bash

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install Three.js if not present
if [ ! -d "node_modules" ]; then
    echo "Installing Three.js..."
    npm init -y
    npm install three
fi

echo "Starting DEM Voxelizer API server..."
echo "Open http://localhost:8000/static/app.html to use the application"
python server.py