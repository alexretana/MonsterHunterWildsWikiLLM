#!/bin/bash

echo "==============================================="
echo " OpenWebUI Frontend Build Script (Unix/Linux)"
echo "==============================================="
echo

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Node.js is installed
echo "[1/4] Checking Node.js installation..."
if ! command_exists node; then
    echo "ERROR: Node.js is not installed or not in PATH."
    echo "Please install Node.js version 22.10 or higher from https://nodejs.org/"
    exit 1
fi

# Check Node.js version
echo "Checking Node.js version..."
NODE_VERSION=$(node --version)
echo "Found Node.js $NODE_VERSION"

# Check if npm is installed
if ! command_exists npm; then
    echo "ERROR: npm is not installed or not in PATH."
    echo "npm should be included with Node.js installation."
    exit 1
fi

# Check if we're in the correct directory
if [ ! -d "external/open-webui-8-6-2025" ]; then
    echo "ERROR: external/open-webui-8-6-2025 directory not found."
    echo "Please run this script from the root of the openwebui-playground directory."
    exit 1
fi

# Navigate to the Open WebUI directory
echo "[2/4] Navigating to Open WebUI directory..."
cd external/open-webui-8-6-2025
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to navigate to external/open-webui-8-6-2025"
    exit 1
fi

# Install dependencies
echo "[3/4] Installing frontend dependencies..."
echo "This may take a few minutes..."
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies."
    echo "Please check your internet connection and try again."
    exit 1
fi

# Build the frontend
echo "[4/4] Building frontend..."
echo "This may take a few minutes..."
npm run build
if [ $? -ne 0 ]; then
    echo "ERROR: Frontend build failed."
    echo "Please check the build output for errors."
    exit 1
fi

echo
echo "==============================================="
echo " Frontend build completed successfully!"
echo "==============================================="
echo
echo "The built frontend files are now ready."
echo "You can now run the backend using start_windows.bat or start.sh"
echo
