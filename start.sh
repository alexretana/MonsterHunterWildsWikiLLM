#!/bin/bash

echo "==============================================="
echo " OpenWebUI Playground Startup Script (Unix/Linux)"
echo "==============================================="
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda is not found in PATH. Please ensure Anaconda or Miniconda is installed and added to PATH."
    exit 1
fi

# Initialize conda for bash (needed for conda activate to work)
eval "$(conda shell.bash hook)"

# Function to check if conda environment exists
env_exists() {
    conda info --envs | grep -q "^$1 "
}

# Check and create openwebui environment
echo "[1/4] Checking openwebui environment..."
if ! env_exists "openwebui"; then
    echo "Environment 'openwebui' not found. Creating from openwebui-environment.yaml..."
    if [ ! -f "openwebui-environment.yaml" ]; then
        echo "ERROR: openwebui-environment.yaml not found in current directory."
        exit 1
    fi
    conda env create -f openwebui-environment.yaml
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create openwebui environment."
        exit 1
    fi
    echo "Successfully created openwebui environment."
else
    echo "Environment 'openwebui' already exists."
fi

# Check and create openwebui-pipelines environment
echo "[2/4] Checking openwebui-pipelines environment..."
if ! env_exists "openwebui-pipelines"; then
    echo "Environment 'openwebui-pipelines' not found. Creating from owu-pipeline-environment.yaml..."
    if [ ! -f "owu-pipeline-environment.yaml" ]; then
        echo "ERROR: owu-pipeline-environment.yaml not found in current directory."
        exit 1
    fi
    conda env create -f owu-pipeline-environment.yaml
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create openwebui-pipelines environment."
        exit 1
    fi
    echo "Successfully created openwebui-pipelines environment."
else
    echo "Environment 'openwebui-pipelines' already exists."
fi

# Check if target directories and scripts exist
echo "[3/4] Checking target directories and scripts..."
if [ ! -f "external/open-webui-8-6-2025/backend/start.sh" ]; then
    echo "ERROR: external/open-webui-8-6-2025/backend/start.sh not found."
    exit 1
fi

if [ ! -f "external/open-webui-pipelines/start.sh" ]; then
    echo "ERROR: external/open-webui-pipelines/start.sh not found."
    exit 1
fi

# Make scripts executable if they aren't already
chmod +x external/open-webui-8-6-2025/backend/start.sh
chmod +x external/open-webui-pipelines/start.sh

# Start both services
echo "[4/4] Starting services..."
echo

# Function to start service in new terminal
start_service() {
    local service_name="$1"
    local env_name="$2"
    local script_path="$3"
    
    # Try different terminal emulators
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal --title="$service_name" -- bash -c "conda activate $env_name && cd $script_path && ./start.sh; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -title "$service_name" -hold -e bash -c "conda activate $env_name && cd $script_path && ./start.sh" &
    elif command -v konsole &> /dev/null; then
        konsole --title "$service_name" --hold -e bash -c "conda activate $env_name && cd $script_path && ./start.sh" &
    elif [[ "$OSTYPE" == "darwin"* ]] && command -v osascript &> /dev/null; then
        # macOS Terminal
        osascript -e "tell application \"Terminal\" to do script \"conda activate $env_name && cd $script_path && ./start.sh\""
    else
        # Fallback: run in background
        echo "No suitable terminal emulator found. Starting $service_name in background..."
        (cd "$script_path" && conda activate "$env_name" && ./start.sh) &
    fi
}

echo "Starting OpenWebUI backend in new terminal..."
start_service "OpenWebUI Backend" "openwebui" "external/open-webui-8-6-2025/backend"

echo "Waiting 5 seconds before starting second service to avoid conda conflicts..."
sleep 5  # Longer pause to prevent conda conflicts

echo "Starting OpenWebUI Pipelines in new terminal..."
start_service "OpenWebUI Pipelines" "openwebui-pipelines" "external/open-webui-pipelines"

echo
echo "==============================================="
echo " Both services are starting in separate terminals"
echo " - OpenWebUI Backend (openwebui environment)"
echo " - OpenWebUI Pipelines (openwebui-pipelines environment)"
echo "==============================================="
echo
echo "Services should be starting in new terminal windows."
echo "If no new windows appeared, the services may be running in background."
echo
