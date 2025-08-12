#!/bin/bash

echo "==============================================="
echo " OpenWebUI Playground Startup Script (Unix/Linux)"
echo "==============================================="
echo

# Initialize variables
OLLAMA_WAS_RUNNING=false
OLLAMA_PORT=11434

# Function to check if port is in use
check_port() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -i :$1 >/dev/null 2>&1
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tuln | grep ":$1 " >/dev/null 2>&1
    else
        # Fallback: try to connect to the port
        timeout 1 bash -c "</dev/tcp/localhost/$1" >/dev/null 2>&1
    fi
}

# Check if Ollama server is running
echo "[1/7] Checking Ollama server status..."
if check_port $OLLAMA_PORT; then
    echo "Ollama server is already running on port $OLLAMA_PORT"
    OLLAMA_WAS_RUNNING=true
else
    echo "Ollama server is not running on port $OLLAMA_PORT"
    
    # Check if ollama is installed
    echo "[2/7] Checking Ollama installation..."
    if ! command -v ollama >/dev/null 2>&1; then
        echo "Ollama is not installed. Attempting to install..."
        
        # Install ollama using curl
        if command -v curl >/dev/null 2>&1; then
            echo "Installing Ollama using curl..."
            curl -fsSL https://ollama.ai/install.sh | sh
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed to install Ollama."
                echo "Please install Ollama manually from https://ollama.ai/download"
                exit 1
            fi
            echo "Ollama installed successfully."
        else
            echo "ERROR: curl is not available for Ollama installation."
            echo "Please install Ollama manually from https://ollama.ai/download"
            exit 1
        fi
    else
        echo "Ollama is already installed."
    fi
fi

# Check required models (only if ollama is installed)
echo "[3/7] Checking required Ollama models..."
if command -v ollama >/dev/null 2>&1; then
    echo "Checking for llama3:8b model..."
    if ! ollama list | grep -q "llama3:8b"; then
        echo "Model llama3:8b not found. Installing..."
        echo "This may take several minutes depending on your internet connection..."
        ollama pull llama3:8b
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install llama3:8b model."
            exit 1
        fi
        echo "Successfully installed llama3:8b model."
    else
        echo "Model llama3:8b is already installed."
    fi
    
    echo "Checking for nomic-embed-text model..."
    if ! ollama list | grep -q "nomic-embed-text"; then
        echo "Model nomic-embed-text not found. Installing..."
        echo "This may take several minutes depending on your internet connection..."
        ollama pull nomic-embed-text
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install nomic-embed-text model."
            exit 1
        fi
        echo "Successfully installed nomic-embed-text model."
    else
        echo "Model nomic-embed-text is already installed."
    fi
else
    echo "Skipping model check - Ollama not available."
fi

# Check if conda is available
echo "[4/7] Checking conda availability..."
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
echo "[5/7] Checking openwebui environment..."
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
echo "[6/7] Checking openwebui-pipelines environment..."
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

# Start Ollama server if it wasn't already running
if [ "$OLLAMA_WAS_RUNNING" = "false" ]; then
    echo "[7/7] Starting Ollama server..."
    echo "Starting Ollama server in new terminal..."
    
    # Start Ollama in a new terminal
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal --title="Ollama Server" -- bash -c "ollama serve; exec bash" &
    elif command -v xterm &> /dev/null; then
        xterm -title "Ollama Server" -hold -e bash -c "ollama serve" &
    elif command -v konsole &> /dev/null; then
        konsole --title "Ollama Server" --hold -e bash -c "ollama serve" &
    elif [[ "$OSTYPE" == "darwin"* ]] && command -v osascript &> /dev/null; then
        # macOS Terminal
        osascript -e "tell application \"Terminal\" to do script \"ollama serve\"" &
    else
        # Fallback: run in background
        echo "No suitable terminal emulator found. Starting Ollama server in background..."
        nohup ollama serve > ollama.log 2>&1 &
    fi
    
    # Wait for Ollama server to start
    echo "Waiting for Ollama server to start up..."
    for i in {1..30}; do
        if check_port $OLLAMA_PORT; then
            echo "Ollama server is now running on port $OLLAMA_PORT"
            break
        fi
        sleep 1
    done
    
    if ! check_port $OLLAMA_PORT; then
        echo "WARNING: Ollama server may not have started properly. Check the Ollama terminal window."
    fi
else
    echo "[7/7] Ollama server was already running, skipping startup."
fi

# Check if target directories and scripts exist
echo "Checking target directories and scripts..."
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

# Start OpenWebUI services
echo "Starting OpenWebUI services..."
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
sleep 5

echo "Starting OpenWebUI Pipelines in new terminal..."
start_service "OpenWebUI Pipelines" "openwebui-pipelines" "external/open-webui-pipelines"

echo
echo "==============================================="
echo " All services are starting in separate terminals:"
if [ "$OLLAMA_WAS_RUNNING" = "false" ]; then
    echo " - Ollama Server (started by this script)"
else
    echo " - Ollama Server (was already running)"
fi
echo " - OpenWebUI Backend (openwebui environment)"
echo " - OpenWebUI Pipelines (openwebui-pipelines environment)"
echo "==============================================="
echo
echo "Services should be starting in new terminal windows."
echo "If no new windows appeared, some services may be running in background."
echo
