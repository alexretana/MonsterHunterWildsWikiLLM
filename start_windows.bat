@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo  OpenWebUI Playground Startup Script (Windows)
echo ===============================================
echo.

REM Initialize variables
set OLLAMA_WAS_RUNNING=false
set OLLAMA_PORT=11434

REM Function to check if port is in use (Ollama server running)
REM Uses netstat to check if port 11434 is listening
echo [1/7] Checking Ollama server status...
netstat -an | findstr ":%OLLAMA_PORT%" | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo Ollama server is already running on port %OLLAMA_PORT%
    set OLLAMA_WAS_RUNNING=true
) else (
    echo Ollama server is not running on port %OLLAMA_PORT%
    
    REM Check if ollama is installed
    echo [2/7] Checking Ollama installation...
    where ollama >nul 2>&1
    if !errorlevel! neq 0 (
        echo Ollama is not installed. Attempting to install...
        
        REM Check if chocolatey is installed
        where choco >nul 2>&1
        if !errorlevel! neq 0 (
            echo ERROR: Chocolatey is not installed.
            echo Please install either:
            echo   1. Chocolatey: https://chocolatey.org/install
            echo   2. Ollama directly: https://ollama.ai/download
            echo Then run this script again.
            pause
            exit /b 1
        )
        
        echo Installing Ollama using Chocolatey...
        choco install ollama -y
        if !errorlevel! neq 0 (
            echo ERROR: Failed to install Ollama using Chocolatey.
            echo Please install Ollama manually from https://ollama.ai/download
            pause
            exit /b 1
        )
        echo Ollama installed successfully.
    ) else (
        echo Ollama is already installed.
    )
)

REM Check required models (only if ollama is installed)
echo [3/7] Checking required Ollama models...
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    echo Checking for llama3:8b model...
    ollama list | findstr "llama3:8b" >nul 2>&1
    if !errorlevel! neq 0 (
        echo Model llama3:8b not found. Installing...
        echo This may take several minutes depending on your internet connection...
        ollama pull llama3:8b
        if !errorlevel! neq 0 (
            echo ERROR: Failed to install llama3:8b model.
            pause
            exit /b 1
        )
        echo Successfully installed llama3:8b model.
    ) else (
        echo Model llama3:8b is already installed.
    )
    
    echo Checking for nomic-embed-text model...
    ollama list | findstr "nomic-embed-text" >nul 2>&1
    if !errorlevel! neq 0 (
        echo Model nomic-embed-text not found. Installing...
        echo This may take several minutes depending on your internet connection...
        ollama pull nomic-embed-text
        if !errorlevel! neq 0 (
            echo ERROR: Failed to install nomic-embed-text model.
            pause
            exit /b 1
        )
        echo Successfully installed nomic-embed-text model.
    ) else (
        echo Model nomic-embed-text is already installed.
    )
) else (
    echo Skipping model check - Ollama not available.
)

REM Check if conda is available
echo [4/7] Checking conda availability...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Conda is not found in PATH. Please ensure Anaconda or Miniconda is installed and added to PATH.
    pause
    exit /b 1
)

REM Check and create openwebui environment
echo [5/7] Checking openwebui environment...
conda info --envs | findstr "openwebui " >nul 2>&1
if %errorlevel% neq 0 (
    echo Environment 'openwebui' not found. Creating from openwebui-environment.yaml...
    if not exist "openwebui-environment.yaml" (
        echo ERROR: openwebui-environment.yaml not found in current directory.
        pause
        exit /b 1
    )
    conda env create -f openwebui-environment.yaml
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create openwebui environment.
        pause
        exit /b 1
    )
    echo Successfully created openwebui environment.
) else (
    echo Environment 'openwebui' already exists.
)

REM Check and create openwebui-pipelines environment
echo [6/7] Checking openwebui-pipelines environment...
conda info --envs | findstr "openwebui-pipelines " >nul 2>&1
if %errorlevel% neq 0 (
    echo Environment 'openwebui-pipelines' not found. Creating from owu-pipeline-environment.yaml...
    if not exist "owu-pipeline-environment.yaml" (
        echo ERROR: owu-pipeline-environment.yaml not found in current directory.
        pause
        exit /b 1
    )
    conda env create -f owu-pipeline-environment.yaml
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create openwebui-pipelines environment.
        pause
        exit /b 1
    )
    echo Successfully created openwebui-pipelines environment.
) else (
    echo Environment 'openwebui-pipelines' already exists.
)

REM Start Ollama server if it wasn't already running
if "%OLLAMA_WAS_RUNNING%"=="false" (
    echo [7/7] Starting Ollama server...
    echo Starting Ollama server in new window...
    start "Ollama Server" cmd /k "ollama serve"
    
    REM Wait for Ollama server to start
    echo Waiting for Ollama server to start up...
    :WAIT_OLLAMA
    ping 127.0.0.1 -n 3 >nul
    netstat -an | findstr ":%OLLAMA_PORT%" | findstr "LISTENING" >nul 2>&1
    if %errorlevel% neq 0 goto WAIT_OLLAMA
    echo Ollama server is now running on port %OLLAMA_PORT%
) else (
    echo [7/7] Ollama server was already running, skipping startup.
)

REM Check if target directories and scripts exist
echo Checking target directories and scripts...
if not exist "external\open-webui-8-6-2025\backend\start_windows.bat" (
    echo ERROR: external\open-webui-8-6-2025\backend\start_windows.bat not found.
    pause
    exit /b 1
)

if not exist "external\open-webui-pipelines\start.bat" (
    echo ERROR: external\open-webui-pipelines\start.bat not found.
    pause
    exit /b 1
)

REM Start both OpenWebUI services
echo Starting OpenWebUI services...
echo.
echo Starting OpenWebUI backend in new window...
start "OpenWebUI Backend" cmd /k "conda activate openwebui && cd external\open-webui-8-6-2025\backend && start_windows.bat"

echo Waiting 5 seconds before starting second service to avoid conda conflicts...
ping 127.0.0.1 -n 6 >nul

echo Starting OpenWebUI Pipelines in new window...
start "OpenWebUI Pipelines" cmd /k "conda activate openwebui-pipelines && cd external\open-webui-pipelines && start.bat"

echo.
echo ===============================================
echo  All services are starting in separate windows:
if "%OLLAMA_WAS_RUNNING%"=="false" (
    echo  - Ollama Server (started by this script)
) else (
    echo  - Ollama Server (was already running)
)
echo  - OpenWebUI Backend (openwebui environment)
echo  - OpenWebUI Pipelines (openwebui-pipelines environment)
echo ===============================================
echo.
echo Press any key to exit this window...
pause >nul
