@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo  OpenWebUI Playground Startup Script (Windows)
echo ===============================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Conda is not found in PATH. Please ensure Anaconda or Miniconda is installed and added to PATH.
    pause
    exit /b 1
)

REM Check and create openwebui environment
echo [1/4] Checking openwebui environment...
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
echo [2/4] Checking openwebui-pipelines environment...
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

REM Check if target directories and scripts exist
echo [3/4] Checking target directories and scripts...
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

REM Start both services
echo [4/4] Starting services...
echo.
echo Starting OpenWebUI backend in new window...
start "OpenWebUI Backend" cmd /k "conda activate openwebui && cd external\open-webui-8-6-2025\backend && start_windows.bat"

echo Waiting 5 seconds before starting second service to avoid conda conflicts...
ping 127.0.0.1 -n 6 >nul

echo Starting OpenWebUI Pipelines in new window...
start "OpenWebUI Pipelines" cmd /k "conda activate openwebui-pipelines && cd external\open-webui-pipelines && start.bat"

echo.
echo ===============================================
echo  Both services are starting in separate windows
echo  - OpenWebUI Backend (openwebui environment)
echo  - OpenWebUI Pipelines (openwebui-pipelines environment)
echo ===============================================
echo.
echo Press any key to exit this window...
pause >nul
