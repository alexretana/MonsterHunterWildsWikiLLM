@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo  OpenWebUI Frontend Build Script (Windows)
echo ===============================================
echo.

REM Check if Node.js is installed
echo [1/4] Checking Node.js installation...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH.
    echo Please install Node.js version 22.10 or higher from https://nodejs.org/
    pause
    exit /b 1
)

REM Check Node.js version
echo Checking Node.js version...
for /f "tokens=1" %%a in ('node --version') do set NODE_VERSION=%%a
echo Found Node.js %NODE_VERSION%

REM Check if we're in the correct directory
if not exist "external\open-webui-8-6-2025" (
    echo ERROR: external\open-webui-8-6-2025 directory not found.
    echo Please run this script from the root of the openwebui-playground directory.
    pause
    exit /b 1
)

REM Navigate to the Open WebUI directory
echo [2/4] Navigating to Open WebUI directory...
cd external\open-webui-8-6-2025
if %errorlevel% neq 0 (
    echo ERROR: Failed to navigate to external\open-webui-8-6-2025
    pause
    exit /b 1
)

REM Install dependencies
echo [3/4] Installing frontend dependencies...
echo This may take a few minutes...
npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

REM Build the frontend
echo [4/4] Building frontend...
echo This may take a few minutes...
npm run build
if %errorlevel% neq 0 (
    echo ERROR: Frontend build failed.
    echo Please check the build output for errors.
    pause
    exit /b 1
)

echo.
echo ===============================================
echo  Frontend build completed successfully!
echo ===============================================
echo.
echo The built frontend files are now ready.
echo You can now run the backend using start_windows.bat or start.sh
echo.
echo Press any key to exit...
pause >nul
