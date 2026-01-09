$ErrorActionPreference = "Stop"

Write-Host ">>> Building QsPlot (VS2022) with Python 3.13..." -ForegroundColor Cyan

# Clean build directory to be safe (Clear CMake Cache)
if (Test-Path "build") {
    # Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    # Remove-Item -Recurse -Force "build"
}

# Forced Python 3.13 Path
$PythonRoot = "C:/Users/mirac/AppData/Local/Programs/Python/Python313"
$PythonExe = "$PythonRoot/python.exe"
$PythonInc = "$PythonRoot/include"

Write-Host "Using Python Root: $PythonRoot" -ForegroundColor Green

# 1. Configure
# Specifying Generator explicitly to avoid Ninja issues without VS Dev Cmd
# Visual Studio 17 2022 is standard for C++20
# Pass MANUAL_PYTHON_INCLUDE to force header inclusion for all targets
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release `
    "-DPython_EXECUTABLE=$PythonExe" `
    "-DMANUAL_PYTHON_INCLUDE=$PythonInc"

# 2. Build
# --config Release is crucial for MSVC
cmake --build build --config Release --parallel

# 3. Install/Move Artifact
$pydFile = Get-ChildItem -Path "build/Release" -Filter "*.pyd" -Recurse | Select-Object -First 1

if ($pydFile) {
    Write-Host "Found module: $($pydFile.Name)" -ForegroundColor Green
    Copy-Item -Path $pydFile.FullName -Destination "examples/" -Force
    Write-Host "Copied to examples/" -ForegroundColor Green
}
else {
    Write-Warning "Could not find generated .pyd file. Build might have failed."
}

Write-Host ">>> Build Complete." -ForegroundColor Cyan
