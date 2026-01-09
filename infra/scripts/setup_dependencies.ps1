$ErrorActionPreference = "Stop"

Write-Host ">>> Setting up QsPlot Dependencies..." -ForegroundColor Cyan

# Ensure infra/dep directory exists
$ExternParams = @{
    Path     = "infra/dep"
    ItemType = "Directory"
    Force    = $true
}
New-Item @ExternParams | Out-Null

Push-Location infra/dep

# -----------------------------------------------------------------------------
# 1. Nanobind (Python Bindings)
# -----------------------------------------------------------------------------
if (-not (Test-Path "nanobind")) {
    Write-Host "Cloning nanobind..." -ForegroundColor Green
    git clone --recursive https://github.com/wjakob/nanobind.git
}
else {
    Write-Host "nanobind already exists." -ForegroundColor Gray
}

# -----------------------------------------------------------------------------
# 2. GLFW (Windowing)
# -----------------------------------------------------------------------------
if (-not (Test-Path "glfw")) {
    Write-Host "Cloning glfw..." -ForegroundColor Green
    git clone https://github.com/glfw/glfw.git
}
else {
    Write-Host "glfw already exists." -ForegroundColor Gray
}

# -----------------------------------------------------------------------------
# 3. Eigen (Linear Algebra)
# -----------------------------------------------------------------------------
if (-not (Test-Path "eigen")) {
    Write-Host "Cloning eigen..." -ForegroundColor Green
    git clone https://gitlab.com/libeigen/eigen.git
}
else {
    Write-Host "eigen already exists." -ForegroundColor Gray
}

# -----------------------------------------------------------------------------
# 4. GLAD (OpenGL Loader)
# -----------------------------------------------------------------------------
# Strategy: Use Python 'glad' package to generate the loader.
# This ensures we get the exact version/profile we want without relying on a repo.
if (-not (Test-Path "glad")) {
    Write-Host "Generating GLAD loader..." -ForegroundColor Green
    
    # Check if 'glad' is installed via pip
    try {
        python -c "import glad"
    }
    catch {
        Write-Host "Installing 'glad' generator via pip..." -ForegroundColor Yellow
        pip install glad
    }

    # Try glad v2 syntax first, then fall back to v1
    # v2: python -m glad --api gl:core=4.1 --out-path glad c
    # v1: python -m glad --api=gl:4.1 --profile=core --generator=c --out-path=glad
    try {
        python -m glad --api "gl:core=4.1" --out-path glad c
    }
    catch {
        Write-Host "Trying legacy glad syntax..." -ForegroundColor Yellow
        python -m glad --api=gl --out-path=glad --profile=core --generator=c
    }
    
    if ((Test-Path "glad/include/glad/glad.h") -or (Test-Path "glad/glad.h")) {
        Write-Host "GLAD generated successfully." -ForegroundColor Green
    }
    else {
        Write-Error "Failed to generate GLAD."
    }
}
else {
    Write-Host "glad already exists." -ForegroundColor Gray
}

# -----------------------------------------------------------------------------
# 5. Dear ImGui (Docking Branch)
# -----------------------------------------------------------------------------
if (-not (Test-Path "imgui")) {
    Write-Host "Cloning imgui (docking)..." -ForegroundColor Green
    git clone -b docking https://github.com/ocornut/imgui.git
}
else {
    Write-Host "imgui already exists." -ForegroundColor Gray
}

Pop-Location
Write-Host ">>> Dependency Setup Complete." -ForegroundColor Cyan
