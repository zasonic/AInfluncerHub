@echo off
setlocal enabledelayedexpansion
title AinfluencerHub - Development Setup
color 0F

echo.
echo ============================================================
echo              AinfluencerHub - Setup and Launch
echo ============================================================
echo.

:: ── 1. Python ────────────────────────────────────────────────────────────────
set PYTHON_CMD=
for %%C in (python3 python py) do (
    if not defined PYTHON_CMD (
        %%C --version >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            for /f "tokens=2" %%V in ('%%C --version 2^>^&1') do (
                for /f "tokens=1,2 delims=." %%A in ("%%V") do (
                    if %%A geq 3 if %%B geq 9 set PYTHON_CMD=%%C
                )
            )
        )
    )
)
if not defined PYTHON_CMD (
    echo [MISSING] Python 3.9+ is required.
    echo           Download from: https://www.python.org/downloads/
    echo           Check "Add Python to PATH" during install.
    start https://www.python.org/downloads/
    pause & exit /b 1
)
echo [OK] Python: %PYTHON_CMD%

:: ── 2. Node.js ───────────────────────────────────────────────────────────────
node --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [MISSING] Node.js is required.
    echo           Download LTS from: https://nodejs.org/
    start https://nodejs.org/
    pause & exit /b 1
)
echo [OK] Node.js: found

:: ── 3. Rust / Cargo ───────────────────────────────────────────────────────────
cargo --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [MISSING] Rust is required to build the Tauri desktop shell.
    echo           Install from: https://rustup.rs/
    echo           Run the installer, restart this window, then try again.
    start https://rustup.rs/
    pause & exit /b 1
)
echo [OK] Rust / Cargo: found

:: ── 4. Python venv ────────────────────────────────────────────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo [..] Creating Python virtual environment...
    %PYTHON_CMD% -m venv venv
    if !ERRORLEVEL! neq 0 ( echo [ERROR] venv creation failed. & pause & exit /b 1 )
)
echo [OK] Python venv ready.

:: ── 5. PyTorch with CUDA ─────────────────────────────────────────────────────
venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [..] Installing PyTorch with CUDA 12.4 support...
    venv\Scripts\pip.exe install torch torchvision ^
        --index-url https://download.pytorch.org/whl/cu124 --quiet --prefer-binary
    echo [OK] PyTorch installed.
) else (
    echo [OK] PyTorch + CUDA already present.
)

:: ── 6. Python backend dependencies ───────────────────────────────────────────
echo [..] Installing Python packages...
venv\Scripts\pip.exe install -r python\requirements.txt --quiet --prefer-binary
echo [OK] Python packages ready.

:: ── 7. Node modules ──────────────────────────────────────────────────────────
if not exist "node_modules" (
    echo [..] Installing Node packages...
    npm install --silent
    if !ERRORLEVEL! neq 0 ( echo [ERROR] npm install failed. & pause & exit /b 1 )
    echo [OK] Node packages installed.
) else (
    echo [OK] Node packages already present.
)

:: ── 8. Output directories ────────────────────────────────────────────────────
if not exist "output\influencers" mkdir output\influencers

:: ── 9. Launch ─────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo  All dependencies ready. Launching AinfluencerHub...
echo  Close this window or press Ctrl+C to stop.
echo ============================================================
echo.

:: Set the Python executable so Tauri's lib.rs can find the venv
set PATH=%CD%\venv\Scripts;%PATH%

npm run tauri:dev

if !ERRORLEVEL! neq 0 (
    echo.
    echo [ERROR] Application exited with an error.
    echo         Check the output above for details.
    pause
)

endlocal
