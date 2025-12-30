@echo off
REM OMA Strategy Setup Script for Windows
REM Automates environment setup for Option Market Albatross

echo.
echo ==========================================
echo Option Market Albatross - Setup Script
echo ==========================================
echo.

REM Check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Conda found
    set USE_CONDA=true
) else (
    echo [WARN] Conda not found, will use venv instead
    set USE_CONDA=false
)

REM Check Python version
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Please install Python 3.11+
    exit /b 1
)

python --version
echo.

REM Create environment
if "%USE_CONDA%"=="true" (
    echo [INFO] Creating Conda environment...
    conda env create -f environment.yml
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create conda environment
        exit /b 1
    )
    echo [OK] Conda environment 'oma-bot' created
    echo.
    echo To activate: conda activate oma-bot
) else (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment
        exit /b 1
    )
    echo [OK] Virtual environment created
    echo.
    echo To activate: venv\Scripts\activate
)

REM Install root dependencies
echo.
echo [INFO] Installing root dependencies...
if "%USE_CONDA%"=="true" (
    call conda activate oma-bot
) else (
    call venv\Scripts\activate.bat
)

python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install root dependencies
    exit /b 1
)
echo [OK] Root dependencies installed

REM Install strategy dependencies
echo.
echo [INFO] Installing strategy dependencies...
cd QuantConnectOmaStrategy
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install strategy dependencies
    exit /b 1
)
cd ..
echo [OK] Strategy dependencies installed

REM Verify installation
echo.
echo [INFO] Verifying installation...
python -c "import numpy, pandas, scipy; print('[OK] Core dependencies: OK')"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Core dependencies verification failed
    exit /b 1
)

echo.
echo ==========================================
echo Setup completed successfully!
echo ==========================================
echo.
echo Next Steps:
echo 1. Activate environment:
if "%USE_CONDA%"=="true" (
    echo    conda activate oma-bot
) else (
    echo    venv\Scripts\activate
)
echo.
echo 2. Upload to QuantConnect:
echo    - Sign up at https://www.quantconnect.com
echo    - Upload QuantConnectOmaStrategy\ folder
echo    - Run backtest in cloud
echo.
echo 3. See INSTALL.md for detailed instructions
echo.

pause

