@echo off
REM Track Error Detection - Setup Script (Windows)

echo.
echo ============================================
echo  Track Error Detection System Setup
echo ============================================
echo.

REM Check Python version
python --version
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo.
echo Creating directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\annotations" mkdir data\annotations
if not exist "models" mkdir models
if not exist "checkpoints" mkdir checkpoints
if not exist "output" mkdir output
if not exist "data\dataset\train" mkdir data\dataset\train
if not exist "data\dataset\valid" mkdir data\dataset\valid
if not exist "data\dataset\test" mkdir data\dataset\test

REM Create data.yaml
echo.
echo Creating data.yaml...
python scripts\create_data_yaml.py

echo.
echo ============================================
echo  Setup completed successfully!
echo ============================================
echo.
echo Next steps:
echo 1. Add your images to data\raw\
echo 2. Run: python scripts\prepare_dataset.py
echo 3. Train: python training\train.py
echo 4. Detect: python inference\detect_realtime.py
echo.

pause

