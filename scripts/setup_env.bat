@echo off
echo Setting up ResNet18 Flower Classification Environment...

rem Create virtual environment
python -m venv venv
call venv\Scripts\activate.bat

rem Upgrade pip
python -m pip install --upgrade pip

rem Install requirements
pip install -r requirements.txt

rem Create necessary directories
mkdir data\synthetic 2>nul
mkdir checkpoints 2>nul
mkdir results 2>nul
mkdir logs 2>nul

echo Environment setup complete!
echo To activate virtual environment: venv\Scripts\activate.bat
pause