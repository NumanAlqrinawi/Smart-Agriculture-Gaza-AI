@echo off
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py"
) else (
    set "PYTHON_CMD=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment...
    %PYTHON_CMD% -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt

if not exist "real_data_nn_model.pkl" (
    python train_model.py
)

echo.
echo Open this address in your browser:
echo http://127.0.0.1:5000
echo.
python app.py
pause
