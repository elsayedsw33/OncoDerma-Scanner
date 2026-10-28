@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else (
    if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe" (
        set "PYTHON_CMD=%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe"
    ) else (
        where py >nul 2>nul
        if not errorlevel 1 (
            py -3.13 --version >nul 2>nul
            if not errorlevel 1 set "PYTHON_CMD=py -3.13"
            if not defined PYTHON_CMD (
                py -3.12 --version >nul 2>nul
                if not errorlevel 1 set "PYTHON_CMD=py -3.12"
            )
            if not defined PYTHON_CMD (
                py -3.11 --version >nul 2>nul
                if not errorlevel 1 set "PYTHON_CMD=py -3.11"
            )
            if not defined PYTHON_CMD (
                py -3.10 --version >nul 2>nul
                if not errorlevel 1 set "PYTHON_CMD=py -3.10"
            )
        )
    )
)

if not defined PYTHON_CMD (
    echo Python 3.10, 3.11, 3.12, or 3.13 was not found.
    echo Install Python 3.13, then open a new Command Prompt and run this file again.
    echo This app's TensorFlow dependency does not support Python 3.14.
    pause
    exit /b 1
)

%PYTHON_CMD% -m streamlit --version >nul 2>nul
if errorlevel 1 (
    echo App dependencies are not installed for the selected Python.
    echo Run: %PYTHON_CMD% -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting OncoDerma Scanner...
%PYTHON_CMD% -m streamlit run app.py

if errorlevel 1 (
    echo.
    echo The app stopped with an error. Check the message above.
    pause
)

endlocal
