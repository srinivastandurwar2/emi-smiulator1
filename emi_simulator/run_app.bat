@echo off
REM Launch the EMI Virtual Lab on Windows (double-click this file).
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python -m streamlit run app.py
pause
