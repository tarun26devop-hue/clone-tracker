@echo off
rem change to project folder
cd /d "C:\path\to\repo"
rem activate venv if you used one (Windows venv)
if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)
rem run app (use python or py)
py -3.10 clone_tracker_full_tk.py
