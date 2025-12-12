@echo off
cd /d "C:\projects\clone-tracker"

if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)

py -3.10 clone_tracker_full_tk.py
