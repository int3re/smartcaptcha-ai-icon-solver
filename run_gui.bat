@echo off
setlocal

rem Dependencies are installed only when the environment is created; rerun
rem "pip install -r requirements_gui.txt" manually after changing requirements.
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements_gui.txt
) else (
  call .venv\Scripts\activate
)

python app_flet.py

endlocal
