@echo off

if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate

python -m pip install --upgrade pip

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python check_env.py

pause
