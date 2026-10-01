@echo off

python -m venv .venv
call .venv\Scripts\activate

python -m pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python check_env.py

pause
