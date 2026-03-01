@echo off
echo Setting up Malayalam Translator...

python -m venv venv
call venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

echo Setup completed successfully!
pause
