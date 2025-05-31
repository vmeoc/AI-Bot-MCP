@echo off
echo Activating virtual environment...
call .venv\Scripts\activate

echo Starting Google Calendar Assistant...
python calendar_assistant.py

pause
