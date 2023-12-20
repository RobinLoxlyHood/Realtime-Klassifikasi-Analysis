@echo off
timeout /t 180
taskkill /F /IM
rem Activate the virtual environment
call "D:\Materi Kuliah\SKRIPSI\skripsiSentiment\Scripts\activate.bat" && python "D:\Materi Kuliah\SKRIPSI\src\realtime.py"