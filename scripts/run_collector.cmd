@echo off
REM One polling pass over the cohort, for Windows Task Scheduler.
REM Deduplication is persistent, so overlapping or repeated runs are safe.
cd /d "%~dp0.."
"C:\Users\jfbaa\AppData\Local\Programs\Python\Python313\python.exe" scripts\collect.py >> data\collector.log 2>&1
