@echo off
chcp 65001 >nul
rem 웨더핏 에이전트 실행: 서버를 켜고, 준비되면 브라우저에 옷차림 추천 화면을 연다.
rem 사용법: 이 파일을 더블클릭한다. 끌 때는 검은 창을 닫는다.

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

rem 가상환경(.venv)이 있으면 그 파이썬을 쓰고, 없으면 기본 python 을 쓴다.
set PY=python
if exist "%~dp0.venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"
if exist "%~dp0..\.venv\Scripts\python.exe" set PY="%~dp0..\.venv\Scripts\python.exe"

echo 웨더핏 에이전트를 시작합니다. 준비되면 브라우저가 자동으로 열립니다.
echo 끝내려면 이 창을 닫으세요.
echo.

rem 서버가 응답할 때까지 기다렸다가 화면을 연다 (최대 약 2분).
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 60;$i++){try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/ -TimeoutSec 2 | Out-Null; Start-Process 'http://127.0.0.1:8000/'; break}catch{Start-Sleep 2}}"

%PY% -m uvicorn server:app --app-dir src --port 8000

echo.
echo 서버가 멈췄습니다. 위 메시지를 확인하세요.
pause
