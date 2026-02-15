@echo off
setlocal

echo Checking Health Endpoint...
for /f %%a in ('curl -s -o NUL -w "%%{http_code}" http://127.0.0.1:8000/health') do set HEALTH=%%a

if "%HEALTH%"=="200" (
    echo Health Check Passed!
) else (
    echo Health Check Failed! Status Code: %HEALTH%
    exit /b 1
)

echo Checking Prediction Endpoint...
rem Create a small dummy container image
python -c "from PIL import Image; Image.new('RGB', (100, 100)).save('smoke_test.jpg')"

for /f %%b in ('curl -s -o NUL -w "%%{http_code}" -X POST -F "file=@smoke_test.jpg" http://127.0.0.1:8000/predict') do set PREDICT=%%b

if "%PREDICT%"=="200" (
    echo Prediction Check Passed!
    del smoke_test.jpg
    exit /b 0
) else (
    echo Prediction Check Failed! Status Code: %PREDICT%
    del smoke_test.jpg
    exit /b 1
)
