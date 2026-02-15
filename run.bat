@echo off
if "%1"=="" goto help

if "%1"=="install" goto install
if "%1"=="preprocess" goto preprocess
if "%1"=="train" goto train
if "%1"=="start-api" goto start-api
if "%1"=="test" goto test
if "%1"=="docker-build" goto docker-build
if "%1"=="smoke" goto smoke

echo Unknown command: %1
goto help

:install
echo Installing dependencies...
pip install -r requirements.txt
goto end

:preprocess
echo Running Preprocessing...
python src/data/preprocess.py
goto end

:train
echo Starting Training...
python src/train.py --epochs 5 --batch_size 32
goto end

:start-api
echo Starting API...
uvicorn api.main:app --reload
goto end

:test
echo Running Tests...
pytest tests/
goto end

:docker-build
echo Building Docker Image...
docker build -f docker/Dockerfile -t cats-dogs-classifier .
goto end

:smoke
echo Running Smoke Tests...
call scripts\smoke_test.bat
goto end

:help
echo Usage: run.bat [command]
echo Commands:
echo   install       Install dependencies
echo   preprocess    Run data preprocessing
echo   train         Train the model
echo   start-api     Start the API locally
echo   test          Run unit tests
echo   docker-build  Build Docker image
echo   smoke         Run smoke tests (requires running API)
goto end

:end
