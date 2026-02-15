# Cats vs Dogs Classification - MLOps Project

This repository contains an end-to-end MLOps pipeline to classify images of cats and dogs using a Convolutional Neural Network (CNN).

## Table of Contents
* [Overview](#overview)
* [Demo Video](#demo-video)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Testing](#testing)
* [CI / CD Pipeline](#ci-cd-pipeline)
* [Deployment](#deployment)
* [Monitoring](#monitoring--observability)

## Overview
This project builds a machine learning classifier to distinguish between cats and dogs. The solution incorporates:
* **Data pipeline**: Versioning with DVC.
* **Model development**: Custom CNN architecture using PyTorch.
* **Experiment tracking**: MLflow integration for metrics and artifacts.
* **Automated testing**: PyTest based testcases.
* **CI / CD**: GitHub Actions pipeline to perform testing, building, and deployment.
* **Monitoring**: Custom middleware for request counting and latency tracking.

## Demo Video
*(Link to demo video to be added)*

## Dataset
Title: Microsoft Cats vs Dogs Dataset
Source: [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=54765) (or Kaggle)

The dataset contains images of cats and dogs.
* Binary target: Cat (0) / Dog (1)

## Project Structure

cnn_catss_dogs_mlops_bits_05454/
```
    ├──src/
    │   ├── api/                       # API end-points
    │   │    ├──main.py
    │   │    ├──schemas.py
    │   │    ├──utils.py
    │   ├── data/                      # Data loading and preprocessing
    │   │    ├──preprocess.py
    │   ├── models/                    # Model definition
    │   │    ├──cnn.py
    │   ├── train.py                   # Training script
    │
    ├──tests/                          # Unit tests (pytest)
    │    ├──test_api.py
    │    ├──test_preprocess.py
    │    ├──test_model_utils.py
    │
    ├──scripts/                        # Pipeline execution scripts
    │    ├──smoke_test.sh
    │    ├──test_image.py
    │    ├──simulate_performance.py
    │    ├──create_dummy_model.py
    │
    ├──deploy/
    │    ├──docker-compose.yml
    │
    ├──docker/
    │    ├──Dockerfile
    │
    ├──monitoring/
    │    ├──logging.py
    │
    ├──.github/
    │   ├── workflows/
    │        ├──ci.yml                 # GitHub Actions CI/CD pipeline
    │
    ├──requirements.txt
    ├──README.md
```

## Installation
### Prerequisites
* Python 3.9+
* Docker
* Git

### Setup

1. Clone the repository
```bash
   git clone https://github.com/chaitudevi/cnn_catss_dogs_mlops_bits_05454.git
   cd cnn_catss_dogs_mlops_bits_05454
```

2. Create a virtual environment (recommended)
```bash
   python -m venv .venv
   .venv\Scripts\activate # Windows
   # source .venv/bin/activate # Linux/Mac
```

3. Install the dependencies
```bash
   pip install -r requirements.txt
```

4. Pull data from DVC
```bash
   dvc pull
```

## Usage

**Train the Model**
```bash
python src/train.py
```

This will:
* Load data
* Train the CNN model
* Log experiments to MLflow
* Save model artifacts to `models/`

**Run the API Locally**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- **Health Check**: `GET /health`
- **Prediction**: `POST /predict` (form-data `file`)
- **Metrics**: `GET /metrics`

**Test the API**
```bash
python scripts/test_image.py path/to/your/image.jpg
```
Or:
```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:8000/predict
```

## Testing

The project includes comprehensive unit tests.

**Run All Tests**
```bash
python -m pytest
```

**Run Performance Simulation**
```bash
python scripts/simulate_performance.py
```

## CI / CD Pipeline
The project uses GitHub Actions for continuous integration. The pipeline runs on push to `master`.

## Pipeline Stages
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Build & Test│────▶│  Push Image  │────▶│    Deploy    │
└──────────────┘     └──────────────┘     └──────────────┘
```
1. **Build & Test**: Installs dependencies, runs pytest.
2. **Push Image**: Builds Docker image and pushes to GitHub Container Registry (GHCR).
3. **Deploy**: Deploys using Docker Compose and runs smoke tests.

## Deployment

### Docker
1. Build the image:
```bash
docker build -t cats-dogs-classifier .
```
2. Run locally:
```bash
docker run -p 8000:8000 cats-dogs-classifier
```

### Docker Compose
Deploy using the pre-built image from GHCR:
```bash
docker-compose -f deploy/docker-compose.yml up -d
```

## Monitoring & Observability
- Request logging enabled via custom middleware.
- Metrics exposed at `/metrics` (request count, latency).
- Performance simulation script logs results to CSV.

