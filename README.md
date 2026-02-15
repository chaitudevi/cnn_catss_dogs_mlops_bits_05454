# MLOps Cats vs Dogs Classifier

## Project Overview
This project implements an end-to-end MLOps pipeline for binary image classification (Cats vs Dogs). It includes data versioning with DVC, experiment tracking with MLflow, model packaging with Docker, CI/CD with GitHub Actions, and deployment configurations for Kubernetes and Docker Compose.

## Architecture
- **Data**: Versioned with DVC.
- **Model**: Simple CNN built with PyTorch.
- **Tracking**: MLflow for metrics and artifacts.
- **API**: FastAPI for inference.
- **Containerization**: Docker.
- **CI/CD**: GitHub Actions (Build, Test, Push to GHCR, Deploy).
- **Deployment**: Docker Compose with self-hosted runner simulation.
- **Monitoring**: Custom middleware for request counting and latency tracking.

## Setup Instructions

### Prerequisites
- Python 3.9+
- Docker
- Git

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Training
To train the model:
```bash
python src/train.py
```

### Inference API
To start the API locally:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- **Health Check**: `GET /health`
- **Prediction**: `POST /predict` (form-data `file`)
- **Metrics**: `GET /metrics`

### Testing
- **Unit Tests**: `python -m pytest`
- **Manual Image Test**:
  ```bash
  python scripts/test_image.py path/to/image.jpg
  ```
- **Performance Simulation**:
  ```bash
  python scripts/simulate_performance.py
  ```

### Docker
Build and run the container:
```bash
docker build -t cats-dogs-classifier .
docker run -p 8000:8000 cats-dogs-classifier
```

## CI/CD Pipeline
- **Continuous Integration**: Triggers on push to `master`. Runs tests, builds Docker image, and pushes to GitHub Container Registry.
- **Continuous Deployment**: Deploys the container using `docker-compose` and runs smoke tests.

## Monitoring
The API exposes a `/metrics` endpoint that provides:
- Total request counts.
- Counts by HTTP status code.
- Average latency.

