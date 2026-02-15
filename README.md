# MLOps Cats vs Dogs Classifier

## Project Overview
This project implements an end-to-end MLOps pipeline for binary image classification (Cats vs Dogs). It includes data versioning with DVC, experiment tracking with MLflow, model packaging with Docker, CI/CD with GitHub Actions, and deployment configurations for Kubernetes and Docker Compose.

## Architecture
- **Data**: Versioned with DVC.
- **Model**: Simple CNN built with PyTorch.
- **Tracking**: MLflow for metrics and artifacts.
- **API**: FastAPI for inference.
- **Containerization**: Docker.
- **CI/CD**: GitHub Actions.
- **Deployment**: Kubernetes / Docker Compose.

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
uvicorn api.main:app --reload
```

### Docker
Build and run the container:
```bash
docker build -t cats-dogs-classifier .
docker run -p 8000:8000 cats-dogs-classifier
```
