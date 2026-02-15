from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from api.schemas import PredictionResponse
from api.utils import load_model, transform_image, get_prediction
from monitoring.logging import setup_logging, LoggingMiddleware
import os

# Setup logging
setup_logging()

app = FastAPI(title="Cats vs Dogs Classifier API")
app.add_middleware(LoggingMiddleware)

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model_path = "models/model.pt"
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model loaded successfully.")
    else:
        print("Warning: Model not found at models/model.pt")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    contents = await file.read()
    try:
        image_tensor = transform_image(contents)
        prediction, probability = get_prediction(model, image_tensor)
        return {
            "filename": file.filename,
            "prediction": prediction,
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
