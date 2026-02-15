from pydantic import BaseModel

class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    probability: float
