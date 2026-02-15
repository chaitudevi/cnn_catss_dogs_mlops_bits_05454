
import torch
import os
import sys

# Add root directory to path
sys.path.append(os.getcwd())

from src.models.cnn import SimpleCNN

def create_dummy_model(path="models/model.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = SimpleCNN()
    torch.save(model.state_dict(), path)
    print(f"Dummy model saved to {path}")

if __name__ == "__main__":
    create_dummy_model()
