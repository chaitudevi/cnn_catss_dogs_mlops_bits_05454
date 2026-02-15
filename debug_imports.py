
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    print("Attempting to import api.utils...")
    import api.utils
    print("Successfully imported api.utils")
except Exception as e:
    print(f"Failed to import api.utils: {e}")

try:
    print("Attempting to import src.models.cnn...")
    from src.models.cnn import SimpleCNN
    print("Successfully imported src.models.cnn")
except Exception as e:
    print(f"Failed to import src.models.cnn: {e}")
