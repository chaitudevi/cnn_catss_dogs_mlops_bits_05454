
import requests
import time
import csv
import random
import io
from PIL import Image

# Configuration
API_URL = "http://localhost:8000"
NUM_REQUESTS = 20
LOG_FILE = "performance_log.csv"

def generate_dummy_image(color='red'):
    """Generates a simple dummy image in memory."""
    img = Image.new('RGB', (224, 224), color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def simulate_traffic():
    print(f"Starting simulation. Sending {NUM_REQUESTS} requests to {API_URL}...")
    
    # Check if API is up
    try:
        requests.get(f"{API_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print("Error: API is not reachable. Make sure it's running (e.g., uvicorn api.main:app).")
        return

    # Open CSV for logging
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Request_ID", "True_Label", "Predicted_Label", "Probability", "Status_Code", "Latency_Seconds"])
        
        successful_preds = 0
        
        for i in range(1, NUM_REQUESTS + 1):
            # Simulate true label (0: Cat, 1: Dog)
            true_label = random.choice(["Cat", "Dog"])
            # Generate image based on logic (just random colors for now, so model prediction is noise)
            color = 'red' if true_label == "Cat" else 'blue'
            image_data = generate_dummy_image(color)
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": ("test_image.jpg", image_data, "image/jpeg")}
                )
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    pred = data.get("prediction", "Unknown")
                    prob = data.get("probability", 0.0)
                    writer.writerow([i, true_label, pred, prob, response.status_code, f"{latency:.4f}"])
                    successful_preds += 1
                else:
                    writer.writerow([i, true_label, "Error", 0.0, response.status_code, f"{latency:.4f}"])
                    
            except Exception as e:
                print(f"Request {i} failed: {e}")
                writer.writerow([i, true_label, "Failed", 0.0, "N/A", "N/A"])
            
            # Rate limiting
            time.sleep(0.1)
            
    print(f"Simulation complete. Logged {successful_preds}/{NUM_REQUESTS} successful requests to {LOG_FILE}")

if __name__ == "__main__":
    simulate_traffic()
