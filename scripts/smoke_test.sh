#!/bin/bash

# Smoke Test Script

echo "Checking Health Endpoint..."
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)

if [ "$HEALTH" -eq 200 ]; then
    echo "Health Check Passed!"
else
    echo "Health Check Failed! Status Code: $HEALTH"
    exit 1
fi

echo "Checking Prediction Endpoint..."
# We need a test image. Let's create a dummy one if not exists or use one from data.
# For simplicity, we assume we invoke this where we can access a file, or we send dummy bytes?
# curl sends form-data.

# Create a small dummy image
python -c "from PIL import Image; Image.new('RGB', (100, 100)).save('smoke_test.jpg')"

PREDICT=$(curl -s -o /dev/null -w "%{http_code}" -X POST -F "file=@smoke_test.jpg" http://localhost:8000/predict)

if [ "$PREDICT" -eq 200 ]; then
    echo "Prediction Check Passed!"
    rm smoke_test.jpg
    exit 0
else
    echo "Prediction Check Failed! Status Code: $PREDICT"
    rm smoke_test.jpg
    exit 1
fi
