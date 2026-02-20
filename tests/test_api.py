from fastapi.testclient import TestClient
from api.main import app
import os
import pytest
from PIL import Image
import io

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint_no_model(mocker):
    # Pretend the model hasn't been loaded yet
    mocker.patch("api.main.model", None)

    # Make up a mimic image string to send over
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    assert response.status_code == 503

# Note: We can't really test the prediction without a heavy model,
# so we just mimic the model's output here.

def test_predict_mocked_model(mocker):
    # Set up a mimic model
    mock_model = mocker.Mock()
    # mimic the helper functions so we don't need real tensors
    mocker.patch("api.main.model", mock_model)
    mocker.patch("api.main.transform_image", return_value="tensor")
    mocker.patch("api.main.get_prediction", return_value=("cat", 0.95))

    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "cat"
    assert data["probability"] == 0.95
