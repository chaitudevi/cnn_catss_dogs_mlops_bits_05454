
import pytest
import torch
from PIL import Image
import io
from api.utils import transform_image, get_prediction
from unittest.mock import MagicMock

def test_transform_image():
    # Create simple RGB image
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    # Test transformation
    tensor = transform_image(img_bytes)
    
    # Check shape: batch_size=1, channels=3, height=224, width=224
    assert tensor.shape == (1, 3, 224, 224)
    # Check type
    assert isinstance(tensor, torch.Tensor)

def test_get_prediction(mocker):
    # Mock model
    mock_model = MagicMock()
    
    # Mock output for "dog" (prob > 0.5)
    mock_model.return_value = torch.tensor([0.8]) 
    # Mock parameters/device
    mock_param = MagicMock()
    mock_param.device = torch.device('cpu')
    mock_model.parameters.side_effect = lambda: iter([mock_param])
    
    # Create dummy tensor
    dummy_tensor = torch.randn(1, 3, 224, 224)
    
    # Test prediction
    prediction, probability = get_prediction(mock_model, dummy_tensor)
    
    assert prediction == "dog"
    assert probability == pytest.approx(0.8)
    
    # Mock output for "cat" (prob <= 0.5)
    mock_model.return_value = torch.tensor([0.2])
    prediction, probability = get_prediction(mock_model, dummy_tensor)
    
    assert prediction == "cat"
    # Probability should be 1 - 0.2 = 0.8
    assert probability == pytest.approx(0.8)
