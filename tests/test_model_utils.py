
import pytest
import torch
from PIL import Image
import io
from api.utils import transform_image, get_prediction
from unittest.mock import MagicMock

def test_transform_image():
    # Make a tiny red square to test with
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    # Run it through our transforms
    tensor = transform_image(img_bytes)

    # Should be batch_size=1, channels=3, height=224, width=224
    assert tensor.shape == (1, 3, 224, 224)
    # Make sure it's an actual PyTorch tensor
    assert isinstance(tensor, torch.Tensor)

def test_get_prediction(mocker):
    # Create a dummy model
    mock_model = MagicMock()

    # Fake a high probability for "dog"
    mock_model.return_value = torch.tensor([0.8])
    # Bypass the need for a real device
    mock_param = MagicMock()
    mock_param.device = torch.device('cpu')
    mock_model.parameters.side_effect = lambda: iter([mock_param])

    # Pass in random noise
    dummy_tensor = torch.randn(1, 3, 224, 224)

    # Check if the logic holds up
    prediction, probability = get_prediction(mock_model, dummy_tensor)

    assert prediction == "dog"
    assert probability == pytest.approx(0.8)

    # Fake a high probability for "cat" this time
    mock_model.return_value = torch.tensor([0.2])
    prediction, probability = get_prediction(mock_model, dummy_tensor)

    assert prediction == "cat"
    # Math check: 1 - 0.2 = 0.8
    assert probability == pytest.approx(0.8)
