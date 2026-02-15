import os
import shutil
from pathlib import Path
from PIL import Image
from src.data.preprocess import process_images
import pytest

@pytest.fixture
def temp_data_dirs(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    os.makedirs(raw_dir / "Cat")
    os.makedirs(raw_dir / "Dog")
    
    # Create dummy images
    img = Image.new('RGB', (100, 100), color='red')
    img.save(raw_dir / "Cat" / "cat1.jpg")
    img.save(raw_dir / "Dog" / "dog1.jpg")
    
    return raw_dir, processed_dir

def test_process_images(temp_data_dirs):
    raw_dir, processed_dir = temp_data_dirs
    
    # Run processing
    process_images(str(raw_dir), str(processed_dir), split_ratio=(0.5, 0.25, 0.25))
    
    # Check if directories created
    assert (processed_dir / "train" / "Cat").exists()
    assert (processed_dir / "val" / "Dog").exists()
    
    # Check if images processed (we have 1 image per class, so split might put it in train or test depending on shuffle)
    # But we should find *some* images in processed directory structure
    total_processed = 0
    for split in ['train', 'val', 'test']:
        for cls in ['Cat', 'Dog']:
            total_processed += len(os.listdir(processed_dir / split / cls))
            
    assert total_processed == 2
    
    # Check image size
    # Find one image
    found = False
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    assert img.size == (224, 224)
                found = True
                break
        if found: break
    assert found
