import os
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

def process_images(raw_dir, processed_dir, split_ratio=(0.8, 0.1, 0.1), img_size=(224, 224)):
    """
    Reads images from raw_dir/Cat and raw_dir/Dog, resizes them,
    and splits them into train/val/test in processed_dir.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    # The dataset unzips into a 'PetImages' folder
    pet_images_path = raw_path / 'PetImages'

    # Check if data exists (look for PetImages/Cat and PetImages/Dog AND images inside), if not, try to download
    has_data = False
    if (pet_images_path / 'Cat').exists() and (pet_images_path / 'Dog').exists():
        cats = [f for f in (pet_images_path / 'Cat').iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        dogs = [f for f in (pet_images_path / 'Dog').iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if len(cats) > 0 and len(dogs) > 0:
            has_data = True

    if not has_data:
        print("Data (PetImages/Cat and PetImages/Dog) not found in raw_dir. Attempting to download from Kaggle...")
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            print("Downloading dataset: bhavikjikadara/dog-and-cat-classification-dataset")
            api.dataset_download_files('bhavikjikadara/dog-and-cat-classification-dataset', path=raw_dir, unzip=True)
            print("Download and unzip complete.")

        except ImportError:
            print("Error: 'kaggle' library not installed.")
            print("Please install it via: pip install kaggle")
            return
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("To fix this:")
            print("1. Ensure KAGGLE_USERNAME and KAGGLE_KEY environment variables are set.")
            print("2. Or place kaggle.json in ~/.kaggle/")
            print("3. Or manually download the dataset and place 'PetImages' folder in data/raw/")
            return

    # Case sensitive directory names as found in data/raw/PetImages
    classes = ['Cat', 'Dog']

    # Create processed directories (clear if exists to avoid mixing)
    if processed_path.exists():
        shutil.rmtree(processed_path)

    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(processed_path / split / cls, exist_ok=True)

    for cls in classes:
        src_folder = pet_images_path / cls
        if not src_folder.exists():
             print(f"Warning: Source folder {src_folder} does not exist. Skipping.")
             continue

        images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        # n_test = n_total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]

        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }

        print(f"Processing {cls}: {n_total} total images.")

        for split, imgs in splits.items():
            print(f"  Saving {len(imgs)} to {split}...")
            for img_name in tqdm(imgs, desc=f"{cls}->{split}"):
                try:
                    src_path = src_folder / img_name
                    dst_path = processed_path / split / cls / img_name

                    with Image.open(src_path) as img:
                        img = img.convert('RGB')
                        img = img.resize(img_size)
                        img.save(dst_path)
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data', type=str, default='data/raw')
    parser.add_argument('--processed_data', type=str, default='data/processed')
    args = parser.parse_args()

    process_images(args.raw_data, args.processed_data)

