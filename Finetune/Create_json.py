import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Pfade definieren
DATA_ROOT = Path(r"C:\Users\KI-Lab2\Documents\Finetune\Images\Original_1024")
OUT_FOLDER = Path(r"C:\Users\KI-Lab2\Documents\Finetune\Images\Masks_1024")
JSON_PATH = Path(r"C:\Users\KI-Lab2\Documents\Finetune\Images\dataset.json")

def find_all_images_and_masks(data_root, mask_root):
    images = sorted(data_root.glob('*.png'))
    data = []

    for image_path in images:
        image_name = image_path.stem
        mask_folder = mask_root / image_name
        masks = sorted(mask_folder.glob('*.png'))
        data.append((image_path, masks))

    return data

def split_data(data, test_size=0.2, val_size=0.25):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)
    return train_data, val_data, test_data

def create_json(train_data, val_data, test_data, json_path):
    json_data = {
        "x_train": [],
        "y_train": [],
        "x_val": [],
        "y_val": [],
        "x_test": [],
        "y_test": []
    }

    for image_path, mask_paths in train_data:
        json_data["x_train"].append(str(image_path))
        json_data["y_train"].append([str(mask_path) for mask_path in mask_paths])

    for image_path, mask_paths in val_data:
        json_data["x_val"].append(str(image_path))
        json_data["y_val"].append([str(mask_path) for mask_path in mask_paths])

    for image_path, mask_paths in test_data:
        json_data["x_test"].append(str(image_path))
        json_data["y_test"].append([str(mask_path) for mask_path in mask_paths])

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def main():
    data = find_all_images_and_masks(DATA_ROOT, OUT_FOLDER)
    train_data, val_data, test_data = split_data(data)

    create_json(train_data, val_data, test_data, JSON_PATH)
    print(f"JSON-Datei erfolgreich erstellt: {JSON_PATH}")

if __name__ == "__main__":
    main()
