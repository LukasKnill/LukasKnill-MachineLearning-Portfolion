from pathlib import Path
from natsort import natsorted
from pathlib import Path
from PIL import Image
import numpy as np

DATA_ROOT = Path(r"C:\Users\KI-Lab2\Documents\Finetune\soil20230711\soil")
OUT_FOLDER = Path(r"C:\Users\KI-Lab2\Documents\Finetune\out")

def find_all_images(data_root):
    data_root_path = Path(data_root)
    images = []
    for image_path in data_root_path.rglob('*'):
        if image_path.is_file() and image_path.suffix.lower() in {'.png'}:
            images.append(image_path)
    images = natsorted(images)
    return images

def process_images(image_paths, out_folder):
    # Farbenspektrum definieren
    r_range = [i for i in range(200, 256, 5)]
    g_range = [i for i in range(0, 101, 5)]
    b_range = [i for i in range(0, 101, 5)]

    # Erstelle den Ausgabepfad, falls nicht existent
    out_folder.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image = Image.open(image_path)
        image_np = np.array(image)

        # Erstelle ein Set, um die Farben zu speichern
        colors = set()

        # Durchsuche das Bild nach den Farben im gewünschten Spektrum
        for x in range(image_np.shape[0]):
            for y in range(image_np.shape[1]):
                pixel = image_np[x, y]
                if pixel[0] in r_range and pixel[1] in g_range and pixel[2] in b_range:
                    colors.add((pixel[0], pixel[1], pixel[2]))

        # Erstelle für jede gefundene Farbe eine Binärmaske
        image_name = image_path.stem
        image_out_folder = out_folder / image_name
        image_out_folder.mkdir(parents=True, exist_ok=True)

        for color in colors:
            mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
            mask[(image_np[:, :, 0] == color[0]) & (image_np[:, :, 1] == color[1]) & (image_np[:, :, 2] == color[2])] = 255
            new_image = Image.fromarray(mask)
            new_image_path = image_out_folder / f"{color[0]}_{color[1]}_{color[2]}_binary.png"
            new_image.save(new_image_path)

def main():
    masks = find_all_images(DATA_ROOT)
    print(f"Number of images found: {len(masks)}")
    process_images(masks, OUT_FOLDER)

if __name__ == "__main__":
    main()
