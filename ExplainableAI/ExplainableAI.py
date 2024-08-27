import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize the VGG16 model
model = VGG16(weights='imagenet')

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Load images
image_paths = {
    "Hund": 'D:\\Studium\\3. Semster\\Bildanalyse\\Assignments\\7\\Bilder\\Hund.png',
    "Katze": 'D:\\Studium\\3. Semster\\Bildanalyse\\Assignments\\7\\Bilder\\Katze.png'
}

# Preprocess images
images = {name: load_and_preprocess_image(path) for name, path in image_paths.items()}

# Predict the class for each image
for name, img in images.items():
    preds = model.predict(img)
    print(f'Predictions for {name}:', decode_predictions(preds, top=3)[0])

# Define the size of the occlusion
occlusion_size = 50
occlusion_stride = 25

# Function to apply occlusion
def apply_occlusion(image, size=occlusion_size, stride=occlusion_stride):
    output = []
    # Use the shape of the image[0] since it has an extra dimension for batch size
    for y in range(0, image.shape[1], stride):
        for x in range(0, image.shape[2], stride):
            occluded_image = image.copy()
            # Ensure the occlusion stays within the image dimensions
            occluded_image[0, max(0, y-size//2):min(y+size//2, image.shape[1]), max(0, x-size//2):min(x+size//2, image.shape[2]), :] = 0
            output.append((x, y, model.predict(occluded_image)))
    return output

# Apply occlusion and analyze results
for name, img in images.items():
    occlusion_results = apply_occlusion(img)
    heatmap = np.zeros((224, 224))
    for x, y, pred in occlusion_results:
        # We use the prediction for the original class
        original_class = np.argmax(model.predict(img))
        # Ensure the assignment is within the bounds of the heatmap
        x_min = max(x - occlusion_size // 2, 0)
        y_min = max(y - occlusion_size // 2, 0)
        x_max = min(x + occlusion_size // 2, 224)
        y_max = min(y + occlusion_size // 2, 224)
        heatmap[y_min:y_max, x_min:x_max] = pred[0][original_class]

    # Normalize the heatmap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title(f'Heatmap for {name}')
    plt.show()
