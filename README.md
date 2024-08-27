# Soil Quality Assessment via Image Segmentation

This repository showcases a project focused on assessing soil quality through AI-driven image segmentation. The project involves analyzing images of soil captured by a tractor to identify and segment clods, with the ultimate goal of inferring soil quality metrics. These metrics are intended to optimize agricultural machinery settings, such as the intensity and speed of operation, to achieve better farming outcomes.

## Project Overview

### Objective
The primary objective of this project is to develop an AI-based system capable of evaluating soil quality by analyzing the shapes and distribution of soil clods in images. By segmenting these clods using deep learning techniques, the system can generate insights that inform decisions about how agricultural machinery should be adjusted in terms of power, depth, and speed. This not only helps in optimizing the effectiveness of soil preparation but also contributes to more sustainable farming practices by minimizing unnecessary strain on the equipment and soil.

### Key Features
- **AI Image Segmentation**: 
  - A deep learning model, specifically a convolutional neural network (CNN), is employed to perform instance segmentation on images of the soil. The model identifies individual clods and segments them from the background, allowing for a detailed analysis of their shape, size, and distribution.
  - The segmentation process is trained on a dataset of soil images, and the model has been iteratively fine-tuned to improve its accuracy in distinguishing clods from other elements like rocks or plant debris.

- **Soil Quality Metrics**:
  - The segmented clod data is analyzed to extract several key metrics, such as clod size distribution, shape irregularity, and density. These metrics are crucial for understanding the soil's physical condition.
  - The metrics are then used to generate recommendations for the optimal operation of agricultural machinery. For example, larger or more irregularly shaped clods may indicate the need for deeper tillage or slower tractor speeds to ensure proper soil breakdown and preparation.
  - The project also includes the development of algorithms that convert these metrics into actionable insights for farmers, enabling them to adjust their machinery settings dynamically based on real-time data from the field.

- **Model Fine-Tuning**:
  - The initial model was trained on a diverse set of soil images to cover various conditions and clod characteristics. However, to enhance its performance, the model underwent several stages of fine-tuning.
  - This fine-tuning involved retraining the model with additional data, adjusting hyperparameters, and experimenting with different network architectures to reduce false positives and improve the segmentation accuracy.
  - The fine-tuning process also included validation against a separate test set of images to ensure that the model generalizes well to new, unseen data. The result is a robust model that provides reliable segmentation even under varying environmental conditions.

### Explanation of the `main` Function

The `main` function orchestrates the entire process of soil image analysis and metric generation. It includes the following key steps:

1. **Data Loading**: Images, point clouds, and disparity maps are loaded from the specified data directory (`DATA_ROOT`).

2. **Image Processing with SAM**: If enabled, the `SamAutomaticMaskGenerator` is used to segment soil clods in the images.

3. **Depth Map Processing**: Disparity maps are processed to create depth maps, which are then used to analyze and create 2D and 3D representations of the soil clods.

4. **Clod Analysis**: The segmented clods are analyzed to generate metrics like the mean weight diameter (MWD) and other relevant soil quality indicators.

5. **Result Compilation**: The results are compiled into CSV files for further analysis.

### Fine-Tuning Process

The fine-tuning of the segmentation model was a critical part of this project, aimed at enhancing its ability to accurately identify and segment soil clods. The process involved several key scripts:

- **`binary_masks.py`**: Generates binary masks from annotated images to serve as ground truth for training.
- **`Create_json.py`**: Creates a JSON annotation file that organizes the dataset.
- **`Finetune_main.py`**: Handles the entire fine-tuning process.
- **`LinearWarmupCosine.py`**: Implements a learning rate scheduler for training optimization.

### My Contributions

I was responsible for developing the parts of the code that create 2D and 3D objects from depth maps, allowing for a detailed analysis of the soil structure. I was responsible for customizing the finetuning and conducting the full training process, ensuring the model was fine-tuned for optimal performance.
