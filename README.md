# Deep Learning Experimental Projects

This repository contains a collection of experimental projects focused on various aspects of deep learning. Each folder represents a different project that explores specific techniques and methodologies within the field of deep learning

## Project Structure

### 1. **Self-SupervisedLearning**
   - **Overview**: This project explores self-supervised learning techniques by implementing a method where the model predicts image rotations as a pretext task. The goal is to train a Convolutional Neural Network (CNN) to learn useful representations from unlabeled data by predicting the correct orientation of rotated images. The project involves data augmentation (rotating images at different angles), training the model, and evaluating its ability to recognize image orientations.
   - **Key Files**:
     - **`Self-SupervisedLearning.py`**: Contains the implementation of the self-supervised learning task, including data augmentation, model definition, training loop, and evaluation.

### 2. **ObjectDetection**
   - **Overview**: This project focuses on object detection using deep learning models. The process involves loading a dataset, initializing a pre-trained model (e.g., Faster R-CNN or YOLO), training the model on the dataset, and visualizing the detection results. The objective is to accurately identify and localize objects within images.
   - **Key Files**:
     - **`Data_Loading.py`**: Handles loading and preprocessing the dataset for training.
     - **`Model_Initialization.py`**: Initializes the object detection model, possibly using a pre-trained backbone.
     - **`Data_Prediction.py`**: Handles the inference process, applying the trained model to detect objects in new images.
     - **`Data_Visualization.py`**: Visualizes the detected objects in images, overlaying bounding boxes and class labels.
     - **`Dataset_Class.py`**: Defines the dataset class, including how data is fed into the model.
     - **`main.py`**: Orchestrates the entire object detection pipeline, from data loading to visualization of results.

### 3. **Image-Segmentation**
   - **Overview**: This project is dedicated to image segmentation, where the task is to partition an image into multiple segments representing different objects or regions. The project involves defining a dataset, loading the data, implementing a neural network for segmentation (such as U-Net), and training the model to perform pixel-wise classification.
   - **Key Files**:
     - **`data_loading_final.py`**: Responsible for loading and preprocessing image data specifically for segmentation tasks.
     - **`Dataset_final.py`**: Defines the dataset class tailored for segmentation, including mask handling and data augmentation.
     - **`neural_network_final.py`**: Implements the neural network architecture for segmentation, potentially using a U-Net or similar model.
     - **`main_final.py`**: Manages the training process, including model training, evaluation, and saving of results.

### 4. **ImageClassification**
   - **Overview**: This project focuses on image classification, where the task is to classify images into predefined categories. The project includes loading a labeled dataset, defining a neural network (such as a simple CNN), training the model, and evaluating its classification accuracy.
   - **Key Files**:
     - **`data_loading.py`**: Handles the loading and preprocessing of image data for classification.
     - **`NN.py`**: Defines the neural network architecture used for classification, possibly a simple CNN.
     - **`main.py`**: Contains the main training loop, including model training, validation, and performance evaluation.

### 5. **ExplainableAI**
   - **Overview**: This project explores Explainable AI (XAI), focusing on making the predictions of neural networks more interpretable. The project implements techniques such as Grad-CAM or LIME to visualize the important regions in an image that the model focuses on when making predictions. This is crucial for understanding and trusting AI decisions, especially in critical applications.
   - **Key Files**:
     - **`ExplainableAI.py`**: Implements various explainability techniques, allowing users to visualize and interpret model decisions, providing insights into the model's reasoning process.

### 6. **ConvolutionalNeuralNetworks**
   - **Overview**: This project focuses on the implementation and training of Convolutional Neural Networks (CNNs), which are widely used in tasks involving image data. The project involves loading data, defining a CNN architecture, training the model, and evaluating its performance on image recognition tasks.
   - **Key Files**:
     - **`data_loading.py`**: Loads and preprocesses image data for CNN training.
     - **`CNN.py`**: Contains the CNN architecture implementation, possibly including layers like convolutional, pooling, and fully connected layers.
     - **`main.py`**: Orchestrates the training process, managing the data flow, model training, and evaluation.
