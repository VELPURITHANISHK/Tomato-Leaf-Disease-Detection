# Tomato Leaf Disease Detection

A deep learning-based web application that detects and classifies tomato leaf diseases from leaf images using a Convolutional Neural Network (CNN). The model is trained on the PlantVillage dataset and can identify multiple tomato leaf diseases with high accuracy.

## Overview

This project leverages Computer Vision and Deep Learning techniques to assist farmers and agricultural professionals in early disease detection. Users can upload a tomato leaf image through the web interface, and the system predicts the disease category in real time.

## Dataset

This project was trained using the PlantVillage dataset, which contains healthy and diseased plant leaf images for various crops, including tomato plants.

##Dataset Source:
https://www.kaggle.com/datasets/arjuntejaswi/plant-village

### Tomato Disease Classes

* Bacterial Spot
* Early Blight
* Late Blight
* Leaf Mold
* Septoria Leaf Spot
* Spider Mites
* Target Spot
* Tomato Yellow Leaf Curl Virus
* Tomato Mosaic Virus
* Healthy

The dataset was preprocessed and used to train a Convolutional Neural Network (CNN) for automated tomato leaf disease classification.



## Features

- Image-based tomato leaf disease classification
- Deep learning model built using TensorFlow and Keras
- Trained on the PlantVillage dataset
- User-friendly web interface
- Real-time disease prediction
- Cloud deployment support

## Dataset

The model is trained using the PlantVillage dataset, which contains thousands of labeled images of healthy and diseased tomato leaves.

Dataset Classes Include:
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Mosaic Virus
- Yellow Leaf Curl Virus
- Healthy

## Tech Stack

### Machine Learning
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

### Deployment
- Google Cloud Functions
- Google Cloud Storage

### Frontend
- HTML
- CSS
- JavaScript

## Model Architecture

The CNN model consists of:

- Conv2D Layers
- MaxPooling Layers

## Project Structure
Tomato-Leaf-Disease-Detection/
│
├── frontend/
│   └── app.html
│
├── model/
│   ├── tomato-2/
│   ├── training-1/
│   └── tomato_leaf_disease_model.keras
│
├── cloud-function/
│   ├── main.py
│   └── requirements.txt
│
├── screenshots/
│
├── README.md
└── .gitignore


## Results

- Training Accuracy: 98.21%
- Validation Accuracy: 97%+
- Efficient disease prediction on unseen leaf images

## Installation

### Clone the Repository
git clone https://github.com/your-username/Tomato-Leaf-Disease-Detection.git
cd Tomato-Leaf-Disease-Detection

##Install Dependencies
pip install -r requirements.txt

##Run the Project
python app.py

Author

Velpuri Thanishk

B.Tech Electronics and Communication Engineering
National Institute of Technology Raipur

License

This project is licensed under the MIT License.


