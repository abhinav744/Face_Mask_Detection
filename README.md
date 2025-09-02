# 😷 Face Mask Detection

A computer vision project that detects whether individuals are wearing face masks, using deep learning techniques.

[Face Mask Detection App](https://xrmbbqmvm3nhk8dw7cocbs.streamlit.app/)

## Project Overview

Objective: Identify and classify faces as wearing a mask or not, supporting real-time detection applications.

Approach: Implemented using Convolutional Neural Networks (CNNs) for high accuracy in image classification and detection.

Use Cases: Ideal for surveillance systems, access control, and health-compliance monitoring in public areas.

## Repository Structure

/Face_Mask_Detection

│── data/                    # Training/testing image sets

│── train_model.ipynb        # Notebook for EDA and model training

│── app.py                   # Streamlit app for live detection

│── requirements.txt         # Dependencies

│── README.md                # Project documentation

## Getting Started

### 1. Clone the repository

git clone https://github.com/abhinav744/Face_Mask_Detection.git

cd Face_Mask_Detection

### 2. (Optional) Set up a virtual environment

python -m venv venv

source venv/bin/activate     # On Windows: venv\Scripts\activate

### 3. Install necessary packages

pip install -r requirements.txt

### 4. Train & Evaluate the Model

Open and run train_model.ipynb to train your mask detection model and evaluate its accuracy.

### 5. Run the Streamlit app

streamlit run app.py

## Future Enhancements

Implement data augmentation (rotations, scaling, lighting variations)

Fine-tune pretrained CNN models (e.g., ResNet, MobileNet)

Add deployment via Flask API for real-time video processing

Improve robustness and fairness using bias-aware datasets
