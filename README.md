# 🩻 X-Ray Based Pneumonia Detection System

## 🙂 Overview

This project uses **Deep Learning Approach along with PyTorch** to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**.

The objective is to demonstrate an end-to-end computer vision workflow, from image preprocessing and class imbalance handling to model training, evaluation, and saving trained weights.
 
-----

## 💡 Problem Statement & Objective

Chest X-ray analysis is an important part of pneumonia assessment. This project explores how deep learning can classify chest X-ray images and assist research into automated medical image analysis.

**Goal:** Develop and evaluate a CNN-based binary image classification model that separates normal chest X-rays from images labeled as pneumonia.

-----

## 📂 Project Structure

```text
X-Ray-Pneumonia-Detection/
│
├── model/
│   └── xray_classifier.pth     
│
├── notebook    
│   └── Notebook.ipynb         
│
├── .gitignore
├── app.py
├── LICENSE
├── README.md
└── requirements.txt
```

-----

## 🔗 Dataset

**Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The dataset contains chest X-ray images categorized into two classes:

- **NORMAL:** Images labeled as normal.
- **PNEUMONIA:** Images labeled as showing pneumonia.

The dataset is used to train, validate, and test the image classification model.

-----

## 🔄 Complete Project Workflow

1. **Data Collection:** Loaded the chest X-ray image dataset and its class labels.
2. **Data Preparation:** Organized the images into training, validation, and testing sets using a 70%:15%:15% split.
3. **Image Preprocessing:** Resized images to 128 × 128 pixels, converted them into PyTorch tensors, and normalized pixel values.
4. **Data Augmentation:** Applied random horizontal flipping and rotation to training images to introduce image variation and support generalization.
5. **Class Imbalance Handling:** Used class-weighted cross-entropy loss to account for differences in class representation during model training.
6. **CNN Architecture:** Developed a custom CNN using convolutional layers, ReLU activation, max pooling, fully connected layers, and dropout.
7. **Model Training:** Trained the network using the Adam optimizer and monitored training and validation loss and accuracy.
8. **Model Evaluation:** Evaluated the trained model on the test dataset.
9. **Model Saving:** Saved the trained model weights in `.pth` format for future use and streamlit deployment.

## 🧠 Model Architecture

The project uses a custom **Convolutional Neural Network (CNN)** implemented with PyTorch.

Key components include:

- Convolutional layers for extracting visual features from X-ray images.
- ReLU activation functions for introducing non-linearity.
- Max-pooling layers for reducing spatial dimensions.
- Dropout for helping reduce overfitting.
- Fully connected layers for final binary classification.
- Class-weighted cross-entropy loss for handling class imbalance.

## 📊 Model Performance

The model achieved the following results on the test dataset:

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | **94.31%** |
| Test Loss | **0.1380** |

-----

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

-----

## 🔮 Future Improvements

- Experiment with different CNN architectures to find a suitable model.
- Train the model with more chest X-ray images.
- Try different image sizes to find the best results.
- Display sample X-ray images along with their predicted labels.
- Apply basic MLOps techniques to track experiments & manage model versions.
