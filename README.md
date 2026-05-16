# 🧬 Malaria Detection using Deep Learning
## 📌 Overview

This project implements a Deep Learning-based system to detect malaria infection from microscopic blood smear images.

The model classifies images into two categories:

Parasitized

Uninfected

Multiple architectures were trained and compared. The final deployed model uses MobileNetV2 (Fine-Tuned Transfer Learning) and is integrated into a Flask web application for real-time prediction.

## 🎯 Problem Statement

Traditional malaria diagnosis relies on manual microscopic examination of blood smear images. This process:

Requires trained medical experts

Is time-consuming

Can be inaccessible in remote areas

May be prone to human error

This project demonstrates how Deep Learning can assist in automating malaria detection and supporting faster preliminary screening.

## 📂 Dataset

Dataset: NIH Malaria Cell Images

Total Images: ~27,000

Classes: 2 (Parasitized, Uninfected)

Train–Validation Split: 80% Training, 20% Validation

Image Size Used: 96 × 96 pixels

## 🔹 Preprocessing

Images resized to 96×96

Pixel values normalized to range [0, 1]

## 🧠 Model Development

Three different models were trained and evaluated:

**1️⃣ Custom CNN**

Validation Accuracy: ~94%

Architecture:

Convolution Layers

ReLU Activation

MaxPooling

Flatten

Dense Layers

Sigmoid Output

**2️⃣ MobileNetV2 (Transfer Learning – Selected Model)**

Validation Accuracy: ~93%

Why selected?

Pretrained on ImageNet

Better generalization

Fine-tuned final layers

Scalable architecture

**3️⃣ EfficientNetB0**

Validation Accuracy: ~50%

Underperformed at 96×96 resolution and was not selected for deployment.

## ⚙️ Training Configuration

Optimizer: Adam

Loss Function: Binary Crossentropy

Epochs: 10

EarlyStopping used

ReduceLROnPlateau used

ModelCheckpoint used

Training Accuracy: ~98%
Validation Accuracy: ~93–94%

The small gap between training and validation accuracy indicates good generalization with minimal overfitting.

## 🌐 Web Application (Flask Deployment)

The trained model is deployed using Flask.

## ✨ Features

Upload microscopic blood cell image

Live image preview before analysis

Real-time prediction

Confidence score display

Class probability breakdown

Processing time measurement

Uploaded image preview in report

Professional dark-themed UI

Medical disclaimer included

## 🔄 Application Workflow

User uploads a blood smear image

Image is resized and normalized

Model performs binary classification

Prediction probabilities are calculated

## Result page displays:

Prediction

Confidence level

Class probabilities

Processing time

Uploaded image preview

Model name

## 📁 Project Structure
MalariaProject/
│
├── app.py
├── train.py
├── model_comparison.py
├── best_model.keras
├── requirements.txt
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    └── uploads/
## 🛠 Installation & Setup
**1️⃣ Create Environment**
conda create -n malaria python=3.10
conda activate malaria
**2️⃣ Install Dependencies**
pip install -r requirements.txt
**3️⃣ Run Application**
python app.py

Open in browser:

**http://127.0.0.1:5000**
## 📊 Results

Validation Accuracy: ~93–94%

Training Accuracy: ~98%

Confusion Matrix generated

Classification report generated

The model demonstrates strong performance with minimal overfitting.

## ⚠️ Limitations

Not a clinical diagnostic tool

Performance depends on image quality

Binary classification only

EfficientNet performance limited due to reduced input resolution

## 📜 Disclaimer

**This project is developed for academic and educational purposes only.
It is not intended for real-world clinical or medical diagnosis.**
