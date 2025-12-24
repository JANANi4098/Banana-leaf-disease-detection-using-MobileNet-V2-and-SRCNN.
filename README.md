# Banana-leaf-disease-detection-using-MobileNet-V2-and-SRCNN.
Project Description:

This project focuses on the automatic detection of diseases in plant leaves using deep learning. The system classifies leaf images into four categories: Cordana, Healthy, Pestalotiopsis, and Sigatoka.

The workflow involves:

Image Preprocessing: Resize and normalize images.

Feature Extraction: Use a pretrained MobileNetV2 model to extract deep features from images.

Classification: Feed extracted features into a Logistic Regression classifier to predict the disease category.

Visualization: Display predictions and training history graphs for analysis.

The model achieves high accuracy by leveraging pretrained CNN features and combining them with a traditional machine learning classifier for robust performance, even with a small dataset.

Tech Stack Used:

Programming Language: Python

Deep Learning Framework: TensorFlow / Keras

Pretrained Model: MobileNetV2 (ImageNet weights)

Machine Learning: Logistic Regression (scikit-learn)

Image Processing: OpenCV, Pillow

Data Handling & Visualization: NumPy, Matplotlib, tqdm

Model Saving: Pickle, H5 format
