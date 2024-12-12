# Color based Image Classification

## Overview
This project implements a deep learning pipeline for image classification using TensorFlow and Keras. It includes a robust preprocessing mechanism, model training, and prediction functionality, targeting the classification of objects into three categories: Black, Transparent, and Colorful objects.

---

## Dataset

The dataset consists of 80 images per category, representing three types of objects:
- Black
- Transparent
- Colorful

### Features of the Dataset:
1. Diverse lighting conditions and backgrounds (white and brown) to enhance generalization.
2. Designed to improve robustness in real-world scenarios.

---

## Project Features

### 1. Preprocessing
A custom preprocessing function is implemented to enhance the image quality and adapt images for training.

#### Steps:
- **Color Space Conversion**: Converts images from RGB to HSV for better feature extraction.
- **Black Image Detection**: Applies edge detection if the image is predominantly black.
- **Image Adjustment**: Modifies brightness and saturation based on thresholds.
- **Final Output**: Either a processed black image or an adjusted color image.

#### Image Pipeline:
- Resizes images to a maximum of 224x224 pixels.
- Saves processed images to a designated directory.

---

### 2. Data Augmentation
Image augmentation is used to improve model generalization.
- **Rescaling**: Scales pixel values between 0 and 1.
- **Validation Split**: Uses 20% of the data for validation.

---

### 3. Model Architecture
A Convolutional Neural Network (CNN) is constructed with the following features:
- **Convolutional Layers**: For feature extraction.
- **Depthwise Convolution**: To enhance spatial features.
- **Max Pooling Layers**: For dimensionality reduction.
- **Fully Connected Layer**: With dropout for regularization.
- **Output Layer**: A softmax layer for multi-class classification.

#### Model Compilation:
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metric: Accuracy

---

### 4. Training and Evaluation
- **Early Stopping**: Stops training when validation loss doesn’t improve for three consecutive epochs.
- **Steps Per Epoch**: Calculated based on the dataset size and batch size.
- **Results**: Achieved 72% accuracy during training.

---

### 5. Object Classification
The trained model predicts the category of objects and directs them to appropriate conveyor belts.

#### Mapping:
- **0**: Black Object → Conveyor Belt A
- **1**: Colorful Object → Conveyor Belt C
- **2**: Transparent Object → Conveyor Belt B

#### Prediction Function:
The function preprocesses a given image, uses the model to classify it, and maps the result to a conveyor belt label.

---

## Usage

### 1. Preprocessing Images
```python
# Example for preprocessing
custom_preprocess(img_array)
```

### 2. Training the Model
```python
# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stopping]
)
```

### 3. Making Predictions
```python
# Predict the object’s category
conveyor_belt = predict_conveyor_belt(image_path)
print(f'The object should be sent to Conveyor Belt: {conveyor_belt}')
```

---

## Libraries Used
- TensorFlow/Keras
- OpenCV
- NumPy
- TQDM

---

## Results
The model successfully classifies objects with the following accuracy levels:
- **Overall Accuracy**: 72%

**Examples:**
1. Correctly identified Black, Transparent, and Colorful objects.
2. Noted cases of misclassification.

---

## Future Enhancements
1. Increase dataset size for improved performance.
2. Experiment with advanced augmentation techniques.
3. Optimize hyperparameters for better accuracy.

---

## Contributors
- Muneeza Iftikhar
- Hafsa Hafeez Siddiqui


---

## Acknowledgements
This project was developed as part of the Deep Learning (AIC-401) course assignment at Bahria University Karachi Campus, Fall 2024. Special thanks to instructor Reema Qaiser Khan.

