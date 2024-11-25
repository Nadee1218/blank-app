# Bee and Wax Moth Classification using TensorFlow

## Project Overview
This project involves building a machine learning model using TensorFlow to classify images of bees and wax moths. The model utilizes a pre-trained MobileNetV2 model with custom layers added on top for the classification task. The goal is to detect and classify images based on two categories: Bees and Wax Moths, which can be useful in bee colony management systems.

## Dataset
The dataset used in this project consists of images of bees and wax moths, which are stored in a directory structure where each class is in its respective folder. The dataset is split into training and validation sets using an 80-20 split, with 20% of the data used for validation.

## Prerequisites
To run this project, you need the following libraries:

- TensorFlow
- NumPy
- Matplotlib
- OpenCV

You can install the required libraries by running the following command:

```bash
pip install -r requirements.txt
tensorflow
numpy
matplotlib
opencv-python

from google.colab import drive
drive.mount('/content/drive')

/content/drive/MyDrive/dataset

history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

model.save('my_model.keras')

loaded_model = tf.keras.models.load_model('my_model.keras')

