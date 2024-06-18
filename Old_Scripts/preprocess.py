import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import requests

# Download and extract the dataset (example code, adjust paths as needed)
def download_and_extract_dataset(url, extract_to='.'):
    response = requests.get(url)
    zip_path = os.path.join(extract_to, 'boat_dataset.zip')
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

dataset_url = "https://www.kaggle.com/datasets/kunalgupta2616/boat-types-recognition/download"
download_and_extract_dataset(dataset_url, './data')

# Assuming the dataset is extracted to ./data/boats
# Load the dataset annotations
annotations = pd.read_csv('./data/boats/annotations.csv')

# Preprocess the dataset (example preprocessing)
def preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0
    return image

# Split the dataset
train_annotations, val_annotations = train_test_split(annotations, test_size=0.2, random_state=42)

# Example data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_annotations,
    directory='./data/boats/images',
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
)

val_generator = val_datagen.flow_from_dataframe(
    val_annotations,
    directory='./data/boats/images',
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
)
