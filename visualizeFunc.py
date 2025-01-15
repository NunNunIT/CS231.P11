from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

from skimage.feature import hog
import joblib
from scipy.sparse import issparse


NUM_CLASSES = 8
NUM_EPOCH = 100
INPUT_SHAPE = (224, 224, 3)
IMG_SIZE = (224, 224)
BATCH_SIZE = 128
LABELS = [
    "shirt, blouse",
    "top, t-shirt, sweatshirt",
    "jacket",
    "pants",
    "skirt",
    "dress",
    "shoe",
    "bag, wallet"
]


def process_image(image):
    if isinstance(image, str):  # Đường dẫn ảnh
        img_array = cv2.imread(image, cv2.IMREAD_COLOR)
    else:  # Đối tượng PIL.Image
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Chuyển từ RGB sang BGR
    img_resized = cv2.resize(img_array, (224, 224))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded


def predict_image(image, model, threshold=0.5):
    """Predict labels and probabilities for the given image."""
    processed_img = process_image(image)
    predictions = model.predict(processed_img)[0]
    predicted_labels = [(LABELS[idx], float(prob)) for idx, prob in enumerate(predictions) if prob > threshold]
    return predicted_labels


def preprocess_image_and_feature_extraction_ml(image):
    RESIZE = (128, 128)
    FEATURE = 'hog'
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, RESIZE)
    features = None
    if FEATURE == 'hog':
        features, _ = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True,
        )
    
    return features


def convert_to_dense(matrix):
    if issparse(matrix):
        dense_matrix = matrix.toarray()
    elif isinstance(matrix, np.ndarray):
        dense_matrix = matrix
    else:
        raise ValueError("Input must be a scipy.sparse matrix or numpy.ndarray.")

    return dense_matrix


def predict_image_from_ml(image, model, threshold=0.5):
    features = preprocess_image_and_feature_extraction_ml(image)
    probas = convert_to_dense(model.predict_proba([features]))[0]
    prediction = convert_to_dense(model.predict([features]))[0]
    predicted_labels = []
    for idx, proba in enumerate(probas):
        if proba >= threshold:
            predicted_labels.append((LABELS[idx], proba))
    return predicted_labels
