import cv2
import numpy as np

from skimage.feature import hog
import joblib
from scipy.sparse import issparse


# Định nghĩa các lớp nhãn cho mô hình đa nhãn
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


def preprocess_image_and_feature_extraction_ml(image):
    RESIZE = (64, 64)
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


def get_predictions_ml(model, features, threshold=0.5):
    probas = convert_to_dense(model.predict_proba([features]))[0]
    prediction = convert_to_dense(model.predict([features]))[0]
    predicted_labels = []
    for idx, proba in enumerate(probas):
        if proba >= threshold:
            predicted_labels.append((LABELS[idx], proba))
    return predicted_labels


def convert_to_dense(probabilities):
    if issparse(probabilities):
        dense_matrix = probabilities.toarray()
    elif isinstance(probabilities, np.ndarray):
        dense_matrix = probabilities
    else:
        raise ValueError("Input must be a scipy.sparse matrix or numpy.ndarray.")

    return dense_matrix