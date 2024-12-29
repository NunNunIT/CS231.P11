from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

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

# Load lại mô hình
loaded_model = tf.keras.models.load_model('./Model/efficient/b0.h5')

# Đảm bảo inference
loaded_model.trainable = False
for layer in loaded_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

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
    print("Processed Image Min/Max:", processed_img.min(), processed_img.max())
    print("Processed Image Shape:", processed_img.shape)
    predictions = model.predict(processed_img)[0]
    print("Predictions:", predictions)
    predicted_labels = [(LABELS[idx], float(prob)) for idx, prob in enumerate(predictions) if prob > threshold]
    print("Labels above threshold:", predicted_labels)
    return predicted_labels

# Example usage:
result = predict_image("./sample_images/test.jpg", loaded_model)
print(result)